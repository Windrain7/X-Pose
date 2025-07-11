# ------------------------------------------------------------------------
# UniPose
# url: https://github.com/IDEA-Research/UniPose
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# ED-Pose
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import math
import os
import random
from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ops.modules import MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import MLP, _get_activation_fn, gen_encoder_output_proposals, gen_sineembed_for_position, get_sine_pos_embed


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation='relu',
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        modulate_hw_attn=False,
        # for deformable encoder
        deformable_encoder=False,
        deformable_decoder=False,
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        use_deformable_box_attn=False,
        box_attn_type='roi_align',
        # init query
        learnable_tgt_init=False,
        decoder_query_perturber=None,
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,
        # two stage
        two_stage_type='no',
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        # evo of #anchors
        dec_layer_number=None,
        rm_enc_query_scale=True,
        rm_dec_query_scale=True,
        rm_self_attn_layers=None,
        key_aware_type=None,
        # layer share
        layer_share_type=None,
        # for detach
        rm_detach=None,
        decoder_sa_type='ca',
        module_seq=['sa', 'ca', 'ffn'],
        # for dn
        embed_init_tgt=False,
        use_detached_boxes_dec_out=False,
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
        binary_query_selection=False,
        ffn_extra_layernorm=False,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.ffn_extra_layernorm = ffn_extra_layernorm
        assert query_dim == 4

        self.binary_query_selection = binary_query_selection
        if self.binary_query_selection:
            self.binary_query_selection_layer = nn.Linear(d_model, 1)
        # assert not binary_query_selection, 'binary_query_selection not implemented yet'

        if num_feature_levels > 1:
            assert deformable_encoder, 'only support deformable_encoder for num_feature_levels > 1'
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                nhead,
                enc_n_points,
                add_channel_attention=add_channel_attention,
                use_deformable_box_attn=use_deformable_box_attn,
                box_attn_type=box_attn_type,
            )
        else:
            raise NotImplementedError

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model, nhead=nhead // 2, dim_feedforward=dim_feedforward // 2, dropout=text_dropout
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model, l_dim=d_model, embed_dim=dim_feedforward // 2, num_heads=nhead // 2, dropout=fusion_dropout, drop_path=fusion_droppath
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            enc_layer_share=enc_layer_share,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                nhead,
                dec_n_points,
                use_text_cross_attention=use_text_cross_attention,
                ffn_extra_layernorm=ffn_extra_layernorm,
            )

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=rm_dec_query_scale,
            dec_layer_share=dec_layer_share,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning(f'num_patterns should be int but {type(num_patterns)}')
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, 'why not learnable_tgt_init'
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], f'unknown param {two_stage_type} of two_stage_type'
        if two_stage_type == 'standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries, f'dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})'
            else:
                assert (
                    dec_layer_number[0] == num_queries * num_patterns
                ), f'dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})'

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            # assert len(rm_self_attn_layers) == num_decoder_layers
            print(f'Removing the self-attn in {rm_self_attn_layers} decoder layers')
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(
        self,
        srcs,
        masks,
        refpoint_embed,
        pos_embeds,
        tgt,
        attn_mask=None,
        attn_mask2=None,
        text_dict=None,
        dn_meta=None,
        targets=None,
        kpt_embed=None,
    ):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # if self.two_stage_type != 'no' and self.two_stage_add_query_num == 0:
        #     assert refpoint_embed is None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict['encoded_text'],
            text_attention_mask=~text_dict['text_token_mask'],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['text_self_attention_masks'],
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict['encoded_text'] = memory_text

        if self.two_stage_type == 'standard':
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            if self.binary_query_selection:
                topk_logits = self.binary_query_selection_layer(output_memory).squeeze(-1)
            else:
                if text_dict is not None:
                    enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
                else:
                    enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

                topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1)  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(f'unknown two_stage_type {self.two_stage_type}')
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if refpoint_embed.isnan().any() | refpoint_embed.isinf().any():
        #         import ipdb; ipdb.set_trace()
        #     if tgt.isnan().any() | tgt.isinf().any():
        #         import ipdb; ipdb.set_trace()

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            tgt_mask2=attn_mask2,
            memory_text=text_dict['encoded_text'],
            text_attention_mask=~text_dict['text_token_mask'],
            text_dict=text_dict,
            dn_meta=dn_meta,
            targets=targets,
            kpt_embed=kpt_embed,
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
                # import ipdb; ipdb.set_trace()
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(text_enhance_layer, num_layers, layer_share=enc_layer_share)
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(feature_fusion_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = torch.arange(n_text, device=memory_text.device).float().unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1)
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(position_ids[..., None], num_pos_feats=256, exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id], output, memory_text, key_padding_mask, text_attention_mask
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output, l=memory_text, attention_mask_v=key_padding_mask, attention_mask_l=text_attention_mask
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(layer, output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask)
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        modulate_hw_attn=False,
        num_feature_levels=1,
        deformable_decoder=False,
        decoder_query_perturber=None,
        dec_layer_number=None,  # number of queries each layer in decoder
        rm_dec_query_scale=False,
        dec_layer_share=False,
        dec_layer_dropout_prob=None,
        use_detached_boxes_dec_out=False,
        num_box_decoder_layers=2,
        num_body_points=68,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, 'support return_intermediate only'
        self.query_dim = query_dim
        assert query_dim in [2, 4], f'query_dim should be 2/4 but {query_dim}'
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.pose_embed = None
        self.pose_hw_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None
        self.num_body_points = num_body_points

        self.hw = nn.Embedding(17, 2)
        self.num_box_decoder_layers = num_box_decoder_layers
        self.kpt_index = [x for x in range(50 * (self.num_body_points + 1)) if x % (self.num_body_points + 1) != 0]
        self.hw_append = nn.Embedding(self.num_body_points - 17, 2)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        tgt_mask2: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_dict: Optional[Tensor] = None,
        dn_meta: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        kpt_embed: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """

        output = tgt
        output += self.hw.weight[0, 0] * 0.0

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        effect_num_dn = dn_meta['pad_size'] if self.training else 0
        inter_select_number = 50
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256, from MLP
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f'output layer_id {layer_id} is nan')
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f'num_nan {num_nan}, num_inf {num_inf}')
                except Exception as e:
                    print(e)

            intermediate.append(self.norm(output))
            # iter update
            if layer_id < self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

            # select # ref points as anchors, object query selection, kpt query selection
            if layer_id == self.num_box_decoder_layers - 1:
                dn_output = output[:effect_num_dn]
                dn_new_reference_points = new_reference_points[:effect_num_dn]
                class_unselected = self.class_embed[layer_id](output.transpose(0, 1), text_dict)[:, effect_num_dn:].transpose(0, 1)
                class_logits, class_ids = class_unselected.max(-1)
                _, topk_proposals = torch.topk(class_logits, inter_select_number, dim=0)  # 50个分数最高的下标

                # 选50个参考点，object qurey, class id
                new_reference_points_for_box = torch.gather(new_reference_points[effect_num_dn:], 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                new_output_for_box = torch.gather(output[effect_num_dn:], 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # 50, bs, 256
                topk_class_ids = class_ids.gather(0, topk_proposals)  # 50, bs

                keypoint_embed = kpt_embed.transpose(0, 1)  # bs, instance, 68, 256 -> instance, bs, 68, 256， 模型根据提示词所生成
                indices = topk_class_ids[..., None, None].expand(-1, -1, *keypoint_embed.shape[2:])  # 50, bs, 68, 256
                new_output_for_keypoint = keypoint_embed.gather(0, indices)  # 50, bs, 68, 256， 根据id选择对应的kpt query
                new_output_for_keypoint = new_output_for_keypoint.transpose(1, 2)  # 50, 68, bs, 256
                # new_output_for_keypoint = keypoint_embed[None, :, :, :].repeat(new_output_for_box.shape[0],1,1,1)

                delta_xy = self.pose_embed[-1](new_output_for_keypoint)[..., :2]
                keypoint_xy = (inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy).sigmoid()
                num_queries, _, bs, _ = keypoint_xy.shape
                aa = torch.cat((self.hw.weight, self.hw_append.weight), dim=0)
                keypoint_wh_weight = aa.unsqueeze(0).unsqueeze(-2).repeat(num_queries, 1, bs, 1).sigmoid()
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1)
                new_reference_points = torch.cat((new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1).flatten(0, 1)
                output = torch.cat((new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1).flatten(0, 1)
                new_reference_points = torch.cat((dn_new_reference_points, new_reference_points), dim=0)
                output = torch.cat((dn_output, output), dim=0)
                tgt_mask = tgt_mask2

            if layer_id >= self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_dn = output[:effect_num_dn]
                output_bbox_norm = output[effect_num_dn:][0 :: (self.num_body_points + 1)]
                reference_before_sigmoid_bbox_dn = reference_before_sigmoid[:effect_num_dn]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[effect_num_dn:][0 :: (self.num_body_points + 1)]
                delta_unsig_dn = self.bbox_embed[layer_id](output_bbox_dn)
                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_dn = delta_unsig_dn + reference_before_sigmoid_bbox_dn
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_dn = outputs_unsig_dn.sigmoid()
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid()
                output_kpt = output[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index, device=output.device))
                delta_xy_unsig = self.pose_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig = (
                    reference_before_sigmoid[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index, device=output.device)).clone()
                )  ##
                delta_hw_unsig = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid()
                bs = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat(
                    (new_reference_points_for_box_norm.unsqueeze(1), new_reference_points_for_keypoint.view(-1, self.num_body_points, bs, 4)), dim=1
                ).flatten(0, 1)
                new_reference_points = torch.cat((new_reference_points_for_box_dn, new_reference_points_norm), dim=0)

            if self.rm_detach and 'dec' in self.rm_detach:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()

            # if layer_id != self.num_layers - 1:
            if self.use_detached_boxes_dec_out:
                ref_points.append(reference_points)
            else:
                ref_points.append(new_reference_points)

        return [[itm_out.transpose(0, 1) for itm_out in intermediate], [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
        add_channel_attention=False,
        use_deformable_box_attn=False,
        box_attn_type='roi_align',
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
        ffn_extra_layernorm=False,
    ):
        super().__init__()

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)
        if ffn_extra_layernorm:
            raise NotImplementedError('ffn_extra_layernorm not implemented')
            self.norm_ext = nn.LayerNorm(d_ffn)
        else:
            self.norm_ext = None

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt, ipdb_flag=False):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4``
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if tgt.isnan().any() | tgt.isinf().any() :
            #         import ipdb; ipdb.set_trace()

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()

            # if tgt.isnan().any() | tgt.isinf().any() :
            #     import ipdb; ipdb.set_trace()

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     tgtk = tgt.clone()
        #     if tgt.isnan().any() | tgt.isinf().any() :
        #         import ipdb; ipdb.set_trace()

        # ffn
        tgt = self.forward_ffn(tgt)
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if tgt.isnan().any() | tgt.isinf().any() :
        #         tgtk = self.forward_ffn(tgtk, ipdb_flag=True)
        #         import ipdb; ipdb.set_trace()

        return tgt


def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber

        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise, w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise
        )

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    binary_query_selection = False
    try:
        binary_query_selection = args.binary_query_selection
    except:
        binary_query_selection = False

    ffn_extra_layernorm = False
    try:
        ffn_extra_layernorm = args.ffn_extra_layernorm
    except:
        print('ffn_extra_layernorm not found, set to False')
        ffn_extra_layernorm = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,
        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,
        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,
        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,
        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,
        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
        binary_query_selection=binary_query_selection,
        ffn_extra_layernorm=ffn_extra_layernorm,
    )
