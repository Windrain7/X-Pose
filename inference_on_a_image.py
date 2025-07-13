import argparse
import os
import sys

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transforms as T
from matplotlib import transforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from models import build_model
from PIL import Image, ImageDraw, ImageFont
from predefined_keypoints import *
from torchvision.ops import nms
from util import box_ops
from util.config import Config
from util.utils import clean_state_dict

# def text_encoding(instance_names, keypoints_names, model, device):
#     ins_text_embeddings = []
#     for cat in instance_names:
#         instance_description = f'a photo of {cat.lower().replace("_", " ").replace("-", " ")}'
#         text = clip.tokenize(instance_description).to(device)
#         text_features = model.encode_text(text)  # 1*512
#         ins_text_embeddings.append(text_features)
#     ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

#     kpt_text_embeddings = []

#     for kpt in keypoints_names:
#         kpt_description = f'a photo of {kpt.lower().replace("_", " ")}'
#         text = clip.tokenize(kpt_description).to(device)
#         with torch.no_grad():
#             text_features = model.encode_text(text)  # 1*512
#         kpt_text_embeddings.append(text_features)

#     kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)

#     return ins_text_embeddings, kpt_text_embeddings


def target_encoding(instance_text_prompt, keypoint_text_prompt, model, device):
    target = {}
    target['keypoints_skeleton_list'] = []

    instance_names = instance_text_prompt.split(',')
    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = f'a photo of {cat.lower().replace("_", " ").replace("-", " ")}'
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)  # 1*512
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)  # [prompts, 512]
    target['instance_text_prompt'] = instance_names
    target['object_embeddings_text'] = ins_text_embeddings.float()

    keypoint_text_examples = keypoint_text_prompt.split(',')
    kpt_text_embeddings = []  # [prompts, 100, 512], pad to 100
    kpt_vis_text = []  # [prompts, 100]
    for kpt_text_example in keypoint_text_examples:
        kpt_names = globals()[kpt_text_example]['keypoints']
        target['keypoints_skeleton_list'].append(globals()[kpt_text_example]['skeleton'])
        text_embeddings = []
        for kpt in kpt_names:
            kpt_description = f'a photo of {kpt.lower().replace("_", " ")}'
            text = clip.tokenize(kpt_description).to(device)
            text_features = model.encode_text(text)  # 1*512
            text_embeddings.append(text_features)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        kpt_vis = torch.ones(text_embeddings.shape[0], device=device)
        kpt_text_embeddings.append(F.pad(text_embeddings, (0, 0, 0, 100 - text_embeddings.shape[0])))
        kpt_vis_text.append(F.pad(kpt_vis, (0, 100 - kpt_vis.shape[0])))
    kpt_text_embeddings = torch.stack(kpt_text_embeddings, dim=0)  # [prompts, 100, 512]
    kpt_vis_text = torch.stack(kpt_vis_text, dim=0)  # [prompts, 100]
    target['keypoint_text_example'] = keypoint_text_examples
    target['kpts_embeddings_text'] = kpt_text_embeddings.float()
    target['kpt_vis_text'] = kpt_vis_text

    return target


def plot_on_image(image_pil, tgt, keypoint_skeletons_list, kpt_vis_text, output_dir, out_fp=None, instance_text_prompt=None):
    H, W = tgt['size']
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect='equal')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    color_kpt = [
        [0.53, 0.81, 0.92],
        [0.82, 0.71, 0.55],
        [1.00, 0.39, 0.28],
        [0.53, 0.81, 0.92],
        [0.50, 0.16, 0.16],
        [0.00, 0.00, 1.00],
        [0.69, 0.88, 0.90],
        [0.00, 1.00, 0.00],
        [0.63, 0.13, 0.94],
        [0.82, 0.71, 0.55],
        [1.00, 0.38, 0.00],
        [0.53, 0.15, 0.34],
        [1.00, 0.39, 0.28],
        [1.00, 0.00, 1.00],
        [0.04, 0.09, 0.27],
        [0.20, 0.63, 0.79],
        [0.94, 0.90, 0.55],
        [0.33, 0.42, 0.18],
        [0.53, 0.81, 0.92],
        [0.71, 0.49, 0.86],
        [0.25, 0.88, 0.82],
        [0.5, 0.0, 0.0],
        [0.0, 0.3, 0.3],
        [1.0, 0.85, 0.73],
        [0.29, 0.0, 0.51],
        [0.7, 0.5, 0.35],
        [0.44, 0.5, 0.56],
        [0.25, 0.41, 0.88],
        [0.0, 0.5, 0.0],
        [0.56, 0.27, 0.52],
        [1.0, 0.84, 0.0],
        [1.0, 0.5, 0.31],
        [0.85, 0.57, 0.94],
        [0.00, 0.00, 0.00],
        [1.00, 1.00, 1.00],
        [1.00, 0.00, 0.00],
        [1.00, 1.00, 0.00],
        [0.50, 0.16, 0.16],
        [0.00, 0.00, 1.00],
        [0.69, 0.88, 0.90],
        [0.00, 1.00, 0.00],
        [0.63, 0.13, 0.94],
        [0.82, 0.71, 0.55],
        [1.00, 0.38, 0.00],
        [0.53, 0.15, 0.34],
        [1.00, 0.39, 0.28],
        [1.00, 0.00, 1.00],
        [0.04, 0.09, 0.27],
        [0.20, 0.63, 0.79],
        [0.94, 0.90, 0.55],
        [0.33, 0.42, 0.18],
        [0.53, 0.81, 0.92],
        [0.71, 0.49, 0.86],
        [0.25, 0.88, 0.82],
        [0.5, 0.0, 0.0],
        [0.0, 0.3, 0.3],
        [1.0, 0.85, 0.73],
        [0.29, 0.0, 0.51],
        [0.7, 0.5, 0.35],
        [0.44, 0.5, 0.56],
        [0.25, 0.41, 0.88],
        [0.0, 0.5, 0.0],
        [0.56, 0.27, 0.52],
        [1.0, 0.84, 0.0],
        [1.0, 0.5, 0.31],
        [0.85, 0.57, 0.94],
    ]
    color = []
    # 根据label_ids选择颜色
    # color_box = color_kpt[tgt['label_ids'].cpu()]
    # color_box = [0.53, 0.81, 0.92]
    polygons = []
    boxes = []

    # 如果提供了instance_text_prompt，将其拆分为列表
    instance_names = None
    if instance_text_prompt is not None:
        if isinstance(instance_text_prompt, str):
            instance_names = instance_text_prompt.split(',')
        else:
            instance_names = instance_text_prompt

    for box, label_id in zip(tgt['boxes'].cpu(), tgt['label_ids'].cpu()):
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2  # cswh to xywh
        # clip bbox
        bbox_x = max(0, min(unnormbbox[0], W))
        bbox_y = max(0, min(unnormbbox[1], H))
        bbox_w = max(0, min(unnormbbox[2], W - bbox_x))
        bbox_h = max(0, min(unnormbbox[3], H - bbox_y))
        boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(color_kpt[label_id])

        # 在边界框框内添加文本标签
        if instance_names is not None and label_id < len(instance_names):
            label_text = instance_names[label_id]
            plt.text(
                bbox_x,
                bbox_y + 10,
                label_text,
                color=color_kpt[label_id],
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5),
            )

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', linestyle='--', edgecolors=color, linewidths=1.5)
    ax.add_collection(p)

    if 'keypoints' in tgt:
        for idx, (keypoint, label_id) in enumerate(zip(tgt['keypoints'], tgt['label_ids'])):
            num_kpts = int(kpt_vis_text[label_id].sum())
            sks = np.array(keypoint_skeletons_list[label_id])
            # import pdb;pdb.set_trace()
            if sks.shape[0] != 0:
                if sks.min() == 1:
                    sks = sks - 1

            kp = np.array(keypoint.cpu())
            Z = kp[: num_kpts * 2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            if len(color) > 0:
                c = color[idx % len(color)]
            else:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=1, color=c)

            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                plt.plot(x[i], y[i], 'o', markersize=4, markerfacecolor=c_kpt, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_axis_off()
    savename = os.path.join(output_dir, out_fp)
    print(f'savename: {savename}')
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    plt.savefig(savename, dpi=dpi)
    plt.close()


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert('RGB')  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = 'cuda' if not cpu_only else 'cpu'
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_unipose_output(
    model, image, instance_text_prompt, keypoint_text_example, box_threshold, iou_threshold, with_logits=True, cpu_only=False, select_min=1
):
    # instance_text_prompt: A, B, C, ...
    # keypoint_text_prompt: skeleton

    device = 'cuda' if not cpu_only else 'cpu'

    # clip_model, _ = clip.load("ViT-B/32", device=device)
    target = target_encoding(instance_text_prompt, keypoint_text_example, model.clip_model, device)
    # import pdb;pdb.set_trace()
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        # 每个image对应一个target
        outputs = model(image[None], [target])

    logits = outputs['pred_logits'].sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'][0]  # (nq, 4)
    keypoints = outputs['pred_keypoints'][0]  # (nq, 68 * 2)
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    keypoints_filt = keypoints.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    if filt_mask.sum() >= select_min:
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        keypoints_filt = keypoints_filt[filt_mask]  # num_filt, 4
    else:
        # 选取分数最大的select_min个
        _, indices = logits_filt.max(dim=1)[0].topk(select_min, dim=0)
        logits_filt = logits_filt[indices]
        boxes_filt = boxes_filt[indices]
        keypoints_filt = keypoints_filt[indices]
    keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)

    # Use keep_indices to filter boxes and keypoints
    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]
    filtered_logits = logits_filt[keep_indices]
    filtered_label_ids = torch.argmax(filtered_logits, dim=1)

    return filtered_boxes, filtered_keypoints, filtered_logits, filtered_label_ids, target['kpt_vis_text'], target['keypoints_skeleton_list']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniPose Inference', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--image_path', '-i', type=str, required=True, help='path to image file')
    parser.add_argument('--instance_text_prompt', '-t', type=str, required=True, help='instance text prompt')
    parser.add_argument('--keypoint_text_example', '-k', type=str, default=None, help='keypoint text prompt')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs', required=True, help='output directory')

    parser.add_argument('--box_threshold', type=float, default=0.1, help='box threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.9, help='box threshold')
    parser.add_argument('--cpu-only', action='store_true', help='running on cpu only!, default=False')
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    instance_text_prompt = args.instance_text_prompt
    keypoint_text_example = args.keypoint_text_example
    instance_list = instance_text_prompt

    output_dir = args.output_dir

    box_threshold = args.box_threshold

    iou_threshold = args.iou_threshold

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(image_path):
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    else:
        image_paths = [image_path]

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    for image_path in image_paths:
        print(f'Processing {image_path}...')
        # load image
        image_pil, image = load_image(image_path)

        # visualize raw image
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        # image_pil.save(os.path.join(output_dir, f'{name}_raw{ext}'))

        # run model
        boxes_filt, keypoints_filt, logits_filt, label_ids_filt, kpt_vis_text, keypoint_skeletons_list = get_unipose_output(
            model, image, instance_text_prompt, keypoint_text_example, box_threshold, iou_threshold, cpu_only=args.cpu_only
        )
        # visualize pred
        size = image_pil.size
        pred_dict = {'boxes': boxes_filt, 'keypoints': keypoints_filt, 'logits': logits_filt, 'label_ids': label_ids_filt, 'size': [size[1], size[0]]}
        # import ipdb; ipdb.set_trace()
        plot_on_image(
            image_pil,
            pred_dict,
            keypoint_skeletons_list,
            kpt_vis_text,
            output_dir,
            out_fp=f'{name}_{{{instance_text_prompt}}}{ext}',
            instance_text_prompt=instance_list,
        )
