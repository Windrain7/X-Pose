import argparse
import json
import os
import pickle

import torch
from inference_on_a_image import load_image, load_model, plot_on_image, text_encoding
from PIL import Image
from predefined_keypoints import *
from torchvision.ops.boxes import nms
from tqdm import tqdm
from util import box_ops


def pos_process(outputs, box_threshold, iou_threshold, select_min=1):
    # Filter boxes
    logits = outputs['pred_logits'].sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'][0]  # (nq, 4)
    keypoints = outputs['pred_keypoints'][0][:, : 2 * len(keypoint_text_prompt)]  # (nq, n_kpts * 2)
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

    return filtered_boxes, filtered_keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniPose COCO Inference', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--coco_path', '-i', type=str, required=True, help='path to COCO format dataset')
    parser.add_argument('--img_root', '-d', type=str, required=True, help='path to image root directory')
    parser.add_argument('--instance_text_prompt', '-t', type=str, required=True, help='instance text prompt')
    parser.add_argument('--keypoint_text_example', '-k', type=str, default=None, help='keypoint text prompt')
    parser.add_argument('--outfp', '-o', type=str, default='outputs', required=True, help='output file path')
    parser.add_argument('--box_threshold', type=float, default=0.1, help='box threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.9, help='box threshold')
    parser.add_argument('--cpu-only', action='store_true', help='running on cpu only!, default=False')
    parser.add_argument('--draw', action='store_true', help='draw keypoints on images, default=False')
    args = parser.parse_args()

    instance_text_prompt = args.instance_text_prompt
    instance_list = instance_text_prompt.split(',')

    if args.keypoint_text_example in globals():
        keypoint_dict = globals()[args.keypoint_text_example]
        keypoint_text_prompt = keypoint_dict.get('keypoints')
        keypoint_skeleton = keypoint_dict.get('skeleton')
    elif args.instance_text_prompt in globals():
        keypoint_dict = globals()[instance_list[0]]
        keypoint_text_prompt = keypoint_dict.get('keypoints')
        keypoint_skeleton = keypoint_dict.get('skeleton')
    else:
        raise ValueError('Invalid keypoint_text_example or instance_text_prompt')

    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)
    device = 'cpu' if args.cpu_only else 'cuda'
    model = model.to(device)

    target = {}
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)
    target['instance_text_prompt'] = instance_list
    target['keypoint_text_prompt'] = keypoint_text_prompt
    target['object_embeddings_text'] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=device)
    target['kpts_embeddings_text'] = torch.cat((kpt_text_embeddings, kpts_embeddings_text_pad), dim=0)
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0], device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0], device=device)
    target['kpt_vis_text'] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)

    with open(args.coco_path) as f:
        coco_data = json.load(f)

    outfp = args.outfp
    predictions = []
    annotations = []

    for image_info in tqdm(coco_data['images'], desc='Processing images'):
        image_path = os.path.join(args.img_root, image_info['file_name'])
        image_pil, image = load_image(image_path)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image[None], [target])

        predictions.append(outputs)

        boxes_filt, keypoints_filt = pos_process(outputs, args.box_threshold, args.iou_threshold)

        if args.draw:
            size = image_pil.size
            pred_dict = {'boxes': boxes_filt, 'keypoints': keypoints_filt, 'size': [size[1], size[0]]}
            directory = os.path.splitext(outfp)[0]
            os.makedirs(directory, exist_ok=True)
            filename = image_info['file_name'].split('/')[-1]
            plot_on_image(image_pil, pred_dict, keypoint_skeleton, keypoint_text_prompt, directory, filename)

        # Convert bbox and keypoints to original image size
        W, H = image_pil.size
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt *= torch.tensor([W, H, W, H])
        num_keypoints = len(keypoint_text_prompt)
        keypoints_filt = keypoints_filt[..., : 2 * num_keypoints]
        keypoints_filt *= torch.tensor([W, H] * num_keypoints)
        keypoints_filt = keypoints_filt.reshape(-1, num_keypoints, 2)
        keypoints_filt = torch.cat([keypoints_filt, torch.ones_like(keypoints_filt[..., :1])], dim=-1)
        keypoints_filt = keypoints_filt.reshape(-1, 3 * num_keypoints)

        # Prepare annotations
        for box, keypoints in zip(boxes_filt, keypoints_filt):
            annotation = {
                'id': len(annotations),
                'image_id': image_info['id'],
                'category_id': coco_data['categories'][0]['id'],  # Assuming single category
                'bbox': box.tolist(),
                'keypoints': keypoints.tolist(),
            }
            annotations.append(annotation)

    # Save results in COCO format
    results = {'images': coco_data['images'], 'annotations': annotations, 'categories': coco_data['categories']}

    os.makedirs(os.path.dirname(outfp), exist_ok=True)
    with open(outfp, 'w') as f:
        json.dump(results, f)
        print(f'Saved results to {outfp}')

    name = outfp.split('/')[-1].split('.')[0]
    # 将predictions保存为pickle文件
    with open(f'{name}_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
        print(f'Saved predictions to {name}_predictions.pkl')
