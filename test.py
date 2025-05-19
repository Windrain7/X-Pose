import argparse
import json
import os
import pickle

import torch
from inference_on_a_image import load_image, load_model, plot_on_image, target_encoding
from PIL import Image
from predefined_keypoints import *
from torchvision.ops.boxes import nms
from tqdm import tqdm
from util import box_ops


def pos_process(outputs, box_threshold, iou_threshold, select_min=1):
    # Filter boxes
    logits = outputs['pred_logits'].sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'][0]  # (nq, 4)
    keypoints = outputs['pred_keypoints'][0]  # (nq, 68*3)
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

    if iou_threshold > 0:
        keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)
    else:
        keep_indices = torch.arange(boxes_filt.shape[0])

    # Use keep_indices to filter boxes and keypoints
    filtered_boxes = boxes_filt[keep_indices]
    filtered_keypoints = keypoints_filt[keep_indices]
    filtered_logits = logits_filt[keep_indices]
    filtered_label_ids = torch.argmax(filtered_logits, dim=1)

    return filtered_boxes, filtered_keypoints, filtered_logits, filtered_label_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniPose COCO Inference', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--coco_path', '-i', type=str, required=True, help='path to COCO format dataset')
    parser.add_argument('--img_root', '-d', type=str, required=True, help='path to image root directory')
    parser.add_argument('--instance_text_prompt', '-t', type=str, required=True, help='instance text prompt')
    parser.add_argument('--keypoint_text_example', '-k', type=str, default=None, help='keypoint text prompt')
    parser.add_argument('--outfp', '-o', type=str, default='outputs', required=True, help='output file path')
    parser.add_argument('--box_threshold', type=float, default=0.0, help='box threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.0, help='nms threshold')
    parser.add_argument('--cpu-only', action='store_true', help='running on cpu only!, default=False')
    parser.add_argument('--draw', action='store_true', help='draw keypoints on images, default=False')
    args = parser.parse_args()

    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)
    device = 'cpu' if args.cpu_only else 'cuda'
    model = model.to(device)

    target = target_encoding(args.instance_text_prompt, args.keypoint_text_example, model.clip_model, device)
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

        boxes_filt, keypoints_filt, scores_filt, label_ids_filt = pos_process(outputs, args.box_threshold, args.iou_threshold)
        predictions.append({'image_id': image_info['id'], 'boxes': boxes_filt, 'keypoints': keypoints_filt, 'scores': scores_filt})

        if args.draw:
            size = image_pil.size
            pred_dict = {'boxes': boxes_filt, 'keypoints': keypoints_filt, 'scores': scores_filt, 'label_ids': label_ids_filt,'size': [size[1], size[0]]}
            directory = os.path.splitext(outfp)[0]
            os.makedirs(directory, exist_ok=True)
            filename = image_info['file_name'].split('/')[-1]
            plot_on_image(image_pil, pred_dict, target['keypoints_skeleton_list'], target['kpt_vis_text'], directory, filename)

        # Convert bbox and keypoints to original image size
        W, H = image_pil.size
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2  # cswh to xywh
        boxes_filt *= torch.tensor([W, H, W, H])

        # num_keypoints = len(keypoint_text_prompt)
        # keypoints_filt = keypoints_filt[..., : 2 * num_keypoints]
        # keypoints_filt *= torch.tensor([W, H] * num_keypoints)
        # keypoints_filt = keypoints_filt.reshape(-1, num_keypoints, 2)
        # keypoints_filt = torch.cat([keypoints_filt, 2 * torch.ones_like(keypoints_filt[..., :1])], dim=-1)
        # keypoints_filt = keypoints_filt.reshape(-1, 3 * num_keypoints)

        # Prepare annotations
        for box, keypoints, score, label_id in zip(boxes_filt, keypoints_filt, scores_filt, label_ids_filt):
            label_id = label_id.item()
            kpt_num = int(target['kpt_vis_text'][label_id].sum())
            keypoints = keypoints[: 2 * kpt_num] * torch.tensor([W, H] * kpt_num)
            keypoints = keypoints.reshape(kpt_num, 2)
            keypoints = torch.cat([keypoints, 2 * torch.ones_like(keypoints[..., :1])], dim=-1)
            keypoints = keypoints.reshape(-1)
            annotation = {
                'id': len(annotations),
                'image_id': image_info['id'],
                'category_id': label_id,  # score中最大值的下标
                # 'category_id': coco_data['categories'][0]['id'],  # Assuming single category
                'bbox': box.tolist(),
                'keypoints': keypoints.tolist(),
                'bbox_score': score.max().item(),
                'score': score.max().item(),
            }
            annotations.append(annotation)

    # Save results in COCO format
    results = {'images': coco_data['images'], 'annotations': annotations, 'categories': coco_data['categories']}

    os.makedirs(os.path.dirname(outfp), exist_ok=True)
    with open(outfp, 'w') as f:
        json.dump(results, f, indent=4)
        print(f'Saved results to {outfp}')

    ext = os.path.splitext(outfp)[1]
    pklname = outfp.replace(f'{ext}', '_predictions.pkl')
    # 将predictions保存为pickle文件
    with open(pklname, 'wb') as f:
        pickle.dump(predictions, f)
        print(f'Saved predictions to {pklname}_predictions.pkl')
