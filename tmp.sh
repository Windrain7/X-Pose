CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i inputs/car-person/1.1-car-tabel-person.png \
    -o outputs/b10pct \
    -t person \
    -k person \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i inputs/car-person/1.1-car-tabel-person.png \
    -o outputs/b10pct \
    -t car \
    -k car \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i inputs/car-person/1.1-car-tabel-person.png \
    -o outputs/b10pct \
    -t person,car \
    -k person,car \
    --box_threshold 0.1 \
    --iou_threshold 0.6


CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t person,dog \
    -k person,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t person \
    -k person \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t person,hand \
    -k person,hand \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t hand \
    -k hand \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t hand,person \
    -k hand,person \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t dog,person \
    -k animal,person \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t cat,person \
    -k animal,person \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t person,dog \
    -k person,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t dog,hand \
    -k animal,hand \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t hand,dog \
    -k hand,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pct \
    -t cat,dog \
    -k animal,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6


CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pctb10pct \
    -t person,cat \
    -k person,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6

CUDA_VISIBLE_DEVICES=0 python inference_on_a_image.py \
    -c config_model/UniPose_SwinT.py \
    -p weights/unipose_swint.pth \
    -i Cat-and-Dog-People-1.jpg \
    -o outputs/b10pctb10pct \
    -t person,cat,dog \
    -k person,animal,animal \
    --box_threshold 0.1 \
    --iou_threshold 0.6