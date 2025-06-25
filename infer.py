import json
import os
import os.path as osp

import gpustat


# 获取GPU显存最少的GPU
def get_gpu_with_least_memory():
    gpus = gpustat.GPUStatCollection.new_query()
    least_memory_gpu = min(gpus, key=lambda gpu: gpu.memory_used)
    return least_memory_gpu.index


config = 'config_model/UniPose_SwinT.py'
weights = 'weights/unipose_swint.pth'
box_thr, iou_thr = 0.1, 0.3

names = ['k5-test']
directorys = ['data/k5-test/imgs']
cats = ['table', 'bed', 'chair', 'sofa']
kpts = ['table', 'bed', 'chair', 'sofa']
for name, directory in zip(names, directorys):
    for i, (cat, kpt) in enumerate(zip(cats, kpts)):
        data_root = osp.join(directory, 'imgs')
        output_dir = osp.join('infer_outputs', f'{name}_{{{cat}}}')
        os.makedirs(output_dir, exist_ok=True)
        gpu_id = get_gpu_with_least_memory()
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python inference_on_a_image.py -c {config} -p {weights} -i {directory} -o {output_dir} -t {cat} -k {kpt} --box_threshold {box_thr} --iou_threshold {iou_thr}'
        print(cmd)
        os.system(cmd)
    gpu_id = get_gpu_with_least_memory()
    output_dir = osp.join('infer_outputs', f'{name}_{{{",".join(cats)}}}')
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python inference_on_a_image.py -c {config} -p {weights} -i {directory} -o {output_dir} -t {",".join(cats)} -k {",".join(kpts)} --box_threshold {box_thr} --iou_threshold {iou_thr}'
    print(cmd)
    os.system(cmd)
