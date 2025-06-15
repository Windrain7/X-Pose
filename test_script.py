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

directory = 'data/k5-test'
filenames = ['copypaste_k4v1.json', '_annotations.coco_v2_recatid.json']
cats = ['table', 'bed', 'chair', 'sofa']
kpts = ['table', 'bed', 'chair', 'sofa']
for filename in filenames:
    input_path = osp.join(directory, filename)
    for i, (cat, kpt) in enumerate(zip(cats, kpts)):
        data_root = osp.join(directory, 'imgs')
        output_path = osp.join('outputs', osp.basename(directory), f'{osp.splitext(filename)[0]}_{{{cat}}}.json')
        gpu_id = get_gpu_with_least_memory()
        cmd = (
            f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t {cat} -k {kpt} -o {output_path}'
        )
        print(cmd)
        os.system(cmd)
        content = json.load(open(output_path))
        for ann in content['annotations']:
            ann['category_id'] = i
        with open(output_path, 'w') as f:
            json.dump(content, f, indent=4)

    gpu_id = get_gpu_with_least_memory()
    output_path = osp.join('outputs', osp.basename(directory), f'{osp.splitext(filename)[0]}_{{{",".join(cats)}}}.json')
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t {",".join(cats)} -k {",".join(kpts)} -o {output_path}'
    print(cmd)
    os.system(cmd)
