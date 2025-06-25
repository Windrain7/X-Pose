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

if 0:
    # directory = 'data/k5-test'
    # filenames = ['copypaste_k4v1.json', '_annotations.coco_v2_recatid.json']
    # img_root = 'imgs'
    directory = 'data/coco'
    filenames = ['annotations/kps-MCII/val2017_k4_hasgt.json']
    img_root = 'val2017'

    cats = ['table', 'bed', 'chair', 'sofa']
    kpts = ['table', 'bed', 'chair', 'sofa']
    for filename in filenames:
        input_path = osp.join(directory, filename)
        for i, (cat, kpt) in enumerate(zip(cats, kpts)):
            data_root = osp.join(directory, img_root)
            output_path = osp.join('outputs', osp.basename(directory), f'{osp.splitext(osp.basename(filename))[0]}_{{{cat}}}.json')
            gpu_id = get_gpu_with_least_memory()
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t {cat} -k {kpt} -o {output_path}'

            print(cmd)
            os.system(cmd)
            content = json.load(open(output_path))
            for ann in content['annotations']:
                ann['category_id'] = i
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=4)

        gpu_id = get_gpu_with_least_memory()
        output_path = osp.join('outputs', osp.basename(directory), f'{osp.splitext(osp.basename(filename))[0]}_{{{",".join(cats)}}}.json')
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t {",".join(cats)} -k {",".join(kpts)} -o {output_path}'
        print(cmd)
        os.system(cmd)

# all in one
if 1:
    infos = [
        dict(dataset='COCO', img_root='', filename='person_keypoints_val2017_hasgt2.json'),
        dict(dataset='AP-10K', img_root='', filename='ap10k-val-split1_animal.json'),
        dict(dataset='HumanArt', img_root='', filename='validation_humanart.json'),
        dict(dataset='300W', img_root='', filename='face_landmarks_300w_test.json'),
        dict(dataset='OneHand10K', img_root='', filename='onehand10k_test.json'),
        dict(dataset='COCO-WholeBody-Hand', img_root='val2017', filename='coco_wholebody_val_v1.0_hand_vis0.6.json'),
        dict(dataset='Keypoint-5', img_root='', filename='table_test.json'),
        dict(dataset='Keypoint-5', img_root='', filename='bed_test.json'),
        dict(dataset='Keypoint-5', img_root='', filename='chair_test.json'),
        dict(dataset='Keypoint-5', img_root='', filename='sofa_test.json'),
        dict(dataset='CarFusion', img_root='', filename='car_keypoints_test.json'),
    ]
    directories = ['data/UniKPT'] * len(infos)
    # fmt: off
    animal_cats = ['antelope', 'argali sheep', 'bison', 'buffalo', 'cow', 'sheep', 'arctic fox', 'dog', 'fox', 'wolf', 'beaver', 'alouatta', 'monkey', 'noisy night monkey', 'spider monkey', 'uakari', 'deer', 'moose', 'hamster', 'elephant', 'horse', 'zebra', 'bobcat', 'cat', 'cheetah', 'jaguar', 'king cheetah', 'leopard', 'lion', 'panther', 'snow leopard', 'tiger', 'giraffe', 'hippo', 'chimpanzee', 'gorilla', 'orangutan', 'rabbit', 'skunk', 'mouse', 'rat', 'otter', 'weasel', 'raccoon', 'rhino', 'marmot', 'squirrel', 'pig', 'mole', 'black bear', 'brown bear', 'panda', 'polar bear', 'bat']
    # animal_cats = ['animal']
    cats = ['person', 'face', 'hand'] +  animal_cats + ['table', 'bed', 'chair', 'sofa', 'car', 'face']
    kpts = ['person', 'face', 'hand'] + ['AP10K'] * len(animal_cats) + ['table', 'bed', 'chair', 'sofa', 'car', 'face']

    for directory, info in zip(directories, infos):
        dataset = info['dataset']
        filename = info['filename']
        img_root = info.get('img_root', '')
        input_path = osp.join(directory, dataset, filename)
        data_root = osp.join(directory, dataset, img_root)
        output_path = osp.join('outputs', osp.basename(directory), f'{dataset}_allinone.json')
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        gpu_id = get_gpu_with_least_memory()
        cmd = (
            f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t "{",".join(cats)}" -k "{",".join(kpts)}" -o {output_path}'
        )
        print(cmd, file=open('test.sh', 'a'), end='\n\n')
        # os.system(cmd)
        # content = json.load(open(output_path))
        # for i, ann in enumerate(content['annotations']):
        #     if 3 + len(animal_cats) > ann['category_id'] >= 3:
        #         ann['category_id'] = 3
        #     elif ann['category_id'] >= 3 + len(animal_cats):
        #         ann['category_id'] -= len(animal_cats) - 1
        # for i, cat in enumerate(content['categories']):
        #     if 3 + len(animal_cats) > cat['id'] >= 3:
        #         cat['id'] = 3
        #     elif cat['id'] >= 3 + len(animal_cats):
        #         cat['id'] -= len(animal_cats) - 1
        # with open(output_path.replace('.json', '_recatid.json'), 'w') as f:
        #     json.dump(content, f, indent=4)

# one by one
if 0:
    infos = [
        # dict(dataset='AP-10K', img_root='', filename='ap10k-val-split1_animal.json'),
        # dict(dataset='CarFusion', img_root='', filename='car_keypoints_test.json'),
        # dict(dataset='COCO', img_root='', filename='person_keypoints_val2017_hasgt2.json'),
        # dict(dataset='OneHand10K', img_root='', filename='onehand10k_test.json'),
        # dict(dataset='Keypoint-5', img_root='', filename='table_test.json'),
        # dict(dataset='Keypoint-5', img_root='', filename='bed_test.json'),
        # dict(dataset='Keypoint-5', img_root='', filename='chair_test.json'),
        # dict(dataset='Keypoint-5', img_root='', filename='sofa_test.json'),
        # dict(dataset='HumanArt', img_root='', filename='validation_humanart.json'),
        dict(dataset='COCO-WholeBody-Hand', img_root='val2017', filename='coco_wholebody_val_v1.0_hand_vis0.6.json'),
    ]
    directories = ['data/UniKPT'] * len(infos)
    cats = ['hand']
    kpts = ['hand']

    for directory, info in zip(directories, infos):
        dataset = info['dataset']
        filename = info['filename']
        img_root = info.get('img_root', '')
        input_path = osp.join(directory, dataset, filename)
        data_root = osp.join(directory, dataset, img_root)
        output_path = osp.join('outputs', f'{osp.basename(directory)}', f'{dataset}_onebyone.json')
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        gpu_id = get_gpu_with_least_memory()
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python test.py -c {config} -p {weights} -i {input_path} -d {data_root} -t "{",".join(cats)}" -k "{",".join(kpts)}" -o {output_path}'
        print(cmd)
        os.system(cmd)
        content = json.load(open(output_path))
