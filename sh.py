import multiprocessing
import os
from multiprocessing import Pool

coco_config = {
    'gt_path': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/COCO/person_keypoints_val2017_hasgt2.json',
    'img_root': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/COCO',
    'metafile': 'configs/_base_/datasets/coco.py',
    'key_words': [
        'person,hand,car',
        # 'hand,car,person',
        # 'car,person,hand',
        'person',
    ],
    'kpt_words': [
        'person,hand,car',
        # 'hand,car,person',
        # 'car,person,hand',
        'person',
    ],
}

humanart_config = {
    'gt_path': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/HumanArt/validation_humanart_re.json',
    'img_root': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/',
    'metafile': 'configs/_base_/datasets/humanart.py',
    'key_words': [
        'person,hand,car',
        # 'hand,car,person',
        # 'car,person,hand',
        'person',
    ],
    'kpt_words': [
        'person,hand,car',
        # 'hand,car,person',
        # 'car,person,hand',
        'person',
    ],
}

hand_config = {
    'gt_path': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/OneHand10K/onehand10k_test.json',
    'img_root': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/OneHand10K',
    'metafile': 'configs/_base_/datasets/onehand10k.py',
    'key_words': [
        'hand,car,person',
        # 'person,hand,car',
        # 'car,person,hand',
        'hand',
    ],
    'kpt_words': [
        'hand,car,person',
        # 'person,hand,car',
        # 'car,person,hand',
        'hand',
    ],
}

car_config = {
    'gt_path': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/CarFusion/car_keypoints_test.json',
    'img_root': '/comp_robot/cv_public_dataset/Human_Data/unipose_data/UniKPT/CarFusion',
    'metafile': 'configs/_base_/datasets/carfusion.py',
    'key_words': [
        'car,person,hand',
        # 'person,hand,car',
        # 'hand,car,person',
        'car',
    ],
    'kpt_words': [
        'car,person,hand',
        # 'person,hand,car',
        # 'hand,car,person',
        'car',
    ],
}

for config in [car_config, hand_config, humanart_config, coco_config]:
    gt_path = config['gt_path']
    out_path = []
    for key_word in config['key_words']:
        out_path.append(f'/home/jiangtao/workspace/X-Pose/outputs/{gt_path.split("/")[-1].split(".")[0]}_{{{key_word}}}.json')
    config['out_path'] = out_path


def get_eval_cmd(config):
    cmds = []
    for out_path in config['out_path']:
        metric_pkl = out_path.replace('.json', '_metrics.pkl')
        metric_json = metric_pkl.replace('.pkl', '.json')
        coco2predictions = f'python tools/dataset_converters/coco2predictions.py {config["gt_path"]} {out_path} {metric_pkl}'
        # print(coco2predictions)
        # os.system(coco2predictions)
        offline_val = (
            f'python tools/offline_eval_uni.py {metric_pkl} {config["gt_path"]} {config["metafile"]} --pck-fmt xyxy_abs --outfp {metric_json}'
        )
        if 'hand' in out_path:
            offline_val += ' --coco-metric-config-path coco21kps.yml'
        # print(offline_val)
        cmds.append((coco2predictions, offline_val))
    return cmds


def run_command(args):
    cmd = args
    print(f'{cmd}')
    os.system(cmd)


def main():
    weight_path = 'weights/unipose_swint.pth'
    commands = []
    for config in [car_config]:
        # 获取所有可用GPU
        gpu_memory_info = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read().strip().split('\n')
        available_gpus = list(range(len(gpu_memory_info)))
        # 按显存剩余大小排序
        available_gpus = sorted(available_gpus, key=lambda x: int(gpu_memory_info[x]))
        for i, (key_word, kpt_word, out_path) in enumerate(zip(config['key_words'], config['kpt_words'], config['out_path'])):
            dev = available_gpus[i]
            cmd = f'CUDA_VISIBLE_DEVICES={dev} python test.py -c config_model/UniPose_SwinT.py -p {weight_path} -i {gt_path} -d {config["img_root"]} -t {key_word} -k {kpt_word} -o {out_path}'
            commands.append(cmd)

    # 使用进程池并行执行
    with Pool(processes=len(commands)) as pool:
        pool.map(run_command, commands)


if __name__ == '__main__':
    main()
