import json
import os.path as osp

import numpy as np


def read_json(fp):
    """Read scalars from a wrong JSON file."""
    with open(fp) as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError:
            # If the file is not a valid JSON, read the last line
            print(f'File {fp} is not a valid JSON, reading the last line instead.')
            f.seek(0)  # Reset file pointer to beginning
            lines = f.readlines()
            last_line = lines[-1].strip()
            content = json.loads(last_line)
    return content


if 1:
    # directory = 'outputs/k5-test'
    # filename = ['copypaste_k4v1', '_annotations.coco_v2_recatid']
    directory = 'outputs/coco'
    filename = ['val2017_k4_hasgt']

    cats = ['table', 'bed', 'chair', 'sofa']
    cats2id = {cat: i for i, cat in enumerate(cats)}
    for name in filename:
        # one by one test
        one_aps = []
        one_pcks_10pct = []
        one_pcks_20pct = []
        one_cars = []
        print(f'--- {name} ---')
        print('--- one by one test ---')
        for cat in cats:
            path = osp.join(directory, f'{name}_{{{cat}}}_metrics.json')
            content = read_json(path)
            one_aps.append(content[f'coco/{cat}/AP'])
            one_pcks_10pct.append(content[f'visible/0.1/cat{cats2id[cat]}/PCK'])
            one_pcks_20pct.append(content[f'visible/0.2/cat{cats2id[cat]}/PCK'])
            one_cars.append(content[f'CAR_cat{cats2id[cat]}'])
        one_aps.append(sum(one_aps) / len(one_aps))
        one_pcks_10pct.append(sum(one_pcks_10pct) / len(one_pcks_10pct))
        one_pcks_20pct.append(sum(one_pcks_20pct) / len(one_pcks_20pct))
        one_cars.append(sum(one_cars) / len(one_cars))
        print('/'.join(f'{ap * 100:.1f}' for ap in one_aps))
        print('/'.join(f'{pck * 100:.1f}' for pck in one_pcks_10pct))
        print('/'.join(f'{pck * 100:.1f}' for pck in one_pcks_20pct))
        print('/'.join(f'{car * 100:.1f}' for car in one_cars))
        print('--- all in one test ---')
        # all in one test
        path = osp.join(directory, f'{name}_{{{",".join(cats)}}}_metrics.json')
        content = read_json(path)
        all_aps = [content[f'coco/{cat}/AP'] for cat in cats]
        all_aps.append(content['coco/AP'])
        all_pcks_10pct = [content[f'visible/0.1/cat{cats2id[cat]}/PCK'] for cat in cats]
        all_pcks_10pct.append(content['visible/0.1/PCK'])
        all_pcks_20pct = [content[f'visible/0.2/cat{cats2id[cat]}/PCK'] for cat in cats]
        all_pcks_20pct.append(content['visible/0.2/PCK'])
        all_cars = [content[f'CAR_cat{cats2id[cat]}'] for cat in cats]
        all_cars.append(content['CAR'])
        print('/'.join(f'{ap * 100:.1f}' for ap in all_aps))
        print('/'.join(f'{pck * 100:.1f}' for pck in all_pcks_10pct))
        print('/'.join(f'{pck * 100:.1f}' for pck in all_pcks_20pct))
        print('/'.join(f'{car * 100:.1f}' for car in all_cars))

if 0:
    fp = 'log/unikpt5/C4_base_withcpmo_wovisCAT_AWARE/20250612_211133/vis_data/scalars.json'
    content = read_json(fp)
    cats = ['table', 'bed', 'chair', 'sofa']
    key = 'recatid_k4v1'
    ap = [f'{content[f"{key}/{cat}/AP"] * 100:.1f}' for i, cat in enumerate(cats)]
    print('/'.join(ap))
    pck_10pct = [f'{content[f"{key}/visible_0.1/cat_{i}/PCK"] * 100:.1f}' for i, cat in enumerate(cats)]
    print('/'.join(pck_10pct))
    pck_20pct = [f'{content[f"{key}/visible_0.2/cat_{i}/PCK"] * 100:.1f}' for i, cat in enumerate(cats)]
    print('/'.join(pck_20pct))
    CAR = [f'{content[f"{key}/CAR_cat_{i}"] * 100:.1f}' for i, cat in enumerate(cats)]
    print('/'.join(CAR))

if 0:
    fp = '/home/jiangtao/workspace/mmpose/log/unikpt5/C4_base_withcpmo/20250614_160041/20250614_160041.json'
    print(f'Loading metrics from {fp}')
    print(f'{fp.split("/")[-2]}')

    content = read_json(fp)
    datasets_keys = ['k5_table', 'k5_bed', 'k5_chair', 'k5_sofa']
    # k4_keys = ['copypaste_k4v1', 'recatid_k4v1']
    k4_keys = ['copypaste_k4v1', 'generate_k4v1']
    cats = ['table', 'bed', 'chair', 'sofa']

    aps = [f'{content[f"{key}/AP"] * 100:.1f}' for key in datasets_keys]
    pck_10pct = [f'{content[f"{key}/visible_0.1/PCK"] * 100:.1f}' for key in datasets_keys]
    pck_20pct = [f'{content[f"{key}/visible_0.2/PCK"] * 100:.1f}' for key in datasets_keys]
    CAR = [f'{content[f"{key}/CAR"] * 100:.1f}' for key in datasets_keys]
    print('--- k4 ---')
    print('/'.join(aps))
    print('/'.join(pck_10pct))
    print('/'.join(pck_20pct))
    print('/'.join(CAR))
    for key in k4_keys:
        print(f'--- {key} ---')
        ap = [f'{content[f"{key}/{cat}/AP"] * 100:.1f}' for i, cat in enumerate(cats)]
        pck_10pct = [f'{content[f"{key}/visible_0.1/cat_{i}/PCK"] * 100:.1f}' for i, cat in enumerate(cats)]
        pck_20pct = [f'{content[f"{key}/visible_0.2/cat_{i}/PCK"] * 100:.1f}' for i, cat in enumerate(cats)]
        CAR = [f'{content[f"{key}/CAR_cat{i}"] * 100:.1f}' for i, cat in enumerate(cats)]
        CAR.append(f'{content[f"{key}/CAR"] * 100:.1f}')
        print('/'.join(ap))
        print('/'.join(pck_10pct))
        print('/'.join(pck_20pct))
        print('/'.join(CAR))
