import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
from copy import deepcopy

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector

from test_space.tools import to_tensor
from workspace.nuscenes_temporal_utils import get_one_pointcloud, get_nuscenes_sensor_pose_in_global


def main(ckpt_path: str, chosen_batch_dict_idx: int):
    cfg_file = f'../tools/cfgs/nuscenes_models/v2x_pointpillar_basic_ego.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger(f'log_v2x_gen_qualitative_performance.txt')
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, 
        batch_size=1,
        dist=False, logger=logger, 
        training=False,
        total_epochs=1, seed=666,
        workers=0)
    
    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(ckpt_path, logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    _batch_dict_idx = 0
    for batch_dict in dataloader:
        if _batch_dict_idx != chosen_batch_dict_idx:
            _batch_dict_idx += 1
            continue
        to_tensor(batch_dict, move_to_gpu=True)
        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)
        
        all_keys = deepcopy(list(batch_dict.keys()))
        for k in all_keys:
            if k not in ['points', 'gt_boxes', 'gt_names', 'metadata', 'final_box_dicts']:
                batch_dict.pop(k)
        
        break

    # get point cloud from others
    ego_lidar_token = batch_dict['metadata'][0]['lidar_token']
    ego_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(dataset.nusc, ego_lidar_token))

    lidar_rec = dataset.nusc.get('sample_data', ego_lidar_token)
    sample = dataset.nusc.get('sample', lidar_rec['sample_token'])
    batch_dict['exchange'] = dict()
    for lidar_name, lidar_token in sample['data'].items():
        if lidar_name not in dataset._lidars_name:
            continue
        lidar_id = int(lidar_name.split('_')[-1])
        if lidar_id == 1:
            continue
        
        agent_points = get_one_pointcloud(dataset.nusc, lidar_token)  # (N, 4)

        glob_se3_agent = get_nuscenes_sensor_pose_in_global(dataset.nusc, lidar_token)
        ego_se3_agent = ego_se3_glob @ glob_se3_agent

        batch_dict['exchange'][lidar_id] = {
            'points': agent_points,
            'ego_se3_agent': ego_se3_agent
        }
        
    _filename = f"for_quali_{ego_lidar_token}.pth"
    torch.save(batch_dict, _filename)
    print(f'batch dict is saved at:{_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--chosen_batch_dict_idx', type=int)
    args = parser.parse_args()
    main(args.ckpt_path, args.chosen_batch_dict_idx)
    # python for_journal_gen_qualitative.py --ckpt_path ../tools/pretrained_models/pillar_pil4sec1_for_qualitative.pth --chosen_batch_dict_idx 1
