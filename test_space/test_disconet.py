import numpy as np
import torch
import argparse
from nuscenes import NuScenes
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector

from test_space.tools import to_tensor
from workspace.o3d_visualization import PointsPainter, print_dict
import lovely_tensors as lt


def make_batch_dict(batch_size: int = 1, target_batch_idx: int = 3):
    np.random.seed(666)
    cfg_file = "../tools/cfgs/nuscenes_models/v2x_pointpillar_disco.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    # shutdown data augmentation
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling'], 
                                                    #    'random_world_flip', 
                                                    #    'random_world_rotation', 'random_world_scaling']
    cfg.DATA_CONFIG.MINI_TRAINVAL_STRIDE = 1
    if cfg.DATA_CONFIG.get('DATASET_DOWNSAMPLING_RATIO', 1) > 1:
        cfg.DATA_CONFIG.DATASET_DOWNSAMPLING_RATIO = 1
    logger = common_utils.create_logger(f'log_v2x_test_disco.txt')

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, 
        batch_size=batch_size,
        dist=False, logger=logger, 
        training=True,
        total_epochs=1, seed=666,
        workers=0)
    
    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda()
    model.eval()

    batch_idx = 0
    for batch_dict in dataloader:
        if batch_idx != target_batch_idx:
            batch_idx += 1
            continue

        to_tensor(batch_dict, move_to_gpu=True)
        
        with torch.no_grad():
            for cur_module in model.module_list:
                batch_dict = cur_module(batch_dict)
        
        print_dict(batch_dict, 'batch_dict')
        
        for k in ['points', 'gt_boxes', 'voxel_features', 'pillar_features', 'voxel_coords', 'spatial_features']:
            batch_dict.pop(k)
        torch.save(batch_dict, './artifact/v2x_disco_batch_dict.pth')
        # ---
        break


def show_batch_dict():
    batch_dict = torch.load('./artifact/v2x_disco_batch_dict.pth', map_location=torch.device('cpu'))
    print_dict(batch_dict, batch_dict)

    metadata = batch_dict['metadata'][0]
    for k, v in metadata['se3_from_ego'].items():
        agent_in_ego = np.linalg.inv(v)
        print(f'agent {k} pix coord: ', (agent_in_ego[:2, -1] + 51.2) / 0.2)


    warped_bev_img = [v[0]for k, v in batch_dict['warped_bev_img'].items()]
    fig, ax = plt.subplots(1, len(warped_bev_img))
    for idx, img in enumerate(warped_bev_img):
        ax[idx].imshow(img, origin='lower')
    plt.show()


if __name__ == '__main__':
    lt.monkey_patch()
    parser = argparse.ArgumentParser('blah')
    parser.add_argument('--make_batch_dict', type=int, default=0)
    parser.add_argument('--show_batch_dict', type=int, default=0)
    args = parser.parse_args()
    if args.make_batch_dict == 1:
        make_batch_dict()
    
    if args.show_batch_dict == 1:
        show_batch_dict()
