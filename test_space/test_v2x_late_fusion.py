import numpy as np
import torch
import argparse

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector

from test_space.tools import to_tensor


def make_batch_dict(batch_size: int = 2, target_batch_idx: int = 3):
    np.random.seed(666)
    cfg_file = "../tools/cfgs/nuscenes_models/v2x_late_fusion.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    # shutdown data augmentation
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling', 'random_world_flip', 
                                                       'random_world_rotation', 'random_world_scaling']
    cfg.DATA_CONFIG.MINI_TRAINVAL_STRIDE = 1
    if cfg.DATA_CONFIG.get('DATASET_DOWNSAMPLING_RATIO', 1) > 1:
        cfg.DATA_CONFIG.DATASET_DOWNSAMPLING_RATIO = 1
    logger = common_utils.create_logger(f'log_v2x_test_late_fusion.txt')

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, 
        batch_size=batch_size,
        dist=False, logger=logger, 
        training=False,
        total_epochs=1, seed=666,
        workers=0)
    
    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)


    batch_idx = 0
    for batch_dict in dataloader:
        if batch_idx != target_batch_idx:
            batch_idx += 1
            continue

        to_tensor(batch_dict, move_to_gpu=True)
        points = torch.clone(batch_dict['points'])
        metadata = np.copy(batch_dict['metadata'])

        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)

        out = {
            'points': points,
            'metadata': metadata,
            'pred_dicts': pred_dicts
        }
        torch.save(out, './artifact/v2x_late_fusion_pred_dict.pth')
        break


def show_batch_dict():
    batch_dict = torch.load('./artifact/v2x_late_fusion_pred_dict.pth', map_location=torch.device('cpu'))

    points = batch_dict['points']
    metadata = batch_dict['metadata']
    pred_dicts = batch_dict['pred_dicts']
    print(f"pred_dicts: {type(pred_dicts)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('blah')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--target_batch_idx', type=int, default=3)
    parser.add_argument('--make_batch_dict', type=int)
    parser.add_argument('--show_batch_dict', type=int)
    args = parser.parse_args()
    if args.make_batch_dict == 1:
        make_batch_dict(args.batch_size, args.target_batch_idx)
    
    if args.show_batch_dict == 1:
        show_batch_dict()
