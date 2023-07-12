import numpy as np
import torch
import argparse
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader

from test_space.tools import build_dataset_for_testing
from test_space.tools import to_tensor
from workspace.bev_maker import BEVMaker


def measure_early_fusion(dataloader, size_dataset: int):
    total_size = 0
    for batch_dict in tqdm(dataloader, total=size_dataset):
        points = batch_dict['points']
        total_size += points.nbytes

    print('avg nbytes: ', total_size / (float(size_dataset) * 1e6))


def measure_late_fusion(dataloader, size_dataset: int):
    total_size = 0
    for batch_dict in tqdm(dataloader, total=size_dataset):
        points = batch_dict['points']
        mask_modar = points[:, 11] > 0
        points = points[mask_modar]
        total_size += points.nbytes

    print('avg nbytes: ', total_size / (float(size_dataset) * 1e6))


def measure_mid_fusion():
    cfg_file = f'../tools/cfgs/nuscenes_models/v2x_pointpillar_disco.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger(f'measure_exchange_data_logger.txt')
    
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, 
        batch_size=1,
        dist=False, logger=logger, 
        training=False,  # but dataset_cfg.INFO_PATH's val == train, set training to False to shutdown database_sampling & data_transformation
        total_epochs=1, seed=666,
        workers=0)

    rsu_bev_maker = BEVMaker(cfg.BEV_MAKER_RSU, 10, dataset)
    car_bev_maker = BEVMaker(cfg.BEV_MAKER_CAR, 10, dataset)
    
    rsu_bev_maker.eval()
    car_bev_maker.eval()

    rsu_bev_maker.cuda()
    car_bev_maker.cuda()


    total_size = 0
    total_sample = 0

    for batch_dict in tqdm(dataloader, total=len(dataset)):
        to_tensor(batch_dict, move_to_gpu=True)

        with torch.no_grad():
            batch_dict = rsu_bev_maker(batch_dict)
            batch_dict = car_bev_maker(batch_dict)

        for agent_idx, _bev_img in batch_dict['bev_img'].items():
            total_size += _bev_img.detach().cpu().numpy().nbytes
            total_sample += 1

    print('avg nbytes: ', total_size / (float(total_sample) * 1e6))


def main(dataset_type: str,
         version: str = 'v2.0-trainval'):
    np.random.seed(666)
    if dataset_type in ('ego', 'ego_early'):
        cfg_file = f'../tools/cfgs/dataset_configs/v2x_sim_dataset_{dataset_type}.yaml'
        dataset, dataloader = build_dataset_for_testing(
            cfg_file, ['car', 'pedestrian'], debug_dataset=False, version=version, batch_size=1, training=False)
        
        if dataset_type == 'ego_early':
            measure_early_fusion(dataloader, len(dataset))
        else:
            measure_late_fusion(dataloader, len(dataset))
    else:
        assert dataset_type == 'ego_disco'
        measure_mid_fusion()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset_type', type=str)
    args = parser.parse_args()
    main(args.dataset_type)
