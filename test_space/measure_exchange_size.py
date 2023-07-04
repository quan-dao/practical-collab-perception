import numpy as np
import argparse
import matplotlib.cm
import pickle
from tqdm import tqdm

from test_space.tools import build_dataset_for_testing


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


def main(dataset_type: str,
         version: str = 'v2.0-trainval'):
    np.random.seed(666)
    assert dataset_type in ('ego', 'ego_early')
    cfg_file = f'../tools/cfgs/dataset_configs/v2x_sim_dataset_{dataset_type}.yaml'
    dataset, dataloader = build_dataset_for_testing(
        cfg_file, ['car', 'pedestrian'], debug_dataset=False, version=version, batch_size=1, training=False)
    
    if dataset_type == 'ego_early':
        measure_early_fusion(dataloader, len(dataset))
    else:
        measure_late_fusion(dataloader, len(dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset_type', type=str)
    args = parser.parse_args()
    main(args.dataset_type)
