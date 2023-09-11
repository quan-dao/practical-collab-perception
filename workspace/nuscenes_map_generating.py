from nuscenes import NuScenes
from pcdet.datasets.nuscenes.nuscenes_map_utils import MapMaker
from tqdm import tqdm
import pickle
import os
import numpy as np
import argparse


def include_nuscenes_data(data_root):
    print('Loading NuScenes dataset')
    nuscenes_infos = []

    infos = ('nuscenes_infos_10sweeps_train.pkl', 'nuscenes_infos_10sweeps_val.pkl')

    for info_path in infos:
        info_path = os.path.join(data_root, info_path)
        if not os.path.exists(info_path):
            continue
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            nuscenes_infos.extend(infos)

    print('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))
    return nuscenes_infos


def main(nusc_ver):
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    bev_img_resolution = 0.2
    normalize_lane_angle = False
    data_root = f'../data/nuscenes/{nusc_ver}'
    map_dir = os.path.join(data_root, 'hd_map')

    nusc = NuScenes(dataroot=data_root, version=nusc_ver, verbose=False)
    nusc_infos = include_nuscenes_data(data_root)
    map_maker = MapMaker(nusc, bev_img_resolution, point_cloud_range, normalize_lane_angle)

    for info in tqdm(nusc_infos):
        sample_rec = nusc.get('sample', info['token'])
        map_file = os.path.join(map_dir, f"map_{info['token']}.npy")
        if os.path.isfile(map_file):
            continue
        img_map = map_maker.make_representation(sample_rec['data']['LIDAR_TOP'])
        np.save(map_file, img_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ver', type=str, default='v1.0-trainval', help='specify nuscenes version')
    args = parser.parse_args()
    main(args.ver)
