import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Dict, List
import argparse
from easydict import EasyDict
import yaml

from pcdet.utils import common_utils
from pcdet.datasets.nuscenes.v2x_sim_dataset import V2XSimDataset_RSU
from workspace.v2x_sim_utils import get_points_and_boxes_of_1lidar, get_nuscenes_sensor_pose_in_global


class V2XSimDataset_CAR(V2XSimDataset_RSU):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        assert dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1) == 1, "don't use MINI_TRAINVAL_STRIDE to reduce dataset, use DATASET_DOWNSAMPLING_RATIO"
        root_path = root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

    def include_v2x_sim_data(self, mode):
        self.logger.info('Loading V2X-Sim dataset')
        v2x_infos = dict([(lidar_id, list()) for lidar_id in range(1, 6)])

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / f"{self._prefix}_{info_path}"
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                for lidar_id, lidar_infos in infos.items():
                    v2x_infos[lidar_id].extend(lidar_infos)

        self.infos: Dict[int, List] = v2x_infos

        self.logger.info(f"Total samples for V2X-Sim dataset used for {self.mode}")
        for lidar_id, _infos in self.infos.items():
            self.logger.info(f"\t\t LIDAR_TOP_id_{lidar_id}: {len(_infos)}")

        if self.training and self.dataset_cfg.get('DATASET_DOWNSAMPLING_RATIO', 1) > 1:
            ratio = float(self.dataset_cfg.DATASET_DOWNSAMPLING_RATIO)
            num_samples_lidar1 = float(len(self.infos[1]))  # LIDAR_TOP_id_1 have most samples, cuz always present
            for lidar_id, lidar_infos in self.infos.items():
                if len(lidar_infos) > 0:    
                    # sort by timestamp
                    lidar_infos.sort(key=lambda e: e['timestamp'])
                    # downsample
                    stride = int((len(lidar_infos) / num_samples_lidar1) * ratio)
                    if stride > 1:
                        self.infos[lidar_id] = lidar_infos[::stride]
                    else:
                        self.infos[lidar_id] = lidar_infos
            
        # merge separate lidar_infos into 1 big list of infos
        merge_infos = list()
        for lidar_id, lidar_infos in self.infos.items():
            if len(lidar_infos) > 0:
                merge_infos.extend(lidar_infos)
        
        self.infos = merge_infos
        self.logger.info('Total samples for V2X-Sim dataset: %d' % (len(self.infos)))

    def build_v2x_sim_info(self) -> None:
        path_trainval_split = self.root_path / Path(f"{self._prefix}_trainval_split.pkl")
        if not path_trainval_split.exists():
            self._build_train_val_split()
        with open(path_trainval_split, 'rb') as f:
            trainval_split = pickle.load(f)

        train_infos = dict([(lidar_id, list()) for lidar_id in range(1, 6)])
        val_infos = dict([(lidar_id, list()) for lidar_id in range(1, 6)])

        for sample in tqdm(self.nusc.sample, total=len(self.nusc.sample), desc='create_info', dynamic_ncols=True):
            for sensor_name, sensor_token in sample['data'].items():
                if 'LIDAR_TOP_id_' not in sensor_name:
                    continue

                lidar_id = int(sensor_name.strip().split('_')[-1])
                if lidar_id == 0:
                    # RSU
                    continue

                stuff = get_points_and_boxes_of_1lidar(self.nusc, 
                                                       sensor_token, 
                                                       self.classes_of_interest, 
                                                       self.dataset_cfg.get('POINTS_IN_BOXES_GPU', False), 
                                                       self.dataset_cfg.get('THRESHOLD_BOXES_BY_POINTS', 5))

                gt_boxes = stuff['boxes_in_lidar']  # (N_gt, 7)
                gt_names = stuff['boxes_name']  # (N_gt,)
                num_points_in_boxes = stuff['num_points_in_boxes']  # (N_gt,)
                assert gt_boxes.shape[0] == gt_names.shape[0] == num_points_in_boxes.shape[0]

                info = dict()
                info['token'] = sample['token']
                info['lidar_token'] = sensor_token
                # for evaluation
                info['glob_se3_lidar'] = get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token'])
                info['gt_boxes'] = gt_boxes  # (N_gt, 7)
                info['gt_names'] = gt_names  # (N_gt,)
                info['num_points_in_boxes'] = num_points_in_boxes  # (N_gt,)
                
                info['lidar_path'] = self.nusc.get_sample_data_path(info['lidar_token'])  # legacy from nuscenes_dataset

                # get timestamp
                sample_data_record = self.nusc.get('sample_data', sensor_token)
                info['timestamp'] = sample_data_record['timestamp']

                if sample['scene_token'] in trainval_split['train']:
                    train_infos[lidar_id].append(info)
                else:
                    val_infos[lidar_id].append(info)
        
        if len(train_infos) > 0:
            path_train_infos = self.root_path / f"{self._prefix}_v2x_sim_car_infos_{self.num_historical_sweeps}sweeps_train.pkl"
            with open(path_train_infos, 'wb') as f:
                pickle.dump(train_infos, f)
            
            self.logger.info(f"v2x-sim {self.dataset_cfg.VERSION} | num samples for training:")
            for lidar_id, _infos in train_infos.items():
                self.logger.info(f"\t\t LIDAR_TOP_id_{lidar_id}: {len(_infos)}")
                

        if len(val_infos) > 0:
            path_val_infos = self.root_path / f"{self._prefix}_v2x_sim_car_infos_{self.num_historical_sweeps}sweeps_val.pkl"
            with open(path_val_infos, 'wb') as f:
                pickle.dump(val_infos, f)

            self.logger.info(f"v2x-sim {self.dataset_cfg.VERSION} | num samples for val:")
            for lidar_id, _infos in val_infos.items():
                self.logger.info(f"\t\t LIDAR_TOP_id_{lidar_id}: {len(_infos)}")


if __name__ == '__main__':
    cfg_file = './tools/cfgs/dataset_configs/v2x_sim_dataset_car.yaml'  # NOTE: launch this from OpenPCDet directory
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_file)))
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--create_v2x_sim_car_infos', action='store_true')
    parser.add_argument('--no-create_v2x_sim_car_infos', dest='create_v2x_sim_car_infos', action='store_false')
    parser.set_defaults(create_v2x_sim_car_infos=False)

    parser.add_argument('--version', type=str, default='v2.0-trainval', help='')

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version

    if args.create_v2x_sim_car_infos:
        v2x_dataset = V2XSimDataset_CAR(dataset_cfg, 
                                        class_names=dataset_cfg.CLASSES_OF_INTEREST, 
                                        root_path=ROOT_DIR / 'data' / 'v2x-sim',
                                        logger=common_utils.create_logger())
        v2x_dataset.build_v2x_sim_info()
        # python -m pcdet.datasets.nuscenes.v2x_sim_dataset_car --create_v2x_sim_car_infos --version v2.0-mini
