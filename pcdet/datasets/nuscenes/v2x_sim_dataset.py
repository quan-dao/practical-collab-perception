import numpy as np
from nuscenes import NuScenes
from typing import Dict, List, Set
from pathlib import Path
import pickle
from tqdm import tqdm
import json
import copy
import argparse
from easydict import EasyDict
import yaml

from nuscenes.eval.detection.config import config_factory

from ..dataset import DatasetTemplate
from pcdet.datasets.nuscenes import nuscenes_utils
from pcdet.utils import common_utils
from workspace.v2x_sim_utils import get_points_and_boxes_of_1lidar, get_nuscenes_sensor_pose_in_global, get_pseudo_sweeps_of_1lidar
from workspace.v2x_sim_eval_utils import transform_det_annos_to_nusc_annos, V2XSimDetectionEval


class V2XSimDataset_RSU(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.infos: List[Dict] = list()        

        self._prefix = 'mini' if 'mini' in self.dataset_cfg.VERSION else 'full'
        self.nusc = NuScenes(dataroot=root_path, version=dataset_cfg.VERSION, verbose=False)
        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE)
        self.classes_of_interest = set(dataset_cfg.get('CLASSES_OF_INTEREST', ['car', 'pedestrian']))
        self.num_sweeps = dataset_cfg.get('NUM_HISTORICAL_SWEEPS', 10) + 1
        self.num_historical_sweeps = dataset_cfg.get('NUM_HISTORICAL_SWEEPS', 10)

        path_train_infos = self.root_path / f"{self._prefix}_v2x_sim_infos_{self.num_historical_sweeps}sweeps_train.pkl"
        print('path_train_infos: ', path_train_infos)
        if not path_train_infos.exists():
            self.logger.warn('dataset infos do not exist, call build_v2x_sim_info')
        else:
            self.include_v2x_sim_data(self.mode)
            self.infos.sort(key=lambda e: e['timestamp'])
            if self.training and self.dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1) > 1:
                self.infos = self.infos[::self.dataset_cfg.MINI_TRAINVAL_STRIDE]  # use 1/4th of the trainval data
            
        self.all_sample_data_tokens = [_info['lidar_token'] for _info in self.infos]  # for evaluation 

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def include_v2x_sim_data(self, mode):
        self.logger.info('Loading V2X-Sim dataset')
        v2x_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = f"{self._prefix}_{info_path}"
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                v2x_infos.extend(infos)

        self.infos.extend(v2x_infos)
        self.logger.info('Total samples for V2X-Sim dataset: %d' % (len(v2x_infos)))

    def _build_train_val_split(self):
        # town 4, 5 for train
        # town 3 for val
        train_locs = set([4, 5])
        val_locs = set([3,])

        train_scenes_token, val_scenes_token = list(), list()

        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            if log['location'] in train_locs:
                train_scenes_token.append(scene['token'])
            else:
                val_scenes_token.append(scene['token'])
        
        if 'mini' not in self.dataset_cfg.VERSION:
            # full
            trainval_split = {
                'train': set(train_scenes_token),
                'val': val_scenes_token
            }
        else:
            # mini -> just for testing functionalities 
            split_tokens = train_scenes_token if len(train_scenes_token) > 0 else val_scenes_token
            trainval_split = {
                'train': set(split_tokens),
                'val': split_tokens
            }
        
        path_ = self.root_path / Path(f"{self._prefix}_trainval_split.pkl")
        with open(path_, 'wb') as f:
            pickle.dump(trainval_split, f)

    def build_v2x_sim_info(self) -> None:
        path_trainval_split = self.root_path / Path(f"{self._prefix}_trainval_split.pkl")
        if not path_trainval_split.exists():
            self._build_train_val_split()
        with open(path_trainval_split, 'rb') as f:
            trainval_split = pickle.load(f)

        lidar_name = 'LIDAR_TOP_id_0'
        train_infos, val_infos = list(), list()

        for sample in tqdm(self.nusc.sample, total=len(self.nusc.sample), desc='create_info', dynamic_ncols=True):
            if lidar_name not in sample['data']:
                continue
            
            stuff = get_points_and_boxes_of_1lidar(self.nusc, 
                                                   sample['data'][lidar_name], 
                                                   self.classes_of_interest, 
                                                   self.dataset_cfg.get('POINTS_IN_BOXES_GPU', False), 
                                                   self.dataset_cfg.get('THRESHOLD_BOXES_BY_POINTS', 5))
            gt_boxes = stuff['boxes_in_lidar']  # (N_gt, 7)
            gt_names = stuff['boxes_name']  # (N_gt,)
            num_points_in_boxes = stuff['num_points_in_boxes']  # (N_gt,)
            assert gt_boxes.shape[0] == gt_names.shape[0] == num_points_in_boxes.shape[0]

            info = dict()
            info['token'] = sample['token']
            info['lidar_token'] = sample['data'][lidar_name]
            # for evaluation
            info['glob_se3_lidar'] = get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token'])
            info['gt_boxes'] = gt_boxes  # (N_gt, 7)
            info['gt_names'] = gt_names  # (N_gt,)
            info['num_points_in_boxes'] = num_points_in_boxes  # (N_gt,)
            
            info['lidar_path'] = self.nusc.get_sample_data_path(info['lidar_token'])  # legacy from nuscenes_dataset

            # get timestamp
            sample_data_record = self.nusc.get('sample_data', sample['data'][lidar_name])
            info['timestamp'] = sample_data_record['timestamp']

            if sample['scene_token'] in trainval_split['train']:
                train_infos.append(info)
            else:
                val_infos.append(info)

        if len(train_infos) > 0:
            path_train_infos = self.root_path / f"{self._prefix}_v2x_sim_infos_{self.num_historical_sweeps}sweeps_train.pkl"
            with open(path_train_infos, 'wb') as f:
                pickle.dump(train_infos, f)
            self.logger.info(f"v2x-sim {self.dataset_cfg.VERSION} | num samples for training: {len(train_infos)}")

        if len(val_infos) > 0:
            path_val_infos = self.root_path / f"{self._prefix}_v2x_sim_infos_{self.num_historical_sweeps}sweeps_val.pkl"
            with open(path_val_infos, 'wb') as f:
                pickle.dump(val_infos, f)
            self.logger.info(f"v2x-sim {self.dataset_cfg.VERSION} | num samples for val: {len(val_infos)}")

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            raise NotImplementedError
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.nuscenes_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

    def nusc_eval(self, det_annos: List[Dict], class_names, **kwargs):
        """
        Args:
            det_annos: each dict is
                {   
                    'metadata': {
                        'token': sample token
                        'lidar_token'
                    }
                    'boxes_lidar': (N, 7) - x, y, z, dx, dy, dz, heading | in LiDAR
                    'score': (N,)
                    'pred_labels': (N,) | int, start from 1
                    'name': (N,) str
                }
        """
        nusc_annos = {
            'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 
                    'use_map': False, 'use_external': False},
            'results': {}
        }
        for info in self.infos:
            nusc_annos['results'].update(
                {info['lidar_token']: list()}  # NOTE: workaround to eval w.r.t lidar_token
            ) 

        transform_det_annos_to_nusc_annos(det_annos, nusc_annos)
        
        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)
        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = V2XSimDetectionEval(nusc=self.nusc, 
                                        config=eval_config, 
                                        result_path=res_path, 
                                        eval_set='', 
                                        output_dir=output_path, 
                                        verbose=True, 
                                        dataset_infos=self.infos)
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
        
        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        info = copy.deepcopy(self.infos[index])

        stuff = get_pseudo_sweeps_of_1lidar(self.nusc, 
                                            info['lidar_token'], 
                                            self.num_historical_sweeps, 
                                            self.classes_of_interest,
                                            points_in_boxes_by_gpu=self.dataset_cfg.get('POINTS_IN_BOXES_GPU', False),
                                            threshold_boxes_by_points=self.dataset_cfg.get('THRESHOLD_BOXES_BY_POINTS', 5))
        
        points = stuff['points']  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
        gt_boxes = stuff['gt_boxes']  # (N_inst, 7)
        gt_names = stuff['gt_names']  # (N_inst,)
        instances_tf = stuff['instances_tf']  # (N_inst, N_sweep, 4, 4)

        input_dict = {
            'points': points,  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
            'gt_boxes': gt_boxes,  # (N_inst, 7)
            'gt_names': gt_names,  # (N_inst,)
            'instances_tf': instances_tf,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {
                'lidar_token': info['lidar_token'],
                'num_sweeps_target': self.num_sweeps,
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    

if __name__ == '__main__':
    cfg_file = './tools/cfgs/dataset_configs/v2x_sim_dataset_rsu.yaml'  # NOTE: launch this from OpenPCDet directory
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_file)))
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--create_v2x_sim_rsu_infos', action='store_true')
    parser.add_argument('--no-create_v2x_sim_rsu_infos', dest='create_v2x_sim_rsu_infos', action='store_false')
    parser.set_defaults(create_v2x_sim_rsu_infos=False)
    
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--no-training', dest='training', action='store_false')
    parser.set_defaults(training=True)

    parser.add_argument('--version', type=str, default='v2.0-trainval', help='')

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version

    if args.create_v2x_sim_rsu_infos:
        v2x_dataset = V2XSimDataset_RSU(dataset_cfg, 
                                        class_names=dataset_cfg.CLASSES_OF_INTEREST, 
                                        training=args.training,
                                        root_path=ROOT_DIR / 'data' / 'v2x-sim',
                                        logger=common_utils.create_logger())
        v2x_dataset.build_v2x_sim_info()
        # python -m pcdet.datasets.nuscenes.v2x_sim_dataset --create_v2x_sim_rsu_infos --training --version v2.0-mini