import numpy as np
import copy
from pathlib import Path

from pcdet.datasets.nuscenes.v2x_sim_dataset_ego import V2XSimDataset_EGO, get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_


class V2XSimDataset_EGO_DISCO(V2XSimDataset_EGO):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.exchange_database = None  # don't need this in early fusion
        if self.dataset_cfg.get('EXCHANGE_PREVIOUS', False):
            self.logger.info('exchange prev feat map for DiscoNet')
            # remove info that do not have prev
            valid_idx = []
            for idx, info in enumerate(self.infos):
                sample = self.nusc.get('sample', info['token'])
                if sample['prev'] != '':
                    valid_idx.append(idx)
            
            self.logger.info(f"num samples have previous: {len(valid_idx)} ({float(len(valid_idx)) / len(self)})")
            self.infos = [self.infos[_i] for _i in valid_idx]

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        info = copy.deepcopy(self.infos[index])

        # ---------------------------
        # ego vehicle's stuff
        # ---------------------------
        ego_stuff = get_pseudo_sweeps_of_1lidar(self.nusc, 
                                            info['lidar_token'], 
                                            self.num_historical_sweeps, 
                                            self.classes_of_interest,
                                            points_in_boxes_by_gpu=self.dataset_cfg.get('POINTS_IN_BOXES_GPU', False),
                                            threshold_boxes_by_points=self.dataset_cfg.get('THRESHOLD_BOXES_BY_POINTS', 5))
        
        points = ego_stuff['points']  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx (for debugging purpose only)
        # replace points' last 2 channels with agent index
        points = np.concatenate([points[:, :5],  # point-5 
                                 np.ones((points.shape[0], 1))  # agent-idx, 1 for ego vehicle
                                 ], axis=1)
        gt_boxes, gt_names = info['gt_boxes'], info['gt_names']
        # gt_boxes: (N_tot, 7)
        # gt_names: (N_tot,)
        num_original = points.shape[0]

        target_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token']))

        # ---------------------------
        # exchange stuff
        # ---------------------------
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        if self.dataset_cfg.get('EXCHANGE_PREVIOUS', False):
            sample_token = sample['prev']
            sample = self.nusc.get('sample', sample_token)
        exchange_metadata = dict([(i, 0.) for i in range(6) if i != 1])
        exchange_points = list()
        se3_from_ego = dict()
        for lidar_name, lidar_token in sample['data'].items():
            if lidar_name not in self._lidars_name:
                continue
            
            lidar_id = int(lidar_name.split('_')[-1])
            if lidar_id == 1:
                continue

            glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
            target_se3_lidar = target_se3_glob @ glob_se3_lidar

            exchange_stuff = get_pseudo_sweeps_of_1lidar(self.nusc, 
                                                         lidar_token, 
                                                         self.num_historical_sweeps,
                                                         self.classes_of_interest,
                                                         points_in_boxes_by_gpu=self.dataset_cfg.get('POINTS_IN_BOXES_GPU', True),
                                                         threshold_boxes_by_points=self.dataset_cfg.get('THRESHOLD_BOXES_BY_POINTS', 1))
            _xpoints = exchange_stuff['points']  # (N_xpts, 5 + 2) - point-5, sweep_idx, inst_idx
            # replace points' last 2 channels with agent index
            _xpoints = np.concatenate([_xpoints[:, :5], 
                                       np.zeros((_xpoints.shape[0], 1)) + lidar_id
                                       ], axis=1)

            if self.dataset_cfg.get('EXCHANGE_CURRENT_ONLY', False):
                mask_current = _xpoints[:, -2].astype(int) == _xpoints[:, -2].max()
                _xpoints = _xpoints[mask_current]

            # map _xpoints to target frame
            _xpoints[:, :3] = apply_se3_(target_se3_lidar, points_=_xpoints[:, :3], return_transformed=True)

            # store
            exchange_metadata[lidar_id] = _xpoints.shape[0]
            exchange_points.append(_xpoints)
            se3_from_ego[lidar_id] = np.linalg.inv(target_se3_lidar)

        if len(exchange_points) > 0:
            # use sample['prev'] != '' to account for 1st sample in a sequence
            points = np.concatenate([points, *exchange_points], axis=0)
        
        # assemble datadict
        input_dict = {
            'points': points,  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
            'gt_boxes': gt_boxes,  # (N_inst, 7)
            'gt_names': gt_names,  # (N_inst,)
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {
                'lidar_token': info['lidar_token'],
                'num_sweeps_target': self.num_sweeps,
                'sample_token': info['token'],
                'lidar_id': 1,
                'num_original': num_original,
                'exchange': exchange_metadata,
                'se3_from_ego': se3_from_ego,
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

