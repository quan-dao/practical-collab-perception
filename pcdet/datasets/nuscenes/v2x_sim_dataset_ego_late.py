import numpy as np
import torch
import copy
from pathlib import Path

from pcdet.datasets.nuscenes.v2x_sim_dataset_ego import V2XSimDataset_EGO, get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_


class V2XSimDataset_EGO_LATE(V2XSimDataset_EGO):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        assert self.mode == 'test', f"late fusion only support validation"

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
        gt_boxes = ego_stuff['gt_boxes']  # (N_inst, 7)
        gt_names = ego_stuff['gt_names']  # (N_inst,)
        num_original = points.shape[0]

        target_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token']))

        # ---------------------------
        # exchange stuff
        # ---------------------------
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        exchange_metadata = dict([(i, 0.) for i in range(6) if i != 1])
        exchange_boxes = list()
        for lidar_name, lidar_token in sample['data'].items():
            if lidar_name not in self._lidars_name:
                continue
            
            lidar_id = int(lidar_name.split('_')[-1])
            
            if self.dataset_cfg.get('EXCHANGE_WITH_RSU_ONLY', False) and lidar_id not in (0, 1):
                continue

            glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
            target_se3_lidar = target_se3_glob @ glob_se3_lidar

            path_modar = self.exchange_database / f"{sample_token}_id{lidar_id}_modar.pth"
            if path_modar.exists():
                modar = torch.load(path_modar, map_location=torch.device('cpu')).numpy()
                # (N_modar, 7 + 2) - box-7, score, label

                # map modar (center & heading) to target frame
                modar[:, :7] = apply_se3_(target_se3_lidar, boxes_=modar[:, :7], return_transformed=True)
            
            # store
            exchange_metadata[lidar_id] = modar.shape[0]
            exchange_boxes.append(modar)

        if len(exchange_boxes) > 0:
            exchange_boxes = np.concatenate(exchange_boxes, axis=0)
        else:
            exchange_boxes = np.zeros((0, 9))
        
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
                'exchange_boxes': exchange_boxes,  # (N_boxes_tot, 7 + 2) - box-7, score, label
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

