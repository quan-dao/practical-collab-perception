import numpy as np
import torch
import pickle
from pathlib import Path
import copy

from pcdet.datasets.nuscenes.v2x_sim_dataset_car import V2XSimDataset_CAR
from workspace.v2x_sim_utils import get_pseudo_sweeps_of_1lidar


class V2XSimDataset_EGO(V2XSimDataset_CAR):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.exchange_database = Path(dataset_cfg.DATABASE_EXCHANGE_DATA)
        assert self.exchange_database.exists(), f"{self.exchange_database} does not exist"

    def include_v2x_sim_data(self, mode):
        self.logger.info('Loading V2X-Sim dataset')

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / f"{self._prefix}_{info_path}"
            assert info_path.exists(), f"{info_path} does not exist"

            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.infos.extend(infos[1])  # ego vehicle's lidar: LIDAR_TOP_id_1
            
        if self.training and self.dataset_cfg.get('DATASET_DOWNSAMPLING_RATIO', 1) > 1:
            self.infos.sort(key=lambda e: e['timestamp'])
            self.infos = self.infos[::self.dataset_cfg.DATASET_DOWNSAMPLING_RATIO]
        self.logger.info('Total samples for V2X-Sim dataset: %d' % (len(self.infos)))

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
        

        # final features: x, y, z, instensity, time-lag | 3-class prob | dx, dy, dz, heading, box-score, box-label | sweep_idx, inst_idx
        points_ = np.zeros((points.shape[0], 5 + 3 + 6 + 2))
        points_[:, :5] = points[:, :5]
        points_[:, -2:] = points[:, -2:]


        # ---------------------------
        # exchange stuff
        # ---------------------------
        max_sweep_idx = points[:, -2].max()
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        for lidar_name, lidar_token in sample['data']:
            if 'LIDAR_TOP_id_' not in lidar_name:
                continue
            
            lidar_id = lidar_name.split('_')[-1]
            if lidar_id == 1:
                continue
            
            if self.dataset_cfg.get('EXCHANGE_WITH_RSU_ONLY', False) and lidar_id != 0:
                continue
            
            exchanged_points = list()

            path_foregr = self.exchange_database / f"{sample_token}_id{lidar_id}_foreground.pth"
            if path_foregr.exists():
                foregr = torch.load(path_foregr, map_location=torch.device('cpu'))  
                # (N_fore, 5 + 3) - point-5, sweep_idx, inst_idx, cls_prob-3
                foregr = torch.cat([foregr[:, :5],  # point-5  
                                    foregr[:, -3:],  # 3-class prob (backgr, stat_foregr, dyn_foregr)
                                    foregr[:, [5, 6]]  # sweep_idx, inst_idx
                                    ], dim=1).contiguous().numpy()
                foregr_ = np.zeros((foregr.shape[0], points_.shape[1]))
                foregr_[:, :8] = foregr[:, :8]
                foregr_[:, -2:] = foregr[:, -2:]
                exchanged_points.append(foregr_)
            
            path_modar = self.exchange_database / f"{sample_token}_id{lidar_id}_modar.pth"
            if path_modar.exists():
                modar = torch.load(path_modar, map_location=torch.device('cpu'))
                # (N_modar, 7 + 2) - box-7, score, label
                modar_ = np.zeros((modar.shape[0], points_.shape[1]))
                modar_[:, :3] = modar[:, :3]
                modar_[:, 8: -2] = modar[:, 3:]
                modar_[:, -2] = max_sweep_idx
                modar_[:, -1] = -1  # dummy instance_idx
                exchanged_points.append(modar_)
            
            if len(exchanged_points) > 0:
                exchanged_points = np.concatenate(exchanged_points)
                points_ = np.concatenate((points_, exchanged_points))
        
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
                # TODO: number of RSU foreground & modar
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

