import numpy as np
import torch
import pickle
from pathlib import Path
import copy
from torch_scatter import scatter

from pcdet.datasets.nuscenes.v2x_sim_dataset_car import V2XSimDataset_CAR
from workspace.v2x_sim_utils import get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_, roiaware_pool3d_utils


class V2XSimDataset_EGO(V2XSimDataset_CAR):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        
        self.exchange_database = self.root_path / dataset_cfg.EXCHANGE_DATABASE_DIRECTORY
        assert self.exchange_database.exists(), f"{self.exchange_database} does not exist"

        self._lidars_name = set([f'LIDAR_TOP_id_{lidar_id}' for lidar_id in range(6)])  # watchout for SEMLIDAR_TOP_id_

    def include_v2x_sim_data(self, mode):
        self.logger.info('Loading V2X-Sim dataset')

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / f"{self._prefix}_{info_path}"
            assert info_path.exists(), f"{info_path} does not exist"

            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            
            for _info in infos[1]:  # ego vehicle's lidar: LIDAR_TOP_id_1
                lidar_rec = self.nusc.get('sample_data', _info['lidar_token'])
                if 'SEM' not in lidar_rec['channel']:
                    self.infos.append(_info)  
            
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
        

        # final features: x, y, z, instensity, time-lag | dx, dy, dz, heading, box-score, box-label | sweep_idx, inst_idx
        points_ = np.zeros((points.shape[0], 5 + 6 + 2))
        points_[:, :5] = points[:, :5]
        points_[:, -2:] = points[:, -2:]
        num_original = points_.shape[0]

        target_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token']))
        # ---------------------------
        # exchange stuff
        # ---------------------------
        max_sweep_idx = points[:, -2].max()
        max_time_lag = points[:, 4].max()
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        exchange_metadata = dict([(i, [0., 0.]) for i in range(6) if i != 1])
        exchange_coord = dict([(i, np.zeros(3)) for i in range(6) if i != 1])
        if sample['prev'] != '':
            prev_sample_token = sample['prev']
            prev_sample = self.nusc.get('sample', prev_sample_token)

            for lidar_name, lidar_token in prev_sample['data'].items():
                if lidar_name not in self._lidars_name:
                    continue
                
                lidar_id = int(lidar_name.split('_')[-1])
                if lidar_id == 1:
                    continue

                if self.dataset_cfg.get('EXCHANGE_WITH_RSU_ONLY', False) and lidar_id != 0:
                    continue
                
                glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
                target_se3_lidar = target_se3_glob @ glob_se3_lidar

                exchange_database = self.exchange_database
                exchanged_points = list()
                path_foregr = exchange_database / f"{prev_sample_token}_id{lidar_id}_foreground.pth"
                if path_foregr.exists() and self.dataset_cfg.EXCHANGE_FOREGROUND:
                    foregr = torch.load(path_foregr, map_location=torch.device('cpu'))  
                    # (N_fore, 5 + 3) - point-5, sweep_idx, inst_idx, cls_prob-3
                    foregr[:, 4] = 0.  # zero time-lag as foregr were corrected
                    foregr = torch.cat([foregr[:, :5],  # point-5  
                                        foregr[:, -3:],  # 3-class prob (backgr, stat_foregr, dyn_foregr)
                                        foregr[:, [5, 6]]  # sweep_idx, inst_idx
                                        ], dim=1).contiguous().numpy()
                    foregr_ = np.zeros((foregr.shape[0], points_.shape[1]))
                    foregr_[:, :8] = foregr[:, :8]
                    foregr_[:, -2:] = foregr[:, -2:]

                    # map foregr_ to target frame
                    foregr_[:, :3] = apply_se3_(target_se3_lidar, points_=foregr_[:, :3], return_transformed=True)

                    exchanged_points.append(foregr_)
                    # log metadata
                    exchange_metadata[lidar_id][0] = foregr_.shape[0]
                
                path_modar = exchange_database / f"{prev_sample_token}_id{lidar_id}_modar.pth"
                if path_modar.exists() and self.dataset_cfg.EXCHANGE_MODAR:
                    modar = torch.load(path_modar)  # on gpu, (N_modar, 7 + 2) - box-7, score, label
                    # move MoDAR at the previous time step according to foregrounds' offset
                    with torch.no_grad():
                        if path_foregr.exists():
                            foregr = torch.load(path_foregr)  # on gpu, (N_fore, 5 + 2 + 3 + 3) - point-5, sweep_idx, inst_idx, cls_prob-3, flow-3
                            
                            # pool
                            box_idx_of_foregr = roiaware_pool3d_utils.points_in_boxes_gpu(
                                foregr[:, :3].unsqueeze(0), modar[:, :7].unsqueeze(0)
                            ).squeeze(0).long()  # (N_foregr,) | == -1 mean not belong to any boxes
                            mask_valid_foregr = box_idx_of_foregr > -1
                            foregr = foregr[mask_valid_foregr]
                            box_idx_of_foregr = box_idx_of_foregr[mask_valid_foregr]
                            
                            unq_box_idx, inv_unq_box_idx = torch.unique(box_idx_of_foregr, return_inverse=True)

                            # weighted sum of foregrounds' offset; weights = foreground's prob dynamic
                            boxes_offset = scatter(foregr[:, -3:], inv_unq_box_idx, dim=0, reduce='mean') * 2.  # (N_modar, 3)
                            # offset modar; here, assume objects maintain the same speed
                            modar[unq_box_idx, :3] += boxes_offset    

                    modar = modar.cpu().numpy()
                    # map modar (center & heading) to target frame
                    modar[:, :7] = apply_se3_(target_se3_lidar, boxes_=modar[:, :7], return_transformed=True)

                    modar_ = np.zeros((modar.shape[0], points_.shape[1]))
                    modar_[:, :3] = modar[:, :3]
                    modar_[:, 4] = max_time_lag  # TODO: after offset, should this be zero?
                    modar_[:, 5: -2] = modar[:, 3:]
                    modar_[:, -2] = max_sweep_idx
                    modar_[:, -1] = -1  # dummy instance_idx
                    exchanged_points.append(modar_)
                    # log metadata
                    exchange_metadata[lidar_id][1] = modar_.shape[0]
                    exchange_coord[lidar_id] = target_se3_lidar[:3, -1]
                
                if len(exchanged_points) > 0:
                    exchanged_points = np.concatenate(exchanged_points)
                    points_ = np.concatenate((points_, exchanged_points))
        
        # assemble datadict
        input_dict = {
            'points': points_,  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
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
                'exchange_coord': exchange_coord
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

