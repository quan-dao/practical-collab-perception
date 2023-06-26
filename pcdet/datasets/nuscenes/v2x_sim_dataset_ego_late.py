import numpy as np
import torch
import copy
from pathlib import Path
from torch_scatter import scatter

from pcdet.datasets.nuscenes.v2x_sim_dataset_ego import V2XSimDataset_EGO, get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_
from workspace.v2x_sim_utils import roiaware_pool3d_utils


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
        exchange_metadata = dict([(i, 0.) for i in range(6) if i != 1])
        exchange_boxes = dict([(i, np.zeros((0, 9))) for i in range(6) if i != 1])
        if sample['prev'] != '':
            prev_sample_token = sample['prev']
            prev_sample = self.nusc.get('sample', prev_sample_token)
            for lidar_name, lidar_token in prev_sample['data'].items():
                if lidar_name not in self._lidars_name:
                    continue
                
                lidar_id = int(lidar_name.split('_')[-1])
                
                if self.dataset_cfg.get('EXCHANGE_WITH_RSU_ONLY', False) and lidar_id not in (0, 1):
                    continue

                glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
                target_se3_lidar = target_se3_glob @ glob_se3_lidar

                if lidar_id == 1:
                    path_modar = self.exchange_database / f"{sample_token}_id{lidar_id}_modar.pth"
                    if path_modar.exists():
                        modar = torch.load(path_modar, map_location=torch.device('cpu')).numpy()
                        # (N_modar, 7 + 2) - box-7, score, label

                        # map modar (center & heading) to target frame
                        modar[:, :7] = apply_se3_(target_se3_lidar, boxes_=modar[:, :7], return_transformed=True)
                    else:
                        modar = np.zeros((1, 9))
                else:
                    path_modar = self.exchange_database / f"{prev_sample_token}_id{lidar_id}_modar.pth"
                    if path_modar.exists():
                        modar = torch.load(path_modar)  # on gpu, (N_modar, 7 + 2) - box-7, score, label
                        # move MoDAR at the previous time step according to foregrounds' offset
                        with torch.no_grad():
                            path_foregr = self.exchange_database / f"{prev_sample_token}_id{lidar_id}_foreground.pth"
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

                # store
                exchange_boxes[lidar_id] = modar
                exchange_metadata[lidar_id] = modar.shape[0]
        else:
            path_modar = self.exchange_database / f"{sample_token}_id1_modar.pth"
            if path_modar.exists():
                modar = torch.load(path_modar, map_location=torch.device('cpu')).numpy()
                # (N_modar, 7 + 2) - box-7, score, label
            else:
                modar = np.zeros((1, 9))
            exchange_boxes[1] = modar
            # store
            exchange_metadata[1] = modar.shape[0]

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

