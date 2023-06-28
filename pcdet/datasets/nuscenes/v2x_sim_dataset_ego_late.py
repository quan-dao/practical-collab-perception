import numpy as np
import torch
import copy
from pathlib import Path
from torch_scatter import scatter
from typing import Dict, Tuple

from pcdet.datasets.nuscenes.v2x_sim_dataset_ego import V2XSimDataset_EGO, get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_
from workspace.v2x_sim_utils import roiaware_pool3d_utils


class V2XSimDataset_EGO_LATE(V2XSimDataset_EGO):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        assert self.mode == 'test', f"late fusion only support validation"

    def _get_prediction_ego(self, sample_token: str) -> np.ndarray:
        path_modar = self.exchange_database / f"{sample_token}_id1_modar.pth"
        modar = torch.load(path_modar, map_location=torch.device('cpu')).numpy() if path_modar.exists() else np.zeros(1, 9)
        return modar
    
    @torch.no_grad()
    def _get_prediction_agent(self, lidar_id: int, lidar_token: str, sample_token: str, exchange_setting: str) -> Tuple[np.ndarray]:
        """
        Predictions are in agent's frame
        """
        assert exchange_setting in ('now', 'prev')
        glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)  # (4, 4)

        path_modar = self.exchange_database / f"{sample_token}_id{lidar_id}_modar.pth"
        if path_modar.exists():
            modar = torch.load(path_modar)  # on gpu, (N_modar, 7 + 2) - box-7, score, label
            # ---
            # propagate modar forward
            path_foregr = self.exchange_database / f"{sample_token}_id{lidar_id}_foreground.pth"
            if path_foregr.exists() and exchange_setting == 'prev':
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
        else:
            modar = np.zeros((0, 9))
        
        return modar, glob_se3_lidar
    
    def _get_lidar_token_of_present_agents(self, sample_token: str) -> Dict[int, str]:
        out = dict()
        if sample_token == '':
            return out
        
        sample = self.nusc.get('sample', sample_token)
        for sensor_name, sensor_token in sample['data'].items():
            if sensor_name not in self._lidars_name:
                continue
            lidar_id = int(sensor_name.split('_')[-1])
            out[lidar_id] = sensor_token
        
        return out
    
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        info = copy.deepcopy(self.infos[index])
        gt_boxes, gt_names = info['gt_boxes'], info['gt_names']
        # gt_boxes: (N_tot, 7)
        # gt_names: (N_tot,)

        ego_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token']))  # (4, 4)

        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)

        # get prediction of the ego vehicle @ now
        exchange_boxes, exchange_metadata = dict(), dict()
        exchange_boxes[1] = self._get_prediction_ego(sample_token)
        exchange_metadata[1] = exchange_boxes[1].shape[0]

        if self.dataset_cfg.EXCHANGE_SETTING == 'now':
            dict_lidar_id_to_token = self._get_lidar_token_of_present_agents(sample_token)
            _token_of_sample_of_interest = sample_token
        elif self.dataset_cfg.EXCHANGE_SETTING == 'prev':
            dict_lidar_id_to_token = self._get_lidar_token_of_present_agents(sample['prev'])
            _token_of_sample_of_interest = sample['prev']
        else:
            raise NotImplementedError(f"EXCHANGE_SETTING := {self.dataset_cfg.EXCHANGE_SETTING} is unknown")

        if len(dict_lidar_id_to_token) > 0:
            for lidar_id, lidar_token in dict_lidar_id_to_token.items():
                if lidar_id == 1:
                    # ego vehicle is already handled above
                    continue
                
                modar, glob_se3_lidar = self._get_prediction_agent(lidar_id, lidar_token, _token_of_sample_of_interest, self.dataset_cfg.EXCHANGE_SETTING)

                # transform modar to ego frame
                ego_se3_lidar = ego_se3_glob @ glob_se3_lidar
                modar[:, :7] = apply_se3_(ego_se3_lidar, boxes_=modar[:, :7], return_transformed=True)

                # store agent's modar
                exchange_boxes[lidar_id] = modar
                exchange_metadata[lidar_id] = modar.shape[0]

        # -----------------
        # format output
        # assemble datadict
        input_dict = {
            'points': np.zeros((1, 7)),  # Dummy | (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
            'gt_boxes': gt_boxes,  # (N_inst, 7)
            'gt_names': gt_names,  # (N_inst,)
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {
                'lidar_token': info['lidar_token'],
                'num_sweeps_target': self.num_sweeps,
                'sample_token': info['token'],
                'lidar_id': 1,
                'num_original': 0,
                'exchange': exchange_metadata,
                'exchange_boxes': exchange_boxes,  # (N_boxes_tot, 7 + 2) - box-7, score, label
            }
        }

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
