import numpy as np
import torch
import pickle
from pathlib import Path
import copy
from torch_scatter import scatter
from easydict import EasyDict
from tqdm import tqdm

from pcdet.models.model_utils import model_nms_utils
from pcdet.datasets.v2x_sim.v2x_sim_dataset_car import V2XSimDataset_CAR
from pcdet.datasets.v2x_sim.v2x_sim_utils import get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_, roiaware_pool3d_utils, get_points_and_boxes_of_1lidar


class V2XSimDataset_EGO(V2XSimDataset_CAR):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.exchange_now = dataset_cfg.EXCHANGE_NOW
        self.logger.info(f'EXCHANGE_NOW: {self.exchange_now}')
        pillar_dir = 'exchange_database_flow'
        second_dir = 'exchange_database_second'
        dummy_dir = Path('blah')
        self.exchange_database = {
            0: self.root_path / pillar_dir,  # remember to change this back when eval five seconds
            1: self.root_path / pillar_dir,
            2: self.root_path / pillar_dir,
            3: self.root_path / pillar_dir,
            4: self.root_path / pillar_dir,
            5: self.root_path / pillar_dir,
        }
        # for _, path_ in self.exchange_database.items():
        #     assert path_.exists(), f"{path_} does not exist"

        self._lidars_name = set([f'LIDAR_TOP_id_{lidar_id}' for lidar_id in range(6)])  # watchout for SEMLIDAR_TOP_id_
        self._nms_config = EasyDict({
            'NMS_TYPE': 'nms_gpu',
            'NMS_THRESH': 0.2,
            'NMS_PRE_MAXSIZE': 10000,
            'NMS_POST_MAXSIZE': 10000,
        })

        if self.dataset_cfg.get('USE_GT_FROM_EVERY_AGENT', True):
            self.logger.info(f'keep only gt_boxes in range {self.dataset_cfg.EVAL_FILTER_GT_BEYOND_RANGE}')
            _path_gt_from_all = self.root_path / f"{self.mode}_gt_from_all_range60.pkl"
            if _path_gt_from_all.exists():
                self.logger.info('load gt_boxes from every agent')
                with open(_path_gt_from_all, 'rb') as f:
                    self.infos = pickle.load(f)
            else:
                self.logger.info('computing gt_boxes from every agent')
                for idx, info in tqdm(enumerate(self.infos), total=len(self.infos), desc='update gt_boxes'):
                    gt_boxes, gt_names = self.get_all_ground_truth(info['lidar_token'])
                    if self.dataset_cfg.get('EVAL_FILTER_GT_BEYOND_RANGE', -1) > 0:
                        mask_kept = np.linalg.norm(gt_boxes[:, :2], axis=1) < self.dataset_cfg.EVAL_FILTER_GT_BEYOND_RANGE
                        if np.any(mask_kept):
                            gt_boxes = gt_boxes[mask_kept]
                            gt_names = gt_names[mask_kept]
                        else:
                            gt_boxes = np.zeros((1, gt_boxes.shape[1]))
                            gt_names = gt_names[[0]]

                    self.infos[idx]['gt_boxes'] = gt_boxes
                    self.infos[idx]['gt_names'] = gt_names
                
                with open(_path_gt_from_all, 'wb') as f:
                    pickle.dump(self.infos, f)
        else:
            self.logger.info('use gt_boxes the ego vehicle only')

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

    def get_all_ground_truth(self, ego_lidar_token: str) -> np.ndarray:
        """
        Returns:
            gt_boxes: (N_tot, 7) - gt boxes from every agent present in the v2x network
        """
        ego_lidar_record = self.nusc.get('sample_data', ego_lidar_token)
        sample_record = self.nusc.get('sample', ego_lidar_record['sample_token'])

        ego_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, ego_lidar_token))

        gt_boxes, gt_names = list(), list()
        for lidar_name, lidar_token in sample_record['data'].items():
            if lidar_name not in self._lidars_name:
                continue
            _info = get_points_and_boxes_of_1lidar(self.nusc, lidar_token, 
                                                   self.classes_of_interest, 
                                                   points_in_boxes_by_gpu=True, 
                                                   threshold_boxes_by_points=1)
            
            boxes = _info['boxes_in_lidar']  # (N_box, 7)
            boxes_name = _info['boxes_name']  # (N_box,)
            
            if boxes.shape[0] > 0:
                # map boxes to ego frame
                glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
                ego_se3_lidar = ego_se3_glob @ glob_se3_lidar
                apply_se3_(ego_se3_lidar, boxes_=boxes)

                # store
                gt_boxes.append(boxes)
                gt_names.append(boxes_name)
        
        # merge by removing duplicate using NMS
        gt_boxes = np.concatenate(gt_boxes)  # (N, 7)
        gt_names = np.concatenate(gt_names)

        gt_boxes = torch.from_numpy(gt_boxes).float().cuda()
        selected, _ = model_nms_utils.class_agnostic_nms(
            box_scores=gt_boxes.new_ones(gt_boxes.shape[0]), 
            box_preds=gt_boxes,
            nms_config=self._nms_config
        )

        # format output
        selected = selected.long().cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()[selected]  # (N_tot, 7)
        gt_names = gt_names[selected]  # (N_tot,)
        return gt_boxes, gt_names

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
        
        # final features: x, y, z, instensity, time-lag | dx, dy, dz, heading, box-score, box-label | pr-bg, pr-stat, pr-dyn | sweep_idx, inst_idx
        points_ = np.zeros((points.shape[0], 5 + 6 + 2))
        points_[:, :5] = points[:, :5]
        points_[:, -2:] = points[:, -2:]
        num_original = points_.shape[0]

        target_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, info['lidar_token']))
        # ---------------------------
        # exchange stuff
        # ---------------------------
        max_sweep_idx = points[:, -2].max()
        sample_token = info['token']
        sample = self.nusc.get('sample', sample_token)
        exchange_metadata = dict([(i, [0., 0.]) for i in range(6) if i != 1])
        exchange_coord = dict([(i, np.zeros(3)) for i in range(6) if i != 1])
        
        if sample['prev'] != '' and not self.exchange_now:
            prev_sample_token = sample['prev']
            prev_sample = self.nusc.get('sample', prev_sample_token)

            for lidar_name, lidar_token in prev_sample['data'].items():
                if lidar_name not in self._lidars_name:
                    continue
                
                lidar_id = int(lidar_name.split('_')[-1])
                if lidar_id == 1:
                    continue
                
                glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
                target_se3_lidar = target_se3_glob @ glob_se3_lidar

                exchange_database = self.exchange_database[lidar_id]
                path_foregr = exchange_database / f"{prev_sample_token}_id{lidar_id}_foreground.pth"
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
                    modar_[:, 4] = 0.  # after offset, trying set it to zero
                    modar_[:, 5: 11] = modar[:, 3:]
                    modar_[:, -2] = max_sweep_idx
                    modar_[:, -1] = -1  # dummy instance_idx
                    # log metadata
                    exchange_metadata[lidar_id][1] = modar_.shape[0]
                    exchange_coord[lidar_id] = target_se3_lidar[:3, -1]
                
                    # merge mordar with point cloud
                    points_ = np.concatenate((points_, modar_))

        elif self.exchange_now:
            for lidar_name, lidar_token in sample['data'].items():
                if lidar_name not in self._lidars_name:
                    continue
                
                lidar_id = int(lidar_name.split('_')[-1])
                if lidar_id == 1:
                    continue
                
                glob_se3_lidar = get_nuscenes_sensor_pose_in_global(self.nusc, lidar_token)
                target_se3_lidar = target_se3_glob @ glob_se3_lidar

                exchange_database = self.exchange_database[lidar_id]
                path_foregr = exchange_database / f"{sample_token}_id{lidar_id}_foreground.pth"
                path_modar = exchange_database / f"{sample_token}_id{lidar_id}_modar.pth"
                if path_modar.exists() and self.dataset_cfg.EXCHANGE_MODAR:
                    modar = torch.load(path_modar)  # on gpu, (N_modar, 7 + 2) - box-7, score, label
                    
                    modar = modar.cpu().numpy()
                    # map modar (center & heading) to target frame
                    modar[:, :7] = apply_se3_(target_se3_lidar, boxes_=modar[:, :7], return_transformed=True)

                    modar_ = np.zeros((modar.shape[0], points_.shape[1]))
                    modar_[:, :3] = modar[:, :3]
                    modar_[:, 4] = 0.  # after offset, trying set it to zero
                    modar_[:, 5: 11] = modar[:, 3:]
                    modar_[:, -2] = max_sweep_idx
                    modar_[:, -1] = -1  # dummy instance_idx
                    # log metadata
                    exchange_metadata[lidar_id][1] = modar_.shape[0]
                    exchange_coord[lidar_id] = target_se3_lidar[:3, -1]
                
                    # merge mordar with point cloud
                    points_ = np.concatenate((points_, modar_))
        
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

