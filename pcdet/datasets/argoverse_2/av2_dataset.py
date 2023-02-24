import numpy as np
import numpy.linalg as LA
from pathlib import Path
from typing import Dict
from av2.utils.io import read_city_SE3_ego

from ..dataset import DatasetTemplate
from av2_utils import AV2Parser, AV2MapHelper, get_log_name_from_sensor_file, get_timestamp_from_feather_file, apply_SE3, yaw_from_rotz


class AV2Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        split = 'train' if training else 'val'
        self.num_sweeps = self.dataset_cfg.NUM_SWEEPS
        self.sweep_stride = self.dataset_cfg.SWEEP_STRIDE        
        self.datadir = Path(self.dataset_cfg.DATA_PATH) / split
        
        all_logs_dir = [x for x in self.datadir.iterdir() if x.is_dir()]
        self.log_name_to_log_info_dict = dict()
        self.all_lidar_files = list()
        all_relative_timestamp_ns = list()  # to sort self.all_lidar_files
        for log_dir in all_logs_dir:
            # get log token
            log_name = log_dir.parts[-1]
            
            # get timestamp of all lidar sweeps
            lidar_dir = log_dir / 'sensors' / 'lidar'
            lidar_files = np.array(list(lidar_dir.glob('*.feather')))
            lidar_timestamp_ns = np.array([get_timestamp_from_feather_file(lidar_file, return_int=True) for lidar_file in lidar_files])
            sorted_idx = np.argsort(lidar_timestamp_ns)  # increase timestamp -> past to presence
            lidar_files = lidar_files[sorted_idx]
            lidar_timestamp_ns = lidar_timestamp_ns[sorted_idx]

            self.log_name_to_log_info_dict[log_name] = {
                'lidar_timestamp_ns': lidar_timestamp_ns.tolist(), 
                'anno_file': log_dir / 'annotations.feather',
                'lidar_dir': lidar_dir,
                'map_dir': log_dir / 'map',
                'log_dir': log_dir
            }

            self.all_lidar_files += lidar_files.tolist()
           
            relative_timestamp_ns = lidar_timestamp_ns - lidar_timestamp_ns[0]
            all_relative_timestamp_ns.append(relative_timestamp_ns)

        # sort self.all_lidar_files according to relative timestamp (w.r.t timestamp of the 1st lidar in the log)
        all_relative_timestamp_ns = np.concatenate(all_relative_timestamp_ns)
        sorted_idx = np.argsort(all_relative_timestamp_ns)
        self.all_lidar_files = np.array(self.all_lidar_files)
        self.all_lidar_files = self.all_lidar_files[sorted_idx].tolist()

        # enable training on a mini partition
        if self.training and self.dataset_cfg.get('DATASET_STRIDE', 1) > 1:
            self.all_lidar_files = self.all_lidar_files[::self.dataset_cfg.DATASET_STRIDE]


    def __len__(self):
        return len(self.all_lidar_files)
    
    def __getitem__(self, idx: int):
        # extract meta data
        current_lidar_file = self.all_lidar_files[idx]
        current_lidar_timestamp_ns = get_timestamp_from_feather_file(current_lidar_file, return_int=True)
        log_name = get_log_name_from_sensor_file(current_lidar_file)
        log_info = self.log_name_to_log_info_dict[log_name]
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_info['log_dir'])

        parser = AV2Parser(log_info, current_lidar_timestamp_ns, self.num_sweeps, self.sweep_stride)
        sweep_info = parser.get_sweep_info()

        list_points, list_gt_boxes, list_gt_names = [], [], []
        dict_id_to_instance_idx: Dict[str, int] = dict()
        inst_idx = 0
        current_ego_SE3_city = LA.inv(timestamp_city_SE3_ego_dict[current_lidar_timestamp_ns].transform_matrix)

        for sweep_idx, lidar_file in enumerate(sweep_info['sweep_files']):
            # -------- #
            # points 
            # -------- #
            # get points in past egovehicle frame
            lidar_timestamp_ns = get_timestamp_from_feather_file(lidar_file, return_int=True)
            points = parser.read_av_lidar_file(lidar_file, lidar_timestamp_ns, use_offset_ns=False)

            # map points in past egovehicle to current egovehicle frame
            city_SE3_ego = timestamp_city_SE3_ego_dict[lidar_timestamp_ns].transform_matrix
            current_ego_SE3_ego = current_ego_SE3_city @ city_SE3_ego  # (4, 4)
            apply_SE3(current_ego_SE3_ego, points)

            # pad points with sweep idx
            points = np.pad(points, pad_width=[(0, 0), (0, 1)], constant_values=sweep_idx)

            # store points for concatenation
            list_points.append(points)

            # -------- #
            # boxes 
            # -------- #
            mask_at_this_ts = sweep_info['annos_timestamp_ns'] == lidar_timestamp_ns
            this_gt_names = sweep_info['annos_category'][mask_at_this_ts]  # (N,)

            # map annos @ this_ts to current egovehicle frame
            ego_SE3_this_annos = sweep_info['ego_SE3_all_annos'][mask_at_this_ts]  # (N, 4, 4)
            current_ego_SE3_this_annos = np.einsum('ij, bjk -> bik', current_ego_SE3_ego, ego_SE3_this_annos)  # (N, 4, 4)

            # get boxes instances index
            this_annos_id = sweep_info['annos_id'][mask_at_this_ts]
            this_boxes_instance_idx = []
            for box_i in range(this_gt_names.shape[0]):
                box_id = this_annos_id[box_i]
                if box_id not in dict_id_to_instance_idx:
                    # new instance
                    this_boxes_instance_idx.append(inst_idx)
                    dict_id_to_instance_idx[box_id] = inst_idx
                    # increase inst_idx (to prepare for the next instance)
                    inst_idx += 1
                else:
                    # existing instance
                    this_boxes_instance_idx.append(dict_id_to_instance_idx[box_id])

            # assembly boxes
            this_gt_boxes = np.concatenate([current_ego_SE3_this_annos[:, :3, -1],  # x, y, z
                                            sweep_info['annos_size'][mask_at_this_ts],  # dx, dy, dz
                                            yaw_from_rotz(current_ego_SE3_this_annos).reshape(-1, 1),
                                            np.array(this_boxes_instance_idx).reshape(-1, 1)], axis=1)  # (N, 8)
            # TODO: given 2 rotation matrix, find a rotation that align 2 z-axis
            this_gt_boxes = np.pad(this_gt_boxes, pad_width=[(0, 0), (0, 1)], constant_values=sweep_idx)  # (N, 9)

            # store gt_boxes
            list_gt_boxes.append(this_gt_boxes)
            list_gt_names.append(this_gt_names)
        
        # merge per-sweep points & boxes
        points = np.concatenate(list_points)  # (N_pts, 6) - x, y, z, intensity, time_sec || sweep_idx  (xyz in egovehicle )
        gt_boxes = np.concatenate(list_gt_boxes)  # (N_boxes, 9) - cx, cy, cz, dx, dy, dz, yaw || instance_idx, sweep_idx  (xyz in egovehicle)
        gt_names = np.concatenate(list_gt_names)

        # ---------------------- #
        # extracting map features 
        # ---------------------- #
        map_helper = AV2MapHelper(log_info['map_dir'], timestamp_city_SE3_ego_dict[current_lidar_timestamp_ns].transform_matrix)

        # ---
        # get points' map semantic features
        # ---
        list_pts_map_feat = list()
        for map_feat in self.dataset_cfg.POINT_MAP_FEATURES:
            if map_feat == 'flag_on_ground':
                pts_feat = map_helper.find_points_on_ground(points)
            elif map_feat == 'flag_on_drivable_area':
                pts_feat = map_helper.find_points_on_drivable_area(points)
            else:
                raise ValueError(f"{map_feat} is not supported, choose in ('flag_on_ground', 'flag_on_drivable_area')")
            list_pts_map_feat.append(pts_feat)

        pts_map_feat = np.stack(list_pts_map_feat, axis=1)  # (N, C_map)
        points = np.concatenate([points[:, self.dataset_cfg.POINT_NUM_RAW_FEATURES], 
                                 pts_map_feat, 
                                 points[:, self.dataset_cfg.POINT_NUM_RAW_FEATURES:]], 
                                 axis=1)  # (N_pts, 8) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx

        # ---
        # pull stuff down the ground
        # ---
        map_helper.compensate_ground_height(points, in_place=True)
        map_helper.compensate_ground_height(gt_boxes, in_place=True)

        # -------------------------------- #
        # points to gt_boxes correspondance 
        # -------------------------------- #
        # TODO: for each instance keep only box has the highest sweep_idx
        # TODO: reorganize gt_boxes & gt_names according to instance_idx

        # -------------------------------------------
        batch_dict = {
            'points': points,  # (N_pts, 9) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx, inst_idx
            'gt_boxes': gt_boxes[:, :7],  # (N_box, 7) - x, y, z, dx, dy, dz, yaw 
            'gt_names': gt_names,  # (N_box,)
            'frame_id': sweep_info['sweep_files'].stem,  # path to lidar file
            'metadata': {
                'log_name': log_name,
                'lidar_timestamp_ns': current_lidar_timestamp_ns,  # int
            }
        }


