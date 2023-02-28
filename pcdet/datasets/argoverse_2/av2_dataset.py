import torch
from torch_scatter import scatter_max
import numpy as np
import numpy.linalg as LA
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import pickle
from av2.utils.io import read_city_SE3_ego

from ..dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from .av2_utils import AV2Parser, AV2MapHelper, get_log_name_from_sensor_file, get_timestamp_from_feather_file, apply_SE3, yaw_from_rotz, \
    transform_det_annos_to_av2_feather


class AV2Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        split = 'train' if training else 'val'
        self.num_sweeps = self.dataset_cfg.NUM_SWEEPS
        self.sweep_stride = self.dataset_cfg.SWEEP_STRIDE        
        self.datadir = root_path / split
        
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
    
    def __getitem__(self, idx: int, **kwargs):
        # extract meta data
        current_lidar_file = self.all_lidar_files[idx]
        current_lidar_timestamp_ns = get_timestamp_from_feather_file(current_lidar_file, return_int=True)
        log_name = get_log_name_from_sensor_file(current_lidar_file)
        log_info = self.log_name_to_log_info_dict[log_name]
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_info['log_dir'])

        parser = AV2Parser(log_info, current_lidar_timestamp_ns, self.num_sweeps, self.sweep_stride, 
                           self.dataset_cfg.DETECTION_CLS if kwargs.get('use_classes', None) is None else kwargs['use_classes'])
        sweep_info = parser.get_sweep_info()

        list_points, list_gt_boxes, list_gt_names, list_gt_poses = [], [], [], []
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
            if not np.any(mask_at_this_ts):
                continue
            this_gt_names = sweep_info['annos_category'][mask_at_this_ts]  # (N,)

            # map annos @ this_ts to current egovehicle frame
            ego_SE3_this_annos = sweep_info['ego_SE3_annos'][mask_at_this_ts]  # (N, 4, 4)
            current_ego_SE3_this_annos = np.einsum('ij, bjk -> bik', current_ego_SE3_ego, ego_SE3_this_annos)  # (N, 4, 4)
            list_gt_poses.append(current_ego_SE3_this_annos)

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

        if len(list_gt_boxes) > 0:
            gt_boxes = np.concatenate(list_gt_boxes)\
                .astype(float)  # (N_boxes, 9) - cx, cy, cz, dx, dy, dz, yaw || instance_idx, sweep_idx  (xyz in egovehicle)
            gt_names = np.concatenate(list_gt_names)  # (N_boxes,)
            gt_poses = np.concatenate(list_gt_poses).astype(float)  # (N_boxes, 4, 4) - to build instance_tf
        else:
            gt_boxes, gt_names, gt_poses = np.zeros((0, 9)), np.zeros(0), np.zeros((0, 4, 4))

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

        points = np.concatenate([points[:, :self.dataset_cfg.POINT_NUM_RAW_FEATURES], 
                                 pts_map_feat, 
                                 points[:, self.dataset_cfg.POINT_NUM_RAW_FEATURES:]], 
                                 axis=1)  # (N_pts, 8) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx

        # ---
        # pull stuff down the ground
        # ---
        if self.dataset_cfg.get('MAP_COMPENSATE_GROUND_HEIGHT', False):
            map_helper.compensate_ground_height(points, in_place=True)

            if gt_boxes.shape[0] > 0:
                map_helper.compensate_ground_height(gt_boxes, in_place=True)

        # ----------------------
        # processing instances
        # ----------------------

        # ---
        # points to gt_boxes correspondance 
        # ---
        if gt_boxes.shape[0] > 0:
            box_idx_of_points = points_in_boxes_gpu(
                torch.from_numpy(points[:, :3]).unsqueeze(0).contiguous().float().cuda(),
                torch.from_numpy(gt_boxes[:, :7]).unsqueeze(0).contiguous().float().cuda(),
            ).long().squeeze(0).cpu().numpy()  # (N_pts,) to index into (N_boxes,)
            points_inst_idx = gt_boxes[box_idx_of_points, -2]  # NOTE: bg points has box_idx_of_points == -1, 
            # but this index will return a non negative value anyway 
            # => set instance idx of background points to -1
            mask_background_pts = box_idx_of_points == -1
            points_inst_idx[mask_background_pts] = -1
            points = np.concatenate([points, 
                                     points_inst_idx.reshape(-1, 1)], 
                                     axis=1)  # (N_pts, 9) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx, inst_idx
        else:
            points = np.pad(points, pad_width=[(0, 0), (0, 1)], constant_values=-1)  # all points are background

        # ---
        # build instance_tf
        # ---
        if gt_boxes.shape[0] > 0:
            gt_boxes_inst_idx = gt_boxes[:, -2].astype(int)
            unq_inst_ids = np.unique(gt_boxes_inst_idx)  # (N_inst,)

            instances_tf = np.tile(np.eye(4)[np.newaxis, np.newaxis, ...], (unq_inst_ids.shape[0], self.num_sweeps, 1, 1))
            for ii, inst_idx in enumerate(unq_inst_ids.tolist()):
                mask_this_instance = gt_boxes_inst_idx == inst_idx
                this_inst_poses = gt_poses[mask_this_instance]  # (N_box_of_inst, 4, 4)
                instances_tf[ii, :this_inst_poses.shape[0]] = np.einsum('ij, bjk -> bik', this_inst_poses[-1], LA.inv(this_inst_poses))
        else:
            instances_tf = np.zeros((0, 4, 4))

        if gt_boxes.shape[0] > 0:
            # for each instance keep only box has the highest sweep_idx
            gt_boxes, gt_names = self._filter_past_gt_boxes(gt_boxes, gt_names)
            # reorganize gt_boxes by instances_idx
            gt_boxes, gt_names = self._sort_gt_boxes_by_instance_idx(gt_boxes, gt_names)

        # -------------------------------------------
        # remove points having NaN
        mask_pts_have_nan = np.isnan(points).any(axis=1)
        points = points[np.logical_not(mask_pts_have_nan)]

        data_dict = {
            'points': points,  # (N_pts, 9) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx, inst_idx
            'gt_boxes': gt_boxes[:, :7],  # (N_box, 7) - x, y, z, dx, dy, dz, yaw 
            'gt_names': gt_names,  # (N_box,)
            'instances_tf': instances_tf,  # (N_inst, N_sweeps, 4, 4)
            'frame_id': sweep_info['sweep_files'][-1].stem,  # path to lidar file
            'metadata': {
                'log_name': log_name,
                'lidar_timestamp_ns': current_lidar_timestamp_ns,  # int
                'num_sweeps': self.num_sweeps
            }
        }
        
        if not kwargs.get('making_gt_database', False):
            # data augmentation & other stuff
            data_dict = self.prepare_data(data_dict=data_dict)

        return data_dict

    @staticmethod
    def _filter_past_gt_boxes(gt_boxes: np.ndarray, gt_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each instance, keep its most recent box (i.e. the one has the highest sweep_idx)

        Args:
            gt_boxes: (N_boxes, 9) - cx, cy, cz, dx, dy, dz, yaw || instance_idx, sweep_idx 
            gt_names: (N_boxes,)
        
        Return:
            gt_boxes: (N_boxes_, 9) - cx, cy, cz, dx, dy, dz, yaw || instance_idx, sweep_idx 
            gt_names: (N_boxes_,)
            NOTE: N_boxes_ << N_boxes
        """
        # group boxes according to instance_idx
        _, inv_indices = torch.unique(torch.from_numpy(gt_boxes[:, -2]).long(), return_inverse=True)
        # find max sweep_idx of each instance
        _, boxes_idx = scatter_max(torch.from_numpy(gt_boxes[:, -1]).long(), inv_indices, dim=0)
        return gt_boxes[boxes_idx], gt_names[boxes_idx]
    
    @staticmethod
    def _sort_gt_boxes_by_instance_idx(gt_boxes: np.ndarray, gt_names: np.ndarray) -> None:
        """
        Args:
            gt_boxes: (N_boxes, 9) - cx, cy, cz, dx, dy, dz, yaw || instance_idx, sweep_idx 
            gt_names: (N_boxes,)
        """
        sorted_idx = np.argsort(gt_boxes[:, -2].astype(int))
        return gt_boxes[sorted_idx], gt_names[sorted_idx]

    def evaluation(self, det_annos: List[Dict], class_names: List[str], **kwargs):
        from av2.evaluation.detection.eval import evaluate
        from av2.evaluation.detection.utils import DetectionCfg
        from av2.utils.io import read_all_annotations
        import pandas as pd

        assert kwargs['eval_metric'] == 'argoverse_2', f"eval_metric {kwargs['eval_metric']} is not supported"

        dataset_dir = Path(self.dataset_cfg.DATA_PATH)
        competition_cfg = DetectionCfg(dataset_dir=dataset_dir)  # Defaults to competition parameters.

        split = "val"
        gts = read_all_annotations(dataset_dir=dataset_dir, split=split)  # Contains all annotations in a particular split.

        detection_df = transform_det_annos_to_av2_feather(det_annos, class_names)

        dts, gts, metrics = evaluate(dts, gts, cfg=competition_cfg)  # Evaluate instances.
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(metrics)

        return '', {'mAP': metrics.loc['AVERAGE_METRICS'].to_numpy().mean().item()}  # TODO

    def create_groundtruth_database(self, use_classes: Tuple = None) -> None:
        database_save_path = self.root_path / f'gt_database_{self.num_sweeps}sweeps'
        database_save_path.mkdir(parents=True, exist_ok=True)

        if use_classes is None:
            use_classes = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'LARGE_VEHICLE', 'WHEELED_DEVICE', 'BUS', 
                           'BOX_TRUCK', 'TRUCK', 'MOTORCYCLE', 'BICYCLIST', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 
                           'MOTORCYCLIST','SCHOOL_BUS', 'WHEELED_RIDER', 'ARTICULATED_BUS', 'RAILED_VEHICLE')
        
        all_db_infos = dict()

        for sample_idx in tqdm(range(len(self.all_lidar_files))):
            data_dict = self.__getitem__(sample_idx, use_classes=use_classes, making_gt_database=True)
            points = data_dict['points']  # (N_pts, 9) - x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx, inst_idx
            gt_boxes = data_dict['gt_boxes']  # (N_box, 7) - x, y, z, dx, dy, dz, yaw 
            gt_names = data_dict['gt_names']  # (N_box,)
            instances_tf = data_dict['instances_tf']  # (N_inst, N_sweeps, 4, 4)
            # NOTE: N_box == N_inst
            
            if gt_boxes.shape[0] == 0:
                # this sample doesn't have any gt
                continue

            assert points.shape[1] == 9, f"expect points.shape[1] == 9, get {points.shape[1]}"
            assert gt_boxes.shape[1] == 7, f"expect gt_boxes.shape[1] == 7, get {gt_boxes.shape[1]}"
            assert gt_boxes.shape[0] == instances_tf.shape[0],\
                f"expect gt_boxes.shape[0] == instances_tf.shape[0], get {gt_boxes.shape[0]} != {instances_tf.shape[0]}" 
            
            points_inst_idx = points[:, -1].astype(int)

            for cur_inst_idx in range(gt_boxes.shape[0]):
                filename = f"{sample_idx}_{gt_names[cur_inst_idx]}_{cur_inst_idx}.bin"
                filepath = database_save_path / filename
                
                gt_points = points[points_inst_idx == cur_inst_idx]
                if gt_points.shape[0] == 0:  # there is no points in this gt_boxes
                    continue
                
                # translate gt_points to frame whose origin @ gt_box center
                gt_points[:, :3] -= gt_boxes[cur_inst_idx, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {
                    'name': gt_names[cur_inst_idx], 
                    'path': db_path, 
                    'image_idx': sample_idx, 
                    'gt_idx': cur_inst_idx,
                    'num_points_in_gt': gt_points.shape[0],
                    'box3d_lidar': gt_boxes[cur_inst_idx],
                    'instance_idx': cur_inst_idx,
                    'instance_tf': instances_tf[cur_inst_idx],  # (max_sweeps, 4, 4)
                }

                if gt_names[cur_inst_idx] in all_db_infos:
                    all_db_infos[gt_names[cur_inst_idx]].append(db_info)
                else:
                    all_db_infos[gt_names[cur_inst_idx]] = [db_info]
        
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        db_info_save_path = self.root_path / f'av2_dbinfos_{self.num_sweeps}sweeps.pkl'
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    from ...utils import common_utils


    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')

    parser.add_argument('--create_groundtruth_database', action='store_true')
    parser.add_argument('--no-create_groundtruth_database', dest='create_groundtruth_database', action='store_false')
    parser.set_defaults(create_groundtruth_database=False)
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

    if args.create_groundtruth_database:
        av2_dataset = AV2Dataset(dataset_cfg, class_names=None, training=True, 
                                 root_path=ROOT_DIR / 'data' / 'argoverse_2',
                                 logger=common_utils.create_logger())
        av2_dataset.create_groundtruth_database()
        # $ python -m pcdet.datasets.argoverse_2.av2_dataset --cfg_file tools/cfgs/dataset_configs/argoverse2_dataset.yaml --create_groundtruth_database
