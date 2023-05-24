import numpy as np
import torch
import copy
import pickle
from pathlib import Path
from typing import List
from tqdm import tqdm
import time

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate


from workspace.rev_get_sweeps_instance_centric import revised_instance_centric_get_sweeps
from workspace.nuscenes_map_helper import MapMaker
from workspace.uda_database_utils import create_database, load_1traj
from workspace.nuscenes_temporal_utils import get_one_pointcloud, apply_se3_, get_sweeps_token, \
    get_nuscenes_sensor_pose_in_global, compute_correction_tf


class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

        self.infos.sort(key=lambda e: e['timestamp'])
        if training and self.dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1) > 1:
            stride = self.dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1)
            self.infos = self.infos[::stride]  # use 1/4th of the trainval data

        self.nusc = NuScenes(dataroot=root_path, version=dataset_cfg.VERSION, verbose=False)
        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE)

        # -----------------------------------
        # prepare database sampling
        database_root = Path(self.dataset_cfg.DATABASE_ROOT)
        if database_root.exists():
            database_classes = [database_root / cls_name for cls_name in class_names]
            self.gt_database = dict()
            for cls_name, database_cls in zip(class_names, database_classes):
                cls_trajs_path = list(database_cls.glob('*.pkl'))
                cls_trajs_path.sort()
            
                # filter traj by length
                invalid_traj_ids = list()
                for traj_idx, traj_path in enumerate(cls_trajs_path):
                    with open(traj_path, 'rb') as f:
                        traj_info = pickle.load(f)
                    if len(traj_info) < self.dataset_cfg.MAX_SWEEPS:
                        invalid_traj_ids.append(traj_idx)

                invalid_traj_ids.reverse()
                for _i in invalid_traj_ids:
                    del cls_trajs_path[_i]
                
                self.gt_database[cls_name] = cls_trajs_path

            self.sample_group = dict()
            for group in self.dataset_cfg.SAMPLE_GROUP:  # ['car: 5', 'pedestrian: 5']
                cls_name, num_to_sample = group.strip().split(':')
                self.sample_group[cls_name] = {
                    'num_to_sample': num_to_sample,
                    'pointer': 0,
                    'indices': np.random.permutation(len(self.gt_database[cls_name])),
                    'num_trajectories': len(self.gt_database[cls_name])
                }

        else:
            print('WARNING | database is not created yet')

    def sample_with_fixed_number(self, class_name: str) -> List[Path]:
        sample_group = self.sample_group[class_name]
        if sample_group['pointer'] + sample_group['num_to_sample'] >= sample_group['num_trajectories']:
            sample_group['indices'] = np.random.permutation(sample_group['num_trajectories'])
            sample_group['pointer'] = 0
        
        pointer, num_to_sample = sample_group['pointer'], sample_group['num_to_sample']
        out = [self.gt_database[class_name][idx] for idx in sample_group['indices'][pointer: pointer + num_to_sample]]
        
        sample_group['pointer'] += num_to_sample
        return out

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = f"{'mini' if 'mini' in self.dataset_cfg.VERSION else 'full'}_{info_path}"
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        num_sweeps = self.dataset_cfg.MAX_SWEEPS

        # -----------------------------
        # get original points
        sample = self.nusc.get('sample', info['token'])
        current_lidar_tk = sample['data']['LIDAR_TOP']
        current_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, current_lidar_tk))

        sweeps_info = get_sweeps_token(self.nusc, current_lidar_tk, n_sweeps=num_sweeps, return_time_lag=True, return_sweep_idx=True)
        points_original = list()
        for (lidar_tk, timelag, sweep_idx) in sweeps_info:
            pcd = get_one_pointcloud(self.nusc, lidar_tk)
            pcd = np.pad(pcd, pad_width=[(0, 0), (0, 3)], constant_values=-1)
            pcd[:, -3] = timelag
            pcd[:, -2] = sweep_idx
            # pcd[:, -1] is instance index which is -1 in the context of UDA

            # map pcd to current frame (EMC)
            glob_se3_past =  get_nuscenes_sensor_pose_in_global(self.nusc, lidar_tk)
            current_se3_past = current_se3_glob @ glob_se3_past
            apply_se3_(current_se3_past, points_=pcd)
            points_original.append(pcd)
            
        points_original = np.concatenate(points_original, axis=0)  # (N_ori_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx] 

        # -----------------------------
        # get sampled points
        if self.training:
            sampled_points, sampled_boxes, sampled_boxes_name, sampled_boxes_velo, instances_tf = list(), list(), list(), list(), list()
            instance_index = 0  # running variable
            for cls_name in self.sample_group.keys():
                sampled_trajectories_path = self.sample_with_fixed_number(cls_name)
                for traj_path in sampled_trajectories_path:
                    traj_points, traj_boxes, _ = load_1traj(traj_path, 
                                                            instance_index,
                                                            num_sweeps_in_target=self.dataset_cfg.MAX_SWEEPS,
                                                            src_frequency=5.,
                                                            target_frequency=20.  # frequency of NuScenes
                                                            )
                    # traj_points: (N_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx] 
                    # traj_boxes: (N_box, 7 + 2) - x, y, z, dx, dy, dz, yaw, [sweep_idx, inst_idx]

                    # compute instance_tf with gt_boxes == info's
                    traj_correction_tf = compute_correction_tf(traj_boxes)  # (N_boxes, 4, 4)
                    padded_traj_correction_tf = np.zeros(self.dataset_cfg.MAX_SWEEPS, 4, 4)  # NOTE: hard-code freq of src < freq of target
                    padded_traj_correction_tf[traj_boxes[:, -2].astype(int)] = traj_correction_tf

                    # compute velo
                    box_velo = (traj_boxes[-1, :2] - traj_boxes[0, :2]) / 0.5  # (2,)

                    # store
                    sampled_points.append(traj_points)
                    sampled_boxes.append(traj_boxes[-1])
                    sampled_boxes_name.append(cls_name)
                    sampled_boxes_velo.append(box_velo)
                    instances_tf.append(padded_traj_correction_tf)

                    # move on to the next traj
                    instance_index += 1

            sampled_points = np.concatenate(sampled_points)  # (N_sampled_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx] 
            sampled_boxes = np.concatenate(sampled_boxes)  # (N_inst, 7 + 2) - x, y, z, dx, dy, dz, yaw, [sweep_idx, inst_idx]
            sampled_boxes_name = np.array(sampled_boxes_name)  # (N_inst,)
            sampled_boxes_velo = np.stack(sampled_boxes_velo, axis=0)  # (N_inst, 2)
            instances_tf = np.stack(instances_tf, axis=0)  # (N_inst, N_sweep in src, 4, 4)

            # check last_box of each traj, see if they overlap with anyone, if yes, remove points
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, :7], sampled_boxes[:, :7])
            iou[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0  # (N_inst, N_inst)
            mask_valid_inst = iou.max(axis=1) == 0  # (N_inst,)
            # remove invalid sampled_boxes & points
            valid_instance_ids = np.arange(sampled_boxes.shape[0])[mask_valid_inst]  # (N_valid_inst,)
            mask_valid_points = np.any(sampled_points[:, -1].astype(int).reshape(-1, 1) == valid_instance_ids.reshape(1, -1), axis=1)  # (N_pts,)
            sampled_points = sampled_points[mask_valid_points]  # (N_sampled_pts, 5 + 2)

            sampled_boxes = sampled_boxes[mask_valid_inst]  # (N_val_inst, 7 + 2) - x, y, z, dx, dy, dz, yaw, [sweep_idx, inst_idx]
            sampled_boxes_name = sampled_boxes_name[mask_valid_inst]  # (N_val_inst,)
            sampled_boxes_velo = sampled_boxes_velo[mask_valid_inst]  # (N_val_inst, 2)
            instances_tf = instances_tf[mask_valid_inst]  # (N_val_inst, N_sweep in src, 4, 4)
            assert sampled_boxes.shape[0] > 0
            
            # remove original points inside each valid sampled box
            points_box_index = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points_original[:, :3]).float(), 
                torch.from_numpy(sampled_boxes[:, :7]).float()
            ).numpy()  # (N_val_inst, N_ori_pts)
            mask_ori_pts_in_box = np.any(points_box_index > 0, axis=0)
            points_original = points_original[np.logical_not(mask_ori_pts_in_box)]

            # merge sampled points and original points
            assert sampled_points.shape[0] > 0
            points = np.concatenate([points_original, sampled_points])
            gt_boxes = np.concatenate([sampled_boxes[:, :7], sampled_boxes_velo], 
                                      axis=1)  # (N_val_inst, 9) - x, y, z, dx, dy, dz, yaw, vx, vy
            gt_names = sampled_boxes_name
        else:
            # validation & testing
            points = points_original
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token'],
                         'num_sweeps_target': num_sweeps},
            'gt_boxes': gt_boxes,
            'gt_names': gt_names  # str
        }

        if self.training:
            input_dict['instances_tf'] = instances_tf
            input_dict['metadata']['num_sampled_boxes'] = gt_boxes.shape[0]

        # data augmentation & other stuff
        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.nuscenes_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

    def nuscenes_eval(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)

        # init nusc_annos to make sure that every sample token of val set is included
        nusc_annos = {
            'results': {},
            'meta': {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }
        }
        for info in self.infos:
            nusc_annos['results'].update({info['token']: []})

        nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc, nusc_annos)

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def kitti_eval(self,  eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils

        map_name_to_kitti = {
            'car': 'car',
            'truck': 'truck',
            'construction_vehicle': 'construction_vehicle',
            'bus': 'bus',
            'trailer': 'trailer',
            'barrier': 'barrier',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle',
            'pedestrian': 'pedestrian',
            'traffic_cone': 'traffic_cone',
            'ignore': 'ignore',
        }
        kitti_utils.transform_annotations_to_kitti_format(self.nusc, eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(self.nusc, eval_gt_annos, map_name_to_kitti=map_name_to_kitti)

        kitti_class_names = [map_name_to_kitti[x] for x in class_names]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names, use_nuscenes_cls=True
        )
        return ap_result_str, ap_dict

    def create_groundtruth_database(self, lyft_root: str, classes_of_interest: list):
        create_database(Path(lyft_root), classes_of_interest=set(classes_of_interest))

    def create_hd_map_database(self):
        assert len(self.infos) > 0, "remember to call create_nuscenes_infos first"
        assert self.dataset_cfg.USE_HD_MAP, f"expect self.dataset_cfg.USE_HD_MAP == True, get {self.dataset_cfg.USE_HD_MAP}"

        map_maker = MapMaker(self.nusc, self.map_point_cloud_range, self.map_bev_image_resolution, 
                            map_layers=('drivable_area', 'ped_crossing', 'walkway', 'carpark_area'))

        database_save_path = self.root_path / "hd_map" if 'trainval' in self.dataset_cfg.VERSION else self.root_path / "mini_hd_map"
        
        if not database_save_path.is_dir():
            database_save_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        for idx in range(len(self.infos)):
            sample_token = self.infos[idx]['token']
            map_img = map_maker.get_map_in_lidar_frame(sample_token)
            # save to hard disk
            filename = database_save_path / f"bin_map_{sample_token}.npy"
            np.save(filename, map_img)

            if idx % 100 == 0:
                avg_exe_time = (time.time() - start_time) / float(idx + 1)
                print(f"finish {idx + 1} / {len(self.infos)}, avg exe time {avg_exe_time}")
                start_time = time.time()


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        info_prefix = 'mini' if 'mini' in version else 'full'
        with open(save_path / f'{info_prefix}_nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'{info_prefix}_nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    
    parser.add_argument('--create_nuscenes_infos', action='store_true')
    parser.add_argument('--no-create_nuscenes_infos', dest='create_nuscenes_infos', action='store_false')
    parser.set_defaults(create_nuscenes_infos=False)

    parser.add_argument('--create_groundtruth_database', action='store_true')
    parser.add_argument('--no-create_groundtruth_database', dest='create_groundtruth_database', action='store_false')
    parser.set_defaults(create_groundtruth_database=False)

    parser.add_argument('--training', action='store_true')
    parser.add_argument('--no-training', dest='training', action='store_false')
    parser.set_defaults(training=True)

    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version

    if args.create_nuscenes_infos:
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

    if args.create_groundtruth_database:
        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=args.training
        )
        assert args.training, "expect args.training == True; get False"
        nuscenes_dataset.create_groundtruth_database(
            lyft_root=ROOT_DIR / 'data' / 'lyft',
            classes_of_interest=['car',]
        )
