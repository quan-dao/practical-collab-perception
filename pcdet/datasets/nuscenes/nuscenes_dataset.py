import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...utils import common_utils
from ..dataset import DatasetTemplate

from workspace.rev_get_sweeps_instance_centric import revised_instance_centric_get_sweeps
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps


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
        if self.dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1) and training:
            stride = self.dataset_cfg.get('MINI_TRAINVAL_STRIDE', 1)
            self.infos = self.infos[::stride]  # use 1/4th of the trainval data

        self.nusc = NuScenes(dataroot=root_path, version=dataset_cfg.VERSION, verbose=False)
        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE)

        num_pts_raw_feat = 5  # x, y, z, intensity, time
        self.map_point_feat2idx = {
            'sweep_idx': num_pts_raw_feat,
            'inst_idx': num_pts_raw_feat + 1,
        }

        if dataset_cfg.get('USE_HD_MAP', False):
            self.prediction_helper = PredictHelper(self.nusc)
            self.map_apis = load_all_maps(self.prediction_helper)
            self.map_point_cloud_range = np.array([-81.0, -81.0, -5.0, 81.0, 81.0, 3.0])  # independent with the dataset 
            self.map_bev_image_resolution = 0.075  # independent with the dataset config
        else:
            self.map_point_cloud_range, self.map_bev_image_resolution = None, None

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

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        num_sweeps = self.dataset_cfg.MAX_SWEEPS
        map_database_dir = self.root_path / "hd_map" if 'trainval' in self.dataset_cfg.VERSION else self.root_path / "mini_hd_map"
        _out = revised_instance_centric_get_sweeps(self.nusc, info['token'], num_sweeps, self.class_names,
                                                   use_hd_map=self.dataset_cfg.get('USE_HD_MAP', False),
                                                   map_database_path=map_database_dir,
                                                   map_point_cloud_range=self.map_point_cloud_range,
                                                   map_bev_image_resolution=self.map_bev_image_resolution)
        points = _out['points']  # (N, C)


        input_dict = {
            'points': points,
            'instances_tf': _out['instances_tf'],
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {
                'token': info['token'],
                'num_sweeps': num_sweeps,
                'num_original_instances': _out['instances_tf'].shape[0],
                'num_original_boxes': _out['gt_boxes'].shape[0],
                'tf_target_from_glob': _out['target_from_glob'],
            }
        }

        # overwrite gt_boxes & gt_names
        input_dict.update({
            'gt_boxes': _out['gt_boxes'],
            'gt_names': _out['gt_names']
        })

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
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

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

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        postfix = '_withmap' if self.dataset_cfg.get('USE_HD_MAP', False) else ''
        if 'trainval' in self.dataset_cfg.VERSION:
            database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo{postfix}'
            db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo{postfix}.pkl'
        else:
            assert 'mini' in self.dataset_cfg.VERSION, f"{self.dataset_cfg.VERSION} is not supported"
            database_save_path = self.root_path / f'mini_gt_database_{max_sweeps}sweeps_withvelo{postfix}'
            db_info_save_path = self.root_path / f'mini_nuscenes_dbinfos_{max_sweeps}sweeps_withvelo{postfix}.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        movable_classes = ('car', 'truck', 'bus')        

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            detection_classes = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle',
                             'bicycle', 'pedestrian', 'traffic_cone')
            _out = revised_instance_centric_get_sweeps(self.nusc, info['token'], self.dataset_cfg.MAX_SWEEPS, detection_classes,
                                                       map_apis=self.map_apis if self.dataset_cfg.get('USE_HD_MAP', False) else None,
                                                       map_binary_layers=self.dataset_cfg.get('MAP_LAYERS', ['drivable_area']),
                                                       map_get_lane_direction=self.dataset_cfg.get('USE_LANE_DIRECTION', False))

            points = _out['points']  # (N, 7) - x, y, z, intensity, time, sweep_idx, inst_idx
            points_inst_idx = points[:, self.map_point_feat2idx['inst_idx']].astype(int)

            gt_boxes = _out['gt_boxes']  # (N_inst, 9) - c_x, c_y, c_z, d_x, d_y, d_z, yaw, vx, vy
            gt_names = _out['gt_names']  # (N_inst,)
            gt_anno_tk = _out['gt_anno_tk']  # (N_inst,)
            instances_tf = _out['instances_tf']  # (N_inst, max_sweeps, 4, 4)

            for i in range(gt_boxes.shape[0]):
                # filter movable class to include only actual moving stuff
                if gt_names[i] in movable_classes:
                    if np.linalg.norm(instances_tf[0, :2, -1]) < 1.0:
                        # translation from the oldest to the presence < 1.0
                        # skip this instance
                        continue

                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[points_inst_idx == i]
                if gt_points.shape[0] == 0:  # there is no points in this gt_boxes
                    continue

                # translate gt_points to frame whose origin @ gt_box center
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                # get sampled gt_boxes velocity in global frame -> will be transformed to target frame @ database_sampler/add_to_scene
                gt_velo = self.nusc.box_velocity(gt_anno_tk[i])  # (vx, vy, vz) in global frame

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {
                        'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                        'num_points_in_gt': gt_points.shape[0],
                        'box3d_lidar': gt_boxes[i],
                        'instance_idx': i,
                        'velo': gt_velo,
                        'instance_tf': instances_tf[i],  # (max_sweeps, 4, 4)
                    }
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def create_hd_map_database(self):

        from _dev_space.tools_box import get_nuscenes_sensor_pose_in_global

        def get_map_name_from_sample_token(nusc: NuScenes, sample_tk: str) -> str:
            sample = nusc.get('sample', sample_tk)
            scene = nusc.get('scene', sample['scene_token'])
            log = nusc.get('log', scene['log_token'])
            return log['location']

        assert len(self.infos) > 0, "remember to call create_nuscenes_infos first"
        assert self.dataset_cfg.USE_HD_MAP, f"expect self.dataset_cfg.USE_HD_MAP == True, get {self.dataset_cfg.USE_HD_MAP}"

        map_binary_layers = ('drivable_area', 'ped_crossing', 'walkway', 'carpark_area')
        bev_image_resolution = self.map_bev_image_resolution
        point_cloud_range = self.map_point_cloud_range
        map_dx, map_dy = point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]  # in meters
        map_size_pixel = (int(map_dx / bev_image_resolution), int(map_dy / bev_image_resolution))  # (W, H)
        
        database_save_path = self.root_path / "hd_map" if 'trainval' in self.dataset_cfg.VERSION else self.root_path / "mini_hd_map"
        database_save_path.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(self.infos))):
            sample_token = self.infos[idx]['token']
            sample_rec = self.nusc.get('sample', sample_token)
            target_sd_token = sample_rec['data']['LIDAR_TOP']
            
            glob_from_target = np.linalg.inv(get_nuscenes_sensor_pose_in_global(self.nusc, target_sd_token))
            ego_x, ego_y = glob_from_target[0, -1], glob_from_target[1, -1]
            ego_yaw = np.arctan2(glob_from_target[1, 0], glob_from_target[0, 0])

            map_name = get_map_name_from_sample_token(self.nusc, sample_token)

            # query map_apis for map around ego vehicle
            patch_box = (ego_x, ego_y, map_dx, map_dy)
            map_masks = self.map_apis[map_name].get_map_mask(patch_box, np.rad2deg(ego_yaw), map_binary_layers, 
                                                             canvas_size=map_size_pixel).astype(float)  # (N_layers, H, W)

            # save to hard disk
            filename = database_save_path / f"bin_map_{sample_token}.npy"
            np.save(filename, map_masks)


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

    parser.add_argument('--create_hd_map_database', action='store_true')
    parser.add_argument('--no-create_hd_map_database', dest='create_hd_map_database', action='store_false')
    parser.set_defaults(create_hd_map_database=False)


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

    if args.create_groundtruth_database or args.create_hd_map_database:
        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        if args.create_groundtruth_database:
            nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
    
        if args.create_hd_map_database:
            nuscenes_dataset.create_hd_map_database()

