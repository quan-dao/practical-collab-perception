import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
from _dev_space.tools_box import show_pointcloud
from _dev_space.viz_tools import viz_boxes, print_dict


# np.random.seed(666)

cfg_file = '../../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger('./dummy_log.txt')
cfg.DATA_PATH = '../../data/nuscenes'
cfg.VERSION = 'v1.0-mini'
cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    # 'random_world_flip',
    # 'random_world_scaling',
    # 'random_world_rotation',
    'gt_sampling',
]

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx']

nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
data_dict = nuscenes_dataset[100]  # 400, 200, 100, 5, 10
print_dict(data_dict)
print('meta: ', data_dict['metadata'])

points = data_dict['points']
gt_boxes = viz_boxes(data_dict['gt_boxes'])
show_pointcloud(points[:, :3], gt_boxes, fgr_mask=points[:, -1] > -1)

# -----------------------
# batch correction
points_instance_idx = points[:, -1].astype(int)
bg_points = points[points_instance_idx == -1]
fg_points = points[points_instance_idx > -1]

instances_tf = data_dict['instances_tf']  # (N_inst, N_sweeps, 4, 4)
instances_tf = instances_tf.reshape(-1, 4, 4)  # (N_inst * N_sweeps, 4, 4)

fg_merge_ids = fg_points[:, -1].astype(int) * cfg.MAX_SWEEPS + fg_points[:, -2].astype(int)  # (N_fg, 3 + C)
instances_traj_4fg = instances_tf[fg_merge_ids]  # (N_fg, 4, 4)

fg_points_xyz1 = np.pad(fg_points[:, :3], pad_width=[(0, 0), (0, 1)], constant_values=1.)  # (N_fg, 4)
fg_points_xyz1 = np.einsum('BCD,BD -> BC', instances_traj_4fg, fg_points_xyz1)  # (N_fg, 4)
fg_points[:, :3] = fg_points_xyz1[:, :3]  # (N_fg, 3)

_points = np.concatenate([bg_points, fg_points], axis=0)
show_pointcloud(_points[:, :3], gt_boxes, fgr_mask=_points[:, -1] > -1)
