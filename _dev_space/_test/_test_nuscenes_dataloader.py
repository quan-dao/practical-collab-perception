import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from _dev_space.tools_box import show_pointcloud
from _dev_space.viz_tools import viz_boxes, print_dict
from _dev_space.instance_centric_tools import correction_numpy


np.random.seed(666)

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

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                                'aug_instance_idx']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                               'aug_instance_idx']

dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                          logger=logger, training=False, total_epochs=1, seed=666)
iter_dataloader = iter(dataloader)
for _ in range(5):
    data_dict = next(iter_dataloader)

print_dict(data_dict)

batch_idx = 1

points = data_dict['points']
points = points[points[:, 0].astype(int) == batch_idx]

gt_boxes = data_dict['gt_boxes'][batch_idx]  # (N_max_gt, 10)
valid_gt_boxes = np.linalg.norm(gt_boxes, axis=1) > 0
gt_boxes = gt_boxes[valid_gt_boxes]

_boxes = viz_boxes(gt_boxes)
print('showing gt instance idx')
show_pointcloud(points[:, 1: 4], _boxes, fgr_mask=points[:, -2] > -1)

print('showing augmented instance idx')
show_pointcloud(points[:, 1: 4], _boxes, fgr_mask=points[:, -1] > -1)

# -----------------------
# correction
points_instance_idx = points[:, -2].astype(int)
bg_points = points[points_instance_idx == -1]
fg_points = points[points_instance_idx > -1]

instances_tf = data_dict['instances_tf']  # (B, N_max_inst, N_sweeps, 3, 4)
n_valid_instances = np.max(points_instance_idx) + 1
instances_tf = instances_tf[batch_idx, :n_valid_instances]  # (N_inst, N_sweeps, 3, 4)

corrected_xyz = correction_numpy(points[:, 1:], instances_tf)
print('showing corrected point cloud')
show_pointcloud(corrected_xyz, _boxes, fgr_mask=points[:, -2] > -1)
