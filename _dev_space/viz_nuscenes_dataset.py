import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
from get_clean_pointcloud import show_pointcloud, apply_transform_to_points
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord
from viz_tools import viz_boxes, print_dict


np.random.seed(666)
cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)

logger = common_utils.create_logger('./dummy_log.txt')

cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    # 'random_world_flip', 'random_world_scaling',
    # 'gt_sampling',
    # 'random_world_rotation',
]

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.VERSION = 'v1.0-mini'


nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
data_dict = nuscenes_dataset[400]  # 400, 200, 100, 5, 10
print_dict(data_dict)
print('meta: ', data_dict['metadata'])

sample_rec = nuscenes_dataset.nusc.get('sample', data_dict['metadata']['token'])
nuscenes_dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'])
plt.show()


pc = data_dict['points']
print('n pts: ', pc.shape[0])
gt_boxes = viz_boxes(data_dict['gt_boxes'])

print('showing EMC accumulated pointcloud')
show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -1] > -1)

print('showing oracle accumulated pointcloud')
pc[:, :2] += pc[:, 5: 7]
print('n pts: ', pc.shape[0])
show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -1] > -1)
