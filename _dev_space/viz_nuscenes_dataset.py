import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord, show_pointcloud
from viz_tools import viz_boxes, print_dict


np.random.seed(666)
cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)

logger = common_utils.create_logger('./dummy_log.txt')

cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    'random_world_flip', 'random_world_scaling',
    'gt_sampling',
    'random_world_rotation',
]

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.VERSION = 'v1.0-mini'


nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
data_dict = nuscenes_dataset[400]  # 400, 200, 100, 5, 10
print_dict(data_dict)
print('meta: ', data_dict['metadata'])

# sample_rec = nuscenes_dataset.nusc.get('sample', data_dict['metadata']['token'])
# nuscenes_dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'])
# plt.show()


pc = data_dict['points']
print('n pts: ', pc.shape[0])
gt_boxes = viz_boxes(data_dict['gt_boxes'])

print('showing EMC accumulated pointcloud')
show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -1] > -1)

# print('showing oracle accumulated pointcloud')
# pc[:, :2] += pc[:, 5: 7]
# print('n pts: ', pc.shape[0])
# show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -1] > -1)

print('showing BEV')
pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
mask_inside = np.all((pc[:, :3] >= pc_range[:3]) & (pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
pc = pc[mask_inside]
mask_fgr = pc[:, -1] > -1
fgr_bev_coord, fgr_bev_feat = compute_bev_coord(pc[mask_fgr, :3], pc_range=pc_range, bev_pix_size=0.2,
                                                pts_feat=pc[mask_fgr, -1].reshape(-1, 1))
fgr_bev_feat = fgr_bev_feat.reshape(-1).astype(int)

fig, ax = plt.subplots()

for instance_id in np.unique(fgr_bev_feat):
    mask_inst = fgr_bev_feat == instance_id
    ax.scatter(fgr_bev_coord[mask_inst, 0], fgr_bev_coord[mask_inst, 1])
    ax.annotate(instance_id, (np.mean(fgr_bev_coord[mask_inst, 0]), np.mean(fgr_bev_coord[mask_inst, 1])))

ax.set_aspect('equal')
plt.show()

print('showing instance points')
chosen_inst_id = 35
mask_chosen_inst = pc[:, -1] == chosen_inst_id
inst_points = pc[mask_chosen_inst]
np.save(f'./artifact/nusc-mini_sample400_inst{chosen_inst_id}.npy', inst_points)

unique_timestamp, inv_indices = np.unique(inst_points[:, 4], return_inverse=True)
palette = plt.cm.rainbow(np.linspace(0, 1, len(unique_timestamp)))[:, :3]
inst_points_color = palette[inv_indices]
show_pointcloud(inst_points[:, :3], pc_colors=inst_points_color)

