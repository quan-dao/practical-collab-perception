import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
from get_clean_pointcloud import show_pointcloud, apply_transform_to_points
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord


def viz_boxes(boxes: np.ndarray):
    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    out = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        dx, dy, dz = box[3: 6].tolist()
        vers = np.concatenate([xs.reshape(-1, 1) * dx, ys.reshape(-1, 1) * dy, zs.reshape(-1, 1) * dz], axis=1)  # (8, 3)
        ref_from_box = np.eye(4)
        yaw = box[6]
        cy, sy = np.cos(yaw), np.sin(yaw)
        ref_from_box[:3, :3] = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        ref_from_box[:3, 3] = box[:3]
        vers = apply_transform_to_points(ref_from_box, vers)
        out.append(vers)
    return out


np.random.seed(666)
cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)

logger = common_utils.create_logger('./dummy_log.txt')

cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    # 'random_world_flip', 'random_world_rotation', 'random_world_scaling',
    # 'gt_sampling',
]

cfg.VERSION = 'v1.0-mini'

# shut down shuffle_points
# for proc_idx, proc in enumerate(cfg.DATA_PROCESSOR):
#     if proc['NAME'] == 'shuffle_points':
#         break
# cfg.DATA_PROCESSOR.pop(proc_idx)

nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
data_dict = nuscenes_dataset[400]  # 400, 200, 100, 5, 10


for k, v in data_dict.items():
    out = f"{k} | {type(v)} | "
    if isinstance(v, str):
        out += v
    elif isinstance(v, np.ndarray):
        out += f"{v.shape}"
    elif isinstance(v, float):
        out += f"{v}"
    elif isinstance(v, np.bool_):
        out += f"{v.item()}"
    print(out)

print(data_dict['metadata'])

sample_rec = nuscenes_dataset.nusc.get('sample', data_dict['metadata']['token'])
nuscenes_dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'])
plt.show()

pc = data_dict['points']
gt_boxes = data_dict['gt_boxes']
print(gt_boxes[:10, -1])
_boxes = viz_boxes(gt_boxes)

indicator = pc[:, -1].astype(int)

show_pointcloud(pc[:, :3], _boxes, fgr_mask=indicator >= 0)

# show target bev
pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
voxel_size = 0.2
bev_size = np.round((pc_range[3: 5] - pc_range[:2]) / voxel_size).astype(int)

bev_bgr_coord, _ = compute_bev_coord(pc[indicator == -1], pc_range, voxel_size)

# foreground offset to cluster center
unq_ind, inv_indices, counts = np.unique(indicator, return_inverse=True, return_counts=True)
cluster_xy_means = np.zeros((unq_ind.shape[0], 2))
np.add.at(cluster_xy_means, inv_indices, pc[:, :2])
cluster_xy_means /= counts[:, None]
offset_to_means = (cluster_xy_means[inv_indices] - pc[:, :2]) / voxel_size  # cvt to pixel

bev_fgr_coord, bev_fgr_o2m = compute_bev_coord(pc[indicator >= 0], pc_range, voxel_size,
                                               pts_feat=offset_to_means[indicator >= 0])

bev_img = np.zeros((bev_size[1], bev_size[0], 3))
bev_img[bev_bgr_coord[:, 1], bev_bgr_coord[:, 0], :] = 0.5
bev_img[bev_fgr_coord[:, 1], bev_fgr_coord[:, 0], 0] = 1.0

fig, ax = plt.subplots()
ax.imshow(bev_img, cmap='gray')

for fidx in range(bev_fgr_coord.shape[0]):
        ax.arrow(bev_fgr_coord[fidx, 0], bev_fgr_coord[fidx, 1], bev_fgr_o2m[fidx, 0], bev_fgr_o2m[fidx, 1],
                 color='g', width=0.01)

plt.show()
