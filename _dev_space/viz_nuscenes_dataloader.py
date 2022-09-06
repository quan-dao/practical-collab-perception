import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from _dev_space.tools_box import show_pointcloud
from _dev_space.viz_tools import viz_boxes, print_dict
from _dev_space.bev_segmentation_utils import assign_target_foreground_seg


cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger('./dummy_log.txt')
cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
cfg.VERSION = 'v1.0-mini'
cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    'random_world_flip', 'random_world_rotation', 'random_world_scaling',
    'gt_sampling',
]
cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']

dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                          logger=logger, training=False, total_epochs=1, seed=666)
iter_dataloader = iter(dataloader)
data_dict = next(iter_dataloader)
# load_data_to_gpu(data_dict)
for k, v in data_dict.items():
    if k in ['frame_id', 'metadata']:
        continue
    elif isinstance(v, np.ndarray):
        data_dict[k] = torch.from_numpy(v)

print_dict(data_dict)
print(f"metadata: {data_dict['metadata']}")

# fig, ax = plt.subplots(1, 2)
# for idx_sample in range(2):
#     sample_token = data_dict['metadata'][idx_sample]['token']
#     sample_rec = dataset.nusc.get('sample', sample_token)
#     dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'], ax=ax[idx_sample])
# plt.show()

target_dict = assign_target_foreground_seg(data_dict, 2)
print_dict(target_dict)

# ---
# viz
# ---
batch_idx = 1
points = data_dict['points']  # (N, 7) - batch_idx, XYZ, C feats, indicator | torch.Tensor
indicator = points[:, -1].int()
mask_curr_batch = points[:, 0].int() == batch_idx

pc = points[mask_curr_batch].numpy()
fgr_mask = (indicator[mask_curr_batch] > -1).numpy()
boxes = viz_boxes(data_dict['gt_boxes'][batch_idx].numpy())
show_pointcloud(pc[:, 1: 4], boxes, fgr_mask=fgr_mask)


bev_cls_label = target_dict['bev_cls_label'].int().numpy()  # (B, H, W)
bev_reg_label = target_dict['bev_reg_label'].numpy()  # (4, B, H, W)

bev_fgr_mask = bev_cls_label[batch_idx] > 0  # (H, W)

fig, ax = plt.subplots(1, 3)
ax[0].set_title('gt class')
ax[0].imshow(bev_fgr_mask, cmap='gray')

ax[1].set_title('gt reg to cluster')
ax[1].imshow(bev_fgr_mask, cmap='gray')
xx, yy = np.meshgrid(np.arange(bev_fgr_mask.shape[0]), np.arange(bev_fgr_mask.shape[1]))
fgr_x, fgr_y = xx[bev_fgr_mask], yy[bev_fgr_mask]
fgr_to_mean = bev_reg_label[:2, batch_idx, bev_fgr_mask]  # (2, N_fgr)
for fidx in range(fgr_to_mean.shape[1]):
    ax[1].arrow(fgr_x[fidx], fgr_y[fidx], fgr_to_mean[0, fidx], fgr_to_mean[1, fidx], color='g', width=0.01)

ax[2].set_title('gt reg to correction')
ax[2].imshow(bev_fgr_mask, cmap='gray')
fgr_to_crt = bev_reg_label[2:, batch_idx, bev_fgr_mask]  # (2, N_fgr)
for fidx in range(fgr_to_crt.shape[1]):
    ax[2].arrow(fgr_x[fidx], fgr_y[fidx], fgr_to_crt[0, fidx], fgr_to_crt[1, fidx], color='g', width=0.01)

plt.show()


# test validity of correction label by inferring from bev crt image
pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
pix_size = 0.4
bev_crt_label = bev_reg_label[2:, batch_idx]  # (2, H, W)
pc_pix_coords = np.floor((pc[:, 1: 3] - pc_range[:2]) / pix_size).astype(int)
pc_crt = bev_crt_label[:, pc_pix_coords[:, 1], pc_pix_coords[:, 0]]
pc[:, 1: 3] += pc_crt.T
show_pointcloud(pc[:, 1: 4], boxes, fgr_mask=fgr_mask)
