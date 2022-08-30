import numpy as np
import torch
from torch_scatter import scatter_mean
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
from _dev_space.tools_box import compute_bev_coord_torch, show_pointcloud
from _dev_space.viz_tools import viz_boxes, print_dict


def assign_target_foreground_seg(data_dict, stride=1) -> dict:
    points = data_dict['points']  # (N, 1+3+C+1) - batch_idx, XYZ, C feats, indicator (-1 bgr, >=0 inst idx)
    indicator = points[:, -1].int()
    pc_range = torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=torch.float, device=indicator.device)
    pix_size = 0.2 * stride
    bev_size = torch.floor((pc_range[3: 5] - pc_range[0: 2]) / pix_size).int()

    bgr_pix_coord, _ = compute_bev_coord_torch(points[indicator == -1], pc_range, pix_size)

    # compute cluster mean with awareness of batch idx
    max_num_inst = torch.max(points[:, -1]) + 1
    mask_fgr = indicator > -1  # (N,)
    merge_batch_and_inst_idx = points[mask_fgr, 0] * max_num_inst + indicator[mask_fgr]  # (N_fgr,) | N_fgr < N
    unq_merge, inv_indices = torch.unique(merge_batch_and_inst_idx, return_inverse=True)  # (N_unq)
    clusters_mean = scatter_mean(points[mask_fgr, 1: 3], inv_indices, dim=0)  # (N_unq, 2)
    # compute offset from each fgr point to its cluster's mean
    fgr_to_cluster_mean = (clusters_mean[inv_indices] - points[mask_fgr, 1: 3]) / pix_size  # (N_fgr, 2)

    fgr_pix_coord, fgr_to_mean = compute_bev_coord_torch(points[mask_fgr], pc_range, pix_size, fgr_to_cluster_mean)

    # format output
    bev_seg_label = -points.new_ones(data_dict['batch_size'], bev_size[1], bev_size[0])
    bev_seg_label[bgr_pix_coord[:, 0], bgr_pix_coord[:, 2], bgr_pix_coord[:, 1]] = 0
    bev_seg_label[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = 1

    bev_reg_label = points.new_zeros(2, data_dict['batch_size'], bev_size[1], bev_size[0])
    for idx_offset in range(2):
        bev_reg_label[idx_offset, fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = \
            fgr_to_mean[:, idx_offset]

    target_dict = {'bev_seg_label': bev_seg_label, 'bev_reg_label': bev_reg_label}
    return target_dict


cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger('./dummy_log.txt')
cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
cfg.VERSION = 'v1.0-mini'
cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    # 'random_world_flip', 'random_world_rotation', 'random_world_scaling',
    # 'gt_sampling',
]

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

target_dict = assign_target_foreground_seg(data_dict)
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


bev_cls_label = target_dict['bev_seg_label'].int().numpy()
bev_reg_label = target_dict['bev_reg_label'].numpy()
fig, ax = plt.subplots(1, 2)
viz_seg_label = np.zeros_like(bev_cls_label[batch_idx], dtype=float)
viz_seg_label[bev_cls_label[batch_idx] < 0] = 0  # unmeasurable region
viz_seg_label[bev_cls_label[batch_idx] == 0] = 0.5  # background
viz_seg_label[bev_cls_label[batch_idx] == 1] = 1.0  # background
ax[0].imshow(viz_seg_label, cmap='gray')

xx, yy = np.meshgrid(np.arange(viz_seg_label.shape[0]), np.arange(viz_seg_label.shape[1]))
fgr_x = xx[bev_cls_label[batch_idx] == 1]
fgr_y = yy[bev_cls_label[batch_idx] == 1]
fgr_to_mean = bev_reg_label[:, batch_idx, bev_cls_label[batch_idx] == 1]
for fidx in range(fgr_to_mean.shape[1]):
    ax[1].arrow(fgr_x[fidx], fgr_y[fidx], fgr_to_mean[0, fidx], fgr_to_mean[1, fidx], color='g', width=0.01)
ax[1].imshow(viz_seg_label, cmap='gray')

plt.show()
