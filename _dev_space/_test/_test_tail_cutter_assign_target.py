import torch
import torch_scatter
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu

from _dev_space.tail_cutter import PointAligner
from _dev_space.viz_tools import print_dict, viz_boxes
from _dev_space.tools_box import show_pointcloud
from _dev_space.tools_4testing import load_data_to_tensor


cfg_file = './tail_cutter_cfg.yaml'
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger('./dummy_log.txt')

dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                          logger=logger, training=False, total_epochs=1, seed=666)
iter_dataloader = iter(dataloader)
for _ in range(5):
    batch_dict = next(iter_dataloader)

load_data_to_tensor(batch_dict)
print_dict(batch_dict)

points = batch_dict['points']
gt_boxes = batch_dict['gt_boxes']  # (B, N_max_gt, 10)

show_point_cloud = True
if show_point_cloud:
    batch_idx = 1
    _points = points[points[:, 0].long() == batch_idx]

    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = _boxes[valid_gt_boxes]
    _boxes = viz_boxes(_boxes.numpy())
    show_pointcloud(_points[:, 1: 4], _boxes, fgr_mask=_points[:, -1] > -1)


model = PointAligner(cfg.MODEL)
# ---
# populate model.forward_return_dict['meta']
# ---
mask_fg = batch_dict['points'][:, -1] > -1
fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instacne_idx
points_batch_idx = batch_dict['points'][:, 0].long()
fg_batch_idx = points_batch_idx[mask_fg]
fg_inst_idx = fg[:, -1].long()
fg_sweep_idx = fg[:, -2].long()

max_num_inst = batch_dict['instances_tf'].shape[1]  # batch_dict['instances_tf']: (batch_size, max_n_inst, n_sweeps, 3, 4)
fg_bi_idx = fg_batch_idx * max_num_inst + fg_inst_idx  # (N,)
fg_bisw_idx = fg_bi_idx * model.cfg.get('NUM_SWEEPS', 10) + fg_sweep_idx

inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=model.training, return_inverse=True)
model.forward_return_dict['meta'] = {'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices}

local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=model.training, return_inverse=True)
# local_bisw: (N_local,)
model.forward_return_dict['meta'].update({'local_bisw': local_bisw,
                                          'local_bisw_inv_indices': local_bisw_inv_indices})

target_dict = model.assign_target(batch_dict)
print_dict(target_dict)


show_inst_assoc = False
if show_inst_assoc:
    batch_idx = 1
    batch_mask = fg[:, 0].long() == batch_idx

    cur_fg = fg[batch_mask]  # (N_cur_fg, 8)

    inst_assoc = target_dict['inst_assoc']  # (N_fg, 2)
    cur_inst_assoc = inst_assoc[batch_mask]  # (N_cur_fg, 2)

    # apply inst_assoc
    cur_fg[:, 1: 3] += cur_inst_assoc

    # viz
    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = _boxes[valid_gt_boxes]
    _boxes = viz_boxes(_boxes.numpy())
    bg = points[torch.logical_not(mask_fg)]
    cur_bg = bg[bg[:, 0].long() == batch_idx]
    pc = torch.cat([cur_bg, cur_fg], dim=0)
    show_pointcloud(pc[:, 1: 4], _boxes, fgr_mask=pc[:, -1] > -1)


show_motion_stat = False
if show_motion_stat:
    inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst)
    fg_motion = inst_motion_stat[inst_bi_inv_indices]

    batch_idx = 1
    batch_mask = fg[:, 0].long() == batch_idx

    cur_fg = fg[batch_mask]  # (N_cur_fg, 8)
    cur_fg_motion = fg_motion[batch_mask] == 1
    cur_fg_color = torch.zeros((cur_fg.shape[0], 3))
    cur_fg_color[cur_fg_motion, 0] = 1
    cur_fg_color[torch.logical_not(cur_fg_motion), 2] = 1

    # viz
    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = _boxes[valid_gt_boxes]
    _boxes = viz_boxes(_boxes.numpy())
    bg = points[torch.logical_not(mask_fg)]
    cur_bg = bg[bg[:, 0].long() == batch_idx]
    pc = torch.cat([cur_bg, cur_fg], dim=0)
    pc_colors = torch.cat((cur_bg.new_zeros(cur_bg.shape[0], 3), cur_fg_color), dim=0)
    show_pointcloud(pc[:, 1: 4], _boxes, pc_colors=pc_colors.numpy())


show_correction = True
if show_correction:
    local_tf = target_dict['local_tf']  # (N_local, 3, 4)
    fg_local_tf = local_tf[local_bisw_inv_indices]  # (N_fg, 3, 4)

    # --
    # only apply correction on fg belongs to moving instance
    # --
    inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst)
    fg_motion = inst_motion_stat[inst_bi_inv_indices]  # (N_fg)

    fg_motion = fg_motion == 1
    dyn_fg = fg[fg_motion]  # (N_dyn, 8)
    dyn_fg_local_tf = fg_local_tf[fg_motion]  # (N_dyn, 3, 4)
    dyn_fg[:, 1: 4] = torch.matmul(dyn_fg_local_tf[:, :3, :3], dyn_fg[:, 1: 4, None]).squeeze(-1) + \
                      dyn_fg_local_tf[:, :, -1]

    # viz
    batch_idx = 1
    cur_dyn_fg = dyn_fg[dyn_fg[:, 0].long() == batch_idx]

    static_fg = fg[torch.logical_not(fg_motion)]
    cur_static_fg = static_fg[static_fg[:, 0].long() == batch_idx]

    bg = points[torch.logical_not(mask_fg)]
    cur_bg = bg[bg[:, 0].long() == batch_idx]

    pc = torch.cat([cur_bg, cur_static_fg, cur_dyn_fg], dim=0)
    pc_colors = torch.zeros((pc.shape[0], 3))
    pc_colors[cur_bg.shape[0]: cur_bg.shape[0] + cur_static_fg.shape[0], 2] = 1  # static fg - blue
    pc_colors[cur_bg.shape[0] + cur_static_fg.shape[0]:, 0] = 1  # dynamic fg - red

    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = _boxes[valid_gt_boxes]
    _boxes = viz_boxes(_boxes.numpy())
    show_pointcloud(pc[:, 1: 4], _boxes, pc_colors=pc_colors.numpy())



