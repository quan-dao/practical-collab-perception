import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from _dev_space.tail_cutter import PointAligner
from _dev_space.viz_tools import print_dict, viz_boxes
from _dev_space.tools_box import show_pointcloud
from tools_4testing import load_data_to_tensor, load_dict_to_gpu, load_dict_to_cpu


def _show_point_cloud(batch_dict, chosen_batch_idx):
    points = batch_dict['points']
    mask_cur = points[:, 0].long() == chosen_batch_idx
    cur_points = points[mask_cur]

    gt_boxes = batch_dict['gt_boxes']  # (B, N_box_max, 7+..)
    cur_gt_boxes = gt_boxes[chosen_batch_idx]
    mask_valid_boxes = cur_gt_boxes[:, -1] > 0
    cur_gt_boxes = cur_gt_boxes[mask_valid_boxes]

    show_pointcloud(cur_points[:, 1: 4], fgr_mask=cur_points[:, -2] > -1, boxes=viz_boxes(cur_gt_boxes))


def _show_target_proposals(model, batch_dict, chosen_batch_idx):
    load_dict_to_gpu(batch_dict)
    model.cuda()

    # invoke forward pass in training mode
    model.train()
    batch_dict = model(batch_dict)
    load_dict_to_cpu(batch_dict)

    target_dict = model.forward_return_dict['target_dict']
    load_dict_to_cpu(target_dict)

    # decode target_proposals
    target_proposals = target_dict['proposals']
    load_dict_to_cpu(target_dict)
    load_dict_to_cpu(model.forward_return_dict)

    center = model.forward_return_dict['meta']['inst_target_center'] + target_proposals['offset']
    size = target_proposals['size']
    yaw = torch.atan2(target_proposals['ori'][:, 0], target_proposals['ori'][:, 1])
    target_boxes = torch.cat([center, size, yaw[:, None]], dim=1)

    # ------
    # viz
    # 1 global group -> 1 color for points & box
    # ------

    # get current boxes
    inst_bi = model.forward_return_dict['meta']['inst_bi']
    max_num_inst = batch_dict['instances_tf'].shape[1]
    inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

    inst_colors = plt.cm.rainbow(np.linspace(0, 1, inst_bi.shape[0]))[:, :3]
    inst_colors = torch.from_numpy(inst_colors)

    cur_target_boxes = target_boxes[inst_batch_idx == chosen_batch_idx]  # (N_cur_inst, 7+...)
    cur_target_boxes_colors = inst_colors[inst_batch_idx == chosen_batch_idx]

    # get current points
    points = batch_dict['points']
    mask_fg = points[:, -1].long() > -1  # Remember the forward pass of training is based on aug_inst_index
    fg, bg = points[mask_fg], points[torch.logical_not(mask_fg)]

    inst_bi_inv_indices = model.forward_return_dict['meta']['inst_bi_inv_indices']
    assert inst_bi_inv_indices.shape[0] == fg.shape[0], f"{inst_bi_inv_indices.shape[0]} != {fg.shape[0]}"

    fg_colors = inst_colors[inst_bi_inv_indices]

    mask_cur_fg = fg[:, 0].long() == chosen_batch_idx
    cur_fg = fg[mask_cur_fg]
    cur_fg_colors = fg_colors[mask_cur_fg]

    cur_bg = bg[bg[:, 0].long() == chosen_batch_idx]
    cur_bg_colors = torch.zeros((cur_bg.shape[0], 3))

    pc = torch.cat([cur_bg, cur_fg])
    pc_colors = torch.cat([cur_bg_colors, cur_fg_colors])
    show_pointcloud(pc[:, 1: 4], pc_colors=pc_colors, boxes=viz_boxes(cur_target_boxes),
                    boxes_color=cur_target_boxes_colors)


def main(**kwargs):
    chosen_batch_idx = kwargs.get('chosen_batch_idx', 1)
    cfg_file = './tail_cutter_cfg.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    for _ in range(kwargs.get('chosen_iteration', 5)):
        batch_dict = next(iter_dataloader)
    load_data_to_tensor(batch_dict)

    model = PointAligner(cfg.MODEL)
    if kwargs.get('print_model', False):
        print('---------\n', model, '\n---------\n')

    if kwargs.get('show_raw_pointcloud', False):
        print('showing raw point cloud')
        _show_point_cloud(batch_dict, chosen_batch_idx)

    if kwargs.get('show_target_proposals'):
        _show_target_proposals(model, batch_dict, chosen_batch_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--show_raw_pointcloud', action='store_true', default=False)
    parser.add_argument('--show_target_proposals', action='store_true', default=False)
    parser.add_argument('--print_model', action='store_true', default=False)
    parser.add_argument('--chosen_iteration', type=int, default=5, help='blah')
    args = parser.parse_args()
    main(**vars(args))



#
# points = batch_dict['points']
# gt_boxes = batch_dict['gt_boxes']  # (B, N_max_gt, 10)
#
# show_point_cloud = True
# if show_point_cloud:
#     batch_idx = 1
#     _points = points[points[:, 0].long() == batch_idx]
#
#     _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
#     valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
#     _boxes = _boxes[valid_gt_boxes]
#     _boxes = viz_boxes(_boxes.numpy())
#     print('showing original point cloud')
#     show_pointcloud(_points[:, 1: 4], _boxes, fgr_mask=_points[:, -1] > -1)
#
#
# model = PointAligner(cfg.MODEL)
# # ---
# # populate model.forward_return_dict['meta']
# # ---
# mask_fg = batch_dict['points'][:, -1] > -1
# fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instacne_idx
# points_batch_idx = batch_dict['points'][:, 0].long()
# fg_batch_idx = points_batch_idx[mask_fg]
# fg_inst_idx = fg[:, -1].long()
# fg_sweep_idx = fg[:, -3].long()
#
# max_num_inst = batch_dict['instances_tf'].shape[1]  # batch_dict['instances_tf']: (batch_size, max_n_inst, n_sweeps, 3, 4)
# fg_bi_idx = fg_batch_idx * max_num_inst + fg_inst_idx  # (N,)
# fg_bisw_idx = fg_bi_idx * model.cfg.get('NUM_SWEEPS', 10) + fg_sweep_idx
#
# inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
# model.forward_return_dict['meta'] = {'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices}
#
# local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
# # local_bisw: (N_local,)
# model.forward_return_dict['meta'].update({'local_bisw': local_bisw,
#                                           'local_bisw_inv_indices': local_bisw_inv_indices})
#
# target_dict = model.assign_target(batch_dict)
# print_dict(target_dict)
#
#
# show_inst_assoc = True
# if show_inst_assoc:
#     # use ORIGINAL inst index
#     batch_idx = 1
#     _fg = points[batch_dict['points'][:, -2].long() > -1]
#     batch_mask = _fg[:, 0].long() == batch_idx
#
#     cur_fg = _fg[batch_mask]  # (N_fg, 9)
#
#     inst_assoc = target_dict['inst_assoc']  # (N_fg, 2)
#     cur_inst_assoc = inst_assoc[batch_mask]  # (N_cur_fg, 2)
#
#     # apply inst_assoc
#     cur_fg[:, 1: 3] += cur_inst_assoc
#
#     # viz
#     _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
#     valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
#     _boxes = _boxes[valid_gt_boxes]
#     _boxes = viz_boxes(_boxes.numpy())
#     _bg = points[points[:, -2].long() == -1]
#     cur_bg = _bg[_bg[:, 0].long() == batch_idx]
#     pc = torch.cat([cur_bg, cur_fg], dim=0)
#     print('showing instance assoc')
#     show_pointcloud(pc[:, 1: 4], _boxes, fgr_mask=pc[:, -2] > -1)
#
#
# show_motion_stat = True
# if show_motion_stat:
#     inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst)
#     fg_motion = inst_motion_stat[inst_bi_inv_indices]
#
#     batch_idx = 1
#     batch_mask = fg[:, 0].long() == batch_idx
#
#     cur_fg = fg[batch_mask]  # (N_cur_fg, 8)
#     cur_fg_motion = fg_motion[batch_mask] == 1
#     cur_fg_color = torch.zeros((cur_fg.shape[0], 3))
#     cur_fg_color[cur_fg_motion, 0] = 1
#     cur_fg_color[torch.logical_not(cur_fg_motion), 2] = 1
#
#     # viz
#     _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
#     valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
#     _boxes = _boxes[valid_gt_boxes]
#     _boxes = viz_boxes(_boxes.numpy())
#     bg = points[torch.logical_not(mask_fg)]
#     cur_bg = bg[bg[:, 0].long() == batch_idx]
#     pc = torch.cat([cur_bg, cur_fg], dim=0)
#     pc_colors = torch.cat((cur_bg.new_zeros(cur_bg.shape[0], 3), cur_fg_color), dim=0)
#     print('showing motion stats')
#     show_pointcloud(pc[:, 1: 4], _boxes, pc_colors=pc_colors.numpy())
#
#
# show_correction = True
# if show_correction:
#     # local_tf is computed on aug inst index -> noisy set of foreground
#     # then, applied to ground truth set of foreground
#     # expected result: good looking point cloud
#     mask_fg = batch_dict['points'][:, -2] > -1
#     fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instacne_idx
#     points_batch_idx = batch_dict['points'][:, 0].long()
#     fg_batch_idx = points_batch_idx[mask_fg]
#     fg_inst_idx = fg[:, -2].long()
#     fg_sweep_idx = fg[:, -3].long()
#
#     max_num_inst = batch_dict['instances_tf'].shape[1]
#     fg_bi_idx = fg_batch_idx * max_num_inst + fg_inst_idx  # (N,)
#     fg_bisw_idx = fg_bi_idx * model.cfg.get('NUM_SWEEPS', 10) + fg_sweep_idx
#
#     _, _inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
#
#     _, _local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
#
#     local_tf = target_dict['local_tf']  # (N_local, 3, 4)
#     fg_local_tf = local_tf[_local_bisw_inv_indices]  # (N_fg, 3, 4)
#
#     # --
#     # only apply correction on fg belongs to moving instance
#     # --
#     inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst)
#     fg_motion = inst_motion_stat[_inst_bi_inv_indices]  # (N_fg)
#
#     fg_motion = fg_motion == 1
#     dyn_fg = fg[fg_motion]  # (N_dyn, 8)
#     dyn_fg_local_tf = fg_local_tf[fg_motion]  # (N_dyn, 3, 4)
#     dyn_fg[:, 1: 4] = torch.matmul(dyn_fg_local_tf[:, :3, :3], dyn_fg[:, 1: 4, None]).squeeze(-1) + \
#                       dyn_fg_local_tf[:, :, -1]
#
#     # viz
#     batch_idx = 1
#     cur_dyn_fg = dyn_fg[dyn_fg[:, 0].long() == batch_idx]
#
#     static_fg = fg[torch.logical_not(fg_motion)]
#     cur_static_fg = static_fg[static_fg[:, 0].long() == batch_idx]
#
#     bg = points[torch.logical_not(mask_fg)]
#     cur_bg = bg[bg[:, 0].long() == batch_idx]
#
#     pc = torch.cat([cur_bg, cur_static_fg, cur_dyn_fg], dim=0)
#     pc_colors = torch.zeros((pc.shape[0], 3))
#     pc_colors[cur_bg.shape[0]: cur_bg.shape[0] + cur_static_fg.shape[0], 2] = 1  # static fg - blue
#     pc_colors[cur_bg.shape[0] + cur_static_fg.shape[0]:, 0] = 1  # dynamic fg - red
#
#     _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
#     valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
#     _boxes = _boxes[valid_gt_boxes]
#     _boxes = viz_boxes(_boxes.numpy())
#     print('showing corrected point cloud')
#     show_pointcloud(pc[:, 1: 4], _boxes, pc_colors=pc_colors.numpy())
#
#
#
