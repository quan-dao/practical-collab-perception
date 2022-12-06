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
import lovely_tensors as lt


lt.monkey_patch()


def _show_point_cloud(batch_dict, chosen_batch_idx):
    points = batch_dict['points']
    print('points cls idx: ', points[:, -1])
    mask_cur = points[:, 0].long() == chosen_batch_idx
    cur_points = points[mask_cur]

    gt_boxes = batch_dict['gt_boxes']  # (B, N_box_max, 7+..)
    cur_gt_boxes = gt_boxes[chosen_batch_idx]
    mask_valid_boxes = cur_gt_boxes[:, -1] > 0
    cur_gt_boxes = cur_gt_boxes[mask_valid_boxes]

    # color points & boxes according to class index
    classes_color = torch.eye(3)

    points_color = cur_points.new_zeros(cur_points.shape[0], 3)
    cur_points_cls_idx = cur_points[:, -1].long() + 1
    mask_fg = cur_points_cls_idx > 0
    points_color[mask_fg] = classes_color[cur_points_cls_idx[mask_fg].long() - 1]

    boxes_color = classes_color[cur_gt_boxes[:, -1].long() - 1]

    show_pointcloud(cur_points[:, 1: 4], boxes=viz_boxes(cur_gt_boxes), pc_colors=points_color, boxes_color=boxes_color)


def _show_target_proposals(model, batch_dict, chosen_batch_idx, color_instance_by_motion=False):
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
    # ------

    # get current boxes
    inst_bi = model.forward_return_dict['meta']['inst_bi']
    max_num_inst = batch_dict['instances_tf'].shape[1]
    inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

    if color_instance_by_motion:
        num_instances = target_boxes.shape[0]
        inst_colors = target_boxes.new_zeros(num_instances, 3)
        inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst,) - long
        inst_motion_mask = inst_motion_stat > 0
        inst_colors[inst_motion_mask] = torch.tensor([1, 0, 0]).float()  # red for moving
        inst_colors[torch.logical_not(inst_motion_mask)] = torch.tensor([0, 0, 1]).float()  # blue for static
    else:
        # 1 instance -> 1 color for points & box
        inst_colors = plt.cm.rainbow(np.linspace(0, 1, inst_bi.shape[0]))[:, :3]
        inst_colors = torch.from_numpy(inst_colors)

    cur_target_boxes = target_boxes[inst_batch_idx == chosen_batch_idx]  # (N_cur_inst, 7+...)
    cur_target_boxes_colors = inst_colors[inst_batch_idx == chosen_batch_idx]

    # get current points
    points = batch_dict['points']
    mask_fg = points[:, -2].long() > -1  # Remember the forward pass of training is based on aug_inst_index
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
    pc_colors = torch.cat([cur_bg_colors, cur_fg_colors]).numpy()
    show_pointcloud(pc[:, 1: 4], pc_colors=pc_colors, boxes=viz_boxes(cur_target_boxes),
                    boxes_color=cur_target_boxes_colors)


def _show_predicted_boxes(model, batch_dict, chosen_batch_idx):
    # invoke forward pass in training mode
    load_dict_to_gpu(batch_dict)
    model.cuda()
    model.train()
    batch_dict = model(batch_dict)

    pred_dict = model.generate_predicted_boxes(batch_dict['batch_size'], debug=True, batch_dict=batch_dict)

    # move stuff to cpu
    load_dict_to_cpu(batch_dict)
    load_dict_to_cpu(model.forward_return_dict)
    for b_idx in range(batch_dict['batch_size']):
        load_dict_to_cpu(pred_dict[b_idx])

    # ---
    # viz
    # ---
    pred_boxes = pred_dict[chosen_batch_idx]['pred_boxes']
    points = batch_dict['points']
    mask_cur = points[:, 0].long() == chosen_batch_idx
    cur_points = points[mask_cur]
    show_pointcloud(cur_points[:, 1: 4], fgr_mask=cur_points[:, -2] > -1, boxes=viz_boxes(pred_boxes))


def _show_corrected_point_cloud(model, batch_dict, chosen_batch_idx):
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
    # ------

    # get current boxes
    inst_bi = model.forward_return_dict['meta']['inst_bi']
    max_num_inst = batch_dict['instances_tf'].shape[1]
    inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

    # color instance by motion
    num_instances = target_boxes.shape[0]
    inst_colors = target_boxes.new_zeros(num_instances, 3)
    inst_motion_stat = target_dict['inst_motion_stat']  # (N_inst,) - long
    inst_motion_mask = inst_motion_stat > 0
    inst_colors[inst_motion_mask] = torch.tensor([1, 0, 0]).float()  # red for moving
    inst_colors[torch.logical_not(inst_motion_mask)] = torch.tensor([0, 0, 1]).float()  # blue for static

    fg, bg = batch_dict['corrected_fg'], batch_dict['bg']

    fg_motion_mask = batch_dict['fg_motion_mask']
    fg_colors = fg.new_zeros(fg.shape[0], 3)
    fg_colors[fg_motion_mask] = torch.tensor([1, 0, 0]).float()  # red for moving
    fg_colors[torch.logical_not(fg_motion_mask)] = torch.tensor([0, 0, 1]).float()  # blue for static

    bg_colors = bg.new_zeros(bg.shape[0], 3)

    pc = torch.cat([fg, bg], dim=0)
    pc_colors = torch.cat([fg_colors, bg_colors], dim=0)

    # extract chosen_batch_idx
    cur_target_boxes = target_boxes[inst_batch_idx == chosen_batch_idx]  # (N_cur_inst, 7+...)
    cur_target_boxes_colors = inst_colors[inst_batch_idx == chosen_batch_idx]
    cur_points_mask = pc[:, 0].long() == chosen_batch_idx
    show_pointcloud(pc[cur_points_mask, 1: 4], pc_colors=pc_colors[cur_points_mask],
                    boxes=viz_boxes(cur_target_boxes),
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

    model = PointAligner(cfg.MODEL, len(cfg.CLASS_NAMES))
    if kwargs.get('print_model', False):
        print('---------\n', model, '\n---------\n')

    if kwargs.get('show_raw_point_cloud', False):
        print('showing raw point cloud')
        _show_point_cloud(batch_dict, chosen_batch_idx)

    if kwargs.get('show_target_proposals', False):
        _show_target_proposals(model, batch_dict, chosen_batch_idx)

    if kwargs.get('show_target_motion_stat', False):
        _show_target_proposals(model, batch_dict, chosen_batch_idx, color_instance_by_motion=True)

    if kwargs.get('show_corrected_point_cloud', False):
        _show_corrected_point_cloud(model, batch_dict, chosen_batch_idx)

    if kwargs.get('test_generate_predicted_boxes', False):
        raise NotImplementedError
        # _show_predicted_boxes(model, batch_dict, chosen_batch_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--show_raw_point_cloud', action='store_true', default=False)
    parser.add_argument('--show_target_proposals', action='store_true', default=False)
    parser.add_argument('--show_target_motion_stat', action='store_true', default=False)
    parser.add_argument('--show_corrected_point_cloud', action='store_true', default=False)
    parser.add_argument('--print_model', action='store_true', default=False)
    parser.add_argument('--chosen_iteration', type=int, default=5, help='blah')
    parser.add_argument('--test_generate_predicted_boxes', action='store_true', default=False)
    args = parser.parse_args()
    main(**vars(args))
