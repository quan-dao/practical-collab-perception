import numpy as np
import matplotlib.pyplot as plt
from utils import *
from einops import rearrange
import torch


def show_points_cls_and_boxes(batch_dict):
    print('----------show_points_cls_and_boxes----------')
    points = batch_dict['points']
    boxes = batch_dict['gt_boxes']

    points_colors = color_points_by_detection_cls(points)
    show_pointcloud(points[:, 1: 4], viz_boxes(boxes[0, :, :7]), pc_colors=points_colors)


def check_aligner_training_compute_fg(batch_dict: dict, map_point_feat2idx: dict, vehicle_class_indices: tuple,
                                      return_mask_fg=False):
    print('----------check_aligner_training_compute_fg----------')
    assert batch_dict['gt_boxes'].shape[1] == batch_dict['instances_tf'].shape[1], \
        f"{batch_dict['gt_boxes'].shape[1]} != {batch_dict['instances_tf'].shape[1]}"
    points = batch_dict['points']
    points_batch_idx = points[:, 0].long()
    num_points = points.shape[0]

    boxes = batch_dict['gt_boxes']
    gt_boxes_cls_idx = rearrange(batch_dict['gt_boxes'][:, :, -1].long(), 'B N_i -> (B N_i)')
    max_num_inst = batch_dict['gt_boxes'].shape[1]
    points_merge_batch_aug_inst_idx = (points_batch_idx * max_num_inst
                                       + points[:, map_point_feat2idx['aug_inst_idx']].long())
    points_aug_cls_idx = gt_boxes_cls_idx[points_merge_batch_aug_inst_idx]
    points_aug_cls_idx[points[:, map_point_feat2idx['aug_inst_idx']].long() == -1] = 0
    mask_fg = points_aug_cls_idx.new_zeros(num_points).bool()
    for vehicle_cls_idx in vehicle_class_indices:
        mask_fg = torch.logical_or(mask_fg, points_aug_cls_idx == vehicle_cls_idx)

    points_color = torch.zeros(num_points, 3)
    points_color[mask_fg] = torch.tensor([1, 0, 0]).float()
    if return_mask_fg:
        return mask_fg
    show_pointcloud(points[:, 1: 4], viz_boxes(boxes[0, :, :7]), pc_colors=points_color)


def check_consistency_2metas(batch_dict: dict, map_point_feat2idx: dict, vehicle_class_indices: tuple):
    print('----------check_consistency_2metas----------')
    points = batch_dict['points']
    num_points = points.shape[0]
    mask_fg = check_aligner_training_compute_fg(batch_dict, map_point_feat2idx, vehicle_class_indices,
                                                return_mask_fg=True)
    max_num_inst = batch_dict['gt_boxes'].shape[1]
    meta = build_meta_dict(points[mask_fg], max_num_inst, map_point_feat2idx['aug_inst_idx'], map_point_feat2idx)

    # ------------------------------------------------------------------
    # compute foreground in PointAligner.assign_target
    # ------------------------------------------------------------------
    points_cls_idx = points[:, map_point_feat2idx['cls_idx']].long()
    mask_fg = points.new_zeros(num_points).bool()
    for vehicle_cls_idx in vehicle_class_indices:
        mask_fg = torch.logical_or(mask_fg, points_cls_idx == vehicle_cls_idx)
    target_meta = build_meta_dict(points[mask_fg], max_num_inst, map_point_feat2idx['inst_idx'], map_point_feat2idx)

    assert target_meta['inst_bi'].shape[0] == meta['inst_bi'].shape[0], \
        f"{target_meta['inst_bi'].shape[0]} == {meta['inst_bi'].shape[0]}"
    assert torch.all(target_meta['inst_bi'] == meta['inst_bi'])


def check_assign_points_wise_target(batch_dict: dict, map_point_feat2idx: dict, vehicle_class_indices: tuple):
    print('----------check_assign_points_wise_target----------')
    points = batch_dict['points']
    num_points = points.shape[0]
    boxes = batch_dict['gt_boxes']
    max_num_inst = batch_dict['gt_boxes'].shape[1]

    points_cls_idx = points[:, map_point_feat2idx['cls_idx']].long()
    mask_fg = points.new_zeros(num_points).bool()
    for vehicle_cls_idx in vehicle_class_indices:
        mask_fg = torch.logical_or(mask_fg, points_cls_idx == vehicle_cls_idx)
    target_fg = mask_fg.long()  # (N_pts,) - use original instance index for foreground seg

    points_color = torch.zeros(num_points, 3)
    points_color[target_fg == 1] = torch.tensor([1, 0, 0]).float()
    print('showing: target_fg')
    show_pointcloud(points[:, 1: 4], viz_boxes(boxes[0, :, :7]), pc_colors=points_color)

    gt_boxes_xy = rearrange(batch_dict['gt_boxes'][:, :, :2], 'B N_i C -> (B N_i) C')
    fg_merge_batch_inst_idx = (points[mask_fg, 0].long() * max_num_inst
                               + points[mask_fg, map_point_feat2idx['inst_idx']].long())
    fg_boxes_xy = gt_boxes_xy[fg_merge_batch_inst_idx]  # (N_fg, 2)
    target_inst_assoc = fg_boxes_xy - points[mask_fg, 1: 3]  # (N_fg, 2)
    print('showing offseted foreground')
    points[mask_fg, 1: 3] = points[mask_fg, 1: 3] + target_inst_assoc

    centers = boxes[0, :, :3]
    centers_colors = centers.new_zeros(centers.shape[0], 3)
    centers_colors[:, 2] = 1  # blue for center
    show_pointcloud(torch.cat((points[:, 1: 4], centers), dim=0),
                    viz_boxes(boxes[0, :, :7]),
                    pc_colors=torch.cat((points_color, centers_colors), dim=0))


def check_assign_instance_wise_target(batch_dict: dict, map_point_feat2idx: dict, vehicle_class_indices: tuple):
    print('----------check_assign_instance_wise_target----------')
    boxes = batch_dict['gt_boxes']
    points = batch_dict['points']
    num_points = points.shape[0]
    max_num_inst = batch_dict['gt_boxes'].shape[1]

    points_cls_idx = points[:, map_point_feat2idx['cls_idx']].long()
    mask_fg = points.new_zeros(num_points).bool()
    for vehicle_cls_idx in vehicle_class_indices:
        mask_fg = torch.logical_or(mask_fg, points_cls_idx == vehicle_cls_idx)

    meta = build_meta_dict(points[mask_fg], max_num_inst, map_point_feat2idx['inst_idx'], map_point_feat2idx)

    instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)
    # target motion
    inst_motion_stat = torch.linalg.norm(instances_tf[:, :, 0, :, -1], dim=-1) > 0.5  # translation more than 0.5m
    inst_motion_stat = rearrange(inst_motion_stat.long(), 'B N_inst_max -> (B N_inst_max)')
    inst_motion_stat = inst_motion_stat[meta['inst_bi']]  # (N_inst)

    # locals' transformation
    instances_tf = rearrange(instances_tf, 'B N_inst_max N_sweep C1 C2 -> (B N_inst_max N_sweep) C1 C2', C1=3, C2=4)
    local_tf = instances_tf[meta['local_bisw']]  # (N_local, 3, 4)

    inst_bi_inv_indices = meta['inst_bi_inv_indices']  # (N_fg)
    fg_motion = inst_motion_stat[inst_bi_inv_indices] == 1  # (N_fg)

    print('showing: fg motion stat')
    points_color = torch.zeros(num_points, 3)
    fg_colors = torch.zeros(mask_fg.long().sum().item(), 3)
    fg_colors[:, 2] = 1.0  # blue for static
    fg_colors[fg_motion] = torch.tensor([1, 0, 0]).float()  # red for dynamic
    points_color[mask_fg] = fg_colors
    show_pointcloud(points[:, 1: 4], viz_boxes(boxes[0, :, :7]), pc_colors=points_color)

    print('showing: oracle point cloud')
    fg, bg = points[mask_fg], points[torch.logical_not(mask_fg)]
    dyn_fg = fg[fg_motion, 1: 4]  # (N_dyn, 3)
    # reconstruct with ground truth
    local_bisw_inv_indices = meta['local_bisw_inv_indices']
    gt_fg_tf = local_tf[local_bisw_inv_indices]  # (N_fg, 3, 4)
    gt_dyn_fg_tf = gt_fg_tf[fg_motion]  # (N_dyn, 3, 4)

    gt_recon_dyn_fg = (torch.matmul(gt_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1)
                       + gt_dyn_fg_tf[:, :, -1])  # (N_dyn, 3)

    fg[fg_motion, 1: 4] = gt_recon_dyn_fg
    show_pointcloud(torch.cat([fg[:, 1: 4], bg[:, 1:4]], dim=0),
                    viz_boxes(boxes[0, :, :7]),
                    pc_colors=torch.cat([fg_colors, torch.zeros(bg.shape[0], 3)], dim=0))


if __name__ == '__main__':
    is_training = True
    batch_size = 1
    target_batch_idx = 1
    batch_dict = make_batch_dict(is_training, batch_size, target_batch_idx)
    load_data_to_tensor(batch_dict)
    print('carrying out test on sample: ', batch_dict['metadata'][0]['token'])
    print_dict(batch_dict, name='batch_dict')

    num_pts_raw_feat = 1 + 5  # batch_idx, x, y, z, intensity, time
    map_point_feat2idx = {
        'sweep_idx': num_pts_raw_feat + 0,
        'inst_idx': num_pts_raw_feat + 1,
        'aug_inst_idx': num_pts_raw_feat + 2,
        'cls_idx': num_pts_raw_feat + 3,
    }

    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    vehicle_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle']
    vehicle_class_indices = tuple([1 + class_names.index(cls_name) for cls_name in vehicle_classes])

    # show_points_cls_and_boxes(batch_dict)
    # check_aligner_training_compute_fg(batch_dict, map_point_feat2idx, vehicle_class_indices)
    # check_consistency_2metas(batch_dict, map_point_feat2idx, vehicle_class_indices) # TODO: test this batch_size=2
    # check_assign_points_wise_target(batch_dict, map_point_feat2idx, vehicle_class_indices)
    check_assign_instance_wise_target(batch_dict, map_point_feat2idx, vehicle_class_indices)
