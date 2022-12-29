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


def check_aligner_training_compute_fg(batch_dict: dict, map_point_feat2idx: dict, vehicle_class_indices: tuple):
    print('----------show_pocheck_aligner_training_compute_fgints_cls_and_boxes----------')
    load_data_to_tensor(batch_dict)
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
    show_pointcloud(points[:, 1: 4], viz_boxes(boxes[0, :, :7]), pc_colors=points_color)


if __name__ == '__main__':
    is_training = True
    batch_size = 1
    target_batch_idx = 1
    batch_dict = make_batch_dict(is_training, batch_size, target_batch_idx)
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

    show_points_cls_and_boxes(batch_dict)
    check_aligner_training_compute_fg(batch_dict, map_point_feat2idx, vehicle_class_indices)
