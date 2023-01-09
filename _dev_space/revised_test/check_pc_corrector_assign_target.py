import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.viz_tools import print_dict
from _dev_space._test.tools_4testing import load_data_to_tensor, load_dict_to_gpu, load_dict_to_cpu
from _dev_space.pc_corrector import PointCloudCorrector
from utils import show_pointcloud, viz_boxes
import sys


def make_target(target_batch_idx=1, batch_size=3, is_training=True):
    cfg_file = './second_corrector_mini.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./a_dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                              batch_size=batch_size, dist=False, logger=logger, training=is_training,
                                              total_epochs=1, seed=666, workers=1)
    iter_dataloader = iter(dataloader)

    batch_dict = None
    for _ in range(target_batch_idx):
        batch_dict = next(iter_dataloader)
    load_data_to_tensor(batch_dict)
    print_dict(batch_dict, 'batch_dict')
    load_dict_to_gpu(batch_dict)

    corrector = PointCloudCorrector(cfg.MODEL.CORRECTOR, num_bev_features=1, voxel_size=[0.1, 0.1, 0.2],
                                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    # extract fg
    points = batch_dict['points']
    mask_fg = points[:, corrector.map_point_feat2idx['inst_idx']] > -1  # all 10 classes
    fg = points[mask_fg]

    # build meta
    meta = corrector.build_meta(fg, batch_dict['gt_boxes'].shape[1], mask_fg)

    # assign target
    target = corrector.assign_target(batch_dict, meta)

    meta['fg'] = fg

    batch_dict['target'] = target
    batch_dict['meta'] = meta

    load_dict_to_cpu(batch_dict)
    torch.save(batch_dict, 'corrector_target_batch_dict.pth')
    return batch_dict


def show_target(batch_dict, chosen_batch_idx=1):
    points = batch_dict['points']
    target = batch_dict['target']
    gt_boxes = batch_dict['gt_boxes']
    meta = batch_dict['meta']
    fg = meta['fg']
    mask_fg = meta['mask_fg']

    points_batch_mask = points[:, 0].long() == chosen_batch_idx

    cur_gt_boxes = gt_boxes[chosen_batch_idx]
    valid_gt_boxes = cur_gt_boxes[:, -1] > 0  # cls_idx > 0
    cur_gt_boxes = cur_gt_boxes[valid_gt_boxes].numpy()
    viz_gt_boxes = viz_boxes(cur_gt_boxes)

    bg = points[torch.logical_not(mask_fg)]

    print('-----\n'
          'showing point cls\n'
          '-----')
    points_cls = target['points_cls']

    color_point_cls = points.new_zeros(points.shape[0], 3)
    color_point_cls[points_cls[:, 1] > 0, 2] = 1.0  # blue: static fg
    color_point_cls[points_cls[:, 2] > 0, 0] = 1.0  # red: dynamic fg

    show_pointcloud(points[points_batch_mask, 1: 4], viz_gt_boxes, color_point_cls[points_batch_mask])

    print('-----\n'
          'showing point embedding\n'
          '-----')
    fg_embedding = target['fg_embedding']  # (N_fg, 2)
    to_center_fg = torch.clone(fg)
    to_center_fg[:, 1: 3] += fg_embedding

    cur_to_center_fg = to_center_fg[fg[:, 0].long() == chosen_batch_idx, 1: 4]
    cur_bg = bg[bg[:, 0].long() == chosen_batch_idx, 1: 4]

    color_to_center = torch.zeros(cur_to_center_fg.shape[0] + cur_bg.shape[0], 3)
    color_to_center[:cur_to_center_fg.shape[0], 0] = 1  # red for all foreground
    show_pointcloud(torch.cat([cur_to_center_fg, cur_bg], dim=0), viz_gt_boxes, color_to_center)

    print('-----\n'
          'showing point offseted\n'
          '-----')
    mask_dyn_fg = target['points_cls'][mask_fg, 2] > 0  # (N_fg,)
    if torch.any(mask_dyn_fg):
        fg_offset = target['fg_offset']  # (N_fg, 3)
        corrected_fg = torch.clone(fg)
        corrected_fg[mask_dyn_fg, 1: 4] += fg_offset[mask_dyn_fg]

        color_corrected_fg = torch.zeros(fg.shape[0], 3)
        color_corrected_fg[mask_dyn_fg, 0] = 1  # red: dynamic fg
        color_corrected_fg[torch.logical_not(mask_dyn_fg), 2] = 1  # blue: static fg

        show_pointcloud(torch.cat([corrected_fg[fg[:, 0].long() == chosen_batch_idx, 1: 4], cur_bg], dim=0), viz_gt_boxes,
                        torch.cat([color_corrected_fg[fg[:, 0].long() == chosen_batch_idx], torch.zeros(cur_bg.shape[0], 3)], dim=0))

    else:
        print('there is no dynamic foreground points')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'make_target':
            batch_dict = make_target(target_batch_idx=1, batch_size=3, is_training=True)
        else:
            raise ValueError
    else:
        batch_dict = torch.load('corrector_target_batch_dict.pth')
        show_target(batch_dict, chosen_batch_idx=2)

