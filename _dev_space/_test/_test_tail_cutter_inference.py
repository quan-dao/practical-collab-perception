import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.viz_tools import print_dict, viz_boxes
from tools_4testing import load_data_to_tensor, load_dict_to_gpu, show_pointcloud
import os
from pcdet.models.detectors.alginer import Aligner
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging


def load_params_from_file(model, filename, logger, to_cpu=False):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['model_state']

    # print('\n--')
    # for k, v in model_state_disk.items():
    #     print(f'{k}: {v.shape}')
    # print('--\n')

    version = checkpoint.get("version", None)
    if version is not None:
        logger.info('==> Checkpoint trained from version: %s' % version)

    state_dict, update_model_state = _load_state_dict(model ,model_state_disk, strict=False)

    for key in state_dict:
        if key not in update_model_state:
            logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))


def _load_state_dict(model, model_state_disk, *, strict=True):
    state_dict = model.state_dict()  # local cache of state_dict
    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_model_state[key] = val
            # logger.info('Update weight %s: %s' % (key, str(val.shape)))

    if strict:
        model.load_state_dict(update_model_state)
    else:
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)
    return state_dict, update_model_state


def inference(ckpt_file: str, target_batch_idx=5, **kwargs):
    if not ckpt_file.endswith('.pth'):
        ckpt_file += '.pth'

    cfg_file = './tail_cutter_full.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    for _ in range(target_batch_idx):
        batch_dict = next(iter_dataloader)

    print_dict(batch_dict)
    print('--\n', batch_dict['metadata'], '\n--')
    load_data_to_tensor(batch_dict)

    model = Aligner(cfg.MODEL, num_class=10, dataset=dataset)
    load_params_from_file(model, f'../from_idris/ckpt/{ckpt_file}', logger, to_cpu=True)

    model.eval()
    model.cuda()
    load_dict_to_gpu(batch_dict)
    with torch.no_grad():
        batch_dict = model(batch_dict)
        correction_dict = model.correct_point_cloud(**kwargs)
        batch_dict['correction_dict'] = correction_dict

    torch.save(batch_dict, f'inference_{ckpt_file}.pth')


def display_inference(ckpt_file: str, **kwargs):
    if not ckpt_file.endswith('.pth'):
        ckpt_file += '.pth'

    batch_dict = torch.load(f'inference_{ckpt_file}.pth', map_location=torch.device('cpu'))
    print_dict(batch_dict)
    print('--\n', batch_dict['metadata'], '\n--')

    chosen_batch_idx = 0
    logger = logging.getLogger()
    if kwargs.get('show_raw_point_cloud', False):
        logger.info('showing raw point cloud w/ g.t boxes & g.t foreground')
        points = batch_dict['points']
        mask_cur = points[:, 0].long() == chosen_batch_idx
        cur_points = points[mask_cur]

        gt_boxes = batch_dict['gt_boxes']  # (B, N_box_max, 7+..)
        cur_gt_boxes = gt_boxes[chosen_batch_idx]
        mask_valid_boxes = cur_gt_boxes[:, -1] > 0
        cur_gt_boxes = cur_gt_boxes[mask_valid_boxes]

        show_pointcloud(cur_points[:, 1: 4], fgr_mask=cur_points[:, -2] > -1, boxes=viz_boxes(cur_gt_boxes))

    if kwargs.get('show_corrected_point_cloud', False):
        correction_dict = batch_dict['correction_dict']

        bg = correction_dict['bg']  # (N_bg, 6[+2])
        bg = bg[bg[:, 0].long() == chosen_batch_idx]

        fg = correction_dict['corrected_fg']  # (N_fg, 6[+2])
        mask_cur_fg = fg[:, 0].long() == chosen_batch_idx

        fg = fg[mask_cur_fg]
        fg_inst_idx = correction_dict['fg_inst_idx'][mask_cur_fg]
        fg_motion_mask = correction_dict['fg_motion_mask'][mask_cur_fg]

        # assign color to points
        color_bg = torch.tensor([0, 0, 0]).repeat(bg.shape[0], 1)
        color_fg = torch.tensor([1, 0, 0]).repeat(fg.shape[0], 1)

        unq_inst_idx, inv_unq_inst_idx = torch.unique(fg_inst_idx, sorted=True, return_inverse=True)
        inst_colors = torch.from_numpy(plt.cm.rainbow(np.linspace(0, 1, unq_inst_idx.shape[0])))[:, :3]
        color_fg_by_inst = inst_colors[inv_unq_inst_idx]

        color_fg_by_motion = torch.clone(color_fg)  # red for dynamic fg
        color_fg_by_motion[torch.logical_not(fg_motion_mask)] = torch.tensor([0, 1, 0])  # green for static fg

        gt_boxes = batch_dict['gt_boxes']  # (B, N_box_max, 7+..)
        cur_gt_boxes = gt_boxes[chosen_batch_idx]
        mask_valid_boxes = cur_gt_boxes[:, -1] > 0
        cur_gt_boxes = cur_gt_boxes[mask_valid_boxes, :7]  # (N_gt_boxes, 7+...)
        color_gt_boxes = torch.tensor([0, 1, 0]).repeat(cur_gt_boxes.shape[0], 1)  # green for g.t

        final_box_dicts = batch_dict['final_box_dicts']  # List of dict
        cur_pred_boxes = final_box_dicts[chosen_batch_idx]['pred_boxes']  # (N_pred_boxes, 7+...)
        color_pred_boxes = torch.tensor([1, 0, 0]).repeat(cur_pred_boxes.shape[0], 1)  # red for predict

        # format visualization input
        viz_points = torch.cat([bg, fg])[:, 1: 4]
        viz_color_points = torch.cat([color_bg, color_fg])
        viz_color_points_by_inst = torch.cat([color_bg, color_fg_by_inst])
        viz_color_points_by_motion = torch.cat([color_bg, color_fg_by_motion])

        viz_bboxes = viz_boxes(torch.cat([cur_gt_boxes, cur_pred_boxes]).numpy())
        viz_color_bboxes = torch.cat([color_gt_boxes, color_pred_boxes])

        logger.info('showing foreground seg')
        show_pointcloud(viz_points, viz_bboxes, viz_color_points, boxes_color=viz_color_bboxes)

        logger.info('showing instance seg')
        show_pointcloud(viz_points, viz_bboxes, viz_color_points_by_inst, boxes_color=viz_color_bboxes)

        logger.info('showing motion seg')
        show_pointcloud(viz_points, viz_bboxes, viz_color_points_by_motion, boxes_color=viz_color_bboxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ckpt_file', type=str, default=None, help='specify the ckpt')
    parser.add_argument('--do_inference', action='store_true', default=False, help='')
    parser.add_argument('--target_batch_idx', type=int, default=5, help='')
    parser.add_argument('--return_instance_index', action='store_true', default=True)
    parser.add_argument('--return_motion_mask', action='store_true', default=True)
    parser.add_argument('--return_foreground_prob', action='store_true', default=True)
    parser.add_argument('--show_raw_point_cloud', action='store_true', default=False)
    parser.add_argument('--show_corrected_point_cloud', action='store_true', default=True)
    args = parser.parse_args()
    if args.do_inference:
        inference(**vars(args))
    display_inference(**vars(args))
