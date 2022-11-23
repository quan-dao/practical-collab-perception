import torch
from einops import rearrange
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import Aligner

from tools_4testing import load_data_to_tensor, load_dict_to_gpu, load_dict_to_cpu
from _dev_space.viz_tools import print_dict

import argparse


def main(**kwargs):
    cfg_file = './tail_cutter_cfg.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    batch_dict = None
    for _ in range(kwargs.get('chosen_iter', 5)):
        batch_dict = next(iter_dataloader)
    load_data_to_tensor(batch_dict)

    model = Aligner(cfg.MODEL, num_class=10, dataset=dataset)
    print('---')
    print(model)
    print('---')

    model.cuda()
    load_dict_to_gpu(batch_dict)
    ret_dict, tb_dict, _ = model(batch_dict)
    load_dict_to_cpu(batch_dict)
    print_dict(batch_dict)

    if kwargs.get('test_assign_target_gt_as_pred', True):
        input_dict = batch_dict['2nd_stage_input']
        gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, 11) -center (3), size (3), yaw, dummy_v (2), instance_index, class
        gt_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        gt_boxes = gt_boxes[input_dict['meta']['inst_bi']]  # (N_inst, 11)

        # overwrite pred_boxes in input_dict
        pseudo_pred_boxes = gt_boxes[:, :7]
        batch_dict['2nd_stage_input']['pred_boxes'] = pseudo_pred_boxes

        target_dict = model.det_head.assign_target(batch_dict)

        refinement = target_dict['boxes_refinement']
        print(refinement)

    if kwargs.get('test_assign_target_perturbed_gt_as_pred', True):
        perturb_transl = torch.tensor([3., 5., 1.])
        perturb_rot = np.pi

        input_dict = batch_dict['2nd_stage_input']
        gt_boxes = batch_dict[
            'gt_boxes']  # (B, N_inst_max, 11) -center (3), size (3), yaw, dummy_v (2), instance_index, class
        gt_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        gt_boxes = gt_boxes[input_dict['meta']['inst_bi']]  # (N_inst, 11)

        pseudo_pred_boxes = torch.clone(gt_boxes[:, :7])
        pseudo_pred_boxes[:, :3] = pseudo_pred_boxes[:, :3] + perturb_transl
        pseudo_pred_boxes[:, -1] = pseudo_pred_boxes[:, -1] + perturb_rot
        batch_dict['2nd_stage_input']['pred_boxes'] = pseudo_pred_boxes

        target_dict = model.det_head.assign_target(batch_dict)

        decode_boxes = model.det_head.decode_boxes_refinement(pseudo_pred_boxes, target_dict['boxes_refinement'])

        diff = decode_boxes - gt_boxes[:, :7]
        print(diff)


if __name__ == '__main__':
    main(test_assign_target_gt_as_pred=False)


