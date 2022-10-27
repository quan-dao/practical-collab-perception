import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.tail_cutter import PointAligner
from _dev_space.viz_tools import print_dict, viz_boxes
from tools_4testing import load_data_to_tensor, load_dict_to_gpu, show_pointcloud
import os
from pcdet.models.detectors.alginer import Aligner
import matplotlib.pyplot as plt
import numpy as np


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


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def inference():
    cfg_file = './tail_cutter_full.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    for _ in range(5):
        batch_dict = next(iter_dataloader)

    print_dict(batch_dict)
    load_data_to_tensor(batch_dict)

    model = Aligner(cfg.MODEL, num_class=10, dataset=dataset)
    load_params_from_file(model, '../from_idris/ckpt/tail_cutter_fatter_head_ep12_nusc4th.pth', logger, to_cpu=True)

    model.eval()
    model.cuda()
    load_dict_to_gpu(batch_dict)
    with torch.no_grad():
        batch_dict = model(batch_dict)

    torch.save(batch_dict, 'inference_tail_cutter_ep10_nusc4th.pth')


def display_inference():
    batch_dict = torch.load('inference_tail_cutter_ep10_nusc4th.pth', map_location=torch.device('cpu'))
    # print(batch_dict)
    print_dict(batch_dict)
    bg = batch_dict['xyz_bg']
    fg = batch_dict['xyz_fg']
    fg_inst_idx = batch_dict['fg_inst_idx']
    gt_boxes = batch_dict['gt_boxes']  # (B, N_max_gt, 10)

    inst_motion_stat = sigmoid(batch_dict['inst_motion_stat'][:, 0]) > 0.5
    inst_bi_inv_indices = batch_dict['forward_meta']['inst_bi_inv_indices']
    fg_motion = inst_motion_stat[inst_bi_inv_indices]  # (N_fg)
    fg_motion = fg_motion == 1  # (N_fg)

    dyn_fg = fg[fg_motion, 1: 4]  # (N_dyn, 3)
    local_transl = batch_dict['local_transl']  # (N_local, 3)
    local_rot_mat = batch_dict['local_rot']  # (N_local, 3, 3)
    local_tf = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)

    # reconstruct with ground truth
    local_bisw_inv_indices = batch_dict['forward_meta']['local_bisw_inv_indices']
    fg_tf = local_tf[local_bisw_inv_indices]  # (N_fg, 3, 4)
    dyn_fg_tf = fg_tf[fg_motion]  # (N_dyn, 3, 4)
    recon_dyn_fg = torch.matmul(dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1) + \
                   dyn_fg_tf[:, :, -1]  # (N_dyn, 3)
    dyn_fg = torch.cat((fg[fg_motion, 0].unsqueeze(1), recon_dyn_fg), dim=1)
    static_fg = fg[torch.logical_not(fg_motion), :4]  # batch_idx, x, y, z
    _recon_fg = torch.cat((dyn_fg, static_fg))
    _recon_fg_color = _recon_fg.new_zeros(_recon_fg.shape[0], 3)
    _recon_fg_color[:dyn_fg.shape[0], 0] = 1
    _recon_fg_color[dyn_fg.shape[0]:, 2] = 1

    batch_idx = 0
    cur_bg = bg[bg[:, 0].long() == batch_idx]
    cur_fg = fg[fg[:, 0].long() == batch_idx]
    cur_fg_inst_idx = fg_inst_idx[fg[:, 0].long() == batch_idx]
    cur_fg_motion = fg_motion[fg[:, 0].long() == batch_idx]

    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = _boxes[valid_gt_boxes]
    _boxes = viz_boxes(_boxes.numpy())

    _original_points = torch.cat((cur_bg, cur_fg))
    _original_points_color = _original_points.new_zeros(_original_points.shape[0], 3)
    _original_points_color[cur_bg.shape[0]:, 0] = 1
    show_pointcloud(_original_points[:, 1: 4], _boxes, _original_points_color)

    # color by instances
    # unq_inst_idx, inv_ids = torch.unique(cur_fg_inst_idx, return_inverse=True)
    # inst_color = plt.cm.rainbow(np.linspace(0, 1, unq_inst_idx.shape[0]))[:, :3]
    # inst_color = torch.from_numpy(inst_color)
    # cur_fg_colors = inst_color[inv_ids]

    # color by motion
    # cur_fg_colors = cur_fg.new_zeros(cur_fg.shape[0], 3)
    # cur_fg_colors[cur_fg_motion, 0] = 1

    # color of reconstruction
    cur_fg_colors = _recon_fg_color[_recon_fg[:, 0].long() == batch_idx]

    cur_bg_color = cur_bg.new_zeros(cur_bg.shape[0], 3)

    cur_points = torch.cat((cur_bg, _recon_fg[_recon_fg[:, 0].long() == batch_idx]))
    # cur_points = torch.cat([cur_bg, cur_fg])
    cur_points_color = torch.cat((cur_bg_color, cur_fg_colors)).numpy()

    show_pointcloud(cur_points[:, 1:], _boxes, pc_colors=cur_points_color)



if __name__ == '__main__':
    inference()
    display_inference()
