from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space._test.tools_4testing import load_data_to_tensor
from _dev_space.viz_tools import print_dict, viz_boxes
from _dev_space.tools_box import show_pointcloud
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_scatter


def make_batch_dict(is_training: bool, batch_size=1, target_batch_idx=5) -> dict:
    cfg_file = './second_aligner_mini.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./a_dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                              batch_size=batch_size, dist=False, logger=logger, training=is_training,
                                              total_epochs=1, seed=666, workers=1)
    iter_dataloader = iter(dataloader)

    batch_dict = None
    for _ in range(target_batch_idx):
        batch_dict = next(iter_dataloader)

    return batch_dict


def color_points_by_detection_cls(points) -> np.ndarray:
    cls_colors = plt.cm.rainbow(np.linspace(0, 1, 11))[:, :3]
    idx_cls = 9
    points_cls_idx = points[:, idx_cls].astype(int)
    points_colors = cls_colors[points_cls_idx]
    # zero-out color of background
    mask_detection_fg = points_cls_idx >= 1
    points_colors *= mask_detection_fg.reshape(-1, 1)
    return points_colors


def build_meta_dict(fg: torch.Tensor, max_num_instances: int, index_of_instance_idx: int,
                    map_point_feat2idx, num_sweeps=10) -> dict:
    """
    Args:
        fg: (N_fg, 10) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_idx, cls_idx
        max_num_instances:
        index_of_instance_idx: where is instance_index in point_feats
    """
    meta = {}

    fg_sweep_idx = fg[:, map_point_feat2idx['sweep_idx']].long()
    fg_batch_idx = fg[:, 0].long()
    fg_inst_idx = fg[:, index_of_instance_idx].long()

    # merge batch_idx & instance_idx
    fg_bi_idx = fg_batch_idx * max_num_instances + fg_inst_idx  # (N,)

    # merge batch_idx, instance_idx & sweep_idx
    fg_bisw_idx = fg_bi_idx * num_sweeps + fg_sweep_idx

    # group foreground points to instance
    inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
    # inst_bi: (N_inst,)
    # inst_bi_inv_indices: (N_fg,)

    # group foreground points to local group
    local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
    # local_bisw: (N_local,)

    meta.update({
        'max_num_inst': max_num_instances,
        'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices,
        'local_bisw': local_bisw, 'local_bisw_inv_indices': local_bisw_inv_indices
    })

    # -----------------------------------------------------------------------------
    # the following is to find the center of each instance's target local group
    # -----------------------------------------------------------------------------
    # get the max sweep_index of each instance
    inst_max_sweep_idx = torch_scatter.scatter_max(fg_sweep_idx, inst_bi_inv_indices)[0]  # (N_inst,)
    # get bisw_index of each instance's max sweep
    inst_target_bisw_idx = inst_bi * num_sweeps + inst_max_sweep_idx  # (N_inst,)
    # for each value in inst_target_bisw_idx find WHERE (i.e., index) it appear in local_bisw
    corr = local_bisw[:, None] == inst_target_bisw_idx[None, :]  # (N_local, N_inst)
    corr = corr.long() * torch.arange(local_bisw.shape[0]).unsqueeze(1).to(fg.device)
    meta['indices_local_to_inst_target'] = corr.sum(dim=0)  # (N_inst)

    # -----------------------------------------------------------------------------
    # the following is to establish correspondence between instances & locals
    # -----------------------------------------------------------------------------
    local_bi = local_bisw // num_sweeps
    # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
    local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
    local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1).to(fg.device)

    meta['local_bi_in_inst_bi'] = local_bi_in_inst_bi.sum(dim=0)  # (N_local)
    # this is identical to indices_instance_to_local which is used to
    # broadcast inst_global_feat from (N_inst, C_inst) to (N_local, C_inst)

    return meta
