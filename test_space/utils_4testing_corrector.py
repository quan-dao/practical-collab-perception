import numpy as np
import torch
from einops import rearrange


def build_meta(fg: torch.Tensor, max_num_inst_in_batch: int, num_sweeps: int, loc_sweep_idx=-2, loc_instance_idx=-1) -> dict:
    assert torch.all(fg[:, loc_instance_idx].long() > -1)
    fg_batch_inst_sw = (fg[:, 0].long() * max_num_inst_in_batch + fg[:, loc_instance_idx].long()) * num_sweeps \
                        + fg[:, loc_sweep_idx].long()
    # group foreground to locals
    locals_bis, locals2fg = torch.unique(fg_batch_inst_sw, sorted=True, return_inverse=True)

    # groups local to instance
    locals_batch_inst = torch.div(locals_bis, num_sweeps, rounding_mode='floor')
    instance_bi, inst2locals = torch.unique(locals_batch_inst, sorted=True, return_inverse=True)

    meta = {
        'locals2fg': locals2fg,
        'inst2locals': inst2locals,
        'locals_bis': locals_bis,
        'instance_bi': instance_bi,
    }
    return meta


def _correct_points(points_: torch.Tensor, instances_tf: torch.Tensor, loc_sweep_idx=-2, loc_instance_idx=-1):
    mask_fg = points_[:, loc_instance_idx].long() > -1
    if not torch.any(mask_fg):
        # no foreground exists -> early return
        return
    _, num_instances, num_sweeps = instances_tf.shape[:3]
    fg = points_[mask_fg]
    meta = build_meta(fg, num_instances, num_sweeps, loc_sweep_idx, loc_instance_idx)

    # extract tf for each foreground 
    all_locals_tf = rearrange(instances_tf, 'B N_i N_sw C1 C2 -> (B N_i N_sw) C1 C2', C1=3, C2=4)
    locals_tf = all_locals_tf[meta['locals_bis']]  # (N_local, 3, 4)

    fg_tf = locals_tf[meta['locals2fg']]  # (N_fg, 3, 4)
    
    # correct foreground
    points_[mask_fg, 1: 4] = torch.matmul(fg_tf[:, :3, :3], points_[mask_fg, 1: 4].unsqueeze(-1)).squeeze(-1) + fg_tf[:, :, -1]


def correct_points(points: np.ndarray, instance_tf: np.ndarray, loc_sweep_idx=-2, loc_instance_idx=-1):
    points  = torch.from_numpy(points).float()
    instance_tf = torch.from_numpy(instance_tf).float()
    _correct_points(points, instance_tf, loc_sweep_idx, loc_instance_idx)
    return points.numpy()
