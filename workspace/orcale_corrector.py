import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict


class OracleCorrector(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, batch_dict: Dict):
        """
        Args:
            batch_dict: {
                points (torch.Tensor): (N, 10) - batch_idx, x, y, z, intensity, time_sec, on_ground, on_drivable || sweep_idx, inst_idx
                instances_tf (torch.Tensor): (N_inst, N_sweeps, 3, 4)
            }
        """
        self.correct_points(batch_dict['points'], batch_dict['instances_tf'])
        return batch_dict

    @staticmethod
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

    @staticmethod
    def correct_points(points_: torch.Tensor, instances_tf: torch.Tensor, loc_sweep_idx=-2, loc_instance_idx=-1):
        mask_fg = points_[:, loc_instance_idx].long() > -1
        if not torch.any(mask_fg):
            # no foreground exists -> early return
            return
        _, num_instances, num_sweeps = instances_tf.shape[:3]
        fg = points_[mask_fg]
        meta = OracleCorrector.build_meta(fg, num_instances, num_sweeps, loc_sweep_idx, loc_instance_idx)

        # extract tf for each foreground 
        all_locals_tf = rearrange(instances_tf, 'B N_i N_sw C1 C2 -> (B N_i N_sw) C1 C2', C1=3, C2=4)
        locals_tf = all_locals_tf[meta['locals_bis']]  # (N_local, 3, 4)

        fg_tf = locals_tf[meta['locals2fg']]  # (N_fg, 3, 4)
        
        # correct foreground
        points_[mask_fg, 1: 4] = torch.matmul(fg_tf[:, :3, :3], points_[mask_fg, 1: 4].unsqueeze(-1)).squeeze(-1) + fg_tf[:, :, -1]


