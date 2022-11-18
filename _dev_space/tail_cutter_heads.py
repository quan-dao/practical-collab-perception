import torch
import torch.nn as nn
from einops import rearrange


class AlignerHead(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO

    @staticmethod
    def generate_corners(boxes: torch.Tensor) -> torch.Tensor:
        """
        Generate coordinate of corners in boxes' canonical frame
        Args:
            boxes: (N, 7) - center(3), size(3), yaw
        Returns:
            corners: (N, 8, 3)
        """
        device = boxes.device
        x = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=device).float()
        y = torch.tensor([-1, 1, 1, -1, -1, 1, 1, -1], device=device).float()
        z = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=device).float()
        corners = torch.stack([x, y, z], dim=1)  # (8, 3)
        corners = rearrange(corners, 'M C -> 1 M C') * rearrange(boxes[:, 3: 6], 'B C -> B 1 C') / 2.0
        return corners

    @torch.no_grad()
    def compute_position_embedding(self, input_dict: dict) -> torch.Tensor:
        """
        Compute position embedding for "corrected" foreground points by
        * map them to their corresponding proposals' local frame
        * pos_embed = [displace w.r.t center & 8 corners]

        Return:
            pos_embed: (N_fg, 27)
        """
        # unzip input
        pred_boxes = input_dict['pred_boxes']  # (N_inst, 7) - center (3), size (3), yaw
        fg = input_dict['fg']  # (N_fg, 7[+2]) - batch_idx, xyz, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]
        indices_inst_to_fg = input_dict['meta']['inst_bi_inv_indices']  # (N_fg)

        # map fg to their corresponding proposals' local frame
        cos, sin = torch.cos(pred_boxes[:, -1]), torch.sin(pred_boxes[:, -1])
        c_x, c_y, c_z = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2]
        zeros, ones = pred_boxes.new_zeros(pred_boxes.shape[0]), pred_boxes.new_ones(pred_boxes.shape[0])

        tf_world_from_boxes = rearrange(torch.stack([
            cos,  -sin,   zeros, c_x,
            sin,   cos,   zeros, c_y,
            zeros, zeros, ones,  c_z
        ], dim=1), 'N_inst (R C) -> N_inst R C', R=3, C=4)

        tf_for_fg = tf_world_from_boxes[indices_inst_to_fg]  # (N_fg, 3, 4)

        fg_in_box = torch.einsum('bij,bj->bi', tf_for_fg[..., :3], fg[:, 1: 4]) + tf_for_fg[..., -1]  # (N_fg, 3)

        # position embedding
        corners = self.generate_corners(pred_boxes)  # (N_inst, 8, 3)
        corners_for_fg = corners[indices_inst_to_fg]  # (N_fg, 8, 3)
        fg_pos_embed = rearrange(fg_in_box, 'N_fg C -> N_fg 1 C') - corners_for_fg  # (N_fg, 8, 3)
        # pad with displacement w.r.t boxes' center (i.e. coordinate in boxes' frame)
        fg_pos_embed = rearrange(
            torch.cat([rearrange(fg_in_box, 'N_fg C -> N_fg 1 C'), fg_pos_embed], dim=1),
            'N_fg M C -> N_fg (M C)', M=9, C=3
        )
        return fg_pos_embed


