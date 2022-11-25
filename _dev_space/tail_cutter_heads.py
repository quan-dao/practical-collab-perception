import torch
import torch.nn as nn
from einops import rearrange
import torch_scatter
from typing import List
import logging
from _dev_space.non_batching_attention import MultiHeadAttention


class AlignerHead(nn.Module):
    def __init__(self, model_cfg, num_bev_features: int, num_instace_features: int):
        super().__init__()
        self.cfg = model_cfg
        attn_cfg = model_cfg.ATTENTION

        self.encoder_fg = self._make_mlp(num_bev_features + 27, [attn_cfg.NUM_FEATURES])
        # + 27 due to the use of position encoding

        self.encoder_inst = self._make_mlp(3 + 2 * num_instace_features, [attn_cfg.NUM_FEATURES])
        # 3 for location of center of target local
        # 2 * NUM_BEV_FEATURES = cat[(target local feat, inst global feat)]

        self.attention_stack = nn.ModuleList([
            MultiHeadAttention(
                attn_cfg.NUM_HEADS, attn_cfg.NUM_FEATURES, return_attn_weight=a_idx == attn_cfg.NUM_LAYERS - 1
            ) for a_idx in range(attn_cfg.NUM_LAYERS)
        ])

        self.decoder_inst = self._make_mlp(attn_cfg.NUM_FEATURES, [*model_cfg.get('DECODER_MID_CHANNELS', []), 8])
        # following residual coder: delta_x, delta_y, delta_z, delta_dx, delta_dy, delta_dz, diff_sin, diff_cos

        self.forward_return_dict = dict()

    @staticmethod
    def _make_mlp(channels_in: int, channels: list):
        """
        Args:
            channels_in: number of channels of input
            channels: [Hidden_0, ..., Hidden_N, Out]
        """
        layers = []
        channels = [channels_in] + channels
        for c_idx in range(len(channels) - 1):
            is_last_layer = c_idx == (len(channels) - 2)
            layers.append(nn.Linear(channels[c_idx], channels[c_idx + 1], bias=is_last_layer))
            if not is_last_layer:
                layers.append(nn.BatchNorm1d(channels[c_idx + 1], eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

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

    def forward(self, batch_dict):
        input_dict = batch_dict['2nd_stage_input']

        # Prepare input for stack of attention layers
        fg_pos_embedding = self.compute_position_embedding(input_dict)  # (N_fg, 27)
        fg_feat = torch.cat([input_dict['fg_feat'], fg_pos_embedding], dim=1)  # (N_fg, C_bev + 27)
        fg_feat = self.encoder_fg(fg_feat)  # (N_fg, C_attn)
        inst_feat = self.encoder_inst(input_dict['inst_global_feat'])  # (N_inst, C_attn)
        indices_inst2fg = input_dict['meta']['inst_bi_inv_indices']  # (N_fg)

        # Invoke attention stack
        for i, attn_layer in enumerate(self.attention_stack):
            if i < len(self.attention_stack) - 1:
                inst_feat = attn_layer(inst_feat, fg_feat, fg_feat, indices_inst2fg)  # (N_inst, C_attn)
            else:
                inst_feat, attn_weights = attn_layer(inst_feat, fg_feat, fg_feat, indices_inst2fg)
                # inst_feat: (N_inst, C_attn)
                # attn_weights: (N_fg,)
                self.forward_return_dict['attn_weights'] = attn_weights  # (N_fg,)

        # Decode final inst_feat to get refinement vector
        inst_refinement = self.decoder_inst(inst_feat)  # (N_inst, 8)
        self.forward_return_dict['pred'] = {'boxes_refinement': inst_refinement}

        if self.training:
            self.forward_return_dict['target'] = self.assign_target(batch_dict)
        else:
            batch_dict['final_box_dicts'] = self.generate_predicted_boxes(batch_dict['batch_size'], batch_dict)
            if self.cfg.get('GENERATE_INSTANCE_PAST_TRAJECTORY', True):
                batch_dict['local_traj'] = self.gen_instances_past_trajectory(batch_dict)
        return batch_dict

    def get_training_loss(self, tb_dict: dict = None):
        if tb_dict is None:
            tb_dict = dict()

        pred_box_refine = self.forward_return_dict['pred']['boxes_refinement']  # (N_inst, 8)
        target_box_refine = self.forward_return_dict['target']['boxes_refinement']  # (N_inst, 8)
        loss_box_refine = nn.functional.smooth_l1_loss(pred_box_refine, target_box_refine, reduction='mean')
        tb_dict['loss_box_refine'] = loss_box_refine.item()
        return loss_box_refine, tb_dict

    @torch.no_grad()
    def assign_target(self, batch_dict):
        input_dict = batch_dict['2nd_stage_input']
        pred_boxes = input_dict['pred_boxes']  # (N_inst, 7) - center (3), size (3), yaw

        gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, 11) -center (3), size (3), yaw, dummy_v (2), instance_index, class
        gt_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        gt_boxes = gt_boxes[input_dict['meta']['inst_bi']]  # (N_inst, 11)

        # map gt_boxes to pred_boxes
        pred_cos, pred_sin = torch.cos(pred_boxes[:, -1]), torch.sin(pred_boxes[:, -1])
        zeros, ones = pred_boxes.new_zeros(pred_boxes.shape[0]), pred_boxes.new_ones(pred_boxes.shape[0])
        pred_rot_inv = rearrange([
            pred_cos,    pred_sin,  zeros,
            -pred_sin,   pred_cos,  zeros,
            zeros,       zeros,     ones
        ], '(r c) N_inst -> N_inst r c', r=3, c=3)

        transl_gt_in_pred = torch.einsum('nij, nj -> ni', pred_rot_inv, gt_boxes[:, :3] - pred_boxes[:, :3])

        # residual coder
        diagonal = torch.sqrt(torch.square(pred_boxes[:, 3: 5]).sum(dim=1, keepdim=True))
        target_transl_xy = transl_gt_in_pred[:, :2] / torch.clamp_min(diagonal, min=1e-2)  # (N_inst, 2)
        target_transl_z = rearrange(transl_gt_in_pred[:, 2] / torch.clamp_min(pred_boxes[:, 5], min=1e-2),
                                    'N_inst -> N_inst 1')
        target_size = torch.log(gt_boxes[:, 3: 6] / torch.clamp_min(pred_boxes[:, 3: 6], min=1e-2))  # (N_inst, 3)
        target_ori = torch.stack([
            torch.cos(-pred_boxes[:, -1] + gt_boxes[:, 6]), torch.sin(-pred_boxes[:, -1] + gt_boxes[:, 6])
        ], dim=1)  # (N_inst, 2) - (cos, sin)

        target_dict = {
            'boxes_refinement': torch.cat([target_transl_xy, target_transl_z, target_size, target_ori], dim=1)
        }
        return target_dict

    @staticmethod
    @torch.no_grad()
    def decode_boxes_refinement(boxes: torch.Tensor, refinement: torch.Tensor):
        """
        Reconstruct rot_gt_in_pred & transl_gt_in_pred -> ^{pred} T_{pred'}
        """
        diagonal = torch.sqrt(torch.square(boxes[:, 3: 5]).sum(dim=1, keepdim=True))
        transl_xy = refinement[:, :2] * diagonal  # (N_inst, 2)
        transl_z = rearrange(refinement[:, 2] * boxes[:, 5], 'N_inst -> N_inst 1')
        transl = torch.cat([transl_xy, transl_z], dim=1)

        cos, sin = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
        zeros, ones = boxes.new_zeros(boxes.shape[0]), boxes.new_ones(boxes.shape[0])
        rot_box = rearrange([
            cos,     -sin,   zeros,
            sin,      cos,   zeros,
            zeros,  zeros,   ones
        ], '(r c) N_inst -> N_inst r c', r=3, c=3)

        new_xyz = torch.einsum('nij, nj -> ni', rot_box, transl) + boxes[:, :3]

        new_yaw = boxes[:, 6] + torch.atan2(refinement[:, -1], refinement[:, -2])
        new_yaw = torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw))

        new_sizes = torch.exp(refinement[:, 3: 6]) * boxes[:, 3: 6]

        return torch.cat([new_xyz, new_sizes, rearrange(new_yaw, 'N_inst -> N_inst 1')], dim=1)

    @torch.no_grad()
    def generate_predicted_boxes(self, batch_size: int, batch_dict) -> List[dict]:
        """
        Element i-th of the returned List represents predicted boxes for the sample i-th in the batch
        Args:
            batch_size:
            batch_dict
        """
        input_dict = batch_dict['2nd_stage_input']

        # decode predicted proposals
        pred_boxes = self.decode_boxes_refinement(input_dict['pred_boxes'],
                                                  self.forward_return_dict['pred']['boxes_refinement'])  # (N_inst, 7)
        if self.cfg.get('DEBUG_NOT_APPLYING_REFINEMENT', False):
            logger = logging.getLogger()
            logger.info('NOT USING REFINEMENT, showing pred_boxes made by 1st stage')
            pred_boxes = input_dict['pred_boxes']

        # compute boxes' score by weighted sum foreground score, weight come from attention matrix
        fg_prob = rearrange(input_dict['fg_prob'], 'N_fg 1 -> N_fg')
        print(f'fg_prob: {fg_prob.shape}')
        attn_weights = self.forward_return_dict['attn_weights']  # (N_fg)
        print(f'attn_weights: {attn_weights.shape}')
        pred_scores = torch_scatter.scatter_mean(fg_prob * attn_weights,
                                                 input_dict['meta']['inst_bi_inv_indices'])  # (N_inst,)

        # separate pred_boxes accodring to batch index
        inst_bi = input_dict['meta']['inst_bi']
        max_num_inst = input_dict['meta']['max_num_inst']
        inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

        out = []
        for b_idx in range(batch_size):
            cur_boxes = pred_boxes[inst_batch_idx == b_idx]
            cur_scores = pred_scores[inst_batch_idx == b_idx]  # (N_cur_inst,)
            cur_labels = cur_boxes.new_ones(cur_boxes.shape[0]).long()
            out.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels
            })
        return out

    @torch.no_grad()
    def gen_instances_past_trajectory(self, batch_dict):
        """
        Returns:
            (N_local, 4) - x, y, z, yaw
        """
        input_dict = batch_dict['2nd_stage_input']

        # instances' pose @ the current time step
        cur_boxes = self.decode_boxes_refinement(input_dict['pred_boxes'],
                                                 self.forward_return_dict['pred']['boxes_refinement'])  # (N_inst, 7)
        if self.cfg.get('DEBUG_NOT_APPLYING_REFINEMENT', False):
            logger = logging.getLogger()
            logger.info('NOT USING REFINEMENT, showing pred_boxes made by 1st stage')
            cur_boxes = input_dict['pred_boxes']

        # extract local_tf
        local_transl = input_dict['local_transl']  # (N_local, 3)
        local_rot = input_dict['local_rot']  # (N_local, 3, 3)
        local_rot_angle = torch.atan2(local_rot[:, 1, 0], local_rot[:, 0, 0])  # (N_local,) - assume only have rot around x

        # reconstruct local poses
        indices_inst2local = input_dict['meta']['local_bi_in_inst_bi']  # (N_local,)
        local_yaw = -local_rot_angle + cur_boxes[indices_inst2local, -1]  # (N_local,)

        offset = cur_boxes[indices_inst2local, :3] - local_transl  # (N_local, 3)
        cos, sin = torch.cos(-local_rot_angle), torch.sin(-local_rot_angle)  # (N_local,)
        local_xyz = torch.stack([
            cos * offset[:, 0] - sin * offset[:, 1],
            sin * offset[:, 0] + cos * offset[:, 1],
            offset[:, 2]
        ], dim=1)  # (N_local, 3)

        return torch.cat([local_xyz, local_yaw[:, None]], dim=1)  # (N_local, 4)



