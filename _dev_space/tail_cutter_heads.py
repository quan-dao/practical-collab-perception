import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch_scatter
from typing import List
import logging
from _dev_space.non_batching_attention import MultiHeadAttention


class AlignerHead(nn.Module):
    def __init__(self, model_cfg, num_bev_features: int, num_instance_features: int):
        super().__init__()
        self.cfg = model_cfg
        obj_cfg = model_cfg.OBJECT_HEAD
        traj_cfg = model_cfg.TRAJECTORY_HEAD

        self.obj_encode_fg = self._make_mlp(num_bev_features + 27, obj_cfg.NUM_FEATURES,
                                            obj_cfg.get('ENC_FG_HIDDEN', []))
        # + 27 due to the use of position encoding

        self.obj_encode_inst = self._make_mlp(num_instance_features, obj_cfg.NUM_FEATURES,
                                              obj_cfg.get('ENC_FG_HIDDEN', []))

        self.obj_attention_stack = nn.ModuleList([
            MultiHeadAttention(
                obj_cfg.NUM_HEADS, obj_cfg.NUM_FEATURES, return_attn_weight=a_idx == obj_cfg.NUM_LAYERS - 1
            ) for a_idx in range(obj_cfg.NUM_LAYERS)
        ])

        self.obj_decoder = self._make_mlp(obj_cfg.NUM_FEATURES, 8, obj_cfg.get('DEC_OBJECT_HIDDEN', []))
        # following residual coder: delta_x, delta_y, delta_z, delta_dx, delta_dy, delta_dz, diff_sin, diff_cos

        self.traj_attention_stack = nn.ModuleList([
            nn.MultiheadAttention(traj_cfg.NUM_FEATURES, traj_cfg.NUM_HEADS, dropout=0.1)
            for _ in range(traj_cfg.NUM_LAYERS)
        ])
        self.traj_mlp = self._make_mlp(traj_cfg.NUM_FEATURES, traj_cfg.NUM_FEATURES, traj_cfg.get('TRAJ_HIDDEN', []))

        self.forward_return_dict = dict()

    @staticmethod
    def _make_mlp(channels_in: int, channels_out: int, channels_hidden: List[int] = None):
        """
        Args:
            channels_in: number of channels of input
            channels: [Hidden_0, ..., Hidden_N, Out]
        """
        if channels_hidden is None:
            channels_hidden = []
        channels = [channels_in] + channels_hidden + [channels_out]
        layers = []
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
        fg = input_dict['foreground']['fg']  # (N_fg, 7[+2]) - batch_idx, xyz, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]
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

    def forward_obj_head_(self, input_dict):
        """
        Mutate self.forward_return_dict
        """
        fg_pos_embedding = self.compute_position_embedding(input_dict)  # (N_fg, 27)
        fg_feat = torch.cat([input_dict['foreground']['fg_feat'], fg_pos_embedding], dim=1)  # (N_fg, C_bev + 27)
        fg_feat = self.obj_encode_fg(fg_feat)  # (N_fg, C_attn)

        inst_feat = self.obj_encode_inst(input_dict['global']['shape_encoding'])  # (N_inst, C_attn)

        indices_inst2fg = input_dict['meta']['inst_bi_inv_indices']  # (N_fg)

        for i, attn_layer in enumerate(self.obj_attention_stack):
            if i < len(self.obj_attention_stack) - 1:
                inst_feat = attn_layer(inst_feat, fg_feat, fg_feat, indices_inst2fg)  # (N_inst, C_attn)
            else:
                inst_feat, attn_weights = attn_layer(inst_feat, fg_feat, fg_feat, indices_inst2fg)
                # inst_feat: (N_inst, C_attn)
                # attn_weights: (N_fg,)
                self.forward_return_dict['obj'] = {'attn_weights': attn_weights}  # (N_fg,)

        inst_refinement = self.obj_decoder(inst_feat)  # (N_inst, 8)
        self.forward_return_dict['obj'].update({'pred_boxes_refinement': inst_refinement})

    def forward(self, batch_dict):
        input_dict = batch_dict['input_2nd_stage']
        self.forward_obj_head_(input_dict)

        if self.training:
            self.forward_return_dict['obj'].update({'target_boxes_refinement': self.obj_assign_target(batch_dict)})
        else:
            batch_dict['final_box_dicts'] = self.generate_predicted_boxes(batch_dict['batch_size'], input_dict)

            if self.cfg.get('GENERATE_INSTANCE_PAST_TRAJECTORY', True):
                batch_dict['local_traj'] = self.gen_instances_past_trajectory(input_dict)
        return batch_dict

    def get_training_loss(self, tb_dict: dict = None):
        if tb_dict is None:
            tb_dict = dict()

        loss_box_refine = nn.functional.smooth_l1_loss(
            self.forward_return_dict['obj']['pred_boxes_refinement'],
            self.forward_return_dict['obj']['target_boxes_refinement'],
            reduction='mean'
        )
        tb_dict['loss_box_refine'] = loss_box_refine.item()
        return loss_box_refine, tb_dict

    @torch.no_grad()
    def obj_assign_target(self, batch_dict):
        input_dict = batch_dict['input_2nd_stage']
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

        target_boxes_refinement = torch.cat([target_transl_xy, target_transl_z, target_size, target_ori], dim=1)
        return target_boxes_refinement

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
    def generate_predicted_boxes(self, batch_size: int, input_dict) -> List[dict]:
        """
        Element i-th of the returned List represents predicted boxes for the sample i-th in the batch
        Args:
            batch_size:
            input_dict
        """
        # decode predicted proposals
        pred_boxes = self.decode_boxes_refinement(input_dict['pred_boxes'],
                                                  self.forward_return_dict['obj']['pred_boxes_refinement'])  # (N_inst, 7)
        if self.cfg.get('DEBUG_NOT_APPLYING_REFINEMENT', False):
            # logger = logging.getLogger()
            # logger.info('NOT USING REFINEMENT, showing pred_boxes made by 1st stage')
            pred_boxes = input_dict['pred_boxes']

        # compute boxes' score by weighted sum foreground score, weight come from attention matrix
        # attn_weights =self.forward_return_dict['obj']['attn_weights']  # (N_fg)
        pred_scores = torch_scatter.scatter_mean(input_dict['foreground']['fg_prob'],
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
    def gen_instances_past_trajectory(self, input_dict: dict, instance_current_boxes: torch.Tensor):
        """
        Args:
            input_dict:
            instance_current_boxes: (N_inst, 7) - x, y, z, dx, dy, dz, yaw
        Returns:
            (N_local, 4) - x, y, z, yaw
        """
        # instances' pose @ the current time step


        # extract local_tf
        local_transl = input_dict['local']['local_transl']  # (N_local, 3)
        local_rot = input_dict['local']['local_rot']  # (N_local, 3, 3)
        local_rot_angle = torch.atan2(local_rot[:, 1, 0], local_rot[:, 0, 0])  # (N_local,) - assume only have rot around x

        # reconstruct local poses
        indices_inst2local = input_dict['meta']['local_bi_in_inst_bi']  # (N_local,)
        local_yaw = -local_rot_angle + instance_current_boxes[indices_inst2local, -1]  # (N_local,)

        offset = instance_current_boxes[indices_inst2local, :3] - local_transl  # (N_local, 3)
        cos, sin = torch.cos(-local_rot_angle), torch.sin(-local_rot_angle)  # (N_local,)
        local_xyz = torch.stack([
            cos * offset[:, 0] - sin * offset[:, 1],
            sin * offset[:, 0] + cos * offset[:, 1],
            offset[:, 2]
        ], dim=1)  # (N_local, 3)

        return torch.cat([local_xyz, local_yaw[:, None]], dim=1)  # (N_local, 4)

    def forward_traj_head_(self, input_dict):
        if self.training:
            inst_cur_boxes = input_dict['pred_boxes']  # (N_inst, 7) - x, y, z, dx, dy, dz, yaw
        else:
            inst_cur_boxes = self.decode_boxes_refinement(
                input_dict['pred_boxes'], self.forward_return_dict['pred']['boxes_refinement']
            )  # (N_inst, 7) - x, y, z, dx, dy, dz, yaw

        local_pose = self.gen_instances_past_trajectory(input_dict, inst_cur_boxes)  # (N_local, 4) - x,y,z,yaw in global
        local_pose = local_pose[:, [0, 1, 3]]  # (N_local, 3) - x, y, yaw (exclude z) in global

        # TODO: map local_pose to instance frame
        indices_inst2local = input_dict['meta']['local_bi_in_inst_bi']  # (N_local,)

        # translate to origin of instance current frame
        local_pose[:, :2] = local_pose[:, :2] - inst_cur_boxes[indices_inst2local, :2]
        # rotate to instance current frame
        cos, sin = torch.cos(inst_cur_boxes[:, 6]), torch.sin(inst_cur_boxes[:, 6])  # (N_inst)
        cos, sin = cos[indices_inst2local], sin[indices_inst2local]  # (N_local,)
        l_x = cos * local_pose[:, 0] + sin * local_pose[:, 1]  # (N_local,)
        l_y = -sin * local_pose[:, 0] + sin * local_pose[:, 1]  # (N_local,)
        local_pose[:, 2] = local_pose[:, 2] - inst_cur_boxes[:, 2]  # (N_local,)
        # overwrite x,y-coord of local pose & put yaw of local pose in range [-pi, pi)
        local_pose[:, :2] = torch.stack([l_x, l_y], dim=1)
        local_pose[:, 2] = torch.atan2(torch.sin(local_pose[:, 2]), torch.cos(local_pose[:, 2]))

        # TODO: compute locals' velocity
        local_sweep_idx = input_dict['meta']['local_bisw'] % self.cfg('NUM_SWEEPS')  # (N_local,)
        inst_max_sweep_idx = input_dict['meta']['inst_max_sweep_idx']  # (N_inst)
        local_time_elapsed = (inst_max_sweep_idx[indices_inst2local] - local_sweep_idx) * 0.05  # (N_local,)
        # * 0.1 because sweeps are produced at 20Hz
        assert torch.all(local_time_elapsed > -1e-5)
        local_velo = torch.zeros_like(local_pose)
        mask_valid_time = local_time_elapsed > 1e-5  # (N_local,)
        local_velo[mask_valid_time] = local_pose[mask_valid_time] / local_time_elapsed[mask_valid_time]

        # compute instance velocity to use as velocity for locals that have invalid time
        unq, inv, counts = torch.unique(indices_inst2local, return_inverse=True, return_counts=True)
        # unq: (n_unq) | inv: (N_local) | counts: (n_unq)
        inst_velo = torch_scatter.scatter_sum(local_velo, inv, dim=0) / torch.clamp_min(counts.float() - 1, min=1.0)
        # - 1 because velo of the most recent locals == 0
        local_velo = local_velo + inst_velo[inv] * torch.logical_not(mask_valid_time).float()

        # TODO: construct local feats for attention
        local_feat = torch.cat(
            [input_dict['local']['shape_encoding'], local_pose, local_velo], dim=1)  # (N_local, C_inst + 6)

        batch_local_feat = local_feat.new_zeros(unq.shape[0], counts.max(), local_feat.shape[-1])
        key_padding_mask = local_feat.new_zeros(unq.shape[0], counts.max())
        for idx in range(unq.shape[0]):
            batch_local_feat[unq[idx], :counts[idx]] = local_feat[indices_inst2local == unq[idx]]
            key_padding_mask[unq[idx], counts[idx]:] = 1  # to be ignored by attention

        batch_local_feat = rearrange(batch_local_feat, 'N_inst N_local_max C -> N_local_max N_inst C').contiguous()
        # to be compatible with setting batch_first=False on torch 1.8.2

        # TODO: global feat can be retrieved from instance index stored in "unq" because order in "batch_local_feat"
        # TODO: is the same as order in "unq"

        # self-attention to refine local feat
        for attn in self.traj_attention_stack:
            batch_local_feat, _ = attn(batch_local_feat, batch_local_feat, batch_local_feat,
                                       key_padding_mask=key_padding_mask)
        batch_local_feat = rearrange(batch_local_feat, 'N_local_max N_inst C -> N_inst N_local_max C').contiguous()

        # PointNet to summarize local feat to new global feat
        batch_local_feat = self.traj_mlp(batch_local_feat)  # (N_inst, N_local_max, C)
        batch_local_feat = batch_local_feat.masked_fill(key_padding_mask[..., None], -np.inf)
        batch_inst_feat = batch_local_feat.max(dim=1)[0]  # (N_inst, C)

        # TODO: [question] does batch_inst_feat (same order as unq) has 1-to-1 corr with inst_cur_boxes

