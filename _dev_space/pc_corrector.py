import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_min, scatter_mean
from einops import rearrange
from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from _dev_space.tail_cutter_utils import bilinear_interpolate_torch, eval_binary_segmentation, quat2mat
from typing import List
from torchmetrics import Precision, Recall


class PointCloudCorrector(nn.Module):
    def __init__(self, model_cfg, num_bev_features, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_sweeps = model_cfg.NUM_SWEEPS
        self.voxel_size = torch.tensor(voxel_size).float().cuda()  # [vox_x, vox_y, vox_z]
        self.bev_image_stride = model_cfg.BEV_IMAGE_STRIDE
        self.point_cloud_range = torch.tensor(point_cloud_range).float().cuda()

        num_pts_raw_feat = 1 + model_cfg.NUM_RAW_POINT_FEATURES  # batch_idx, x, y, z, intensity, time
        idx_offset = model_cfg.POINT_FEAT_INDEX_OFFSET_FROM_RAW_FEAT
        self.map_point_feat2idx = {
            'sweep_idx': num_pts_raw_feat + idx_offset.SWEEP_INDEX,
            'inst_idx': num_pts_raw_feat + idx_offset.INSTANCE_INDEX,
            'aug_inst_idx': num_pts_raw_feat + idx_offset.AUGMENTED_INSTANCE_INDEX,
            # 'cls_idx': num_pts_raw_feat + idx_offset.CLASS_INDEX,  # NOT USED!
        }

        self.forward_return_dict = dict()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                num_bev_features, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        # -------
        # Points Head
        # -------
        num_points_features = self.model_cfg.SHARED_CONV_CHANNEL
        self.points_seg = self._make_mlp(num_points_features, 3, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 3 cls:= bg, static fg, dynamic fg

        self.points_reg = self._make_mlp(num_points_features, 3, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 3:= offset_x, offset_y, offset_z

        self.points_embedding = self._make_mlp(num_points_features, 2, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 2:= offset_toward_instance_center_x|_y

        # -------
        # Instances Head
        # -------
        self.local_shape_encoder = self._make_mlp(3, num_points_features,
                                                  model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))

        self.local_tf_decoder = self._make_mlp(2 * num_points_features + 6, 7,
                                               model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))
        # in := local_feat | global_feat | local_centroid | target_local_centroid
        # 7 := 3 (translation vector)  + 4 (quaternion)

        self.instance_motion_seg = self._make_mlp(num_points_features + 6, 1,
                                                  model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))
        # in := global_feat | init_local_centroid | target_local_centroid

        # -------
        # Loss
        # -------
        self.loss_points_seg = CELovaszLoss(num_classes=3)
        self.loss_instance_mos = CELovaszLoss(num_classes=2)

        # --
        self.eval_segmentation = model_cfg.get('EVAL_SEGMENTATION_WHILE_TRAINING', False)
        self.precision_points_cls = Precision(task='multiclass', average='macro', num_classes=3, threshold=0.5, top_k=1)
        self.recall_points_cls = Recall(task='multiclass', average='macro', num_classes=3, threshold=0.5, top_k=1)

    @staticmethod
    def _make_mlp(in_c: int, out_c: int, mid_c: List = None, use_drop_out=False):
        if mid_c is None:
            mid_c = []
        channels = [in_c] + mid_c + [out_c]
        layers = []
        for c_idx in range(1, len(channels)):
            c_in = channels[c_idx - 1]
            c_out = channels[c_idx]
            is_last = c_idx == len(channels) - 1
            layers.append(nn.Linear(c_in, c_out, bias=is_last))
            if not is_last:
                layers.append(nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(True))
                if c_idx == len(channels) - 2 and use_drop_out:
                    layers.append(nn.Dropout(p=0.5))

        return nn.Sequential(*layers)

    def build_meta(self, foreground: torch.Tensor, max_num_instances_in_batch: int, mask_fg: torch.Tensor):
        fg_batch_inst_sw = (foreground[:, 0].long() * max_num_instances_in_batch
                            + foreground[:, self.map_point_feat2idx['inst_idx']].long()) * self.num_sweeps \
                           + foreground[:, self.map_point_feat2idx['sweep_idx']].long()
        # group foreground to locals
        locals_bis, locals2fg = torch.unique(fg_batch_inst_sw, sorted=True, return_inverse=True)

        # groups local to instance
        locals_batch_inst = torch.div(locals_bis, self.num_sweeps, rounding_mode='floor')
        instance_bi, inst2locals = torch.unique(locals_batch_inst, sorted=True, return_inverse=True)

        # find target local for each instance
        locals_sweep = locals_bis - locals_batch_inst * self.num_sweeps  # (N_local,)
        instance_max_sweep, indices_locals_max_sweep = scatter_max(locals_sweep, inst2locals, dim=0)
        # indices_locals_max_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        # find init local for each instance
        instance_min_sweep, indices_locals_min_sweep = scatter_min(locals_sweep, inst2locals, dim=0)
        # indices_locals_min_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        meta = {'locals2fg': locals2fg,
                'inst2locals': inst2locals,
                'indices_locals_max_sweep': indices_locals_max_sweep,
                'indices_locals_min_sweep': indices_locals_min_sweep,
                'locals_bis': locals_bis,
                'instance_bi': instance_bi,
                'mask_fg': mask_fg,
                'num_fg': mask_fg.long().sum().item()}
        return meta

    def forward(self, batch_dict: dict):
        assert batch_dict['gt_boxes'].shape[1] == batch_dict['instances_tf'].shape[1], \
            f"{batch_dict['gt_boxes'].shape[1]} != {batch_dict['instances_tf'].shape[1]}"
        self.forward_return_dict = {}

        points = batch_dict['points']
        points_batch_idx = points[:, 0].long()
        num_points = points.shape[0]

        spatial_features_2d = batch_dict['spatial_features_2d']
        bev_img = self.shared_conv(spatial_features_2d)  # (B, num_pts_feat, H, W)

        points_bev_coord = (points[:, 1: 3] - self.point_cloud_range[:2]) / (self.voxel_size[:2] * self.bev_image_stride)
        points_feat = points.new_zeros(num_points, bev_img.shape[1])
        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            cur_points_feat = bilinear_interpolate_torch(_img, points_bev_coord[batch_mask, 0],
                                                         points_bev_coord[batch_mask, 1])
            points_feat[batch_mask] = cur_points_feat

        # -------
        # invoke Point Heads
        # -------
        points_cls_logit = self.points_seg(points_feat)  # (N, 3)
        points_offset = self.points_reg(points_feat)  # (N, 3)
        points_embedding = self.points_embedding(points_feat)  # (N, 2)

        prediction_dict = {
            'points_cls_logit': points_cls_logit,
            'points_offset': points_offset,
            'points_embedding': points_embedding
        }

        if self.training:
            # -------
            # instance stuff
            # -------
            mask_fg = points[:, self.map_point_feat2idx['inst_idx']] > -1  # all 10 classes
            fg = points[mask_fg]
            fg_feat = points_feat[mask_fg]  # (N_fg, C)

            meta = self.build_meta(fg, batch_dict['gt_boxes'].shape[1], mask_fg)

            # compute locals' shape encoding
            locals_centroid = scatter_mean(fg[:, 1: 4], meta['locals2fg'], dim=0)  # (N_local, 3)
            centered_fg = fg[:, 1: 4] - locals_centroid[meta['locals2fg']]  # (N_fg, 3)
            locals_shape_encoding = scatter_max(self.local_shape_encoder(centered_fg), meta['locals2fg'],
                                                dim=0)[0]  # (N_local, C)

            # compute locals feat
            locals_feat = scatter_max(fg_feat, meta['locals2fg'], dim=0)[0]  # (N_local, C)
            locals_feat = locals_feat + locals_shape_encoding  # (N_local, C)

            # compute globals stuff
            globals_feat = scatter_max(locals_feat, meta['inst2locals'], dim=0)[0]  # (N_global, C)
            globals_init_local_center = locals_centroid[meta['indices_locals_min_sweep']]  # (N_global, 3)
            globals_target_local_center = locals_centroid[meta['indices_locals_max_sweep']]  # (N_global, 3)

            # invoke instance head
            locals_feat = torch.cat((locals_feat,
                                     globals_feat[meta['inst2locals']],
                                     locals_centroid,
                                     globals_target_local_center[meta['inst2locals']]), dim=1)  # (N_local, 2*C + 6)
            locals_tf = self.local_tf_decoder(locals_feat)  # (N_local, 7)

            inst_feat = torch.cat((globals_feat, globals_init_local_center, globals_target_local_center), dim=1)
            inst_mos = self.instance_motion_seg(inst_feat)  # (N_global, 1)

            prediction_dict.update({
                'locals_tf': locals_tf,
                'inst_mos': inst_mos
            })

            # invoke assign target
            target_dict = self.assign_target(batch_dict, meta)

            # save prediction & target for computing loss
            meta['fg'] = fg
            self.forward_return_dict.update({
                'prediction': prediction_dict,
                'target': target_dict,
                'meta': meta
            })

        if not self.training or self.model_cfg.get('CORRECT_POINTS_WHILE_TRAINING', False):
            # apply points_offset on dynamic foreground points
            points_all_cls_prob = nn.functional.softmax(points_cls_logit, dim=1)  # (N, 3)
            points_cls_prob, points_cls_indices = torch.max(points_all_cls_prob, dim=1)  # (N,), (N,)
            mask_dyn_fg = torch.logical_and(points_cls_prob > self.model_cfg.get('THRESH_CLS_PROB', 0.5),
                                            points_cls_indices == 2)  # (N,)

            # mutate xyz-coord of points where mask_dyn_fg == True using predict offset
            points[mask_dyn_fg, 1: 4] += points_offset[mask_dyn_fg]

        return batch_dict

    def assign_target(self, batch_dict, meta):

        # -------------------
        # Instances target
        # -------------------
        all_locals_tf = rearrange(batch_dict['instances_tf'], 'B N_i N_sw C1 C2 -> (B N_i N_sw) C1 C2', C1=3, C2=4)
        locals_tf = all_locals_tf[meta['locals_bis']]  # (N_local, 3, 4)

        # instances' motion status
        # translation of the init_local_tf >= 0.5m
        all_inst_mos = torch.linalg.norm(batch_dict['instances_tf'][:, :, 0, :, -1], dim=-1) > 0.5  # (B, N_inst_max)
        mask_inst_mos = rearrange(all_inst_mos, 'B N_inst_max -> (B N_inst_max)')[meta['instance_bi']]  # (N_i,)

        # -------------------
        # Point-wise target
        # -------------------
        points = batch_dict['points']  # (N, C)
        num_points = points.shape[0]

        # --
        # points' class
        # --
        points_cls = points.new_zeros(num_points, 3)  # bg | static fg | dynamic fg
        mask_fg = meta['mask_fg']
        # background
        points_cls[torch.logical_not(mask_fg), 0] = 1

        # broadcast mask_inst_mos -> local -> foreground
        mask_locals_mos = mask_inst_mos[meta['inst2locals']]
        mask_fg_mos = mask_locals_mos[meta['locals2fg']]
        if meta['num_fg'] > 0:
            fg_cls = mask_fg.new_zeros(meta['num_fg'], 2).float()  # static fg | dynamic fg
            fg_cls.scatter_(dim=1, index=mask_fg_mos.view(meta['num_fg'], 1).long(), value=1.0)
            points_cls[mask_fg, 1:] = fg_cls

        # --
        # points' embedding
        # --
        gt_boxes_xy = rearrange(batch_dict['gt_boxes'][:, :, :2], 'B N_inst_max C -> (B N_inst_max) C')[meta['instance_bi']]
        locals_box_xy = gt_boxes_xy[meta['inst2locals']]  # # (N_local, 2)
        fg_box_xy = locals_box_xy[meta['locals2fg']]  # (N_fg, 2)
        fg_embedding = fg_box_xy - points[mask_fg, 1: 3]  # (N_fg, 2)

        # --
        # points' offset
        # --
        if torch.any(points_cls[:, 2] > 0):  # if there are any dynamic fg
            # broadcast locals_tf -> fg
            fg_tf = locals_tf[meta['locals2fg']]  # (N_fg, 3, 4)
            # correct fg
            fg = points[mask_fg]  # (N_fg, C)
            corrected_fg = torch.matmul(fg_tf[:, :3, :3], fg[:, 1: 4].unsqueeze(-1)).squeeze(-1) + fg_tf[:, :, -1]
            # target fg offset
            fg_offset = corrected_fg - fg[:, 1: 4]  # (N_fg, 3)
        else:
            fg_offset = None

        target_dict = {'locals_tf': locals_tf,
                       'inst_mos': mask_inst_mos.long(),
                       'points_cls': points_cls.long(),
                       'fg_embedding': fg_embedding,
                       'fg_offset': fg_offset}

        # --
        # remove gt_boxes that are outside of pc range
        # --
        valid_gt_boxes = list()
        gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, C)
        max_num_valid_boxes = 0
        for bidx in range(batch_dict['batch_size']):
            mask_in_range = torch.logical_and(gt_boxes[bidx, :, :3] >= self.point_cloud_range[:3],
                                              gt_boxes[bidx, :, :3] < self.point_cloud_range[3:]).all(dim=1)
            valid_gt_boxes.append(gt_boxes[bidx, mask_in_range])
            max_num_valid_boxes = max(max_num_valid_boxes, valid_gt_boxes[-1].shape[0])

        batch_valid_gt_boxes = gt_boxes.new_zeros(batch_dict['batch_size'], max_num_valid_boxes, gt_boxes.shape[2])
        for bidx, valid_boxes in enumerate(valid_gt_boxes):
            batch_valid_gt_boxes[bidx, :valid_boxes.shape[0]] = valid_boxes

        batch_dict.pop('gt_boxes')
        batch_dict['gt_boxes'] = batch_valid_gt_boxes

        return target_dict

    def get_training_loss(self, tb_dict=None):
        if tb_dict is None:
            tb_dict = dict()

        pred = self.forward_return_dict['prediction']
        target = self.forward_return_dict['target']
        meta = self.forward_return_dict['meta']

        mask_fg = meta['mask_fg']
        device = mask_fg.device

        # -----------
        # point-wise loss
        # -----------

        # ---
        # points cls
        # ---
        # convert points_cls from one-hot encoding to integer
        target_points_cls = torch.argmax(target['points_cls'], dim=1)  # (N,)

        l_points_cls = self.loss_points_seg(pred['points_cls_logit'], target_points_cls)
        tb_dict['l_points_cls'] = l_points_cls.item()

        # ---
        # [foreground] points embedding
        # ---
        l_points_embed = nn.functional.smooth_l1_loss(pred['points_embedding'][mask_fg], target['fg_embedding'],
                                                      reduction='none').sum(dim=1).mean()
        tb_dict['l_points_embed'] = l_points_embed.item()

        # ---
        # [dynamic foreground] points offset
        # ---
        if target['fg_offset'] is not None:
            # compute offset loss for every fg, then zero-out loss of static fg
            pred_fg_offset = pred['points_offset'][mask_fg]  # (N_fg, 3)
            l_dyn_fg_offset = nn.functional.smooth_l1_loss(pred_fg_offset, target['fg_offset'], reduction='none')  # (N_fg, 3)

            mask_dyn_fg = target['points_cls'][mask_fg, 2] > 0  # (N_fg,)
            assert torch.any(mask_dyn_fg)
            l_dyn_fg_offset = torch.sum(l_dyn_fg_offset * mask_dyn_fg.float().unsqueeze(1), dim=1).mean()

        else:
            l_dyn_fg_offset = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['l_dyn_fg_offset'] = l_dyn_fg_offset.item()

        # -----------
        # instance-wise loss
        # -----------
        l_inst_mos = self.loss_instance_mos(pred['inst_mos'], target['inst_mos'])
        tb_dict['l_inst_mos'] = l_inst_mos.item()

        mask_inst_mos = target['inst_mos'] > 0
        mask_locals_mos = mask_inst_mos[meta['inst2locals']]  # (N_local,)
        if torch.any(mask_locals_mos):
            target_dyn_locals_tf = target['locals_tf'][mask_locals_mos]  # (N_dyn_locals, 3, 4)

            # translation
            pred_dyn_locals_trans = pred['locals_tf'][mask_locals_mos, :3]  # (N_dyn_locals, 3)
            l_locals_transl = nn.functional.smooth_l1_loss(pred_dyn_locals_trans, target_dyn_locals_tf[:, :, -1],
                                                           reduction='none').sum(dim=1).mean()

            # rotation
            pred_locals_rot = quat2mat(pred['locals_tf'][:, 3:])  # (N_local, 3, 3)
            pred_dyn_locals_rot = pred_locals_rot[mask_locals_mos]  # (N_dyn_locals, 3, 3)
            l_locals_rot = torch.linalg.norm(pred_dyn_locals_rot - target_dyn_locals_tf[:, :, :3],
                                             dim=(1, 2), ord='fro').mean()

            # reconstruction
            mask_fg_mos = mask_locals_mos[meta['locals2fg']]  # (N_fg,)
            dyn_fg_xyz = meta['fg'][mask_fg_mos, 1: 4]  # (N_dyn_fg, 3)

            fg_tf = target['locals_tf'][meta['locals2fg']]  # (N_fg, 3, 4)
            dyn_fg_tf = fg_tf[mask_fg_mos]  # (N_dyn_fg, 3, 4)

            gt_corrected_dyn_fg = (torch.matmul(dyn_fg_tf[:, :3, :3], dyn_fg_xyz.unsqueeze(-1)).squeeze(-1)
                                   + dyn_fg_tf[:, :3, -1])  # (N_dyn_fg, 3)
            # --
            pred_local_tf = torch.cat((pred_locals_rot, pred['locals_tf'][:, :3].unsqueeze(-1)), dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = pred_local_tf[meta['locals2fg']]
            pred_dyn_fg_tf = pred_fg_tf[mask_fg_mos]

            corrected_dyn_fg = (torch.matmul(pred_dyn_fg_tf[:, :3, :3], dyn_fg_xyz.unsqueeze(-1)).squeeze(-1)
                                + pred_dyn_fg_tf[:, :3, -1])  # (N_dyn_fg, 3)

            l_recon = nn.functional.smooth_l1_loss(corrected_dyn_fg, gt_corrected_dyn_fg,
                                                   reduction='none').sum(dim=1).mean()

            # -----
            # consistency loss
            # -----
            assert target['fg_offset'] is not None
            pred_dyn_fg_offset = pred_fg_offset[mask_fg_mos]  # (N_dyn_fg, 3)
            offseted_dyn_fg = pred_dyn_fg_offset + dyn_fg_xyz
            l_consist = nn.functional.smooth_l1_loss(offseted_dyn_fg, corrected_dyn_fg, reduction='none').sum(dim=1).mean()

        else:
            l_locals_transl = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            l_locals_rot = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            l_recon = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            l_consist = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)

        tb_dict.update({
            'l_locals_transl': l_locals_transl.item(),
            'l_locals_rot': l_locals_rot.item(),
            'l_recon': l_recon.item(),
            'l_consist': l_consist.item()
        })

        loss = (l_points_cls + l_points_embed + l_dyn_fg_offset
                + l_inst_mos + l_locals_transl + l_locals_rot + l_recon
                + l_consist)

        with torch.no_grad():
            if self.eval_segmentation:
                points_cls_precision = self.precision_points_cls(pred['points_cls_logit'], target_points_cls)
                points_cls_recall = self.recall_points_cls(pred['points_cls_logit'], target_points_cls)
                tb_dict.update({
                    'points_cls_P': points_cls_precision.item(),
                    'points_cls_R': points_cls_recall.item()
                })

        return loss, tb_dict
