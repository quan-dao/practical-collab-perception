import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from easydict import EasyDict as edict
from typing import List, Dict, Tuple
from functools import partial
from einops import rearrange

from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from workspace.sc_conv import conv_bn_relu
from workspace.hunter_toolbox import interpolate_points_feat_from_bev_img, nn_make_mlp, remove_gt_boxes_outside_range, bev_scatter, quat2mat, \
    hard_mining_regression_loss


class HunterObjectHead(nn.Module):
    def __init__(self, num_point_features: int, mlp_hidden_channels: List[int] = None, use_drop_out: bool = False):
        super().__init__()
        _make_mlp = partial(nn_make_mlp, hidden_channels=mlp_hidden_channels, use_drop_out=use_drop_out)
        self.num_local_feat = num_point_features  # hard coded because locals_feat = locals_feat + locals_shape_encoding

        self.points_shape_encoder = _make_mlp(3, 
                                              num_point_features, 
                                              is_head=False)
        
        self.local_feat_encoder = _make_mlp(2 * self.num_local_feat + 3 + 3, 
                                            self.num_local_feat, 
                                            is_head=False)
        
        self.local_tf_decoder = _make_mlp(self.num_local_feat, 7, hidden_channels=[])
        # in := local_feat | global_feat | local_centroid | target_local_centroid
        # out == 7 := 3 (translation vector)  + 4 (quaternion)
    
    def forward(self, fg_xyz: torch.Tensor, fg_feat: torch.Tensor, meta: Dict[str, torch.Tensor]):
        """
        Args:
            fg_xyz: (N, 3)
            fg_feats: (N, C)
            meta:
        """
        assert fg_xyz.shape[1] == 3, f"expect xyz, get {fg_xyz.shape}"

        # compute locals' shape encoding
        locals_centroid = torch_scatter.scatter_mean(fg_xyz, meta['locals2fg'], dim=0)  # (N_local, 3)
        centered_fg = fg_xyz - locals_centroid[meta['locals2fg']]  # (N_fg, 3)
        locals_shape_encoding = torch_scatter.scatter_max(self.points_shape_encoder(centered_fg), 
                                                          meta['locals2fg'], dim=0)[0]  # (N_local, C_pts)
        
        # compute raw locals feat
        locals_feat = torch_scatter.scatter_max(fg_feat, meta['locals2fg'], dim=0)[0]  # (N_local, C_pts)
        locals_feat = locals_feat + locals_shape_encoding  # (N_local, C_local)

        # compute globals stuff
        globals_feat = torch_scatter.scatter_max(locals_feat, meta['inst2locals'], dim=0)[0]  # (N_global, C_local)
        globals_target_local_center = locals_centroid[meta['indices_locals_max_sweep']]  # (N_global, 3)

        # compute locals_feat
        locals_feat = torch.cat((locals_feat, 
                                 globals_feat[meta['inst2locals']], 
                                 locals_centroid, 
                                 globals_target_local_center[meta['inst2locals']]), 
                                 dim=1)  # (N_local, 2*C_local + 6)
        locals_feat = self.local_feat_encoder(locals_feat)  # (N_local, C_local)

        # predict locals_tf
        locals_tf = self.local_tf_decoder(locals_feat)  # (N_local, 7)

        return locals_tf, locals_feat


class HunterPointHead(nn.Module):
    def __init__(self, num_point_features: int, mlp_hidden_channels: List[int] = None, use_drop_out: bool = False):
        """
        points_feat -> `local_feat_predictor` -> pts_local_feat -> predict point task & distill
        """
        super().__init__()
        _make_mlp = partial(nn_make_mlp, use_drop_out=use_drop_out)

        self.local_feat_predictor = _make_mlp(num_point_features, num_point_features, 
                                              hidden_channels=mlp_hidden_channels, 
                                              is_head=False)

        self.seg = _make_mlp(num_point_features, 3)
        self.reg_flow3d = _make_mlp(num_point_features, 3)
        self.instance_embedding = _make_mlp(num_point_features, 2)
    
    def forward(self, points_feat: torch.Tensor) -> Tuple[torch.Tensor]:
        pts_local_feat = self.local_feat_predictor(points_feat)  # (N, C_local)
        
        pts_final_feat = points_feat + pts_local_feat

        points_cls_logit = self.seg(pts_final_feat)  # (N, 3)
        points_flow3d = self.reg_flow3d(pts_final_feat)  # (N, 3)
        points_inst_embed = self.instance_embedding(pts_final_feat)  # (N, 2)

        return pts_local_feat, points_cls_logit, points_flow3d, points_inst_embed
    
    def get_loss_distill(self, pts_local_feat: torch.Tensor, locals_feat: torch.Tensor, meta: Dict[str, torch.Tensor]):
        if not torch.any(meta['mask_fg']):
            # no foreground exists
            return torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=pts_local_feat.device)
        
        fg_local_feat = pts_local_feat[meta['mask_fg']]  # (N_fg, C_locals)
        label_fg_local_feat = locals_feat[meta['locals2fg']]  # (N_fg, C_locals)
        loss = F.smooth_l1_loss(fg_local_feat, label_fg_local_feat, reduction='none').sum(dim=1).mean() * 0.1
        return loss
        

class HunterJr(nn.Module):
    def __init__(self, model_cfg: edict, num_bev_features: int, voxel_size: List, point_cloud_range: List):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_sweeps = model_cfg.get('NUM_SWEEPS')
        self.bev_image_stride = model_cfg.get('BEV_IMAGE_STRIDE')
        self.voxel_size = torch.tensor(voxel_size).float().cuda()  # [vox_x, vox_y, vox_z]
        self.point_cloud_range = torch.tensor(point_cloud_range).float().cuda()

        self._meta_pts_feat_loc_sweep_idx = model_cfg.get('META_POINTS_FEAT_LOCATION_SWEEP_IDX', -2)
        self._meta_pts_feat_loc_inst_idx = model_cfg.get('META_POINTS_FEAT_LOCATION_INSTANCE_IDX', -1)

        self.num_points_feat = num_bev_features

        # Stem input
        # -------------
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv_input = conv_bn_relu(num_bev_features, num_bev_features, padding=1, norm_layer=norm_layer)
        
        # Point Head to predict point-wise cls, embed, flow3d
        # -----------------------------------------------------
        self.point_head = HunterPointHead(num_bev_features, model_cfg.get('POINT_HEAD_HIDDEN_CHANNELS'), use_drop_out=False)

        # Object Head to predict locals_feat, locals_tf
        # -----------------------------------------------------
        if self.training:
            self.object_head = HunterObjectHead(num_bev_features, model_cfg.get('OBJ_HEAD_HIDDEN_CHANNELS'), use_drop_out=False)
        else:
            self.object_head = None

        # Weightor to compute weight for fusing corrected bev img & original bev img
        # -----------------------------------------------------
        self.thresh_point_cls_prob = model_cfg.get('THRESHOLD_POINT_CLS_PROB', 0.3)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv_weightor = nn.Sequential(
            conv_bn_relu(2 * num_bev_features, 2 * num_bev_features, padding=1, norm_layer=norm_layer),
            nn.Conv2d(2 * num_bev_features, 2, kernel_size=3, padding=1)  # 2 set of weights, 1 for each BEV image
        )

        self.forward_return_dict = dict()

        # Loss fnc
        # ---------
        self.loss_points_seg = CELovaszLoss(num_classes=3)

    def _build_meta(self, foreground: torch.Tensor, max_num_instances_in_this_batch: int) -> Dict[str, torch.Tensor]:
        merged_batch_inst_sw = (foreground[:, 0].long() * max_num_instances_in_this_batch + 
                                foreground[:, self._meta_pts_feat_loc_inst_idx].long()) * self.num_sweeps + \
                                foreground[:, self._meta_pts_feat_loc_sweep_idx].long()
        
        # group foreground to locals
        locals_bis, locals2fg = torch.unique(merged_batch_inst_sw, sorted=True, return_inverse=True)

        # groups local to instance
        locals_batch_inst = torch.div(locals_bis, self.num_sweeps, rounding_mode='floor')
        instance_bi, inst2locals = torch.unique(locals_batch_inst, sorted=True, return_inverse=True)

        # find target local for each instance
        locals_sweep = locals_bis - locals_batch_inst * self.num_sweeps  # (N_local,)
        instance_max_sweep, indices_locals_max_sweep = torch_scatter.scatter_max(locals_sweep, inst2locals, dim=0)
        # indices_locals_max_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        # find init local for each instance
        instance_min_sweep, indices_locals_min_sweep = torch_scatter.scatter_min(locals_sweep, inst2locals, dim=0)
        # indices_locals_min_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        meta = {'locals2fg': locals2fg,
                'inst2locals': inst2locals,
                'indices_locals_max_sweep': indices_locals_max_sweep,
                'indices_locals_min_sweep': indices_locals_min_sweep,
                'locals_bis': locals_bis,
                'instance_bi': instance_bi}
        return meta
        
    def assign_target(self, batch_dict: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Local Target
        # -----------------------
        all_locals_tf = rearrange(batch_dict['instances_tf'], 'B N_i N_sw C1 C2 -> (B N_i N_sw) C1 C2', C1=3, C2=4)
        locals_tf = all_locals_tf[meta['locals_bis']]  # (N_local, 3, 4)

        # Point Target
        # -----------------------
        points = batch_dict['points']  # (N, C)
        
        # points_cls
        # ---
        points_cls = points.new_zeros(points.shape[0], 3)  # bg | static fg | dynamic fg
        mask_fg = meta['mask_fg']
        # background
        points_cls[torch.logical_not(mask_fg), 0] = 1.

        # to find dynamic foreground: 
        # 1. find dynmic inst
        # 2. broadcast mask_inst_mos -> local -> foreground
        all_inst_mos = torch.linalg.norm(batch_dict['instances_tf'][:, :, 0, :, -1], dim=-1) > 0.5  # (B, N_inst_max)
        mask_inst_mos = rearrange(all_inst_mos, 'B N_inst_max -> (B N_inst_max)')[meta['instance_bi']]  # (N_inst,)
        mask_locals_mos = mask_inst_mos[meta['inst2locals']]
        mask_fg_mos = mask_locals_mos[meta['locals2fg']]
        if torch.any(mask_fg):
            num_fg = int(mask_fg.sum().item())
            fg_cls = mask_fg.new_zeros(num_fg, 2).float()  # static fg | dynamic fg
            fg_cls.scatter_(dim=1, index=mask_fg_mos.view(num_fg, 1).long(), value=1.0)
            points_cls[mask_fg, 1:] = fg_cls
        
        # fg 's instance embedding
        # ---
        gt_boxes_xy = rearrange(batch_dict['gt_boxes'][:, :, :2], 'B N_inst_max C -> (B N_inst_max) C')[meta['instance_bi']]
        locals_box_xy = gt_boxes_xy[meta['inst2locals']]  # # (N_local, 2)
        fg_box_xy = locals_box_xy[meta['locals2fg']]  # (N_fg, 2)
        fg_embedding = fg_box_xy - points[mask_fg, 1: 3]  # (N_fg, 2)

        # fg 's flow3d
        # ---
        fg = points[mask_fg]  # (N_fg, C)
        if fg.shape[0] > 0:  # if there are any dynamic fg
            # broadcast locals_tf -> fg
            fg_tf = locals_tf[meta['locals2fg']]  # (N_fg, 3, 4)
            # correct fg
            corrected_fg = torch.matmul(fg_tf[:, :3, :3], fg[:, 1: 4].unsqueeze(-1)).squeeze(-1) + fg_tf[:, :, -1]
            # target fg offset
            fg_offset = corrected_fg - fg[:, 1: 4]  # (N_fg, 3)
        else:
            # either no fg at all
            # or no dynamic fg
            fg_offset = None

        target = {
            'locals_tf': locals_tf,  # (N_locals, 7)
            'points_cls': points_cls,  # (N_pts, 3)
            'fg_embedding': fg_embedding,  # (N_fg, 2)
            'fg_offset': fg_offset,  # (N_fg, 3)
            'meta': {'mask_locals_mos': mask_locals_mos}  # for computing loss
        }
        return target
    
    def correct_bev_image(self, 
                          points: torch.Tensor, 
                          points_feat: torch.Tensor,
                          points_bev_coord: torch.Tensor, 
                          points_cls_logit: torch.Tensor, 
                          points_flow3d: torch.Tensor, 
                          bev_img: torch.Tensor) -> torch.Tensor:
        
        points_all_cls_prob = nn.functional.sigmoid(points_cls_logit)  # (N, 3)
        points_cls_prob, points_cls_indices = torch.max(points_all_cls_prob, dim=1)  # (N,), (N,)
        mask_dyn_fg = torch.logical_and(points_cls_prob > self.thresh_point_cls_prob, 
                                        points_cls_indices == 2)  # (N,)
        
        # mutate xyz-coord of points where mask_dyn_fg == True using predict offset
        points[mask_dyn_fg, 1: 4] = points[mask_dyn_fg, 1: 4] + points_flow3d[mask_dyn_fg]
        
        if torch.any(mask_dyn_fg):
            correct_feat, correct_bev_coord = interpolate_points_feat_from_bev_img(bev_img, 
                                                                                   points, 
                                                                                   self.point_cloud_range, 
                                                                                   self.voxel_size[:2] * self.bev_image_stride,
                                                                                   return_bev_coord=True)
            # update feat of dynamic foreground
            points_feat = points_feat * (1.0 - mask_dyn_fg.float().unsqueeze(1)) + correct_feat * mask_dyn_fg.float().unsqueeze(1)
        else:
            correct_bev_coord = points_bev_coord

        # scatter points_feat back to BEV image
        corrected_bev_img = bev_scatter(correct_bev_coord, points[:, 0].long(), points_feat, bev_img.shape[2:])

        # fuse bev_img & correct_bev_img
        weights = self.conv_weightor(torch.cat([bev_img, corrected_bev_img], dim=1))
        weights = torch.softmax(weights, dim=1)

        fused_bev_img = bev_img * weights[:, [0]] + corrected_bev_img * weights[:, [1]]

        return fused_bev_img

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        """
        Args:
            batch_dict: {
                points: (N, 1 + 5 + 2) - batch_idx, point-5, sweep_idx, instance_idx
                spatial_features_2d: (B, num_bev_feat, H, W)
            }
        """
        points = batch_dict['points']  # (N, 1 + 10 + 2) - batch_idx | x, y, z, intensity, time | map_feat (5) | sweep_idx, instance_idx
        bev_img = self.conv_input(batch_dict['spatial_features_2d'])  # (B, conv_input_channels, H, W)

        points_feat, points_bev_coord = interpolate_points_feat_from_bev_img(bev_img, 
                                                                             points, 
                                                                             self.point_cloud_range, 
                                                                             self.voxel_size[:2] * self.bev_image_stride, 
                                                                             return_bev_coord=True)
        
        # invoke Point Head
        pts_local_feat, points_cls_logit, points_flow3d, points_inst_embed = self.point_head(points_feat)
        prediction_dict = {
            'points_cls_logit': points_cls_logit,  # (N, 3) - bg | static fg | dynamic fg
            'points_flow3d': points_flow3d,  # (N, 3)
            'points_embedding': points_inst_embed  # (N, 2) - offset to center of instance's gt box
        }

        if self.training:
            mask_fg = points[:, self._meta_pts_feat_loc_inst_idx] > -1
            fg = points[mask_fg]
            fg_feat = points_feat[mask_fg]

            # build meta
            meta = self._build_meta(fg, batch_dict['gt_boxes'].shape[1])
            meta['mask_fg'] = mask_fg
            self.forward_return_dict['meta'] = meta

            # invoke Object Head
            locals_tf, locals_feat = self.object_head(fg[:, 1: 4], fg_feat, meta)
            # locals_tf: (N_local, 7)
            # locals_feat: (N_local, C_local)
            prediction_dict['locals_tf'] = locals_tf

            loss_dtl_locals_feat = self.point_head.get_loss_distill(pts_local_feat, locals_feat, meta)
            self.forward_return_dict['loss_dtl_locals_feat'] = loss_dtl_locals_feat

            target_dict = self.assign_target(batch_dict, meta)

            # save target_dict & points (before being corrected) for debugging
            if self.model_cfg.get('DEBUG', False):
                batch_dict['target_dict'] = target_dict
                batch_dict['points_original'] = torch.clone(points)
                batch_dict['hunter_meta'] = meta

            # save prediction & target for computing loss
            meta['fg'] = fg
            self.forward_return_dict.update({
                'prediction': prediction_dict,
                'target': target_dict,
                'meta': meta
            })
        
        # correct BEV image
        fused_bev_img = self.correct_bev_image(points, points_feat, points_bev_coord, points_cls_logit, points_flow3d, bev_img)

        # distill BEV image here
        if self.training and 'teacher_spatial_features_2d' in batch_dict:
            _B, _C, _H, _W = fused_bev_img.shape

            teacher_bev_img = batch_dict['teacher_spatial_features_2d']  # (B, C, H, W)
            # only distill where teacher_bev_img's magnitude > 1e-3
            teacher_bev_img = rearrange(teacher_bev_img, 'B C H W -> (B H W) C')
            fused_bev_img = rearrange(fused_bev_img, 'B C H W -> (B H W) C')
            mask_valid_teacher_loc = torch.linalg.norm(teacher_bev_img, dim=1) > 1e-3

            loss_dtl_bev_img = F.smooth_l1_loss(fused_bev_img[mask_valid_teacher_loc], 
                                                teacher_bev_img[mask_valid_teacher_loc], 
                                                reduction='none').sum(dim=1).mean()
            self.forward_return_dict['loss_dtl_bev_img'] = loss_dtl_bev_img

            fused_bev_img = rearrange(fused_bev_img, '(B H W) C -> B C H W', B=_B, H=_H, W=_W, C=_C)

        # overwrite
        batch_dict.pop('spatial_features_2d')
        batch_dict['spatial_features_2d'] = fused_bev_img.contiguous()

        # remove gt_boxes outside range
        if self.training:
            batch_dict = remove_gt_boxes_outside_range(batch_dict, self.point_cloud_range)
        
        if not self.training and self.model_cfg.get('GENERATING_EXCHANGE_DATA', False):
            # make points to be sent away
            points_cls_prob = torch.sigmoid(points_cls_logit)  # (N, 3)
            mask_send = points_cls_prob[:, 0] < 0.3  # prob background is sufficiently small
            if torch.any(mask_send):
                points_to_send = torch.cat([points[mask_send, 1:],  # exclude batch_idx | point-5, sweep_idx, inst_idx
                                            points_cls_prob[mask_send],
                                            points_flow3d[mask_send]], 
                                            dim=1)  # (N_pts_send, 5 + 3 + 3) - point-5, sweep_idx, inst_idx, cls_prob-3, flow-3
                
                points_to_send_batch_idx = points[mask_send, 0].long()
                for b_idx, metadata in enumerate(batch_dict['metadata']):
                    sample_token = metadata['sample_token']
                    lidar_id = metadata['lidar_id']
                    sample_points = points_to_send[points_to_send_batch_idx == b_idx]
                    if sample_points.shape[0] > 0:
                        save_path = f"{self.model_cfg.DATABASE_EXCHANGE_DATA}/{sample_token}_id{lidar_id}_foreground.pth"
                        torch.save(sample_points, save_path)

        return batch_dict

    def get_training_loss(self, tb_dict=None):
        pred = self.forward_return_dict['prediction']
        target = self.forward_return_dict['target']
        meta = self.forward_return_dict['meta']

        mask_fg = meta['mask_fg']
        device = mask_fg.device

        # Point loss
        # -------------------
        
        # cls loss
        target_points_cls = torch.argmax(target['points_cls'], dim=1)  # (N,)

        l_points_cls = self.loss_points_seg(pred['points_cls_logit'], target_points_cls)
        tb_dict['l_points_cls'] = l_points_cls.item()

        # [foreground] points embedding
        # ---
        l_points_embed = nn.functional.smooth_l1_loss(pred['points_embedding'][mask_fg], target['fg_embedding'],
                                                      reduction='none').sum(dim=1).mean()
        tb_dict['l_points_embed'] = l_points_embed.item()

        # [foreground] points offset
        # ---
        if target['fg_offset'] is not None:
            pred_fg_offset = pred['points_flow3d'][mask_fg]  # (N_fg, 3)
            loss_fg_offset = nn.functional.smooth_l1_loss(pred_fg_offset, target['fg_offset'], reduction='none').sum(dim=1)  # (N_fg,)
            mask_dyn_fg = target['points_cls'][mask_fg, 2] > 0  # (N_fg,)
            l_fg_offset = hard_mining_regression_loss(loss_fg_offset, 
                                                      mask_dyn_fg, 
                                                      device, 
                                                      self.model_cfg.get('LOSS_HARD_MINING_STATIC_FG_COEF', 1))
        else:
            l_fg_offset = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['l_fg_offset'] = l_fg_offset.item()

        # [dynamic locals] locals_tf
        # ---
        mask_locals_mos = target['meta']['mask_locals_mos']
        if torch.any(mask_fg):
            # translation
            # ---
            loss_locals_transl = F.smooth_l1_loss(pred['locals_tf'][:, :3], 
                                                  target['locals_tf'][:, :, -1], reduction='none').sum(dim=1)  # (N_locals)
            l_locals_transl = hard_mining_regression_loss(loss_locals_transl, 
                                                          mask_locals_mos, 
                                                          device, 
                                                          self.model_cfg.get('LOSS_HARD_MINING_STATIC_LOCALS_COEF', 1))

            # rotation
            # ---
            pred_locals_rot = quat2mat(pred['locals_tf'][:, 3:])  # (N_locals, 3, 3)
            loss_locals_rot = torch.linalg.norm(pred_locals_rot - target['locals_tf'][:, :, :3], dim=(1, 2), ord='fro')  # (N_locals)
            l_locals_rot = hard_mining_regression_loss(loss_locals_rot,
                                                       mask_locals_mos,
                                                       device, 
                                                       self.model_cfg.get('LOSS_HARD_MINING_STATIC_LOCALS_COEF', 1))

            # reconstruction
            # ---
            fg_xyz = meta['fg'][:, 1: 4]  # (N_dyn_fg, 3)

            fg_tf = target['locals_tf'][meta['locals2fg']]  # (N_fg, 3, 4)
            gt_corrected_fg = torch.matmul(fg_tf[:, :3, :3], fg_xyz.unsqueeze(-1)).squeeze(-1) + fg_tf[:, :3, -1]  # (N_dyn_fg, 3)

            pred_local_tf = torch.cat((pred_locals_rot, pred['locals_tf'][:, :3].unsqueeze(-1)), dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = pred_local_tf[meta['locals2fg']]
            corrected_fg = torch.matmul(pred_fg_tf[:, :3, :3], fg_xyz.unsqueeze(-1)).squeeze(-1) + pred_fg_tf[:, :3, -1]  # (N_dyn_fg, 3)

            loss_recon = nn.functional.smooth_l1_loss(corrected_fg, gt_corrected_fg, reduction='none').sum(dim=1) 
            mask_fg_mos = mask_locals_mos[meta['locals2fg']]  # (N_fg,)
            l_recon = hard_mining_regression_loss(loss_recon, 
                                                  mask_fg_mos, 
                                                  device, 
                                                  self.model_cfg.get('LOSS_HARD_MINING_STATIC_FG_COEF', 1)) * 0.1
        else:
            l_locals_transl = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            l_locals_rot = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            l_recon = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)

        tb_dict.update({
            'l_locals_transl': l_locals_transl.item(),
            'l_locals_rot': l_locals_rot.item(),
            'l_recon': l_recon.item()
        })

        tb_dict['l_dtl_locals_feat'] = self.forward_return_dict['loss_dtl_locals_feat'].item()

        loss = l_points_cls + l_points_embed + l_fg_offset + \
                l_locals_transl + l_locals_rot + l_recon + \
                self.forward_return_dict['loss_dtl_locals_feat']
        
        return loss, tb_dict

