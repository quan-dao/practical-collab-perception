import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from einops import rearrange
from typing import List
from kornia.losses.focal import BinaryFocalLossWithLogits
from torchmetrics.functional import precision_recall
from sklearn.cluster import DBSCAN
from _dev_space.unet_2d import UNet2D
from _dev_space.resnet import PoseResNet
from _dev_space.instance_centric_tools import quat2mat
from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from _dev_space.external_drop import DropBlock2d


def to_ndarray(x):
    return x if isinstance(x, np.ndarray) else np.array(x)


def to_tensor(x, is_cuda=True):
    t = torch.from_numpy(to_ndarray(x))
    t = t.float()
    if is_cuda:
        t = t.cuda()
    return t


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class PillarEncoder(nn.Module):
    def __init__(self, n_raw_feat, n_bev_feat, pc_range, voxel_size):
        assert len(voxel_size) == 3, f"voxel_size: {voxel_size}"
        assert len(pc_range) == 6, f"pc_range: {pc_range}"
        super().__init__()
        self.pointnet = nn. Sequential(
            nn.Linear(n_raw_feat + 5, n_bev_feat, bias=False),
            nn.BatchNorm1d(n_bev_feat, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.pc_range = to_tensor(pc_range, is_cuda=True)
        self.voxel_size = to_tensor(voxel_size, is_cuda=True)
        self.bev_size = (to_ndarray(pc_range)[3: 5] - to_ndarray(pc_range)[:2]) / (to_ndarray(voxel_size[:2]))
        self.bev_size = self.bev_size.astype(int)  # [size_x, size_y]
        self.scale_xy = self.bev_size[0] * self.bev_size[1]
        self.scale_x = self.bev_size[0]
        self.xy_offset = self.voxel_size[:2] / 2.0 + self.pc_range[:2]
        self.n_raw_feat = n_raw_feat
        self.n_bev_feat = n_bev_feat

    def pillarize(self, points: torch.Tensor):
        """
        Args:
            points: (N, 6 + C) - batch_idx, x, y, z, intensity, time, [...]
        Returns:
            points_feat: (N, 6 + 5) - "+ 5" because x,y,z-offset to mean of points in pillar & x,y-offset to pillar center
            points_bev_coord: (N, 2) - bev_x, bev_y
            pillars_flatten_coord: (P,) - P: num non-empty pillars,
                flatten_coord = batch_idx * scale_xy + y * scale_x + x
            idx_pillar_to_point: (N,)
        """
        # check if all points are inside range
        mask_in = torch.all((points[:, 1: 4] >= self.pc_range[:3]) & (points[:, 1: 4] < self.pc_range[3:] - 1e-3))
        assert mask_in

        points_bev_coord = torch.floor((points[:, [1, 2]] - self.pc_range[:2]) / self.voxel_size[:2]).long()  # (N, 2): x, y
        points_flatten_coord = points[:, 0].long() * self.scale_xy + \
                               points_bev_coord[:, 1] * self.scale_x + \
                               points_bev_coord[:, 0]

        pillars_flatten_coord, idx_pillar_to_point = torch.unique(points_flatten_coord, return_inverse=True)
        # pillars_flatten_coord: (P)
        # idx_pillar_to_point: (N)

        pillars_mean = torch_scatter.scatter_mean(points[:, 1: 4], idx_pillar_to_point, dim=0)
        f_mean = points[:, 1: 4] - pillars_mean[idx_pillar_to_point]  # (N, 3)

        f_center = points[:, 1: 3] - (points_bev_coord.float() * self.voxel_size[:2] + self.xy_offset)  # (N, 2)

        features = torch.cat([points[:, 1: 1 + self.n_raw_feat], f_mean, f_center], dim=1)

        return features, points_bev_coord, pillars_flatten_coord, idx_pillar_to_point

    def forward(self, batch_dict: dict):
        """
        Args:
            batch_dict:
                'points': (N, 6 + C) - batch_idx, x, y, z, intensity, time, [...]
        """
        # remove points outside of pc-range
        mask_in = torch.all((batch_dict['points'][:, 1: 4] >= self.pc_range[:3]) &
                            (batch_dict['points'][:, 1: 4] < self.pc_range[3:] - 1e-3), dim=1)
        batch_dict['points'] = batch_dict['points'][mask_in]  # (N, 6 + C)

        # compute pillars' coord & feature
        points_feat, _, pillars_flatten_coord, idx_pillar_to_point = \
            self.pillarize(batch_dict['points'])  # (N, 6 + 5)
        points_feat = self.pointnet(points_feat)  # (N, C_bev)
        pillars_feat = torch_scatter.scatter_max(points_feat, idx_pillar_to_point, dim=0)[0]  # (P, C_bev)

        # scatter pillars to BEV
        batch_bev_img = points_feat.new_zeros(batch_dict['batch_size'] * self.scale_xy, self.n_bev_feat)
        batch_bev_img[pillars_flatten_coord] = pillars_feat
        batch_bev_img = rearrange(batch_bev_img, '(B H W) C -> B C H W', B=batch_dict['batch_size'], H=self.bev_size[1],
                                  W=self.bev_size[0]).contiguous()
        return batch_bev_img


class PointAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        pillar_cfg = cfg.PILLAR_ENCODER
        self.pillar_encoder = PillarEncoder(
            pillar_cfg.NUM_RAW_FEATURES, pillar_cfg.NUM_BEV_FEATURES, pillar_cfg.POINT_CLOUD_RANGE,
            pillar_cfg.VOXEL_SIZE)

        map_net_cfg = cfg.MAP_NET
        if map_net_cfg.ENABLE:
            if map_net_cfg.NAME == 'MapFusion':
                map_net_channels = [map_net_cfg.NUM_MAP_LAYERS] + map_net_cfg.MAP_FUSION_CHANNELS
                map_net_layers = []
                for ch_idx in range(len(map_net_channels) - 1):
                    is_last_layer = ch_idx == len(map_net_channels) - 2
                    map_net_layers.append(
                        nn.Conv2d(map_net_channels[ch_idx], map_net_channels[ch_idx + 1], 3, padding=1, bias=is_last_layer)
                    )
                    if not is_last_layer:
                        map_net_layers.append(nn.BatchNorm2d(map_net_channels[ch_idx + 1], eps=1e-3, momentum=0.01))
                        map_net_layers.append(nn.ReLU(True))
                self.map_net = nn.Sequential(*map_net_layers)
                num_map_features = map_net_cfg.MAP_FUSION_CHANNELS[-1]
            else:
                self.map_net = PoseResNet(map_net_cfg, map_net_cfg.NUM_MAP_LAYERS)
                num_map_features = self.map_net.num_out_features
        else:
            self.map_net = None
        self.drop_map = DropBlock2d(map_net_cfg.DROP_PROB, map_net_cfg.DROP_BLOCK_SIZE)

        if map_net_cfg.ENABLE:
            self.backbone2d = UNet2D(num_map_features + pillar_cfg.NUM_BEV_FEATURES, cfg.BEV_BACKBONE)
        else:
            self.backbone2d = UNet2D(pillar_cfg.NUM_BEV_FEATURES, cfg.BEV_BACKBONE)

        # ----
        backbone_out_c = self.backbone2d.n_output_feat
        # point heads
        self.point_fg_seg = self._make_mlp(backbone_out_c, 1, cfg.get('HEAD_MID_CHANNELS', None))
        self.point_inst_assoc = self._make_mlp(backbone_out_c, 2, cfg.get('HEAD_MID_CHANNELS', None))

        # ---
        self.inst_global_mlp = self._make_mlp(backbone_out_c, cfg.INSTANCE_OUT_CHANNELS,
                                              cfg.get('INSTANCE_MID_CHANNELS', None))
        self.inst_local_mlp = self._make_mlp(3, cfg.INSTANCE_OUT_CHANNELS, cfg.get('INSTANCE_MID_CHANNELS', None))
        # input channels == 3 because: x - \bar{x}, y - \bar{y}, z - \bar{z}; \bar{} == center

        # ----
        # instance heads
        self.inst_motion_seg = self._make_mlp(cfg.INSTANCE_OUT_CHANNELS, 1,
                                              cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)

        self.inst_proposal_gen = self._make_mlp(3 + 2 * cfg.INSTANCE_OUT_CHANNELS, 8,
                                                cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # in_ch == 3 + 2 * cfg.INSTANCE_OUT_CHANNELS because
        # 3 = |centroid of the local @ target time step|
        # 2 * cfg.INSTANCE_OUT_CHANNELS = |concatenation of feat of local @ target & global|
        # out_ch == 8 because c_x, c_y, c_z, d_x, d_y, d_z, sin_yaw, cos_yaw

        self.inst_local_transl = self._make_mlp(6 + 3 * cfg.INSTANCE_OUT_CHANNELS, 3,
                                                cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # out == 3 for 3 components of translation vector

        self.inst_local_rot = self._make_mlp(6 + 3 * cfg.INSTANCE_OUT_CHANNELS, 4,
                                             cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # out == 4 for 4 components of quaternion
        fill_fc_weights(self.inst_local_rot)
        # ---
        self.forward_return_dict = dict()
        # loss func
        if cfg.LOSS.SEGMENTATION_LOSS == 'focal':
            self.seg_loss = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction='sum')
        elif cfg.LOSS.SEGMENTATION_LOSS == 'ce_lovasz':
            self.seg_loss = CELovaszLoss(num_classes=2)  # binary classification for both fg seg & motion seg
        else:
            raise NotImplementedError

        self.clusterer = DBSCAN(eps=cfg.CLUSTER.EPS, min_samples=cfg.CLUSTER.MIN_POINTS)
        self.has_2nd_stage = cfg.get('HAS_2ND_STAGE', False)
        self.num_instance_features = cfg.INSTANCE_OUT_CHANNELS

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

    def forward(self, batch_dict: dict):
        """
        Args:
            batch_dict:
                'points': (N, 6 + C) - batch_idx, x, y, z, intensity, time, [...]
        """
        self.forward_return_dict = dict()  # clear previous output

        bev_img = self.pillar_encoder(batch_dict)  # (B, C_bev, H, W), (N, 2)-bev_x, bev_y

        # concatenate hd_map with bev_img before passing to backbone_2d
        if self.map_net is not None:
            map_img = self.map_net(batch_dict['img_map'])
            map_img = self.drop_map(map_img)
            bev_img = torch.cat([bev_img, map_img], dim=1)  # (B, C_bev + C_map, H, W)

        bev_img = self.backbone2d(bev_img)  # (B, 64, H, W)

        # interpolate points feature from bev_img
        points_bev_coord = (batch_dict['points'][:, [1, 2]] - self.pillar_encoder.pc_range[:2]) / \
                           self.pillar_encoder.voxel_size[:2]
        points_batch_idx = batch_dict['points'][:, 0].long()
        points_feat = bev_img.new_zeros(points_batch_idx.shape[0], bev_img.shape[1])
        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            points_feat[batch_mask] = bilinear_interpolate_torch(_img, points_bev_coord[batch_mask, 0],
                                                                 points_bev_coord[batch_mask, 1])

        # invoke point heads
        pred_points_fg = self.point_fg_seg(points_feat)  # (N, 1)
        pred_points_inst_assoc = self.point_inst_assoc(points_feat)  # (N, 2) - x-,y-offset to mean of points in instance
        pred_dict = {
            'fg': pred_points_fg,  # (N, 1)
            'inst_assoc': pred_points_inst_assoc  # (N, 2)
        }

        # ---------------------------------------------------------------
        # INSTANCE stuff
        # ---------------------------------------------------------------

        if self.training or not self.cfg.REAL_INFERENCE:
            # Training 1st stage or Freeze 1st stage to train 2nd stage
            # ==> use AUGMENTED instance index
            mask_fg = batch_dict['points'][:, -1].long() > -1
            fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instance_idx
            fg_inst_idx = fg[:, -1].long()  # (N_fg,)
            max_num_inst = batch_dict['instances_tf'].shape[1]  # batch_dict['instances_tf']: (batch_size, max_n_inst, n_sweeps, 3, 4)
            fg_batch_idx = points_batch_idx[mask_fg]
            fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)
            fg_prob = sigmoid(pred_points_fg.detach())[mask_fg]  # (N_fg_valid, 1)
        else:
            if self.cfg.get('FG_THRESH', 0.5) > 0:
                mask_fg = rearrange(sigmoid(pred_points_fg), 'N 1 -> N') > self.cfg.FG_THRESH
            else:
                # use ground truth foreground
                mask_fg = batch_dict['points'][:, -2] > -1
            fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instance_idx
            fg_batch_idx = points_batch_idx[mask_fg]  # (N_fg,)
            fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)
            fg_inst_idx = fg.new_zeros(fg.shape[0]).long()
            # ==
            # DBSCAN to get instance_idx during inference
            # ==
            # apply inst_assoc
            fg_embed = fg[:, 1: 3] + pred_points_inst_assoc[mask_fg]  # (N_fg, 2)

            # invoke dbscan
            max_num_inst = 0
            for batch_idx in range(batch_dict['batch_size']):
                mask_batch = fg_batch_idx == batch_idx
                cur_fg_embed_numpy = fg_embed[mask_batch].detach().cpu().numpy()  # # (N_cur, 2)
                self.clusterer.fit(cur_fg_embed_numpy)
                fg_labels = self.clusterer.labels_  # (N_cur,)

                # update fg_inst_idx
                fg_inst_idx[mask_batch] = torch.from_numpy(fg_labels).long().to(fg.device)

                # update max_num_inst
                max_num_inst = max(max_num_inst, np.max(fg_labels))

            # remove noisy foreground (i.e. foreground that don't get assigned to any clusters)
            valid_fg = fg_inst_idx > -1
            fg = fg[valid_fg]  # (N_fg_valid, 8)
            fg_batch_idx = fg_batch_idx[valid_fg]  # (N_fg_valid,)
            fg_inst_idx = fg_inst_idx[valid_fg]  # (N_fg_valid,)
            fg_feat = fg_feat[valid_fg]  # (N_fg_valid, C_bev)
            fg_prob = sigmoid(pred_points_fg)[mask_fg]  # (N_fg_valid, 1)
            fg_prob = fg_prob[valid_fg]

        with torch.no_grad():
            fg_sweep_idx = fg[:, -3].long()

            # merge batch_idx & instance_idx
            fg_bi_idx = fg_batch_idx * max_num_inst + fg_inst_idx  # (N,)

            # merge batch_idx, instance_idx & sweep_idx
            fg_bisw_idx = fg_bi_idx * self.cfg.get('NUM_SWEEPS', 10) + fg_sweep_idx

            # --
            # group foreground points to instance
            # ---
            inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
            # inst_bi: (N_inst,)
            # inst_bi_inv_indices: (N_fg,)

            # --
            # group foreground points to local group
            # ---
            local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
            # local_bisw: (N_local,)

            self.forward_return_dict['meta'] = {
                'max_num_inst': max_num_inst,
                'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices,
                'local_bisw': local_bisw, 'local_bisw_inv_indices': local_bisw_inv_indices
            }

        # ------------
        # compute instance global feature
        fg_feat4glob = self.inst_global_mlp(fg_feat)  # (N_fg, C_inst)
        inst_global_feat = torch_scatter.scatter_max(fg_feat4glob, inst_bi_inv_indices, dim=0)[0]  # (N_inst, C_inst)

        # use inst_global_feat to predict motion stat
        pred_inst_motion_stat = self.inst_motion_seg(inst_global_feat)  # (N_inst, 1)
        pred_dict['inst_motion_stat'] = pred_inst_motion_stat

        # ------------
        # compute instance local shape encoding
        local_center = torch_scatter.scatter_mean(fg[:, 1: 4], local_bisw_inv_indices, dim=0)  # (N_local, 3)
        fg_centered_xyz = fg[:, 1: 4] - local_center[local_bisw_inv_indices]  # (N_fg, 3)
        fg_shape_enc = self.inst_local_mlp(fg_centered_xyz)  # (N_fg, C_inst)

        local_shape_enc = torch_scatter.scatter_max(fg_shape_enc, local_bisw_inv_indices, dim=0)[0]  # (N_local, C_inst)

        # ------------
        # prepare input for prediction @ the 2nd stage
        self.forward_return_dict['input_2nd_stage'] = {
            'foreground': {
                'fg': fg,  # (N_fg, 7[+2]) - batch_idx x, y, z, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]
                'fg_feat': fg_feat.detach(),  # (N_fg, C_bev)
                'fg_prob': rearrange(fg_prob.detach(), 'N_fg 1 -> N_fg'),
                'fg_inst_idx': fg_inst_idx,  # (N_fg,)
            },
            'local': {
                'shape_encoding': local_shape_enc.detach(),  # (N_local, C_inst)
            },
            'global': {
                'shape_encoding': inst_global_feat.detach(),  # (N_inst, C_inst)
                'motion_stat': sigmoid(rearrange(pred_inst_motion_stat.detach(), 'N_inst 1 -> N_inst')),
            }
        }

        # ------------
        # compute instance shape encoding of the local @ target timestep (i.e. local having the largest sweep idx)
        with torch.no_grad():
            # get the max sweep_index of each instance
            inst_max_sweep_idx = torch_scatter.scatter_max(fg_sweep_idx, inst_bi_inv_indices)[0]  # (N_inst,)
            # get bisw_index of each instance's max sweep
            inst_target_bisw_idx = inst_bi * self.cfg.get('NUM_SWEEPS', 10) + inst_max_sweep_idx  # (N_inst,)

            # for each value in inst_target_bisw_idx find WHERE (i.e., index) it appear in local_bisw
            corr = local_bisw[:, None] == inst_target_bisw_idx[None, :]  # (N_local, N_inst)
            corr = corr.long() * torch.arange(local_bisw.shape[0]).unsqueeze(1).to(fg.device)
            corr = corr.sum(dim=0)  # (N_inst)

        inst_target_center = local_center[corr]  # (N_inst, 3)
        self.forward_return_dict['meta'].update({
            'inst_target_center': inst_target_center,  # for correction  - (N_inst, 3)
            'inst_max_sweep_idx': inst_max_sweep_idx,  # for prediction  - (N_inst,)
        })

        inst_target_center_shape = torch.cat((inst_target_center, local_shape_enc[corr]), dim=1)  # (N_inst, 3+C_inst)
        inst_global_feat = torch.cat((inst_global_feat, inst_target_center_shape), dim=1)  # (N_inst, 3+2*C_inst)

        # ------------
        # generate 3D proposals
        # ------------
        proposals = self.inst_proposal_gen(inst_global_feat)  # (N_inst, 8)
        pred_dict['proposals'] = proposals  # (N_inst, 8)

        # ------------
        with torch.no_grad():
            # broadcast inst_global_feat from (N_inst, C_inst) to (N_local, C_inst)
            local_bi = local_bisw // self.cfg.get('NUM_SWEEPS', 10)
            # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
            local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
            local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1).to(fg.device)
            local_bi_in_inst_bi = local_bi_in_inst_bi.sum(dim=0)  # (N_local)
            self.forward_return_dict['meta']['local_bi_in_inst_bi'] = local_bi_in_inst_bi

        # ---
        local_global_feat = inst_global_feat[local_bi_in_inst_bi]  # (N_local, 3+2*C_inst)

        # concatenate feat of local
        local_feat = torch.cat((local_global_feat, local_center, local_shape_enc), dim=1)  # (N_local, 6+3*C_inst)

        # use local_feat to predict local_tf of size (N_local, 3, 4)
        pred_local_transl = self.inst_local_transl(local_feat)  # (N_local, 3)
        pred_dict['local_transl'] = pred_local_transl

        pred_local_rot = self.inst_local_rot(local_feat)  # (N_local, 4)
        pred_dict['local_rot'] = quat2mat(pred_local_rot)  # (N_local, 3, 3)

        # update forward_return_dict with prediction & target for computing loss
        self.forward_return_dict['pred_dict'] = pred_dict
        if self.training:
            self.forward_return_dict['target_dict'] = self.assign_target(batch_dict)

        # -------------------------
        # prepare input for the 2nd stage
        pred_boxes = self.decode_proposals()  # (N_inst, 7)
        fg_motion_mask = self.correct_point_cloud_()

        self.forward_return_dict['input_2nd_stage']['local'].update({
            'local_transl': pred_dict['local_transl'].detach(),  # (N_local, 3)
            'local_rot': pred_dict['local_rot'].detach()  # (N_local, 3, 3)
        })
        self.forward_return_dict['input_2nd_stage']['pred_boxes'] = pred_boxes.detach(),  # (N_inst, 7)

        batch_dict['input_2nd_stage'] = {
            'meta': self.forward_return_dict['meta']
        }
        batch_dict['input_2nd_stage'].update(self.forward_return_dict['input_2nd_stage'])

        if not self.training:
            batch_dict['input_2nd_stage']['fg_motion_mask'] = fg_motion_mask
            batch_dict['input_2nd_stage']['bg'] = batch_dict['points'][torch.logical_not(mask_fg)]  # (N_bg, 7[+2])

        return batch_dict

    @torch.no_grad()
    def assign_target(self, batch_dict):
        """
        Args:
            batch_dict:
                'points': (N, 8) - batch_idx, x, y, z, intensity, time, sweep_idx, instance_idx
                    * inst_indicator: -1 for background, >= 0 for foreground
                'instances_tf': (B, N_inst_max, N_sweeps, 3, 4)
        """
        points = batch_dict['points']
        # -------------------------------------------------------
        # Point-wise target
        # use ORIGINAL instance index
        # -------------------------------------------------------
        # --------------
        # target foreground
        mask_fg = points[:, -2].long() > -1
        target_fg = mask_fg.long()  # (N,) - use original instance index for foreground seg

        # --------------
        # target of inst_assoc as offset toward mean of points inside each instance
        _fg = points[mask_fg]
        max_num_inst = batch_dict['instances_tf'].shape[1]
        _fg_bi_idx = _fg[:, 0].long() * max_num_inst + _fg[:, -2].long()
        _, _inst_bi_inv_indices = torch.unique(_fg_bi_idx, sorted=True, return_inverse=True)

        inst_mean_xy = torch_scatter.scatter_mean(points[mask_fg, 1: 3], _inst_bi_inv_indices, dim=0)  # (N_inst, 2)
        target_inst_assoc = inst_mean_xy[_inst_bi_inv_indices] - points[mask_fg, 1: 3]

        # -------------------------------------------------------
        # Instance-wise target
        # use AUGMENTED instance index
        # -------------------------------------------------------
        instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)

        # --------------
        inst_bi = self.forward_return_dict['meta']['inst_bi']
        # target motion
        inst_motion_stat = torch.linalg.norm(instances_tf[:, :, 0, :, -1], dim=-1) > self.cfg.TARGET_CONFIG.MOTION_THRESH
        inst_motion_stat = rearrange(inst_motion_stat.long(), 'B N_inst_max -> (B N_inst_max)')
        inst_motion_stat = inst_motion_stat[inst_bi]  # (N_inst)

        # target proposal
        gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, 11) -center (3), size (3), yaw, dummy_v (2), instance_index, class
        assert gt_boxes.shape[1] == instances_tf.shape[1], f"{gt_boxes.shape[1]} != {instances_tf.shape[1]}"
        batch_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        batch_boxes = batch_boxes[inst_bi]  # (N_inst, 11)

        target_proposals = {
            'offset': batch_boxes[:, :3] - self.forward_return_dict['meta']['inst_target_center'],
            'size': batch_boxes[:, 3: 6],
            'ori': torch.stack([torch.sin(batch_boxes[:, 6]), torch.cos(batch_boxes[:, 6])], dim=1)
        }

        # --------------
        # locals' transformation
        local_bisw = self.forward_return_dict['meta']['local_bisw']
        # local_bisw: (N_local,)

        local_tf = rearrange(instances_tf, 'B N_inst_max N_sweep C1 C2 -> (B N_inst_max N_sweep) C1 C2', C1=3, C2=4)
        local_tf = local_tf[local_bisw]  # (N_local, 3, 4)

        # format output
        target_dict = {
            'fg': target_fg,
            'inst_assoc': target_inst_assoc,
            'inst_motion_stat': inst_motion_stat,
            'local_tf': local_tf,
            'proposals': target_proposals
        }
        return target_dict

    @staticmethod
    def l2_loss(pred: torch.Tensor, target: torch.Tensor, dim=-1, reduction='sum'):
        """
        Args:
            pred
            target
            dim: dimension where the norm take place
            reduction
        """
        assert len(pred.shape) >= 2
        assert pred.shape == target.shape
        assert reduction in ('sum', 'mean', 'none')
        diff = torch.linalg.norm(pred - target, dim=dim)
        if reduction == 'none':
            return diff
        elif reduction == 'sum':
            return diff.sum()
        else:
            return diff.mean()

    def get_training_loss(self, batch_dict, tb_dict=None, debug=False):
        if tb_dict is None:
            tb_dict = dict()
        debug_dict = dict() if debug else None

        pred_dict = self.forward_return_dict['pred_dict']
        target_dict = self.forward_return_dict['target_dict']

        # -----------------------
        # Point-wise loss
        # -----------------------
        # ---
        # foreground seg
        fg_logit = pred_dict['fg']  # (N, 1)
        fg_target = target_dict['fg']  # (N,)
        assert torch.all(torch.isfinite(fg_logit))

        num_gt_fg = fg_target.sum().item()
        if self.cfg.LOSS.SEGMENTATION_LOSS == 'focal':
            loss_fg = self.seg_loss(fg_logit, fg_target[:, None].float()) / max(1., num_gt_fg)
        else:
            loss_fg = self.seg_loss(fg_logit, fg_target, tb_dict, loss_name='fg')
        tb_dict['loss_fg'] = loss_fg.item()

        # ---
        # instance assoc
        inst_assoc = pred_dict['inst_assoc']  # (N, 2)
        if num_gt_fg > 0:
            inst_assoc_target = target_dict['inst_assoc']  # (N_fg, 2) ! target was already filtered by foreground mask
            loss_inst_assoc = self.l2_loss(inst_assoc[fg_target == 1], inst_assoc_target, dim=1, reduction='mean')
        else:
            loss_inst_assoc = self.l2_loss(inst_assoc, torch.clone(inst_assoc).detach(), dim=1, reduction='mean')
        tb_dict['loss_inst_assoc'] = loss_inst_assoc.item()

        # -----------------------
        # Instance-wise loss
        # -----------------------
        # ---
        # motion seg
        inst_mos_logit = pred_dict['inst_motion_stat']  # (N_inst, 1)
        inst_mos_target = target_dict['inst_motion_stat']  # (N_inst,)

        num_gt_dyn_inst = float(inst_mos_target.sum().item())
        if self.cfg.LOSS.SEGMENTATION_LOSS == 'focal':
            loss_inst_mos = self.seg_loss(inst_mos_logit, inst_mos_target[:, None].float()) / max(1., num_gt_dyn_inst)
        else:
            loss_inst_mos = self.seg_loss(inst_mos_logit, inst_mos_target, tb_dict, loss_name='mos')
        tb_dict['loss_inst_mos'] = loss_inst_mos.item()

        # ---
        # Local tf - regression loss | ONLY FOR LOCAL OF DYNAMIC INSTANCE

        local_transl = pred_dict['local_transl']  # (N_local, 3)
        local_rot_mat = pred_dict['local_rot']  # (N_local, 3, 3)

        local_tf_target = target_dict['local_tf']  # (N_local, 3, 4)

        assert torch.all(torch.isfinite(local_transl))
        assert torch.all(torch.isfinite(local_rot_mat))
        assert torch.all(torch.isfinite(local_tf_target))

        # which local are associated with dynamic instances
        with torch.no_grad():
            local_bi_in_inst_bi = self.forward_return_dict['meta']['local_bi_in_inst_bi']  # (N_local,)
            local_mos_target = inst_mos_target[local_bi_in_inst_bi]  # (N_local,)
            local_mos_mask = local_mos_target == 1  # (N_local,)

        # translation
        # logger = logging.getLogger()
        if torch.any(local_mos_mask):
            # logger.info('loss_local_transl has ground truth')
            loss_local_transl = self.l2_loss(local_transl[local_mos_mask], local_tf_target[local_mos_mask, :, -1],
                                             dim=-1, reduction='mean')
        else:
            # logger.info('loss_local_transl does not have ground truth')
            loss_local_transl = self.l2_loss(local_transl, torch.clone(local_transl).detach(),
                                             dim=-1, reduction='mean')
        tb_dict['loss_local_transl'] = loss_local_transl.item()

        # rotation
        if torch.any(local_mos_mask):
            # logger.info('loss_local_rot has ground truth')
            loss_local_rot = torch.linalg.norm(
                local_rot_mat[local_mos_mask] - local_tf_target[local_mos_mask, :, :3], dim=(1, 2), ord='fro').mean()
        else:
            # logger.info('loss_local_rot does not have ground truth')
            loss_local_rot = self.l2_loss(local_rot_mat.reshape(-1, 1),
                                          torch.clone(local_rot_mat).detach().reshape(-1, 1),
                                          dim=-1, reduction='mean')
        tb_dict['loss_local_rot'] = loss_local_rot.item()

        # ---
        # Local tf - reconstruction loss

        # get motion mask of foreground
        inst_bi_inv_indices = self.forward_return_dict['meta']['inst_bi_inv_indices']  # (N_fg)
        fg_motion = inst_mos_target[inst_bi_inv_indices] == 1  # (N_fg)

        aug_fg_mask = batch_dict['points'][:, -1].long() > -1  # (N,) - use aug inst index
        fg = batch_dict['points'][aug_fg_mask]  # (N_fg)

        if torch.any(fg_motion):
            # extract dyn fg points
            dyn_fg = fg[fg_motion, 1: 4]  # (N_dyn, 3)

            # reconstruct with ground truth
            local_bisw_inv_indices = self.forward_return_dict['meta']['local_bisw_inv_indices']
            gt_fg_tf = local_tf_target[local_bisw_inv_indices]  # (N_fg, 3, 4)
            gt_dyn_fg_tf = gt_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            gt_recon_dyn_fg = torch.matmul(gt_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1) + \
                              gt_dyn_fg_tf[:, :, -1]  # (N_dyn, 3)
            if debug:
                debug_dict['gt_recon_dyn_fg'] = gt_recon_dyn_fg  # (N_dyn, 3)

            # reconstruct with prediction
            local_tf_pred = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = local_tf_pred[local_bisw_inv_indices]  # (N_fg, 3, 4)
            pred_dyn_fg_tf = pred_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            pred_recon_dyn_fg = torch.matmul(pred_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1) + \
                                pred_dyn_fg_tf[:, :, -1]  # (N_dyn, 3)

            loss_recon = self.l2_loss(pred_recon_dyn_fg, gt_recon_dyn_fg, dim=-1, reduction='mean')
        else:
            # there is no moving fg -> dummy loss
            local_bisw_inv_indices = self.forward_return_dict['meta']['local_bisw_inv_indices']
            local_tf_pred = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = local_tf_pred[local_bisw_inv_indices]  # (N_fg, 3, 4)
            pred_recon_dyn_fg = torch.matmul(pred_fg_tf[:, :, :3], fg[:, 1: 4, None]).squeeze(-1) + \
                                pred_fg_tf[:, :, -1]  # (N_dyn, 3)
            loss_recon = self.l2_loss(pred_recon_dyn_fg, torch.clone(pred_recon_dyn_fg).detach(), dim=-1, reduction='mean')
        tb_dict['loss_recon'] = loss_recon.item()

        # add proposals loss
        pred_proposals = pred_dict['proposals']  # (N_inst, 8)
        target_proposals = target_dict['proposals']

        loss_prop_offset = self.l2_loss(pred_proposals[:, :3], target_proposals['offset'], reduction='mean')
        tb_dict['loss_prop_offset'] = loss_prop_offset.item()

        loss_prop_size = self.l2_loss(pred_proposals[:, 3: 6], target_proposals['size'], reduction='mean')
        tb_dict['loss_prop_size'] = loss_prop_size.item()

        loss_prop_ori = self.l2_loss(pred_proposals[:, 6:], target_proposals['ori'], reduction='mean')
        tb_dict['loss_prop_ori'] = loss_prop_ori.item()

        loss_prop = loss_prop_offset + loss_prop_size + loss_prop_ori
        tb_dict['loss_prop'] = loss_prop.item()

        # --------------
        # total loss
        # --------------
        loss = loss_fg + loss_inst_assoc + loss_inst_mos + loss_local_transl + loss_local_rot + loss_recon + loss_prop
        tb_dict['loss'] = loss.item()

        # eval foregound seg, motion seg during training
        with torch.no_grad():
            pred_fg_prob = sigmoid(fg_logit.detach()).squeeze(-1)  # (N,)
            precision_fg, recall_fg = precision_recall(pred_fg_prob, fg_target, threshold=0.5)
            tb_dict['fg_P'] = precision_fg.item()
            tb_dict['fg_R'] = recall_fg.item()

            inst_mos_prob = sigmoid(inst_mos_logit.detach()).squeeze(-1)  # (N_inst)
            precision_mos, recall_mos = precision_recall(inst_mos_prob, inst_mos_target, threshold=0.5)
            tb_dict['mos_P'] = precision_mos.item()
            tb_dict['mos_R'] = recall_mos.item()
        out = [loss, tb_dict]
        # if debug:
        #     out.append(debug_dict)
        return out

    def decode_proposals(self) -> torch.Tensor:
        pred_proposals = self.forward_return_dict['pred_dict']['proposals']  # (N_inst, 8)
        center = self.forward_return_dict['meta']['inst_target_center'] + pred_proposals[:, :3]
        size = pred_proposals[:, 3: 6]
        yaw = torch.atan2(pred_proposals[:, 6], pred_proposals[:, 7])
        pred_boxes = torch.cat([center, size, yaw[:, None]], dim=1)  # (N_inst, 7)
        return pred_boxes

    @torch.no_grad()
    def generate_predicted_boxes(self, batch_size: int, debug=False, batch_dict=None) -> List[dict]:
        """
        Element i-th of the returned List represents predicted boxes for the sample i-th in the batch
        Args:
            batch_size:
            debug: to use target_proposals as pred_proposals
            batch_dict: only necessary for debugging
        """
        pred_dict = self.forward_return_dict['pred_dict']

        # decode predicted proposals
        if not debug:
            pred_boxes = self.decode_proposals()  # (N_inst, 7)
        else:
            target_dict = self.forward_return_dict['target_dict']
            target_proposals = target_dict['proposals']
            center = self.forward_return_dict['meta']['inst_target_center'] + target_proposals['offset']
            size = target_proposals['size']
            yaw = torch.atan2(target_proposals['ori'][:, 0], target_proposals['ori'][:, 1])
            pred_boxes = torch.cat([center, size, yaw[:, None]], dim=1)  # (N_inst, 7)

        # compute boxes' score by averaging foreground score, using inst_bi_inv_indices
        if not debug:
            fg_prob = self.forward_return_dict['foreground']['fg_prob']
        else:
            assert self.training
            fg_prob = sigmoid(pred_dict['fg'])  # (N_points, 1)
            mask_fg = batch_dict['points'][:, -1].long() > -1
            fg_prob = fg_prob[mask_fg]
        pred_scores = torch_scatter.scatter_mean(fg_prob,
                                                 self.forward_return_dict['meta']['inst_bi_inv_indices'],
                                                 dim=0)  # (N_inst, 1)

        # separate pred_boxes accodring to batch index
        inst_bi = self.forward_return_dict['meta']['inst_bi']
        max_num_inst = self.forward_return_dict['meta']['max_num_inst']
        inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

        out = []
        for b_idx in range(batch_size):
            cur_boxes = pred_boxes[inst_batch_idx == b_idx]
            cur_scores = rearrange(pred_scores[inst_batch_idx == b_idx], 'N_cur_inst 1 -> N_cur_inst')
            cur_labels = cur_boxes.new_ones(cur_boxes.shape[0]).long()
            out.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels
            })
        return out

    def correct_point_cloud_(self):
        """
        Overwrite self.forward_return_dict['foreground']['fg']
        """
        pred_dict = self.forward_return_dict['pred_dict']
        fg = self.forward_return_dict['input_2nd_stage']['foreground']['fg']  # (N_fg, 7[+2])
        # 7[+2] - batch_idx x, y, z, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]

        # get motion mask of foreground to find dynamic foreground
        inst_motion_stat = sigmoid(pred_dict['inst_motion_stat'][:, 0]) > 0.5  # TODO: use this for motion pred
        inst_bi_inv_indices = self.forward_return_dict['meta']['inst_bi_inv_indices']  # (N_fg)
        fg_motion_mask = inst_motion_stat[inst_bi_inv_indices] == 1  # (N_fg)

        if torch.any(fg_motion_mask):
            # only perform correction if there are some dynamic foreground points
            dyn_fg = fg[fg_motion_mask, 1: 4]  # (N_dyn, 3)

            local_tf = torch.cat([
                pred_dict['local_rot'],
                rearrange(pred_dict['local_transl'], 'N_local C -> N_local C 1', C=3)
            ], dim=-1)  # (N_local, 3, 4)

            # pair local_tf to foreground
            fg_tf = local_tf[self.forward_return_dict['meta']['local_bisw_inv_indices']]  # (N_fg, 3, 4)

            # extract local_tf of dyn_fg
            dyn_fg_tf = fg_tf[fg_motion_mask]  # (N_dyn, 3, 4)

            # apply transformation on dyn_fg
            dyn_fg = torch.matmul(dyn_fg_tf[:, :, :3], dyn_fg.unsqueeze(-1)).squeeze(-1) + dyn_fg_tf[:, :, -1]  # (N_dyn, 3)

            # overwrite coordinate of dynamic foreground with dyn_fg (computed above)
            fg[fg_motion_mask, 1: 4] = dyn_fg

        return fg_motion_mask


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
