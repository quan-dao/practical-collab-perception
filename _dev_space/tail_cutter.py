import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from einops import rearrange
from functools import partial
from _dev_space.unet_2d import UNet2D


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
        points_feat, points_bev_coord, pillars_flatten_coord, idx_pillar_to_point = \
            self.pillarize(batch_dict['points'])  # (N, 6 + 5)
        points_feat = self.pointnet(points_feat)  # (N, C_bev)
        pillars_feat = torch_scatter.scatter_max(points_feat, idx_pillar_to_point, dim=0)[0]  # (P, C_bev)

        # scatter pillars to BEV
        batch_bev_img = points_feat.new_zeros(batch_dict['batch_size'] * self.scale_xy, self.n_bev_feat)
        batch_bev_img[pillars_flatten_coord] = pillars_feat
        batch_bev_img = rearrange(batch_bev_img, '(B H W) C -> B C H W', B=batch_dict['batch_size'], H=self.bev_size[1],
                                  W=self.bev_size[0]).contiguous()
        return batch_bev_img, points_bev_coord


class PointSegmenter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pillar_encoder = PillarEncoder(cfg.NUM_RAW_FEATURES, cfg.NUM_BEV_FEATURES, cfg.POINT_CLOUD_RANGE,
                                            cfg.VOXEL_SIZE)
        self.backbone2d = UNet2D(cfg.NUM_BEV_FEATURES)
        self.head_fg_seg = self._make_head(cfg.HEAD_MID_CHANNELS, 1)
        self.head_motion_seg = self._make_head(cfg.HEAD_MID_CHANNELS, 1)
        self.head_inst_assoc = self._make_head(cfg.HEAD_MID_CHANNELS, 2)

    def _make_head(self, n_channels_mid, n_channels_out):
        return nn.Sequential(
            nn.Linear(self.cfg.NUM_BEV_FEATURES, n_channels_mid, bias=False),
            nn.BatchNorm1d(n_channels_mid, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(n_channels_mid, n_channels_out)
        )

    def forward(self, batch_dict: dict):
        """
        Args:
            batch_dict:
                'points': (N, 6 + C) - batch_idx, x, y, z, intensity, time, [...]
        """
        bev_img, points_bev_coord = self.pillar_encoder(batch_dict)  # (B, C_bev, H, W), (N, 2)-bev_x, bev_y
        bev_img = self.backbone2d(bev_img)  # (B, 64, H, W)

        # interpolate points feature from bev_img
        points_batch_idx = batch_dict['points'][:, 0].long()
        points_feat = points_batch_idx.new_zeros(points_batch_idx.shape[0], self.cfg.NUM_BEV_FEATURES)
        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            points_feat[batch_mask] = bilinear_interpolate_torch(_img, points_bev_coord[batch_mask, 0],
                                                                 points_bev_coord[batch_mask, 1])

        # invoke heads
        pred_points_fg = self.head_fg_seg(points_feat)  # (N, 1)
        pred_points_motion = self.head_motion_seg(points_feat)  # (N, 1)
        pred_points_inst_assoc = self.head_inst_assoc(points_feat)  # (N, 2) - x-,y-offset to mean of points in instance

    def assign_target(self, batch_dict):
        """
        Args:
            batch_dict:
                'points': (N, 6 + C) - ..., moving_status, inst_indicator
                    * moving_status: -1 for background, 0 for static foreground, 1 for moving foreground
                    * inst_indicator: -1 for background, >= 0 for foreground
        """
        points = batch_dict['points']
        # sanity check
        mask_bg = points[:, -2].long() == -1
        assert torch.all(points[mask_bg, -1].long() == -1), "inconsistency between moving stat & inst_indicator"

        target_fg = (points[:, -1].long() > -1).long()
        target_motion = (points[:, -2].long() > 0).long()  # only supervise this for g.t foreground points

        # target of inst_assoc by offset toward mean of points inside each instance
        unq_inst_idx, idx_inst_to_point = torch.unique(points[:, -1].long(), return_inverse=True)
        inst_mean_xy = torch_scatter.scatter_mean(points[:, 1: 3], idx_inst_to_point, dim=0)  # (N_inst, 2)
        target_inst_assoc = inst_mean_xy[idx_inst_to_point] - points[:, 1: 3]

        # format output
        target_dict = {'fg': target_fg, 'motion': target_motion, 'inst_assoc': target_inst_assoc}
        return target_dict


