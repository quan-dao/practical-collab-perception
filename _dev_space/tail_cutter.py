import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from einops import rearrange


def to_ndarray(x):
    return x if isinstance(x, np.ndarray) else np.array(x)


def to_tensor(x, is_cuda=True):
    t = torch.from_numpy(to_ndarray(x))
    t = t.float()
    if is_cuda:
        t = t.cuda()
    return t


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

        return features, pillars_flatten_coord, idx_pillar_to_point

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
        points_feat, pillars_flatten_coord, idx_pillar_to_point = self.pillarize(batch_dict['points'])  # (N, 6 + 5)
        points_feat = self.pointnet(points_feat)  # (N, C_bev)
        pillars_feat = torch_scatter.scatter_max(points_feat, idx_pillar_to_point, dim=0)[0]  # (P, C_bev)

        # scatter pillars to BEV
        batch_bev_img = points_feat.new_zeros(batch_dict['batch_size'] * self.scale_xy, self.n_bev_feat)
        batch_bev_img[pillars_flatten_coord] = pillars_feat
        batch_bev_img = rearrange(batch_bev_img, '(B H W) C -> B C H W', B=batch_dict['batch_size'], H=self.bev_size[1],
                                  W=self.bev_size[0]).contiguous()
        return batch_bev_img


