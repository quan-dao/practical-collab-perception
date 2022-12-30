'''
source: https://github.com/prs-eth/PCAccumulation
'''
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN


def voxel_down_sample(points: torch.Tensor, point_cloud_range: torch.Tensor, voxel_size=0.2):
    """
    Args:
        points: (N, C) - batch_idx, x, y, z, ...
        point_cloud_range:
        voxel_size
    Returns:
        vox_coords: (N_v, 2) - x, y
        indices_vox2pts: (N,)
    """
    assert torch.all(points[:, 0].long() == points[0, 0].long()), "this function is only designed for batch_size=1"
    coords = torch.floor((points[:, 1: 3] - point_cloud_range[:2]) / voxel_size).long()  # (N, 2) - x, y
    size_x = torch.floor((point_cloud_range[3] - point_cloud_range[0]) / voxel_size).long()  # (size_x, size_y)
    assert torch.all(coords[:, 0] < size_x),  f"x-coord must be smaller than size_x"
    merged_coords = coords[:, 1] * size_x + coords[:, 0]
    vox_merged_coords, indices_vox2pts = torch.unique(merged_coords, sorted=True, return_inverse=True)
    vox_coords = torch.stack([
        vox_merged_coords % size_x,
        vox_merged_coords // size_x
    ], dim=1)  # (N_v, 2)
    return vox_coords, indices_vox2pts


class Cluster(nn.Module):
    def __init__(self, cfg, point_cloud_range):
        super(Cluster, self).__init__()
        cluster_cfg = cfg['cluster']
        self.min_p_cluster = cluster_cfg['min_p_cluster']
        self.voxel_size = cluster_cfg['voxel_size']
        self.cluster_estimator = DBSCAN(min_samples=cluster_cfg['min_samples_dbscan'],
                                        metric=cluster_cfg['cluster_metric'], eps=cluster_cfg['eps_dbscan'])
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def cluster_per_batch(self, voxels):
        """
        Input:
            voxels: [N, 2] - x, y
        Output:
            voxels_inst_idx: [N], voxels_inst_idx = -1 means background/ignored voxels
        """
        # 1. cluster the objects
        voxels_inst_idx = self.cluster_estimator.fit_predict(voxels.detach().cpu().numpy())

        # 2. Ignore clusters with less than self.min_p_cluster voxels
        unique_inst_idx = np.unique(voxels_inst_idx).tolist()
        for inst_idx in unique_inst_idx:
            instance_mask = voxels_inst_idx == inst_idx
            n_vox_of_this_inst = instance_mask.astype(int).sum()
            if n_vox_of_this_inst < self.min_p_cluster:
                voxels_inst_idx[instance_mask] = -1

        return torch.from_numpy(voxels_inst_idx).to(voxels.device)

    @torch.no_grad()
    def forward(self, batch_size: int, points: torch.Tensor, voxel_size=0.2):
        """
        Args:
            points: (N, C) - batch_idx, x,y, z, ...  | actually is offsetted foreground
        """
        points_inst_idx = points.new_zeros(points.shape[0])
        max_num_inst = 0
        for b_idx in range(batch_size):
            mask_current = points[:, 0].long() == b_idx
            voxels_coord, indices_vox2pts = voxel_down_sample(points[mask_current], self.point_cloud_range, voxel_size)
            voxels_inst_idx = self.cluster_per_batch(voxels_coord)  # (N_v,)
            # update maximum number of instances in this batch
            num_inst = voxels_inst_idx.max().item()
            if num_inst >= 0:
                # actually find some instances
                num_inst += 1  # num instance = max inst idx + 1
            max_num_inst = max(max_num_inst, num_inst)
            # update points instance index
            points_inst_idx[mask_current] = voxels_inst_idx[indices_vox2pts]  # (N_p,)
        return points_inst_idx, max_num_inst
