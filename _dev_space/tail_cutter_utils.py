import torch
import torch.nn as nn
import numpy as np
import torch_scatter
from einops import rearrange
from torchmetrics.functional import precision_recall
from typing import List


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


def quat2mat(quat):
    """
    convert quaternion to rotation matrix ([x, y, z, w] to follow scipy
    :param quat: (B, 4) four quaternion of rotation
    :return: rotation matrix [B, 3, 3]
    """
    # norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    # w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


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


@torch.no_grad()
def eval_binary_segmentation(prediction: torch.Tensor, target: torch.Tensor, prediction_was_activated: bool,
                             threshold=0.5):
    """
    Args:
        prediction: (N,)
        target: (N,)
        prediction_was_activated:
        threshold
    """
    assert len(prediction.shape) == len(target.shape) == 1
    if not prediction_was_activated:
        prediction = sigmoid(prediction)

    precision, recall = precision_recall(prediction, target, threshold=threshold)
    return precision.item(), recall.item()


@torch.no_grad()
def correct_point_cloud(fg_: torch.Tensor, inst_motion_prob: torch.Tensor, local_tf: torch.Tensor, meta_dict: dict):
    """
    Args:
        fg: (N_fg, 7[+2]) - batch_idx x, y, z, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]
        inst_motion_prob: (N_inst,) activated motion probability
        local_tf: (N_local, 3, 4) - local rotation | local translation
        meta_dict: == PointAligner.forward_return_dict['meta']
    Return:
        corrected_fg: (N_fg, 7[+2]) - batch_idx x, y, z, intensity, time, sweep_idx, [inst_idx, aug_inst_idx]
        fg_motion_mask: (N_fg,) - True for foreground points of moving instance
    """

    # get motion mask of foreground to find dynamic foreground
    inst_motion_mask = inst_motion_prob > 0.5
    inst_bi_inv_indices = meta_dict['inst_bi_inv_indices']  # (N_fg)
    fg_motion_mask = inst_motion_mask[inst_bi_inv_indices] == 1  # (N_fg)

    if torch.any(fg_motion_mask):
        # only perform correction if there are some dynamic foreground points
        dyn_fg = fg_[fg_motion_mask, 1: 4]  # (N_dyn, 3)

        # pair local_tf to foreground
        fg_tf = local_tf[meta_dict['local_bisw_inv_indices']]  # (N_fg, 3, 4)

        # extract local_tf of dyn_fg
        dyn_fg_tf = fg_tf[fg_motion_mask]  # (N_dyn, 3, 4)

        # apply transformation on dyn_fg
        dyn_fg = torch.matmul(dyn_fg_tf[:, :, :3], dyn_fg.unsqueeze(-1)).squeeze(-1) + dyn_fg_tf[..., -1]  # (N_dyn, 3)

        # overwrite coordinate of dynamic foreground with dyn_fg (computed above)
        fg_[fg_motion_mask, 1: 4] = dyn_fg

    return fg_, fg_motion_mask


@torch.no_grad()
def decode_proposals(proposals: torch.Tensor, instances_most_recent_center: torch.Tensor) -> torch.Tensor:
    """
    Args:
        proposals: (N_inst, 8) - delta_x, delta_y, delta_z, dx, dy, dz, sin_yaw, cos_yaw
        instances_most_recent_center: (N_inst, 3) - x, y, z
    Returns:
        pred_boxes: (N_inst, 7) - c_x, c_y, c_z, dx, dy, dz, yaw
    """
    center = instances_most_recent_center + proposals[:, :3]
    size = proposals[:, 3: 6]
    yaw = torch.atan2(proposals[:, 6], proposals[:, 7])
    pred_boxes = torch.cat([center, size, yaw[:, None]], dim=1)  # (N_inst, 7)
    return pred_boxes


@torch.no_grad()
def generate_predicted_boxes(batch_size: int, proposals: torch.Tensor, fg_prob: torch.Tensor, meta_dict: dict) -> List[dict]:
    """
    Element i-th of the returned List represents predicted boxes for the sample i-th in the batch
    Args:
        batch_size:
        proposals: (N_inst, 8) - delta_x, delta_y, delta_z, dx, dy, dz, sin_yaw, cos_yaw
        fg_prob: (N_fg,)
        meta_dict: == PointAligner.forward_return_dict['meta']
    """
    pred_boxes = decode_proposals(proposals, meta_dict['inst_target_center'])  # (N_inst, 7)

    # compute boxes' score by averaging foreground score, using inst_bi_inv_indices
    pred_scores = torch_scatter.scatter_mean(fg_prob, meta_dict['inst_bi_inv_indices'], dim=0)  # (N_inst, )

    # separate pred_boxes according to batch index
    inst_bi = meta_dict['inst_bi']
    max_num_inst = meta_dict['max_num_inst']
    inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)

    out = []
    for b_idx in range(batch_size):
        cur_boxes = pred_boxes[inst_batch_idx == b_idx]
        cur_scores = pred_scores[inst_batch_idx == b_idx]
        cur_labels = cur_boxes.new_ones(cur_boxes.shape[0]).long()
        out.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    return out


@torch.no_grad()
def voxelize(points_coord: torch.Tensor, points_feat: torch.Tensor, points_score: torch.Tensor,
             voxel_size: torch.Tensor, point_cloud_range: torch.Tensor, return_xyz=True):
    """
    Args:
        points_coord: (N, 4) - batch_idx, x, y, z
        points_feat: (N, C)
        voxel_size: (3) - vx, vy, vz
        point_cloud_range: (6) - x_min, y_min, z_min, x_max, y_max, z_max
    """
    grid_size = torch.round((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).long()  # [size_x, size_y, size_z]
    points_vox_coord = torch.floor((points_coord[:, 1:] - point_cloud_range[:3]) / voxel_size).long()  # (N, 3)

    mask_valid_points = torch.logical_and(points_vox_coord >= 0, points_vox_coord < grid_size).all(dim=1)  # (N,)
    points_feat = points_feat[mask_valid_points]
    points_score = points_score[mask_valid_points]
    points_vox_coord = points_vox_coord[mask_valid_points]
    points_batch_idx = points_coord[mask_valid_points, 0].long()

    volume = grid_size[0] * grid_size[1] * grid_size[2]
    area_xy = grid_size[0] * grid_size[1]
    size_x = grid_size[0]
    points_merge_coord = (points_batch_idx * volume
                          + points_vox_coord[:, 2] * area_xy  # z
                          + points_vox_coord[:, 1] * size_x   # y
                          + points_vox_coord[:, 0])  # x
    unq_coord, inv = torch.unique(points_merge_coord, return_inverse=True)

    mean_feat = torch_scatter.scatter_mean(points_feat, inv, dim=0)
    max_score = torch_scatter.scatter_max(points_score, inv, dim=0)[0]

    voxels_coord = torch.stack((unq_coord // volume,  # batch_idx
                                (unq_coord % volume) // area_xy,  # z
                                (unq_coord % area_xy) // size_x,  # y
                                unq_coord % size_x,  # x
                                ), dim=1)
    if return_xyz:
        voxels_coord = voxels_coord[:, [0, 3, 2, 1]]

    return voxels_coord.contiguous(), mean_feat.contiguous(), max_score.contiguous()
