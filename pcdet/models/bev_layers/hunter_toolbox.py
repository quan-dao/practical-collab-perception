import torch
import torch.nn as nn
import torch_scatter
from einops import rearrange
from typing import List, Dict


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


def bev_scatter(points_bev_coord: torch.Tensor, points_batch_idx: torch.Tensor, points_feat: torch.Tensor,
                bev_img_size: tuple):
    """
    Args:
        points_bev_coord: (N, 2) - bev_x, bev_y
        points_batch_idx: (N,)
        points_feat: (N, C)
        bev_img_size: (height, width)
    """
    batch_size = torch.max(points_batch_idx).item() + 1
    num_channels = points_feat.shape[1]
    height, width = bev_img_size
    area = height * width

    # make sure points are in range
    mask_in = ((points_bev_coord[:, 0] > 0) & (points_bev_coord[:, 0] < width)
               & (points_bev_coord[:, 1] > 0) & (points_bev_coord[:, 1] < height))

    points_bev_coord = points_bev_coord[mask_in].long()  # (N, 2) - bev_x, bev_y
    points_merge = points_batch_idx[mask_in] * area + points_bev_coord[:, 1] * width + points_bev_coord[:, 0]

    unq, inv = torch.unique(points_merge, return_inverse=True)
    bev_img = points_feat.new_zeros(batch_size * height * width, num_channels)
    bev_img[unq] = torch_scatter.scatter_mean(points_feat[mask_in], inv, dim=0)

    bev_img = rearrange(bev_img, '(B H W) C -> B C H W', B=batch_size, H=height, W=width).contiguous()
    return bev_img


def interpolate_points_feat_from_bev_img(bev_img: torch.Tensor, 
                                         points: torch.Tensor, 
                                         point_cloud_range: torch.Tensor, 
                                         bev_pixel_size: torch.Tensor,
                                         return_bev_coord: bool = False) -> torch.Tensor:
    """
    Args:
        bev_img: (B, C, H, W)
        points: (N, 1 + 10 + 2) - batch_idx | x, y, z, intensity, time | map_feat (5) | sweep_idx, instance_idx
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_pixel_size: [pix_size_x, pix_size_y] = voxel_size * bev_img_stride
    
    Returns:
        points_feat: (N, C)
    """
    points_feat = bev_img.new_zeros(points.shape[0], bev_img.shape[1])

    points_bev_coord = (points[:, 1: 3] - point_cloud_range[:2]) / bev_pixel_size
    
    points_batch_idx = points[:, 0].long()

    batch_size = bev_img.shape[0]
    for b_idx in range(batch_size):
        _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
        batch_mask = points_batch_idx == b_idx
        cur_points_feat = bilinear_interpolate_torch(_img, 
                                                     points_bev_coord[batch_mask, 0], 
                                                     points_bev_coord[batch_mask, 1])
        points_feat[batch_mask] = cur_points_feat

    if return_bev_coord:
        return points_feat, points_bev_coord

    return points_feat


def nn_make_mlp(c_in: int, c_out: int, hidden_channels: List[int] = None, is_head: bool = True, use_drop_out: bool = False) -> nn.Module:
    if hidden_channels is None:
        hidden_channels = []

    channels = [c_in] + hidden_channels + [c_out]
    layers = []

    for c_idx in range(1, len(channels)):
        c_in = channels[c_idx - 1]
        c_out = channels[c_idx]
        if use_drop_out:
            layers.append(nn.Dropout(p=0.5))
        
        is_last = c_idx == len(channels) - 1
        
        if is_last:
            if is_head:
                layers.append(nn.Linear(c_in, c_out, bias=True))
            else:
                # mlp for feat encoding
                layers.append(nn.Linear(c_in, c_out, bias=False))
                layers.append(nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(True))    
        else:
            layers.append(nn.Linear(c_in, c_out, bias=False))
            layers.append(nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01))
            layers.append(nn.ReLU(True))
        
    return nn.Sequential(*layers)


def remove_gt_boxes_outside_range(batch_dict: Dict[str, torch.Tensor], point_cloud_range: torch.Tensor) -> None:
    valid_gt_boxes = list()
    gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, C)
    max_num_valid_boxes = 0
    for bidx in range(batch_dict['batch_size']):
        mask_in_range = torch.logical_and(gt_boxes[bidx, :, :3] >= point_cloud_range[:3], 
                                          gt_boxes[bidx, :, :3] < point_cloud_range[3:]).all(dim=1)
        valid_gt_boxes.append(gt_boxes[bidx, mask_in_range])
        max_num_valid_boxes = max(max_num_valid_boxes, valid_gt_boxes[-1].shape[0])

    batch_valid_gt_boxes = gt_boxes.new_zeros(batch_dict['batch_size'], max_num_valid_boxes, gt_boxes.shape[2])
    for bidx, valid_boxes in enumerate(valid_gt_boxes):
        batch_valid_gt_boxes[bidx, :valid_boxes.shape[0]] = valid_boxes

    batch_dict.pop('gt_boxes')
    batch_dict['gt_boxes'] = batch_valid_gt_boxes
    return batch_dict


def hard_mining_regression_loss(loss_all: torch.Tensor, 
                              mask_positive: torch.Tensor, 
                              device: torch.device, 
                              negative_to_positve_ratio: int = 1,
                              num_negtiave_when_no_positive: int = 100):
    """
    Args:
        loss_all: (N,)
        mask_postive: (N,)
        device:
        negative_to_positve_ratio:
        num_negtiave_when_no_positive:
    """
    num_positive = int(mask_positive.sum().item())
    if num_positive == 0:
        # all negative
        if num_negtiave_when_no_positive < loss_all.shape[0]:
            top_loss, _ = torch.topk(loss_all, k=num_negtiave_when_no_positive)
            return top_loss.mean()
        else:
            return loss_all.mean()

    loss_positive = loss_all[mask_positive].mean()

    num_negative = loss_all.shape[0] - num_positive
    if num_negative > 0:
        num_chosen_negative = min(num_positive * negative_to_positve_ratio, num_negative)
        # extract top "num_chosen_negative" loss
        if num_chosen_negative < num_negative:
            top_loss_negative, _ = torch.topk(loss_all[torch.logical_not(mask_positive)], k=num_chosen_negative)
        else:
            top_loss_negative = loss_all[torch.logical_not(mask_positive)]
        
        loss_negative = top_loss_negative.mean()
    else:
        # no negative
        loss_negative = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)

    loss = loss_positive + loss_negative
    return loss
