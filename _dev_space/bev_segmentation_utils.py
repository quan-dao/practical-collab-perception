import torch
import numpy as np
from torch_scatter import scatter_mean
from tqdm import tqdm

from _dev_space.tools_box import compute_bev_coord_torch


@torch.no_grad()
def assign_target_foreground_seg(data_dict, input_stride) -> dict:
    points = data_dict['points']  # (N, 1+3+C+1) - batch_idx, XYZ, C feats, indicator (-1 bgr, >=0 inst idx)
    indicator = points[:, -1].int()
    pc_range = torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=torch.float, device=indicator.device)
    pix_size = 0.2 * input_stride
    bev_size = torch.floor((pc_range[3: 5] - pc_range[0: 2]) / pix_size).int()

    # bgr_pix_coord, _ = compute_bev_coord_torch(points[indicator == -1], pc_range, pix_size)

    # compute cluster mean with awareness of batch idx
    max_num_inst = torch.max(points[:, -1]) + 1
    mask_fgr = indicator > -1  # (N,)
    merge_batch_and_inst_idx = points[mask_fgr, 0] * max_num_inst + indicator[mask_fgr]  # (N_fgr,) | N_fgr < N
    unq_merge, inv_indices = torch.unique(merge_batch_and_inst_idx, return_inverse=True)  # (N_unq)
    clusters_mean = scatter_mean(points[mask_fgr, 1: 3], inv_indices, dim=0)  # (N_unq, 2)
    # compute offset from each fgr point to its cluster's mean
    fgr_to_cluster_mean = (clusters_mean[inv_indices] - points[mask_fgr, 1: 3])  # (N_fgr, 2)

    fgr_reg_target = torch.cat((fgr_to_cluster_mean, points[mask_fgr, 6: 8]), dim=1)
    fgr_pix_coord, fgr_to_mean = compute_bev_coord_torch(points[mask_fgr], pc_range, pix_size, fgr_reg_target)

    # format output
    bev_cls_label = points.new_zeros(data_dict['batch_size'], bev_size[1], bev_size[0])
    # bev_cls_label[bgr_pix_coord[:, 0], bgr_pix_coord[:, 2], bgr_pix_coord[:, 1]] = 0
    bev_cls_label[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = 1

    bev_reg_label = points.new_zeros(4, data_dict['batch_size'], bev_size[1], bev_size[0])
    # ATTENTION: bev_reg_label is (C, B, H, W), not (B, C, H, W)
    for idx_offset in range(4):
        bev_reg_label[idx_offset, fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = \
            fgr_to_mean[:, idx_offset]

    target_dict = {'bev_cls_label': bev_cls_label, 'bev_reg_label': bev_reg_label}
    return target_dict


@torch.no_grad()
def compute_cls_stats(pred: torch.Tensor, label: torch.Tensor, threshold: float):
    # https://www.researchgate.net/figure/Calculation-of-Precision-Recall-and-Accuracy-in-the-confusion-matrix_fig3_336402347
    assert len(pred.shape) == 1 and len(label.shape) == 1 and pred.shape[0] == label.shape[0]
    # pred's shape: (N_obs,)

    # partition label into True & False
    label_pos = label > 0  # (N_obs,)
    label_neg = ~label_pos

    # partition prediction into Pos & Neg
    pred_pos = pred > threshold
    pred_neg = ~pred_pos

    # stats
    true_positive = (label_pos & pred_pos).int().sum().item()
    false_postive = (label_neg & pred_pos).int().sum().item()
    false_negative = (label_pos & pred_neg).int().sum().item()  # TODO: something not right here

    if true_positive + false_postive > 0:
        precision = float(true_positive) / float(true_positive + false_postive)
    else:
        precision = 0.

    if true_positive + false_negative > 0:
        recall = float(true_positive) / float(true_positive + false_negative)
    else:
        recall = 0.

    if precision + recall > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


@torch.no_grad()
def eval_segmentation(model, dataloader, threshold_list=None):
    model.eval()

    stats_dict = dict()
    if threshold_list is None:
        threshold_list = [0.3, 0.5, 0.7]

    for threshold in threshold_list:
        for stat in ['precision', 'recall', 'f1_score']:
            stats_dict[f"{stat}_{threshold}"] = 0.

    for data_dict in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        load_data_to_gpu(data_dict)
        # forward pass
        for cur_module in model.module_list:
            data_dict = cur_module(data_dict)

        # generate target
        target_dict = data_dict['bev_target_dict']
        target_cls = target_dict['bev_cls_label']  # (B, H, W)

        # compute stats
        pred_dict = data_dict['bev_pred_dict']
        pred_cls = sigmoid(pred_dict['bev_cls_pred'].squeeze(1).contiguous())  # (B, H, W)
        for threshold in threshold_list:
            stats = compute_cls_stats(pred_cls.reshape(-1), target_cls.reshape(-1), threshold)
            for stat, v in stats.items():
                stats_dict[f"{stat}_{threshold}"] += v

    for stat, v in stats_dict.items():
        stats_dict[stat] = v / len(dataloader)

    return stats_dict


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


