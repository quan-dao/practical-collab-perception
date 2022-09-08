import torch
import numpy as np
from torch_scatter import scatter_mean
from tqdm import tqdm
from pcdet.utils.transform_utils import bin_depths
from _dev_space.tools_box import compute_bev_coord_torch
from _dev_space.viz_tools import print_dict


@torch.no_grad()
def assign_target_foreground_seg(data_dict, input_stride, crt_mag_max=15., crt_num_bins=40, crt_dir_num_bins=80) -> dict:
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
    fgr_to_cluster_mean = (clusters_mean[inv_indices] - points[mask_fgr, 1: 3]) / pix_size  # (N_fgr, 2)

    # scatter regression target to pixels
    fgr_reg_target = torch.cat((fgr_to_cluster_mean,  # measured in pixels
                                points[mask_fgr, 6: 8]),  # measured in meters
                               dim=1)
    fgr_pix_coord, fgr_reg_label = compute_bev_coord_torch(points[mask_fgr], pc_range, pix_size, fgr_reg_target)
    # fgr_reg_label: (N_fgr, 4)

    # format output
    bev_cls_label = points.new_zeros(data_dict['batch_size'], bev_size[1], bev_size[0])
    bev_cls_label[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = 1

    bev_reg2mean_label = points.new_zeros(2, data_dict['batch_size'], bev_size[1], bev_size[0])
    # ATTENTION: bev_reg_label is (C, B, H, W), not (B, C, H, W)
    for idx_offset in range(2):
        bev_reg2mean_label[idx_offset, fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = \
            fgr_reg_label[:, idx_offset]

    # ---
    # label for regression to crt
    # ---
    crt_mag = torch.linalg.norm(fgr_reg_label[:, 2:], dim=1, keepdim=True)  # (N_fgr, 1)
    crt_class = bin_depths(crt_mag, mode='LID', depth_min=0., depth_max=crt_mag_max, num_bins=crt_num_bins,
                           target=True)  # (N_fgr, 1)
    bev_crt_class = crt_class.new_zeros(data_dict['batch_size'], bev_size[1], bev_size[0])  # (B, H, W) - contain cls index
    bev_crt_class[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = crt_class.squeeze(1)

    crt_angle = torch.atan2(fgr_reg_label[:, 3], fgr_reg_label[:, 2]).unsqueeze(1)  # (N_fgr, 1)
    crt_dir_class = bin_depths(crt_angle, mode='UD', depth_min=-np.pi, depth_max=np.pi - 1e-3,
                               num_bins=crt_dir_num_bins, target=True)  # (N_fgr, 1)

    bev_crt_dir_cls = crt_dir_class.new_zeros(data_dict['batch_size'], bev_size[1], bev_size[0])
    bev_crt_dir_cls[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = crt_dir_class.squeeze(1)

    crt_dir_residue = crt_angle - (2 * np.pi - 1e-3) * crt_dir_class / crt_dir_num_bins
    bev_crt_dir_res = crt_dir_residue.new_zeros(data_dict['batch_size'], bev_size[1], bev_size[0])
    bev_crt_dir_res[fgr_pix_coord[:, 0], fgr_pix_coord[:, 2], fgr_pix_coord[:, 1]] = crt_dir_residue.squeeze(1)

    target_dict = {'target_cls': bev_cls_label, 'target_to_mean': bev_reg2mean_label,
                   'target_crt_cls': bev_crt_class,
                   'target_crt_dir_cls': bev_crt_dir_cls, 'target_crt_dir_res': bev_crt_dir_res}
    return target_dict


@torch.no_grad()
def compute_cls_stats(pred: torch.Tensor, label: torch.Tensor, threshold: float, return_counts=False):
    # https://www.researchgate.net/figure/Calculation-of-Precision-Recall-and-Accuracy-in-the-confusion-matrix_fig3_336402347
    assert len(pred.shape) == 1 and len(label.shape) == 1 and pred.shape[0] == label.shape[0]
    # pred's shape: (N_obs,)

    # partition label into True & False
    label_pos = label > 0  # (N_obs,)
    label_neg = ~label_pos
    # print('---\n', f'num gt positive: {label_pos.int().sum()}')

    # partition prediction into Pos & Neg
    pred_pos = pred > threshold
    pred_neg = ~pred_pos

    # stats
    true_positive = (label_pos & pred_pos).int().sum().item()
    false_positive = (label_neg & pred_pos).int().sum().item()
    false_negative = (label_pos & pred_neg).int().sum().item()  # TODO: something not right here
    if return_counts:
        return {'true_positive': true_positive, 'false_positive': false_positive, 'false_negative': false_negative}

    if true_positive + false_positive > 0:
        precision = float(true_positive) / float(true_positive + false_positive)
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

    counts_dict = dict()
    if threshold_list is None:
        threshold_list = [0.3, 0.5, 0.7]

    for threshold in threshold_list:
        for cnt_type in ['true_positive', 'false_positive', 'false_negative']:
            counts_dict[f"{cnt_type}_{threshold}"] = 0

    for data_dict in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True):
        load_data_to_gpu(data_dict)
        # forward pass
        for cur_module in model.module_list:
            data_dict = cur_module(data_dict)

        # generate target
        target_dict = data_dict['bev_target_dict']
        target_cls = target_dict['target_cls']  # (B, H, W)

        # compute stats
        pred_dict = data_dict['bev_pred_dict']
        pred_cls = sigmoid(pred_dict['pred_cls'].squeeze(1).contiguous())  # (B, H, W)
        for threshold in threshold_list:
            cnt_dict = compute_cls_stats(pred_cls.reshape(-1), target_cls.reshape(-1), threshold, return_counts=True)

            for cnt_type, cnt in cnt_dict.items():
                counts_dict[f"{cnt_type}_{threshold}"] += cnt

    stats_dict = dict()
    for threshold in threshold_list:
        tp = counts_dict[f"true_positive_{threshold}"]
        fp = counts_dict[f"false_positive_{threshold}"]
        fn = counts_dict[f"false_negative_{threshold}"]

        precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.
        recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.
        f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.

        # log stats
        stats_dict[f"precision_{threshold}"] = precision
        stats_dict[f"recall_{threshold}"] = recall
        stats_dict[f"f1_score_{threshold}"] = f1_score

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


