import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu

import matplotlib.pyplot as plt
from _dev_space.viz_tools import viz_boxes, print_dict
from _dev_space.tools_box import show_pointcloud
from _dev_space.bev_segmentation_utils import sigmoid, eval_segmentation, compute_cls_stats


def numpy2torch_(data_dict):
    for k, v in data_dict.items():
        if k in ['frame_id', 'metadata']:
            continue
        elif isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v)


@torch.no_grad()
def inference(cfg_file, pretrained_model):
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')
    # ---
    # adjust dataset to use NuScenes Mini
    # ---
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        'random_world_flip', 'random_world_rotation', 'random_world_scaling',
        'gt_sampling',
    ]
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    cfg.DATA_CONFIG.USE_MINI_TRAINVAL = False
    print(cfg)
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666,
                                              workers=1)
    iter_dataloader = iter(dataloader)
    for _ in range(3):
        data_dict = next(iter_dataloader)
    load_data_to_gpu(data_dict)
    # numpy2torch_(data_dict)
    print_dict(data_dict)

    # ---
    # model
    # ---
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    print(model)
    model.cuda()
    model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)

    # ---
    # inference
    # ---
    model.eval()
    for cur_module in model.module_list:
        data_dict = cur_module(data_dict)

    torch.save(data_dict, './_output/inference_result.pth')
    # bev_loss, tb_dict = model.backbone_2d.get_loss()
    # print_dict(tb_dict)
    # print(f'---\nbev_loss: {bev_loss}',)


def viz_inference_result():
    batch_idx = 1
    data_dict = torch.load('./_output/inference_result.pth', map_location=torch.device('cpu'))
    print_dict(data_dict)
    print_dict(data_dict['bev_pred_dict'])

    # ---
    # viz 3D
    # ---
    points = data_dict['points']  # (N, 7) - batch_idx, XYZ, C feats, indicator | torch.Tensor

    pc = points.numpy()
    batch_mask = pc[:, 0].astype(int) == batch_idx
    indicator = pc[:, -1].astype(int)
    fgr_mask = indicator > -1
    boxes = viz_boxes(data_dict['gt_boxes'][batch_idx].numpy())

    print('showing EMC accumulated pointcloud')
    show_pointcloud(pc[batch_mask, 1: 4], boxes, fgr_mask=fgr_mask[batch_mask])

    bev_cls_pred = sigmoid(data_dict['bev_pred_dict']['pred_cls'])[batch_idx].detach()  # (1, 256, 256)
    bev_reg_pred = data_dict['bev_pred_dict']['pred_to_mean'][batch_idx].detach()  # (2, 256, 256)
    bev_crt_cls = data_dict['bev_pred_dict']['pred_crt_cls'][batch_idx].detach().numpy()  # (D+1, 256, 256)
    bev_crt_dir_cls = data_dict['bev_pred_dict']['pred_crt_dir_cls'][batch_idx].detach().numpy()  # (D'+1, 256, 256)
    bev_crt_dir_res = data_dict['bev_pred_dict']['pred_crt_dir_res'][batch_idx].detach().numpy()  # (1, 256, 256)

    target_dict = data_dict['bev_target_dict']
    target_cls = target_dict['target_cls'][batch_idx]  # (H, W)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(bev_cls_pred[0].numpy(), cmap='gray')
    ax[0].set_title('fgr prob')

    ax[1].imshow(target_cls.numpy(), cmap='gray')
    ax[1].set_title('ground truth')

    # --
    threshold = 0.5
    bev_size = bev_cls_pred.shape[1]
    pred_bev_fgr = bev_cls_pred[0] > threshold
    xx, yy = np.meshgrid(np.arange(bev_size), np.arange(bev_size))
    fgr_x = xx[pred_bev_fgr]
    fgr_y = yy[pred_bev_fgr]
    fgr_to_mean = bev_reg_pred[:, pred_bev_fgr]
    for fidx in range(fgr_to_mean.shape[1]):
        ax[2].arrow(fgr_x[fidx], fgr_y[fidx], fgr_to_mean[0, fidx], fgr_to_mean[1, fidx], color='g', width=0.01)

    ax[2].imshow(bev_cls_pred[0].numpy() > threshold, cmap='gray')
    ax[2].set_title('predicted fgr')

    fig.tight_layout()
    plt.show()

    print('showing oracle accumulated pointcloud')
    oracle_pc = np.copy(pc)
    oracle_pc[:, 1: 3] += oracle_pc[:, 6: 8]
    show_pointcloud(oracle_pc[batch_mask, 1: 4], boxes, fgr_mask=fgr_mask[batch_mask])

    print('showing corrected pointcloud')

    pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    pix_size = 0.4
    num_bins = 40
    crt_dir_num_bins = 80
    pc_pix_coords = np.floor((pc[batch_mask, 1: 3] - pc_range[:2]) / pix_size).astype(int)  # (N, 2)

    pred_fgr_img = bev_cls_pred[0].numpy()  # (256, 256)
    pc_fgr_prob = pred_fgr_img[pc_pix_coords[:, 1], pc_pix_coords[:, 0]]
    pc_fgr_mask = pc_fgr_prob > threshold  # batch_mask >> pc_fgr_mask

    # mag
    pc_crt_cls_prob = bev_crt_cls[:, pc_pix_coords[pc_fgr_mask, 1], pc_pix_coords[pc_fgr_mask, 0]].T  # (N, D+1)
    pc_crt_cls = np.argmax(pc_crt_cls_prob, axis=1)
    mask_invalid_crt_cls = pc_crt_cls == num_bins
    pc_crt_mag = (15. / (40 * 41)) * (pc_crt_cls * (pc_crt_cls + 1))
    pc_crt_mag[mask_invalid_crt_cls] = 0

    # dir
    pc_crt_dir_cls_prob = bev_crt_dir_cls[:, pc_pix_coords[pc_fgr_mask, 1], pc_pix_coords[pc_fgr_mask, 0]] .T  # (N, D'+1)
    pc_crt_dir_cls = np.argmax(pc_crt_dir_cls_prob, axis=1)

    pc_crt_dir_res = bev_crt_dir_res[0, pc_pix_coords[pc_fgr_mask, 1], pc_pix_coords[pc_fgr_mask, 0]]  # (N)
    pc_crt_dir = pc_crt_dir_res + (2 * np.pi - 1e-3) * pc_crt_dir_cls / crt_dir_num_bins

    # correct fgr
    curr_pc = pc[batch_mask, 1: 4]
    bgr = curr_pc[np.logical_not(pc_fgr_mask)]

    fgr = curr_pc[pc_fgr_mask]
    crt_fgr = np.copy(fgr)
    crt_fgr[:, :2] = fgr[:, :2] + pc_crt_mag[:, None] * np.stack([np.cos(pc_crt_dir), np.sin(pc_crt_dir)], axis=1)

    _fgr_mask = np.zeros(curr_pc.shape[0], dtype=bool)
    _fgr_mask[bgr.shape[0]:] = True
    show_pointcloud(np.concatenate([bgr, crt_fgr], axis=0), boxes, fgr_mask=_fgr_mask)


def eval_model(cfg_file, pretrained_model):
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')
    # ---
    # adjust dataset to use NuScenes Mini
    # ---
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        'random_world_flip', 'random_world_rotation', 'random_world_scaling',
        'gt_sampling',
    ]
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    cfg.DATA_CONFIG.USE_MINI_TRAINVAL = False

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666,
                                              workers=1)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    # print(model)
    model.cuda()
    model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)

    threshold_list = np.arange(0.1, 1.0, 0.1).tolist()
    stats_dict = eval_segmentation(model, dataloader, threshold_list=threshold_list)
    # print_dict(stats_dict)

    _stats = {'precision': [], 'recall': [], 'f1_score': []}
    for threshold in threshold_list:
        for stat_type in _stats.keys():
            _stats[stat_type].append(stats_dict[f"{stat_type}_{threshold}"])

    fig, ax = plt.subplots(1, 3)
    idx_ax = 0
    colors = ['r', 'g', 'b']
    for stat_type, stat in _stats.items():
        ax[idx_ax].plot(threshold_list, stat, color=colors[idx_ax])
        ax[idx_ax].set_xlabel('threshold')
        ax[idx_ax].set_ylabel(stat_type)
        idx_ax += 1

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    cfg_file = './from_idris/_cbgs_dyn_pp_centerpoint.yaml'
    pretrained_model = './from_idris/ckpt/bev_seg_caddn_style_fullnusc_ep5.pth'
    # inference(cfg_file, pretrained_model)
    # viz_inference_result()
    eval_model(cfg_file, pretrained_model)
