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
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
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
    bev_loss, tb_dict = model.backbone_2d.get_loss()
    print_dict(tb_dict)
    print(f'---\nbev_loss: {bev_loss}',)


def viz_inference_result():
    # NOTE: batch_size always = 1
    data_dict = torch.load('./_output/inference_result.pth', map_location=torch.device('cpu'))
    print_dict(data_dict)
    print_dict(data_dict['bev_pred_dict'])

    # ---
    # viz 3D
    # ---
    points = data_dict['points']  # (N, 7) - batch_idx, XYZ, C feats, indicator | torch.Tensor
    indicator = points[:, -1].int()
    pc = points.numpy()
    fgr_mask = (indicator > -1).numpy()
    boxes = viz_boxes(data_dict['gt_boxes'][0].numpy())
    show_pointcloud(pc[:, 1: 4], boxes, fgr_mask=fgr_mask)

    bev_cls_pred = sigmoid(data_dict['bev_pred_dict']['bev_cls_pred'])[0].detach()  # (1, 512, 512)
    bev_reg_pred = data_dict['bev_pred_dict']['bev_reg_pred'][0].detach()  # (2, 512, 512)

    target_dict = data_dict['bev_target_dict']
    target_cls = target_dict['bev_cls_label'][0]  # (H, W)

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

    plt.show()

    mask_observed = target_cls > -1
    for threshold in [0.3, 0.5, 0.7]:
        print('---')
        print(f'threshold: {threshold}')
        stats_dict = compute_cls_stats(bev_cls_pred[0][mask_observed], target_cls[mask_observed], threshold)
        print_dict(stats_dict)
        print('---\n')


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
    print(model)
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
    plt.show()


if __name__ == '__main__':
    cfg_file = './from_idris/_cbgs_dyn_pp_centerpoint.yaml'
    pretrained_model = './from_idris/ckpt/bev_seg_focal_fullnusc_ep5.pth'
    # inference(cfg_file, pretrained_model)
    viz_inference_result()
    # eval_model(cfg_file, pretrained_model)
