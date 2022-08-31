import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset, build_dataloader
from pcdet.models import load_data_to_gpu

from _dev_space.revise_cam import PointCloudCorrector
from _dev_space.viz_tools import viz_boxes, print_dict, viz_clusters2d
from _dev_space.tools_box import show_pointcloud
from _dev_space.bev_segmentation_utils import sigmoid

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


np.random.seed(666)
logger = common_utils.create_logger('./dummy_log.txt')

cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
cfg.VERSION = 'v1.0-mini'
cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    # 'random_world_flip', 'random_world_rotation', 'random_world_scaling',
    # 'gt_sampling',
]


def main(data_batch_idx, viz_batch_idx=0):
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                              logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    data_dict = None
    for _ in range(data_batch_idx):
        data_dict = next(iter_dataloader)

    print('----------------\n')
    print('showing original pointcloud & boxes')
    boxes = viz_boxes(data_dict['gt_boxes'][viz_batch_idx])
    points = data_dict['points'][data_dict['points'][:, 0].astype(int) == viz_batch_idx]
    # (N, 7): batch_idx, xyz, intensity, time, indicator
    show_pointcloud(points[:, 1: 4], boxes, fgr_mask=points[:, -1].astype(int) == 0)
    print('----------------\n')

    load_data_to_gpu(data_dict)

    corrector = PointCloudCorrector()
    # print(corrector)
    print('----------\n', 'invoking corrector')
    data_dict = corrector(data_dict)
    torch.save(data_dict, './_output/revise_cam_corrector_out.pth')
    print_dict(data_dict)


def show_corrected_pointcloud(chosen_batch_idx=0, show_foreground_prob=False, show_offset=False):
    data_dict = torch.load('./_output/revise_cam_corrector_out.pth', map_location=torch.device('cpu'))
    print('----------------\n')
    print('showing corrected pointcloud & boxes')
    boxes = viz_boxes(data_dict['gt_boxes'][chosen_batch_idx])
    points = data_dict['points'][data_dict['points'][:, 0].int() == chosen_batch_idx].numpy()
    # (N, 7 [+3]): batch_idx, xyz, intensity, time, [foreground_prob, offset_xy], indicator
    if not show_foreground_prob:
        show_pointcloud(points[:, 1: 4], boxes, fgr_mask=points[:, -1].int() == 0)
    else:
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = points[:, 6]
        show_pointcloud(points[:, 1: 4], boxes, pc_colors=colors, fgr_mask=points[:, 6] > 0.5, fgr_offset=points[:, 7: 9])


def show_bev_seg(batch_idx=0, threshold=0.5):
    data_dict = torch.load('./_output/revise_cam_corrector_out.pth', map_location=torch.device('cpu'))
    print_dict(data_dict['bev_pred_dict'])
    print('---------')
    print_dict(data_dict['bev_target_dict'])

    pred_dict = data_dict['bev_pred_dict']
    pred_cls = pred_dict['bev_cls_pred'][batch_idx]  # (1, 256, 256) | stride=2
    pred_cls = sigmoid(pred_cls[0])  # (256, 256)
    pred_reg = pred_dict['bev_reg_pred'][batch_idx]  # (2, 256, 256)

    target_dict = data_dict['bev_target_dict']
    target_cls = target_dict['bev_cls_label'][batch_idx]  # (256, 256)

    fig, ax = plt.subplots(1, 4)
    for _ax in ax:
        _ax.set_aspect('equal')

    ax[0].set_title('ground truth')
    ax[0].imshow(target_cls.numpy(), cmap='gray')

    ax[1].set_title('fgr prob')
    ax[1].imshow(pred_cls.numpy(), cmap='gray')

    ax[2].set_title('predicted fgr')
    ax[2].imshow(pred_cls.numpy() > threshold, cmap='gray')
    pred_fgr = (pred_cls > threshold) & (torch.linalg.norm(pred_reg, dim=0) < 10)  # (256, 256)
    xx, yy = np.meshgrid(np.arange(pred_cls.shape[1]), np.arange(pred_cls.shape[0]))
    fgr_x = xx[pred_fgr]  # (N_fgr,)
    fgr_y = yy[pred_fgr]  # (N_fgr,)
    vector_fgr2centroid = pred_reg[:, pred_fgr]  # (2, N_fgr)
    for fidx in range(vector_fgr2centroid.shape[1]):
        ax[2].arrow(fgr_x[fidx], fgr_y[fidx], vector_fgr2centroid[0, fidx], vector_fgr2centroid[1, fidx],
                    color='g', width=0.01)

    # DBSCAN on pred_cls
    ax[3].set_title('clustering result')
    ax[3].imshow(pred_cls.numpy() > threshold, cmap='gray')
    fgr_xy = np.stack((fgr_x, fgr_y), axis=1)
    moved_fgr = fgr_xy + vector_fgr2centroid.numpy().T  # (N_fgr, 2)
    scanner = DBSCAN(eps=1.0, min_samples=3)
    scanner.fit(moved_fgr)
    viz_clusters2d(fgr_xy, scanner.labels_, ax[3], prefix='cl_')

    # mask_targeted_cl = scanner.labels_ == 1
    # vector_fgr2centroid = vector_fgr2centroid.numpy().T
    # cl_fgr2centroid = vector_fgr2centroid[mask_targeted_cl]
    # fig2, ax2 = plt.subplots()
    # counts, bins, _ = ax2.hist(np.linalg.norm(cl_fgr2centroid, axis=1))
    # print('counts: ', counts)
    # print('bins: ', bins)

    plt.show()


if __name__ == '__main__':
    chosen_batch_idx = 0
    main(data_batch_idx=2, viz_batch_idx=chosen_batch_idx)
    show_corrected_pointcloud(chosen_batch_idx, show_foreground_prob=True)

    # show_bev_seg(chosen_batch_idx)
