import os
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from _dev_space.viz_tools import print_dict, viz_boxes
from _dev_space.tools_box import show_pointcloud
from _dev_space.bev_segmentation_utils import sigmoid
import matplotlib.pyplot as plt


torch.cuda.empty_cache()
np.random.seed(666)


@torch.no_grad()
def make_prediction(sample_idx, is_training, use_nusc_mini=False):
    cfg_file = './cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml'
    ckpt = './pretrained_models/pp_nusc4th_ep10.pth'

    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('../output/demo/demo_log.txt')
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        'random_world_flip', 'random_world_rotation', 'random_world_scaling',
        # 'gt_sampling',
    ]
    if use_nusc_mini:
        cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    cfg.DATA_CONFIG.USE_MINI_TRAINVAL = False

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                              dist=False, logger=logger, training=is_training, total_epochs=1, seed=666,
                                              workers=1)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()

    iter_dataloader = iter(dataloader)
    for _ in range(sample_idx + 1):  # 5
        data_dict = next(iter_dataloader)

    load_data_to_gpu(data_dict)

    data_dict['_points_before'] = torch.clone(data_dict['points'])

    model.eval()
    pred_dicts, recall_dicts = model(data_dict)
    data_dict['pred_dicts'] = pred_dicts

    filename = f'../output/demo/{cfg.DATA_CONFIG.VERSION}_sampleIdx_{sample_idx}_PPillars_data+pred.pth'
    torch.save(data_dict, filename)


def show_prediction(use_mini, sample_idx, is_training=False):
    nusc_version = 'v1.0-trainval' if not use_mini else 'v1.0-mini'
    filename = f'../output/demo/{nusc_version}_sampleIdx_{sample_idx}_PPillars_data+pred.pth'
    if not os.path.exists(filename):
        make_prediction(sample_idx, is_training, use_mini)

    data_dict = torch.load(filename, map_location=torch.device('cpu'))
    print_dict(data_dict)
    points = data_dict['points'].numpy()
    _points_before = data_dict['_points_before'].numpy()
    pred_dict = data_dict['pred_dicts'][0]
    gt_boxes = viz_boxes(data_dict['gt_boxes'][0].numpy())
    pred_boxes = pred_dict['pred_boxes'].numpy()  # (N_b, 9): cx,cy,cz,dx,dy,dz,yaw,vx,vy
    pred_scores = pred_dict['pred_scores'].numpy()  # (N_b)
    pred_labels = pred_dict['pred_labels'].numpy().astype(int) - 1  # (N_b)
    classes = np.array(['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'])
    cls_colors = plt.cm.rainbow(np.linspace(0, 1., len(classes)))[:, :3]
    pred_classes = classes[pred_labels]
    pred_colors = cls_colors[pred_labels]
    box_disp_mask = (pred_classes == 'car') | (pred_classes == 'pedestrian') | (pred_classes == 'bus')
    disp_pred_boxes = viz_boxes(pred_boxes[box_disp_mask, :-2])
    disp_pred_boxes_colors = pred_colors[box_disp_mask]

    print('showing point cloud before correction with ground truth boxes and foreground points')
    show_pointcloud(_points_before[:, 1: 4], gt_boxes, fgr_mask=_points_before[:, -1] > -1)
    print('---')

    print('showing point cloud after correction with predicted boxes and predicted foreground points')
    show_pointcloud(points[:, 1: 4], disp_pred_boxes, fgr_mask=points[:, 8] > 0.5, boxes_color=disp_pred_boxes_colors)
    print('---')


if __name__ == '__main__':
    show_prediction(use_mini=True, sample_idx=5, is_training=False)
