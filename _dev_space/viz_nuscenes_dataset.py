import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord, show_pointcloud, get_nuscenes_sensor_pose_in_ego_vehicle, apply_tf
from viz_tools import viz_boxes, print_dict, show_image_
import argparse


np.random.seed(666)


def modify_dataset_cfg(cfg, use_hd_map=False):
    cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        # 'random_world_flip',
        # 'random_world_scaling',
        'gt_sampling',
        # 'random_world_rotation',
    ]

    cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx',
                                                    'instance_idx',
                                                    'aug_instance_idx']
    cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                                   'aug_instance_idx']
    cfg.VERSION = 'v1.0-mini'
    if use_hd_map:
        cfg.USE_HD_MAP = True
        cfg.NORMALIZE_LANE_ANGLE = True


def main(**kwargs):
    cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    modify_dataset_cfg(cfg, use_hd_map=kwargs.get('use_hd_map', False))
    logger = common_utils.create_logger('./dummy_log.txt')
    nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
    data_dict = nuscenes_dataset[200]  # 400, 200, 100, 5, 10
    print_dict(data_dict)
    print('meta: ', data_dict['metadata'])

    pc = data_dict['points']
    gt_boxes = viz_boxes(data_dict['gt_boxes'])

    if kwargs.get('show_raw_pointcloud', False):
        print('-----------------------\n'
              'showing EMC accumulated pointcloud')
        show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -2] > -1)

    if kwargs.get('show_oracle_pointcloud', False):
        raise NotImplementedError

    if kwargs.get('show_bev', False):
        pc_range = np.array(cfg.POINT_CLOUD_RANGE)
        bev_img_size = int((pc_range[3] - pc_range[0]) / cfg.get('BEV_IMAGE_RESOLUTION', 0.2))
        xlims = ylims = (0, bev_img_size)

        # ---
        # point cloud to BEV
        # ---
        mask_inside = np.all((pc[:, :3] >= pc_range[:3]) & (pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
        pc = pc[mask_inside]

        bev_coord, bev_feat = compute_bev_coord(pc[:, :3], pc_range=pc_range,
                                                bev_pix_size=cfg.get('BEV_IMAGE_RESOLUTION', 0.2),
                                                pts_feat=pc[:, 3].reshape(-1, 1))

        # ---
        # gt boxes to BEV
        # ---
        boxes_in_bev = [(box[:, :2] - pc_range[:2]) / cfg.get('BEV_IMAGE_RESOLUTION', 0.2) for box in gt_boxes]

        # ---
        # drawing
        # ---

        # Intensity
        fig, ax = plt.subplots()
        img_intesity = np.zeros((bev_img_size, bev_img_size))
        img_intesity[bev_coord[:, 1], bev_coord[:, 0]] = bev_feat[:, 0]
        img_intesity = img_intesity / np.max(img_intesity)
        show_image_(ax, img_intesity, 'intensity', xlims, ylims, boxes_in_bev, cmap='gray')

        # Drivable area & lane
        if kwargs.get('use_hd_map', False):
            img_map = data_dict['img_map']
            fig2, ax2 = plt.subplots(1, 2)
            for ax_id, (layer_name, layer_idx) in enumerate(zip(['drivable_are', 'lanes'], [0, -1])):
                show_image_(ax2[ax_id], img_map[layer_idx], layer_name, xlims, ylims, boxes_in_bev)

        # cam_front
        fig3, ax3 = plt.subplots()
        sample_rec = nuscenes_dataset.nusc.get('sample', data_dict['metadata']['token'])
        nuscenes_dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'], ax=ax3)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--show_raw_pointcloud', action='store_true', default=False)
    parser.add_argument('--show_oracle_pointcloud', action='store_true', default=False)
    parser.add_argument('--show_bev', action='store_true', default=False)
    parser.add_argument('--use_hd_map', action='store_true', default=False)

    args = parser.parse_args()
    main(**vars(args))
