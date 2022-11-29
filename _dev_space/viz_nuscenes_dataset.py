import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset, build_dataloader
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord, show_pointcloud, get_nuscenes_sensor_pose_in_ego_vehicle, apply_tf
from viz_tools import viz_boxes, print_dict, show_image_
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import lovely_tensors as lt
from einops import rearrange


np.random.seed(666)
lt.monkey_patch()


def modify_dataset_cfg(cfg):
    cfg.CLASS_NAMES = ['car']

    cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        'random_world_flip',
        'random_world_scaling',
        'gt_sampling',
        'random_world_rotation',
    ]

    cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx',
                                                    'instance_idx',
                                                    'aug_instance_idx']
    cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                                   'aug_instance_idx']
    cfg.VERSION = 'v1.0-mini'

    cfg.USE_HD_MAP = True
    cfg.NORMALIZE_LANE_ANGLE = True

    cfg.POSSIBLE_NUM_SWEEPS = [10]
    cfg.DROP_BACKGROUND = edict({
        'ENABLE': True,
        'DISTANCE_THRESHOLD': 10,
        'DROP_PROB': 0.4
    })
    cfg.PRED_VELOCITY = False

    cfg.POINT_CLOUD_RANGE = [-60, -60, -5.0, 60, 60, 3.0]


def rot_z(yaw):
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        cos, -sin, 0,
        sin, cos, 0,
        0, 0, 1
    ]).reshape(3, 3)
    return rot


def xyyaw2pose(x, y, yaw):
    out = np.eye(4)
    out[:3, :3] = rot_z(yaw)
    out[:2, -1] = [x, y]
    return out


def main(**kwargs):
    cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    modify_dataset_cfg(cfg)
    logger = common_utils.create_logger('./dummy_log.txt')

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                              logger=logger, training=True, total_epochs=1, seed=666)
    batch_idx = 1

    if kwargs.get('viz_dataset', False):
        print('visualizing dataset')
        data_dict = dataset[200]  # 400, 200, 100, 5, 10
        pc = data_dict['points']
        gt_boxes = viz_boxes(data_dict['gt_boxes'])
        # convert waypoints to poses
        poses = [xyyaw2pose(*data_dict['instances_waypoints'][wp_idx].tolist()[:-1])
                 for wp_idx in range(data_dict['instances_waypoints'].shape[0])]
    else:
        assert kwargs.get('viz_dataloader', False)
        iter_dataloader = iter(dataloader)
        for _ in range(5):
            data_dict = next(iter_dataloader)

        points = data_dict['points']

        # 1 global group -> 1 color for points & box
        fg = points[points[:, -2].astype(int) > -1]  # (N_fg, 8)
        max_num_inst = data_dict['instances_tf'].shape[1]
        fg_bi = fg[:, 0].astype(int) * max_num_inst + fg[:, -2].astype(int)
        inst_bi, inst_bi_inv = np.unique(fg_bi, return_inverse=True)
        # inst_bi: (N_inst,) - does not necessary contain a contiguous sequence of number (i.e., [0, 3, 10, 15] )

        inst_colors = plt.cm.rainbow(np.linspace(0, 1, inst_bi.shape[0]))[:, :3]
        fg_colors = inst_colors[inst_bi_inv]  # (N_fg, 3)

        # extract current_fg, _target_boxes
        mask_cur_fg = fg[:, 0].astype(int) == batch_idx
        cur_fg = fg[mask_cur_fg]  # (N_cur_fg, 8)
        cur_fg_colors = fg_colors[mask_cur_fg]

        # the following 3 lines only correct if 1-to-1 correspondence between target_boxes & inst_bi
        gt_boxes = data_dict['gt_boxes']  # (B, N_inst_max, 7+...)
        gt_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        gt_boxes = gt_boxes[inst_bi]

        inst_batch_idx = inst_bi // max_num_inst  # (N_inst,)
        cur_target_boxes = gt_boxes[inst_batch_idx == batch_idx]  # (N_cur_inst, 7+...)
        cur_target_boxes_colors = inst_colors[inst_batch_idx == batch_idx]

        bg = points[points[:, -2].astype(int) == -1]  # (N_fg, 8)
        cur_bg = bg[bg[:, 0].astype(int) == batch_idx]
        cur_bg_colors = np.zeros((cur_bg.shape[0], 3))

        pc = np.concatenate([cur_bg, cur_fg])[:, 1: 4]
        pc_colors= np.concatenate([cur_bg_colors, cur_fg_colors])
        gt_boxes = viz_boxes(cur_target_boxes)

        batch_inst_wpts = data_dict['instances_waypoints']
        cur_inst_wpts = batch_inst_wpts[batch_inst_wpts[:, 0].astype(int) == batch_idx]
        print('---\n cur_inst_wpts inst idx:\n', np.unique(cur_inst_wpts[:, -1].astype(int)), '\n---')
        print('---\n cur_fg inst idx:\n', np.unique(cur_fg[:, -2].astype(int)), '\n---')
        poses = [xyyaw2pose(*cur_inst_wpts[wp_idx].tolist()[1:-1]) for wp_idx in range(cur_inst_wpts.shape[0])]

    print_dict(data_dict)
    print('meta: ', data_dict['metadata'])
    print('data_dict[instance_future_waypoints]:\n', data_dict['instances_waypoints'])

    if kwargs.get('show_raw_pointcloud', False):
        print('-----------------------\n'
              'showing EMC accumulated pointcloud')
        # dataset.nusc.render_sample(data_dict['metadata']['token'])
        # plt.show()
        if kwargs.get('viz_dataset', False):
            show_pointcloud(pc[:, :3], boxes=gt_boxes, poses=poses)
        else:
            show_pointcloud(pc, pc_colors=pc_colors, boxes=gt_boxes, boxes_color=cur_target_boxes_colors, poses=poses)

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
            if kwargs.get('viz_dataloader', False):
                img_map = img_map[batch_idx]
            fig2, ax2 = plt.subplots(1, 2)
            for ax_id, (layer_name, layer_idx) in enumerate(zip(['drivable_are', 'lanes'], [0, -1])):
                show_image_(ax2[ax_id], img_map[layer_idx], layer_name, xlims, ylims, boxes_in_bev)

        # cam_front
        fig3, ax3 = plt.subplots()
        if kwargs.get('viz_dataset', False):
            token = data_dict['metadata']['token']
        else:
            t_ = data_dict['metadata'][batch_idx]
            token = t_['token']
        sample_rec = dataset.nusc.get('sample', token)
        dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'], ax=ax3)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--show_raw_pointcloud', action='store_true', default=False)
    parser.add_argument('--show_oracle_pointcloud', action='store_true', default=False)
    parser.add_argument('--show_bev', action='store_true', default=False)
    parser.add_argument('--viz_dataset', action='store_true', default=False)
    parser.add_argument('--viz_dataloader', action='store_true', default=False)

    args = parser.parse_args()
    main(**vars(args))
