import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset
import matplotlib.pyplot as plt
from tools_box import compute_bev_coord, show_pointcloud, get_nuscenes_sensor_pose_in_ego_vehicle, apply_tf
from viz_tools import viz_boxes, print_dict


np.random.seed(666)
cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)

logger = common_utils.create_logger('./dummy_log.txt')

cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    'random_world_flip', 'random_world_scaling',
    'gt_sampling',
    'random_world_rotation',
]

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                                'aug_instance_idx']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                               'aug_instance_idx']
cfg.VERSION = 'v1.0-mini'


nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)
data_dict = nuscenes_dataset[200]  # 400, 200, 100, 5, 10
print_dict(data_dict)
print('meta: ', data_dict['metadata'])

# sample_rec = nuscenes_dataset.nusc.get('sample', data_dict['metadata']['token'])
# nuscenes_dataset.nusc.render_sample_data(sample_rec['data']['CAM_FRONT'])
# plt.show()


pc = data_dict['points']
print('n pts: ', pc.shape[0])
gt_boxes = viz_boxes(data_dict['gt_boxes'])

print('showing EMC accumulated pointcloud')
pc_colors = np.zeros((pc.shape[0], 3))
# pc_colors[pc[:, -2] == 0] = np.array([0, 0, 1])
pc_colors[pc[:, -2] > -1] = np.array([1, 0, 0])
show_pointcloud(pc[:, :3], boxes=gt_boxes, pc_colors=pc_colors)

showing_orcale = False
if showing_orcale:
    print('showing oracle accumulated pointcloud')
    pc[:, :2] += pc[:, 5: 7]
    print('n pts: ', pc.shape[0])
    show_pointcloud(pc[:, :3], boxes=gt_boxes, fgr_mask=pc[:, -1] > -1)

showing_bev = True
if showing_bev:
    print('showing BEV')
    pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    mask_inside = np.all((pc[:, :3] >= pc_range[:3]) & (pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
    pc = pc[mask_inside]
    bev_coord, bev_feat = compute_bev_coord(pc[:, :3], pc_range=pc_range, bev_pix_size=0.2,
                                            pts_feat=pc[:, 3].reshape(-1, 1))

    fig, ax = plt.subplots()
    img = np.zeros((512, 512), dtype=float)
    img[bev_coord[:, 1], bev_coord[:, 0]] = bev_feat[:, 0]
    img = img / np.max(img)
    ax.set_title('bev img (intensity)')
    ax.imshow(img)

    # draw LiDAR frame on BEV image
    lidar_frame = np.array([
        [0, 0],  # origin
        [3, 0],  # x-axis
        [0, 3]   # y-axis
    ])
    lidar_frame_in_bev = np.floor((lidar_frame - pc_range[:2]) / 0.2)
    ax.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
             lidar_frame_in_bev[1, 0] - lidar_frame_in_bev[0, 0], lidar_frame_in_bev[1, 1] - lidar_frame_in_bev[0, 1],
             color='r', width=3)  # x-axis

    ax.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
             lidar_frame_in_bev[2, 0] - lidar_frame_in_bev[0, 0], lidar_frame_in_bev[2, 1] - lidar_frame_in_bev[0, 1],
             color='b', width=3)  # y-axis

    # draw ego-pose on BEV image
    sample_tk = data_dict['metadata']['token']
    sample_rec = nuscenes_dataset.nusc.get('sample', sample_tk)
    ego_from_lidar = get_nuscenes_sensor_pose_in_ego_vehicle(nuscenes_dataset.nusc, sample_rec['data']['LIDAR_TOP'])
    ego_frame = np.array([
        [0, 0, 0],  # origin
        [10, 0, 0],  # x-axis
        [0, 10, 0]   # y-axis
    ])
    ego_frame = apply_tf(np.linalg.inv(ego_from_lidar), ego_frame)
    ego_frame_in_bev = np.floor((ego_frame[:, :2] - pc_range[:2]) / 0.2)
    ax.arrow(ego_frame_in_bev[0, 0], ego_frame_in_bev[0, 1],
             ego_frame_in_bev[1, 0] - ego_frame_in_bev[0, 0], ego_frame_in_bev[1, 1] - ego_frame_in_bev[0, 1],
             color='r', width=3, linestyle='--')  # x-axis

    ax.arrow(ego_frame_in_bev[0, 0], ego_frame_in_bev[0, 1],
             ego_frame_in_bev[2, 0] - ego_frame_in_bev[0, 0], ego_frame_in_bev[2, 1] - ego_frame_in_bev[0, 1],
             color='b', width=3, linestyle='--')  # y-axis

    ax.set_aspect('equal')
    plt.show()
