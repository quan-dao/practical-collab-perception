import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import *

from _dev_space.tools_box import apply_tf, tf, get_nuscenes_sensor_pose_in_global, \
    get_nuscenes_sensor_pose_in_ego_vehicle
from _dev_space.viz_tools import viz_boxes

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import NuScenesDataset


def print_record(rec, rec_type=None):
    print(f'--- {rec_type}' if rec_type is not None else '---')
    for k, v in rec.items():
        print(f"{k}: {v}")
    print('---\n')


def draw_lidar_frame_(pc_range, resolution, ax_):
    lidar_frame = np.array([
        [0, 0],  # origin
        [3, 0],  # x-axis
        [0, 3]  # y-axis
    ])
    lidar_frame_in_bev = np.floor((lidar_frame - pc_range[:2]) / resolution)
    ax_.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
              lidar_frame_in_bev[1, 0] - lidar_frame_in_bev[0, 0],
              lidar_frame_in_bev[1, 1] - lidar_frame_in_bev[0, 1],
              color='r', width=3)  # x-axis

    ax_.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
              lidar_frame_in_bev[2, 0] - lidar_frame_in_bev[0, 0],
              lidar_frame_in_bev[2, 1] - lidar_frame_in_bev[0, 1],
              color='b', width=3)  # y-axis


def draw_boxes_in_bev_(boxes, pc_range, resolution, ax_, color='r'):
    """
    Args:
        boxes (List[np.ndarray]): each box - (8, 3) forward: 0-1-2-3, backward: 4-5-6-7, up: 0-1-5-4
        ax_
    """
    for box in boxes:
        box_in_bev = np.floor((box[:, :2] - pc_range[:2]) / resolution)  # (8, 2)
        top_face = box_in_bev[[0, 1, 5, 4, 0]]
        ax_.plot(top_face[:, 0], top_face[:, 1], c=color)

        # draw heading
        center = (box_in_bev[0] + box_in_bev[5]) / 2.0
        mid_01 = (box_in_bev[0] + box_in_bev[1]) / 2.0
        heading_line = np.stack([center, mid_01], axis=0)
        ax_.plot(heading_line[:, 0], heading_line[:, 1], c=color)


def _set_up_cfg_(cfg):
    cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        'placeholder',
        'random_world_flip', 'random_world_scaling',
        'gt_sampling',
        'random_world_rotation',
    ]
    cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx',
                                                    'instance_idx',
                                                    'aug_instance_idx']
    cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'sweep_idx', 'instance_idx',
                                                   'aug_instance_idx']
    cfg.VERSION = 'v1.0-mini'


class StaticLayerRenderer:
    def __init__(self, nuscenes_api: NuScenes, helper: PredictHelper,
                 resolution: float = 0.2, # meters / pixel
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        self.nusc = nuscenes_api
        self.helper = helper
        self.maps = load_all_maps(helper)

        self.layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255)]

        # BEV image config
        self.point_cloud_range = np.array(point_cloud_range)
        self.resolution = resolution
        self.img_size_length = self.point_cloud_range[3] - self.point_cloud_range[0]
        self.img_size_length_pixels = int(self.img_size_length / self.resolution)
        self.canvas_size = (self.img_size_length_pixels, self.img_size_length_pixels)

    def make_representation(self, sample_data_token: str):
        sample_data_rec = self.nusc.get('sample_data', sample_data_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_data_rec['sample_token'])

        glob_from_sensor = get_nuscenes_sensor_pose_in_global(self.nusc, sample_data_token)  # (4, 4)
        x, y = glob_from_sensor[:2, -1]
        yaw = quaternion_yaw(Quaternion(matrix=glob_from_sensor))

        patch_box = get_patchbox(x, y, self.img_size_length)
        masks = self.maps[map_name].get_map_mask(patch_box, np.rad2deg(yaw), self.layer_names,
                                                 canvas_size=self.canvas_size)

        images = [masks[ch_idx] for ch_idx in range(masks.shape[0])]

        return images


if __name__ == '__main__':
    np.random.seed(666)
    cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    _set_up_cfg_(cfg)
    logger = common_utils.create_logger('./dummy_log.txt')
    nuscenes_dataset = NuScenesDataset(cfg, cfg.CLASS_NAMES, training=True, logger=logger)

    data_dict = nuscenes_dataset[200]  # 400, 200, 100, 5, 10

    gt_boxes = viz_boxes(data_dict['gt_boxes'])

    # -----------------------
    # Map stuff
    # -----------------------
    nusc = nuscenes_dataset.nusc
    helper = PredictHelper(nusc)
    renderer = StaticLayerRenderer(nusc, helper)

    sample = nusc.get('sample', data_dict['metadata']['token'])

    images = renderer.make_representation(sample['data']['LIDAR_TOP'])

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title('drivable_are @ LiDAR')
    ax[0].set_aspect('equal')
    ax[0].imshow(images[0])
    draw_lidar_frame_(renderer.point_cloud_range, renderer.resolution, ax[0])
    draw_boxes_in_bev_(gt_boxes, renderer.point_cloud_range, renderer.resolution, ax[0])

    ax[1].set_title('walkway @ LiDAR')
    ax[1].set_aspect('equal')
    ax[1].imshow(images[1])
    draw_lidar_frame_(renderer.point_cloud_range, renderer.resolution, ax[1])
    draw_boxes_in_bev_(gt_boxes, renderer.point_cloud_range, renderer.resolution, ax[1])

    plt.show()
