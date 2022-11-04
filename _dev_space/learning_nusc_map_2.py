import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import *
import colorsys

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


def draw_lane_in_bev_(lane, pc_range, resolution, ax_, discretization_meters=1):
    """
    Args:
        lane (np.ndarray): (N, 3) - x, y, yaw in frame where BEV is generated (default: LiDAR frame)
    """
    lane_xy_in_bev = np.floor((lane[:, :2] - pc_range[:2]) / resolution)  # (N, 2)
    for _i in range(lane.shape[0]):
        cos, sin = discretization_meters * np.cos(lane[_i, -1]), discretization_meters * np.sin(lane[_i, -1])

        normalized_rgb_color = colorsys.hsv_to_rgb(np.rad2deg(lane[_i, -1]) / 360, 1., 1.)

        ax_.arrow(lane_xy_in_bev[_i, 0], lane_xy_in_bev[_i, 1], cos, sin, color=normalized_rgb_color, width=0.75)


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


def put_in_2pi(angles: np.ndarray):
    """
    Args:
        angles: (N,) in range [-pi, pi)
    """
    assert np.all(np.abs(angles) < (np.pi + 1e-3)), "angles must be put in range [-pi, pi) first"
    mask_neg = angles < 0
    angles[mask_neg] = angles[mask_neg] + 2 * np.pi
    return angles


class MapMaker:
    def __init__(self, nuscenes_api: NuScenes,
                 resolution: float = 0.2, # meters / pixel
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        self.nusc = nuscenes_api
        self.helper = PredictHelper(nusc)
        self.maps = load_all_maps(self.helper)

        self.layer_names = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']
        self.colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255), (255, 191, 0)]

        # BEV image config
        self.point_cloud_range = np.array(point_cloud_range)
        self.resolution = resolution
        self.img_size_length = self.point_cloud_range[3] - self.point_cloud_range[0]
        self.img_size_length_pixels = int(self.img_size_length / self.resolution)
        self.canvas_size = (self.img_size_length_pixels, self.img_size_length_pixels)
        self.lane_thickness = 10  # in pixels

    def map_lanes_to_sensor(self, sensor_token: str, lanes: dict) -> dict:
        """
        Args:
            sensor_token:
            lanes: {str: list[(center_line_x, center_line_y, yaw)_i | i=0,..., N]}
        Returns:
            lanes_in_sensor: {str: (N, 3) - center_line_x, center_line_y, yaw}
        """
        lanes_in_sensor = dict()  # {lane_tk: np.ndarray - (N, 3) - x, y, yaw}

        glob_from_sensor = get_nuscenes_sensor_pose_in_global(self.nusc, sensor_token)  # (4, 4)
        sensor_from_glob = np.linalg.inv(glob_from_sensor)  # (4, 4)

        for lane_token in lanes:
            lane = np.array(lanes[lane_token])  # (N, 3) - x, y, yaw in global
            # location
            lane_xyz = np.pad(lane[:, :2], pad_width=[(0, 0), (0, 1)], constant_values=0)
            lane_xyz = apply_tf(sensor_from_glob, lane_xyz)  # (N, 3) - x, y, dummy_z in lidar frame

            # orientation
            cos, sin = np.cos(lane[:, -1]), np.sin(lane[:, -1])  # (N,)
            zeros, ones = np.zeros(cos.shape[0]), np.ones(cos.shape[0])  # (N,)
            lane_ori = np.stack([
                cos, -sin, zeros,
                sin, cos, zeros,
                zeros, zeros, ones
            ], axis=1)  # (N, 9) - in global
            lane_ori = lane_ori.reshape((-1, 3, 3))  # (N, 3, 3) - in global

            lane_ori = np.matmul(sensor_from_glob[np.newaxis, :3, :3], lane_ori)  # (N, 3, 3) - in sensor frame

            lane_yaw = np.arctan2(lane_ori[:, 1, 0], lane_ori[:, 0, 0])  # (N,) - in [-pi, pi)
            lane_yaw = put_in_2pi(lane_yaw)  # (N,) - in [0, 2*pi)

            # save output
            lanes_in_sensor[lane_token] = np.concatenate([lane_xyz[:, :2], lane_yaw.reshape(-1, 1)], axis=1)
        return lanes_in_sensor

    def compute_bev_coord(self, points: np.ndarray, to_int=False) -> np.ndarray:
        """
        Args:
            points: (N, 2 or 3) - x, y, [z]
            to_int: whether to convert to int
        """
        assert points.shape[1] >= 2, f"points must have (N, 2 or 3), get {points.shape}"
        bev_coord = (points[:, :2] - self.point_cloud_range[:2]) / self.resolution
        if to_int:
            bev_coord = np.floor(bev_coord).astype(int)
        return bev_coord

    def draw_lane_in_bev(self, lanes: dict):
        lane_img = np.zeros(self.canvas_size)
        max_angle = 2 * np.pi + 1e-3
        for _, lane in lanes.items():
            lane_xy_in_bev = self.compute_bev_coord(lane, to_int=True)  # (N, 2)
            assert np.all(lane[:, -1] >= 0)
            assert np.all(lane[:, -1] < max_angle)
            for p_idx in range(lane_xy_in_bev.shape[0] - 1):
                color = lane[p_idx, -1] / max_angle
                cv2.line(lane_img, lane_xy_in_bev[p_idx, :2], lane_xy_in_bev[p_idx + 1, :2], color,
                         self.lane_thickness)
        return lane_img

    def make_representation(self, sample_data_token: str, return_lanes=False):
        """
        Args:
            sample_data_token:
            return_lanes:
        Returns:
            (5, H, W) - 4 binary layers & 1 float layer representing lane dir
        """
        sample_data_rec = self.nusc.get('sample_data', sample_data_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_data_rec['sample_token'])

        # ---------
        # sensor pose in global frame
        # ---------
        glob_from_sensor = get_nuscenes_sensor_pose_in_global(self.nusc, sample_data_token)  # (4, 4)
        x, y = glob_from_sensor[:2, -1]
        yaw = quaternion_yaw(Quaternion(matrix=glob_from_sensor))

        # ---------
        # Map's binary layers
        # ---------
        patch_box = get_patchbox(x, y, self.img_size_length)
        masks = self.maps[map_name].get_map_mask(patch_box, np.rad2deg(yaw), self.layer_names,
                                                 canvas_size=self.canvas_size)  # (N_layers, H, W)

        # ---------
        # Lanes
        # ---------
        lanes = get_lanes_in_radius(x, y, 60, discretization_meters=1, map_api=self.maps[map_name])
        lanes_in_sensor = self.map_lanes_to_sensor(sample_data_token, lanes)
        lanes_img = self.draw_lane_in_bev(lanes_in_sensor)

        out = np.concatenate([masks.astype(float), lanes_img[np.newaxis]], axis=0)
        if return_lanes:
            return out, lanes_in_sensor
        else:
            return out

    def render_map_in_color_(self, map_image: np.ndarray, ax: Axes, lanes=None, boxes=None, draw_sensor_frame=False):
        """
        Args:
            map_image: (5, H, W) - 4 binary layers & 1 float layer representing lane dir
            ax:
            lanes (dict):
            boxes (list): each element (8, 3) - x, y, z of 8 vertices
        """
        assert map_image.shape[0] == 5, f"expect (5, H, W), get {map_image.shape}"
        ax.set_title('colorize map @ sensor frame')
        ax.set_xlim([0, self.img_size_length_pixels])
        ax.set_ylim([0, self.img_size_length_pixels])

        colorized_binary_layers = np.zeros((*self.canvas_size, 3))
        for l_id, color in enumerate(self.colors):
            colorized_binary_layers[map_image[l_id].astype(bool)] = np.array(color)
        ax.imshow(colorized_binary_layers)

        if lanes is not None:
            for _, lane in lanes.items():
                draw_lane_in_bev_(lane, self.point_cloud_range, self.resolution, ax, discretization_meters=1)

        if boxes is not None:
            draw_boxes_in_bev_(gt_boxes, self.point_cloud_range, self.resolution, ax)

        if draw_sensor_frame:
            draw_lidar_frame_(self.point_cloud_range, self.resolution, ax)


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
    renderer = MapMaker(nusc)

    sample = nusc.get('sample', data_dict['metadata']['token'])

    map_img, lanes_in_lidar = renderer.make_representation(sample['data']['LIDAR_TOP'], return_lanes=True)

    fig, ax = plt.subplots(1, 2)
    for _i, (layer, layer_idx) in enumerate(zip(['drivable_are', 'lanes'], [0, -1])):
        ax[_i].set_title(f' {layer} @ LiDAR')
        ax[_i].set_aspect('equal')
        ax[_i].set_xlim([0, renderer.img_size_length_pixels])
        ax[_i].set_ylim([0, renderer.img_size_length_pixels])
        ax[_i].imshow(map_img[layer_idx])

        draw_lidar_frame_(renderer.point_cloud_range, renderer.resolution, ax[_i])
        draw_boxes_in_bev_(gt_boxes, renderer.point_cloud_range, renderer.resolution, ax[_i])

    fig2, ax2 = plt.subplots()
    renderer.render_map_in_color_(map_img, ax2, lanes_in_lidar, gt_boxes, draw_sensor_frame=True)

    plt.show()
