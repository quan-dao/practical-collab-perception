import numpy as np
from matplotlib.axes import Axes
import cv2
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps, quaternion_yaw, get_patchbox, \
    get_lanes_in_radius
from pyquaternion import Quaternion
from _dev_space.tools_box import apply_tf, get_nuscenes_sensor_pose_in_global
from _dev_space.viz_tools import draw_lane_in_bev_, draw_boxes_in_bev_, draw_lidar_frame_
import logging


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
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 normalize_lane_angle=False):
        self.nusc = nuscenes_api
        self.helper = PredictHelper(self.nusc)
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
        self.normalize_lane_angle = normalize_lane_angle

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

    def draw_lane_in_bev(self, lanes: dict, normalize_lane_angle=True):
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

        if not normalize_lane_angle:
            mask_has_angle = lane_img > -1
            lane_img[mask_has_angle] = lane_img[mask_has_angle] * max_angle

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
        logging.disable(logging.INFO)
        patch_box = get_patchbox(x, y, self.img_size_length)
        masks = self.maps[map_name].get_map_mask(patch_box, np.rad2deg(yaw), self.layer_names,
                                                 canvas_size=self.canvas_size)  # (N_layers, H, W)
        logging.disable(logging.NOTSET)
        # ---------
        # Lanes
        # ---------
        lanes = get_lanes_in_radius(x, y, 51.2, discretization_meters=1, map_api=self.maps[map_name])
        lanes_in_sensor = self.map_lanes_to_sensor(sample_data_token, lanes)
        lanes_img = self.draw_lane_in_bev(lanes_in_sensor, self.normalize_lane_angle)

        out = np.concatenate([masks.astype(float), lanes_img[np.newaxis]], axis=0)  # (5, H, W)
        if return_lanes:
            return out, lanes_in_sensor
        else:
            return out

    def render_map_in_color_(self, map_image: np.ndarray, ax_: Axes, lanes=None, boxes=None, draw_sensor_frame=False):
        """
        Args:
            map_image: (5, H, W) - 4 binary layers & 1 float layer representing lane dir
            ax_:
            lanes (dict):
            boxes (list): each element (8, 3) - x, y, z of 8 vertices
        """
        assert map_image.shape[0] == 5, f"expect (5, H, W), get {map_image.shape}"
        ax_.set_title('colorize map @ sensor frame')
        ax_.set_xlim([0, self.img_size_length_pixels])
        ax_.set_ylim([0, self.img_size_length_pixels])

        colorized_binary_layers = np.zeros((*self.canvas_size, 3))
        for l_id, color in enumerate(self.colors):
            colorized_binary_layers[map_image[l_id].astype(bool)] = np.array(color)
        ax_.imshow(colorized_binary_layers)

        if lanes is not None:
            for _, lane in lanes.items():
                draw_lane_in_bev_(lane, self.point_cloud_range, self.resolution, ax_, discretization_meters=1)

        if boxes is not None:
            draw_boxes_in_bev_(boxes, self.point_cloud_range, self.resolution, ax_)

        if draw_sensor_frame:
            draw_lidar_frame_(self.point_cloud_range, self.resolution, ax_)
