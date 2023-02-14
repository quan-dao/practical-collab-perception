import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps, get_lanes_in_radius
from einops import rearrange
import cv2
from _dev_space.tools_box import get_nuscenes_sensor_pose_in_ego_vehicle, get_nuscenes_sensor_pose_in_global, apply_tf


def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def put_in_2pi(angles: np.ndarray):
    """
    Args:
        angles: (N,) in range [-pi, pi)
    """
    assert np.all(np.abs(angles) < (np.pi + 1e-3)), "angles must be put in range [-pi, pi) first"
    mask_neg = angles < 0
    angles[mask_neg] = angles[mask_neg] + 2 * np.pi
    return angles


class MapMaker(object):
    def __init__(self, nuscenes_api: NuScenes, point_cloud_range: np.ndarray, map_resolution: float, 
                map_layers=('drivable_area', 'ped_crossing', 'walkway', 'carpark_area')):
        self.nusc = nuscenes_api
        self.point_cloud_range = point_cloud_range
        self.map_resolution = map_resolution
        self.map_layers = map_layers
        prediction_helper = PredictHelper(self.nusc)
        self.map_apis = load_all_maps(prediction_helper)
        self.map_dx, self.map_dy = point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]
        self.map_size_pixel = (int(self.map_dx / map_resolution), int(self.map_dy / map_resolution))  # (W, H)
        self.lane_thickness = int(5.0 / self.map_resolution)

    def get_map_name_from_sample_token(self, sample_tk: str) -> str:
        sample = self.nusc.get('sample', sample_tk)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        return log['location']

    def get_binary_layers_in_ego_frame(self, sample_tk: str, return_channel_last=True) -> np.ndarray:
        sample_data_tk = self.nusc.get('sample', sample_tk)['data']['LIDAR_TOP']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', sample_data_tk)['ego_pose_token'])
        ego_x, ego_y, _ = ego_pose['translation']
        ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

        patch_box = (ego_x, ego_y, self.map_dx, self.map_dy)
        map_name = self.get_map_name_from_sample_token(sample_tk)
        map_layers = self.map_apis[map_name].get_map_mask(patch_box, -np.rad2deg(correct_yaw(ego_yaw)), self.map_layers,
                                                          canvas_size=self.map_size_pixel).astype(float)  # (N_layers, H, W)
        if return_channel_last:
            map_layers = rearrange(map_layers, 'C H W -> H W C')

        return map_layers
    
    def get_binary_layers_in_lidar_frame(self, sample_tk: str, return_channel_last=True) -> np.ndarray:
        map_in_ego = self.get_binary_layers_in_ego_frame(sample_tk, return_channel_last=True)  # (H, W, C)

        rows, cols = map_in_ego.shape[:2]

        # ------------------------------- #
        # transform map from ego to lidar
        # ------------------------------- #

        # rotation | fix at -90 because sensor frame = Rot(z, -90)[ego frame]
        rot_mat = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1.0)
        map_in_lidar = cv2.warpAffine(map_in_ego, rot_mat, (cols, rows))

        # translation
        lidar_tk = self.nusc.get('sample', sample_tk)['data']['LIDAR_TOP']
        sensor_from_ego = np.linalg.inv(get_nuscenes_sensor_pose_in_ego_vehicle(self.nusc, lidar_tk))
        transl_ego_from_sensor_pixel = sensor_from_ego[:2, -1] / self.map_resolution
        transl_mat = np.array([
            [1.0, 0.0, transl_ego_from_sensor_pixel[0]],
            [0.0, 1.0, transl_ego_from_sensor_pixel[1]],
        ])
        map_in_lidar = cv2.warpAffine(map_in_lidar, transl_mat, (cols, rows))

        if not return_channel_last:
            # channel first
            map_in_lidar = rearrange(map_in_lidar, 'H W C -> C H W')
        return map_in_lidar

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

    def rasterize_lane(self, lanes_dict: dict, normalize_lane_direction=False) -> np.ndarray:
        lane_img = np.zeros((self.map_size_pixel[1], self.map_size_pixel[0])).astype(float)
        
        for _, lane in lanes_dict.items():
            lane_bev_coord = np.floor((lane[:, :2] - self.point_cloud_range[:2]) / self.map_resolution).astype(int)
            lane_angle = np.clip(lane[:, -1], a_min=0.0, a_max=2 * np.pi - 1e-3)
            for idx in range(lane_bev_coord.shape[0] - 1):
                color = lane_angle[idx] / (2 * np.pi)
                cv2.line(lane_img, lane_bev_coord[idx, :2], lane_bev_coord[idx + 1, :2], color, thickness=self.lane_thickness)

        if not normalize_lane_direction:
            # to return actual lane angle
            lane_img = lane_img * (2 * np.pi)

        return lane_img

    def get_rasterized_lanes_in_lidar_frame(self, sample_tk: str, normalize_lane_direction=False) -> np.ndarray:
        sample_data_tk = self.nusc.get('sample', sample_tk)['data']['LIDAR_TOP']
        glob_from_sensor = get_nuscenes_sensor_pose_in_global(self.nusc, sample_data_tk)  # (4, 4)
        x, y = glob_from_sensor[:2, -1]
        
        map_name = self.get_map_name_from_sample_token(sample_tk)
        lanes = get_lanes_in_radius(x, y, self.point_cloud_range[3], discretization_meters=1, map_api=self.map_apis[map_name])  # in global

        lanes_in_sensor = self.map_lanes_to_sensor(sample_data_tk, lanes)
        
        lane_img = self.rasterize_lane(lanes_in_sensor, normalize_lane_direction)
        
        return lane_img



