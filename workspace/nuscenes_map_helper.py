import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from einops import rearrange
import cv2
from _dev_space.tools_box import get_nuscenes_sensor_pose_in_ego_vehicle


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



