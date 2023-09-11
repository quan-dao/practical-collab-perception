import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from pcdet.datasets.nuscenes._legacy_tools_box import get_nuscenes_sensor_pose_in_global, get_nuscenes_sensor_pose_in_ego_vehicle
from pyquaternion import Quaternion
import cv2
from einops import rearrange


def get_map_name_from_sample_token(nusc: NuScenes, sample_tk: str) -> str:
    sample = nusc.get('sample', sample_tk)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    return log['location']


def to_bev_coord(xy: np.ndarray, point_cloud_range: np.ndarray, bev_resolution: float) -> np.ndarray:
    return (xy - point_cloud_range[:2]) / bev_resolution


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


def main():
    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    prediction_helper = PredictHelper(nusc)
    map_apis = load_all_maps(prediction_helper)
    map_point_cloud_range = np.array([-81.0, -81.0, -5.0, 81.0, 81.0, 3.0])  # independent with the 
    map_bev_image_resolution = 0.075  # independent with the dataset config
    map_binary_layers = ('drivable_area', 'ped_crossing', 'walkway', 'carpark_area')

    map_dx, map_dy = map_point_cloud_range[3] - map_point_cloud_range[0], map_point_cloud_range[4] - map_point_cloud_range[1]
    map_size_pixel = (int(map_dx / map_bev_image_resolution), int(map_dy / map_bev_image_resolution))  # (W, H)


    # TODO: get 1 ego pose, show drivable area to have a starting point

    scene = nusc.scene[0]
    sample_tk = scene['first_sample_token']
    for _ in range(15):
        sample_rec = nusc.get('sample', sample_tk)
        sample_tk = sample_rec['next']
    

    sample_rec = nusc.get('sample', sample_tk)
    target_sd_token = sample_rec['data']['LIDAR_TOP']
    glob_from_target = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))

    ego_pose = nusc.get('ego_pose', nusc.get('sample_data', target_sd_token)['ego_pose_token'])

    ego_x, ego_y, _ = ego_pose['translation']
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

    map_name = get_map_name_from_sample_token(nusc, sample_tk)
    patch_box = (ego_x, ego_y, map_dx, map_dy)
    map_masks = map_apis[map_name].get_map_mask(patch_box, np.rad2deg(correct_yaw(ego_yaw)), map_binary_layers, 
                                                canvas_size=map_size_pixel).astype(float)  # (N_layers, H, W)
    map_masks = rearrange(map_masks, 'C H W -> H W C')
    
    # transform map_masks from ego frame to LiDAR frame
    ego_from_sensor = get_nuscenes_sensor_pose_in_ego_vehicle(nusc, target_sd_token)
    print('ego_from_sensor:\n', ego_from_sensor)
    yaw_ego_from_sensor = Quaternion(matrix=ego_from_sensor).yaw_pitch_roll[0]
    print('yaw_ego_from_sensor (in deg): ', np.rad2deg(yaw_ego_from_sensor))
    
    sensor_from_ego = np.linalg.inv(ego_from_sensor)


    rows, cols = map_masks.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -90, 1.0)
    map_in_lidar = cv2.warpAffine(map_masks, rot_mat, (cols, rows))
    print('map_in_lidar: ', map_in_lidar.shape)

    transl_ego_from_sensor_pixel = sensor_from_ego[:2, -1] / map_bev_image_resolution
    transl_mat = np.array([
        [1.0, 0.0, transl_ego_from_sensor_pixel[0]],
        [0.0, 1.0, transl_ego_from_sensor_pixel[1]],
    ])
    map_in_lidar = cv2.warpAffine(map_in_lidar, transl_mat, (cols, rows))

    # display ego frame
    cos, sin = np.cos(0), np.sin(0)
    arrow_length = 2.0  # meters
    ego_x_offset = arrow_length * np.array([cos, sin]) / map_bev_image_resolution
    ego_y_offset = arrow_length * np.array([-sin, cos]) / map_bev_image_resolution

    fig, axe = plt.subplots(1, 2)
    axes_limit = map_point_cloud_range[3]
    # draw ego frame
    axe[0].scatter([map_size_pixel[0] / 2], [map_size_pixel[1] / 2], marker='P', s=10)
    axe[0].arrow(map_size_pixel[0] / 2, map_size_pixel[1] / 2, ego_x_offset[0], ego_x_offset[1], color='r')
    axe[0].arrow(map_size_pixel[0] / 2, map_size_pixel[1] / 2, ego_y_offset[0], ego_y_offset[1], color='g')
    axe[0].imshow(map_masks[..., 0], origin='lower')

    axe[1].scatter([map_size_pixel[0] / 2], [map_size_pixel[1] / 2], marker='P', s=10)
    axe[1].arrow(map_size_pixel[0] / 2, map_size_pixel[1] / 2, ego_x_offset[0], ego_x_offset[1], color='r')
    axe[1].arrow(map_size_pixel[0] / 2, map_size_pixel[1] / 2, ego_y_offset[0], ego_y_offset[1], color='g')
    axe[1].imshow(map_in_lidar[..., 0], origin='lower')


    fig2, ax2 = plt.subplots()
    nusc.render_sample_data(target_sd_token, ax=ax2, axes_limit=map_point_cloud_range[3])

    plt.show()



if __name__ == '__main__':
    main()
    from nuscenes.prediction.input_representation.interface import InputRepresentation
    from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
    from nuscenes.prediction.helper import angle_of_rotation

