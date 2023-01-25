import numpy as np
from ...utils import box_utils
import copy
from nuscenes import NuScenes
from typing import List
from nuscenes.utils.geometry_utils import transform_matrix, Quaternion


def transform_annotations_to_kitti_format(nusc: NuScenes, annos: List, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        nusc:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    nu_lidar_to_kitti = kitti_to_nu_lidar.inverse  # (4, 4)

    for anno_idx in range(len(annos)):
        anno = copy.deepcopy(annos[anno_idx])
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))  # 2D bounding box | Ignore
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))

        if 'boxes_lidar' in anno:  # prediction dict
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        elif 'gt_boxes_lidar' in anno:  # kitti
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()
        else:  # nuscenes
            gt_boxes_lidar = anno['gt_boxes'].copy()  # (N, 7 [+2]) - x, y, z, dx, dy, dz, yaw, [velocity_x, _y]

        # map gt_boxes to KITTI
        if 'metadata' in anno:
            sample_tk = anno['metadata']['token']
        elif 'token' in anno:
            sample_tk = anno['token']
        else:
            print('anno:\n', anno.keys(), '\n----')
            raise KeyError

        # Get sample data.
        sample = nusc.get('sample', sample_tk)
        cam_front_token = sample['data']['CAM_FRONT']
        lidar_token = sample['data']['LIDAR_TOP']

        # Retrieve sensor records.
        sd_record_cam = nusc.get('sample_data', cam_front_token)
        sd_record_lid = nusc.get('sample_data', lidar_token)
        cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        # Combine transformations and convert to KITTI format.
        # Note: cam uses same conventions in KITTI and nuScenes.
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                      inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                      inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)  # (4, 4)

        if gt_boxes_lidar.shape[0] > 0:
            cos, sin = np.cos(gt_boxes_lidar[:, 6]), np.sin(gt_boxes_lidar[:, 6])
            zeros, ones = np.zeros(gt_boxes_lidar.shape[0]), np.ones(gt_boxes_lidar.shape[0])
            boxes_to_nu_lidar = np.stack([
                cos, -sin, zeros, gt_boxes_lidar[:, 0],
                sin, cos, zeros, gt_boxes_lidar[:, 1],
                zeros, zeros, ones, gt_boxes_lidar[:, 2],
                zeros, zeros, zeros, ones
            ], axis=1).reshape((gt_boxes_lidar.shape[0], 4, 4))
            boxes_to_cam_kitti = np.einsum('ij, jk, bkh -> bih', velo_to_cam_kitti,
                                           nu_lidar_to_kitti.transformation_matrix, boxes_to_nu_lidar)  # (N, 4, 4)

            locs = boxes_to_cam_kitti[:, :3, -1]  # (N, 3)
            locs[:, 2] += gt_boxes_lidar[:, 5] / 2  # + because CAM_FRONT's y-axis points downward
            anno['location'] = locs

            dxdydz = gt_boxes_lidar[:, 3: 6]
            anno['dimensions'] = dxdydz[:, [2, 1, 0]]

            anno['rotation_y'] = np.arctan2(-boxes_to_cam_kitti[:, 2, 0], boxes_to_cam_kitti[:, 0, 0])

            anno['alpha'] = -10 * np.ones(gt_boxes_lidar.shape[0])  # dummy
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        annos[anno_idx] = anno

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2