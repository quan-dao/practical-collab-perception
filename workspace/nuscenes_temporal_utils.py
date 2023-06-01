import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from typing import List, Union, Tuple


def tf(translation, rotation):
    """
    Build transformation matrix
    """
    if not isinstance(rotation, Quaternion):
        assert isinstance(rotation, list) or isinstance(rotation, np.ndarray), f"{type(rotation)} is not supported"
        rotation = Quaternion(rotation)
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rotation.rotation_matrix
    tf_mat[:3, 3] = translation
    return tf_mat


def apply_tf(tf: np.ndarray, points: np.ndarray):
    assert points.shape[1] == 3
    assert tf.shape == (4, 4)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out = tf @ points_homo.T
    return out[:3, :].T


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    return np.arctan2(rot[1, 0], rot[0, 0])


def apply_se3_(se3_tf: np.ndarray, 
               points_: np.ndarray = None, 
               boxes_: np.ndarray = None, boxes_has_velocity: bool = False, 
               vector_: np.ndarray = None,
               return_transformed: bool = False) -> None:
    """
    Inplace function

    Args:
        se3_tf: (4, 4) - homogeneous transformation
        points_: (N, 3 + C) - x, y, z, [others]
        boxes_: (N, 7 + 2 + C) - x, y, z, dx, dy, dz, yaw, [vx, vy], [others]
        boxes_has_velocity: make boxes_velocity explicit
        vector_: (N, 2 [+ 1]) - x, y, [z]
    """
    if return_transformed:
        points = points_.copy() if points_ is not None else None
        boxes = boxes_.copy() if boxes_ is not None else None
        vectors = vector_.copy() if vector_ is not None else None
        apply_se3_(se3_tf, 
                   points_=points, 
                   boxes_=boxes, 
                   vector_=vectors, 
                   boxes_has_velocity=boxes_has_velocity)
        out = [ele for ele in [points, boxes, vectors] if ele is not None]
        if len(out) == 1:
            return out[0]
        else:
            return out

    if points_ is not None:
        assert points_.shape[1] >= 3, f"points_.shape: {points_.shape}"
        points_[:, :3] = points_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]

    if boxes_ is not None:
        assert boxes_.shape[1] >= 7, f"boxes_.shape: {boxes_.shape}"
        boxes_[:, :3] = boxes_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]
        boxes_[:, 6] += rotation_matrix_to_yaw(se3_tf[:3, :3])
        boxes_[:, 6] = np.arctan2(np.sin(boxes_[:, 6]), np.cos(boxes_[:, 6]))
        if boxes_has_velocity:
            boxes_velo = np.pad(boxes_[:, 7: 9], pad_width=[(0, 0), (0, 1)], constant_values=0.0)  # (N, 3) - vx, vy, vz
            boxes_velo = boxes_velo @ se3_tf[:3, :3].T
            boxes_[:, 7: 9] = boxes_velo[:, :2]

    if vector_ is not None:
        if vector_.shape[1] == 2:
            vector = np.pad(vector_, pad_width=[(0, 0), (0, 1)], constant_values=0.)
            vector_[:, :2] = (vector @ se3_tf[:3, :3].T)[:, :2]
        else:
            assert vector_.shape[1] == 3, f"vector_.shape: {vector_.shape}"
            vector_[:, :3] = vector_ @ se3_tf[:3, :3].T

    return


def get_nuscenes_sensor_pose_in_ego_vehicle(nusc: NuScenes, curr_sd_token: str):
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_cs_rec = nusc.get('calibrated_sensor', curr_rec['calibrated_sensor_token'])
    ego_from_curr = tf(curr_cs_rec['translation'], curr_cs_rec['rotation'])
    return ego_from_curr


def get_nuscenes_sensor_pose_in_global(nusc: NuScenes, curr_sd_token: str):
    ego_from_curr = get_nuscenes_sensor_pose_in_ego_vehicle(nusc, curr_sd_token)
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_ego_rec = nusc.get('ego_pose', curr_rec['ego_pose_token'])
    glob_from_ego = tf(curr_ego_rec['translation'], curr_ego_rec['rotation'])
    glob_from_curr = glob_from_ego @ ego_from_curr
    return glob_from_curr


def get_sweeps_token(nusc: NuScenes, curr_sd_token: str, n_sweeps: int, return_time_lag=True, return_sweep_idx=False) -> list:
    ref_sd_rec = nusc.get('sample_data', curr_sd_token)
    ref_time = ref_sd_rec['timestamp'] * 1e-6
    sd_tokens_times = []
    for s_idx in range(n_sweeps):
        curr_sd = nusc.get('sample_data', curr_sd_token)
        if not return_sweep_idx:
            sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6))
        else:
            sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6, n_sweeps - 1 - s_idx))
        # s_idx: the higher, the closer to the current
        # move to previous
        if curr_sd['prev'] != '':
            curr_sd_token = curr_sd['prev']

    # organize from PAST to PRESENCE
    sd_tokens_times.reverse()

    if return_time_lag:
        return sd_tokens_times
    else:
        return [token for token, _ in sd_tokens_times]


def get_one_pointcloud(nusc: NuScenes, sweep_token: str) -> np.ndarray:
    """
    Args:
        nusc:
        sweep_token: sample data token

    Return:
        pointcloud: (N, 4) - x, y, z, reflectant
    """
    pcfile = nusc.get_sample_data_path(sweep_token)
    # pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)

    points = np.fromfile(str(pcfile), dtype=np.float32, count=-1)
    if points.shape[0] % 5 != 0:
        points = points[: points.shape[0] - (points.shape[0] % 5)]
    points = points.reshape([-1, 5])[:, :4]

    return points


def quaternion_to_yaw(q: Quaternion) -> float:
    return np.arctan2(q.rotation_matrix[1, 0], q.rotation_matrix[0, 0])


def make_rotation_around_z(yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])
    return out


def make_se3(translation: Union[List[float], np.ndarray], yaw: float = None, rotation_matrix: np.ndarray = None):
    if yaw is None:
        assert rotation_matrix is not None
    else:
        assert rotation_matrix is None
    
    if rotation_matrix is None:
        rotation_matrix = make_rotation_around_z(yaw)

    out = np.zeros((4, 4))
    out[-1, -1] = 1.0

    out[:3, :3] = rotation_matrix

    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    out[:3, -1] = translation.reshape(3)

    return out


def compute_correction_tf(boxes: np.ndarray) -> np.ndarray:
    """
    Args:
        boxes: (N_box, 7 + C) - x, y, z, dx, dy, dz, yaw, [+ C] 
    
    Returns:
        instance_tf: (N_box, 4, 4)
    """
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    zeros, ones = np.zeros(boxes.shape[0]), np.ones(boxes.shape[0])
    poses = np.stack([
        cos,    -sin,   zeros,      boxes[:, 0],
        sin,    cos,    zeros,      boxes[:, 1],
        zeros,  zeros,  ones,       boxes[:, 2],
        zeros,  zeros,  zeros,      ones,
    ], axis=1).reshape(-1, 4, 4)
    correction_tf = np.einsum('ij, bjk -> bik', poses[-1], np.linalg.inv(poses))
    return correction_tf


def get_sweeps(nusc: NuScenes, sample_token: str, num_sweeps: int) -> Tuple[np.ndarray]:
    """
    Args:
        nusc:
        sample_token:
        num_sweeps:

    Returns:
        points: (N_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx (== -1)]
        glob_se3_current: (4, 4) - transformation from the current lidar frame to global frame
    """
    sample = nusc.get('sample', sample_token)
    current_lidar_tk = sample['data']['LIDAR_TOP']
    
    glob_se3_current = get_nuscenes_sensor_pose_in_global(nusc, current_lidar_tk)
    current_se3_glob = np.linalg.inv(glob_se3_current)

    sweeps_info = get_sweeps_token(nusc, current_lidar_tk, n_sweeps=num_sweeps, return_time_lag=True, return_sweep_idx=True)
    points, list_lidar_coord = list(), list()
    for (lidar_tk, timelag, sweep_idx) in sweeps_info:
        pcd = get_one_pointcloud(nusc, lidar_tk)
        
        # remove ego points
        mask_ego_points = np.all(np.abs(pcd[:, :2]) < 2.0, axis=1)
        pcd = pcd[np.logical_not(mask_ego_points)]

        pcd = np.pad(pcd, pad_width=[(0, 0), (0, 3)], constant_values=-1)
        pcd[:, -3] = timelag
        pcd[:, -2] = sweep_idx
        # pcd[:, -1] is instance index which is -1 in the context of UDA

        # map pcd to current frame (EMC)
        glob_se3_past =  get_nuscenes_sensor_pose_in_global(nusc, lidar_tk)
        current_se3_past = current_se3_glob @ glob_se3_past
        apply_se3_(current_se3_past, points_=pcd)
        points.append(pcd)
        
        # log coord of lidar, when this sweep in collected, in "current lidar" frame
        lidar_coord = np.pad(current_se3_past[:3, -1], pad_width=[(0, 1)], constant_values=sweep_idx)  # (x, y, z, sweep_idx)
        list_lidar_coord.append(lidar_coord)
        
    points = np.concatenate(points, axis=0)  # (N_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx] 

    return points, glob_se3_current


def map_points_on_traj_to_local_frame(points: np.ndarray, 
                                      boxes: np.ndarray, 
                                      num_sweeps: int) -> np.ndarray:
    """
    Map points scatter along a trajectory of an object to the object's local frame. 
    Note points & boxes must be in the same frame (e.g., global frame or lidar frame)

    Args:
        points: (N_pts, 3 + C + 2) - x, y, z, C-channel, [sweep_idx, instance_idx (always=-1)]
        boxes: (N_b, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
        num_sweeps:
    
    Returns:
        points_in_box: (N_pts, 3 + C + 2)
    """
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    zeros, ones = np.zeros(boxes.shape[0]), np.ones(boxes.shape[0])
    batch_glob_se3_boxes = np.stack([
        cos,    -sin,       zeros,      boxes[:, 0],
        sin,     cos,       zeros,      boxes[:, 1],
        zeros,   zeros,     ones,       boxes[:, 2],
        zeros,   zeros,     zeros,      ones
    ], axis=1).reshape(-1, 4, 4)
    batch_boxes_se3_glob = np.linalg.inv(batch_glob_se3_boxes)  # (N_valid_sw, 4, 4)

    boxes_sweep_idx = boxes[:, -1].astype(int)
    assert np.unique(boxes_sweep_idx).shape[0] == boxes_sweep_idx.shape[0]
    assert num_sweeps >= boxes_sweep_idx.shape[0]  # assert N_sw >= N_valid_sw 
    
    pad_batch_boxes_se3_glob = np.tile(np.eye(4).reshape(1, 4, 4), (num_sweeps,1, 1))
    pad_batch_boxes_se3_glob[boxes_sweep_idx] = batch_boxes_se3_glob  # (N_sw, 4, 4)  ! N_sw >= N_valid_sw
    
    points_sweep_idx = points[:, -2].astype(int)  # (N_pts,)
    perpoint_box_se3_glob = pad_batch_boxes_se3_glob[points_sweep_idx]  # (N_pts, 4, 4)
    points_in_box = np.einsum('bik, bk -> bi', perpoint_box_se3_glob[:, :3, :3], points[:, :3]) \
        + perpoint_box_se3_glob[:, :3, -1]
    
    return points_in_box
