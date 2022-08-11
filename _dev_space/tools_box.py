from pyquaternion import Quaternion
import numpy as np
from nuscenes.nuscenes import NuScenes
from get_clean_pointcloud import show_pointcloud


DYNAMIC_CLASSES = ('vehicle', 'human')  # 'human'
CENTER_RADIUS = 1.


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


def get_nuscenes_pointcloud(nusc: NuScenes, sample_data_token: str, center_radius=CENTER_RADIUS, ground_height=None,
                            return_foreground_mask=False, dyna_cls=DYNAMIC_CLASSES, box_tol=5e-2,
                            pc_time=None) -> tuple:
    pcfile = nusc.get_sample_data_path(sample_data_token)
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)
    if pc_time is not None:
        assert isinstance(pc_time, float)
        pc = np.concatenate([pc, np.tile(np.array([[pc_time]]), (pc.shape[0], 1))], axis=1)
    # remove ego points
    too_close_mask = np.square(pc[:, :2]).sum(axis=1) < center_radius**2
    pc = pc[np.logical_not(too_close_mask)]
    # remove ground using lidar height
    if ground_height is not None:
        assert isinstance(ground_height, float)
        above_ground_mask = pc[:, 2] > ground_height
        pc = pc[above_ground_mask]

    if return_foreground_mask:
        boxes = nusc.get_boxes(sample_data_token)  # in global
        mask_foreground = -np.ones(pc.shape[0])
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, sample_data_token)
        for bidx, box in enumerate(boxes):
            if box.name.split('.')[0] not in dyna_cls:
                continue
            anno_rec = nusc.get('sample_annotation', box.token)
            if anno_rec['num_lidar_pts'] == 0:
                continue
            # map pc to box's local frame
            box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
            pts_wrt_box = apply_tf(box_from_glob @ glob_from_curr, pc[:, :3])
            mask_inside_box = np.all(
                np.abs(pts_wrt_box / np.array([box.wlh[1], box.wlh[0], box.wlh[2]])) < (0.5 + box_tol),
                axis=1
            )
            mask_foreground[mask_inside_box] = bidx

        return pc, mask_foreground
    else:
        return pc, None


def get_nuscenes_pointcloud_in_target_frame(nusc: NuScenes, curr_sd_token: str, target_sd_token: str = None,
                                            center_radius=CENTER_RADIUS, ground_height: float = None,
                                            return_foreground_mask=False,
                                            pc_range: list = None):
    """
    Get pointcloud and map its points to the target frame represented by target_sd_token

    Args:
        nusc:
        curr_sd_token: sample data from which pointcloud is obtained
        target_sd_token: sample data w.r.t pointcloud is expressed
        center_radius: to filter ego vehicle points
        ground_height: to filter points on & below ground
        return_foreground_mask:
        pc_range:
    Returns:
        (np.ndarray): (N, 3+C)
    """
    pc, mask_foreground = get_nuscenes_pointcloud(nusc, curr_sd_token, center_radius, ground_height,
                                                  return_foreground_mask)
    if target_sd_token is not None:
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
        glob_from_target = get_nuscenes_sensor_pose_in_global(nusc, target_sd_token)
        target_from_curr = np.linalg.inv(glob_from_target) @ glob_from_curr
        pc[:, :3] = apply_tf(target_from_curr, pc[:, :3])

    if pc_range is not None:
        assert len(pc_range) == 6
        if not isinstance(pc_range, np.ndarray):
            pc_range = np.array(pc_range)
        in_range_mask = np.all((pc[:, :3] >= pc_range[:3]) & (pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
        pc = pc[in_range_mask]
        if mask_foreground is not None:
            mask_foreground = mask_foreground[in_range_mask]

    return pc, mask_foreground


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


def get_pointcloud_propagated_by_annotations(nusc, curr_sd_token: str, target_sd_token: str,
                                             dyna_cls=DYNAMIC_CLASSES, center_radius=CENTER_RADIUS,
                                             pc_range=None, ground_height=None,
                                             tol=1e-2):
    curr_pc = get_nuscenes_pointcloud(nusc, curr_sd_token, center_radius, ground_height)  # in curr_sd frame
    glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
    curr_pc[:, :3] = apply_tf(glob_from_curr, curr_pc[:, :3])  # in global

    curr_boxes = nusc.get_boxes(curr_sd_token)  # in global
    # partition curr_pc
    curr_annos = {}  # instance_token: points inside
    for box in curr_boxes:
        if box.name.split('.')[0] not in dyna_cls:
            continue
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['num_lidar_pts'] == 0:
            continue
        box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
        pts_wrt_box = apply_tf(box_from_glob, curr_pc[:, :3])
        in_box_mask = np.all(np.abs(pts_wrt_box / np.array([box.wlh[1], box.wlh[0], box.wlh[2]])) < (0.5 + tol), axis=1)
        # store points inside the box
        curr_annos[anno_rec['instance_token']] = pts_wrt_box[in_box_mask]  # in box
        # remove points inside the box from pointcloud
        curr_pc = curr_pc[~in_box_mask]  # in global

    # propagate to target_sd frame
    glob_from_target = get_nuscenes_sensor_pose_in_global(nusc, target_sd_token)
    dyn_points_in_target = []
    for box in nusc.get_boxes(target_sd_token):
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['instance_token'] in curr_annos:
            box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
            box_from_target = box_from_glob @ glob_from_target
            target_from_box = np.linalg.inv(box_from_target)
            dyn_points_in_target.append(apply_tf(target_from_box, curr_annos[anno_rec['instance_token']]))

    curr_pc[:, :3] = apply_tf(np.linalg.inv(glob_from_target), curr_pc[:, :3])
    prop_pc = np.vstack((curr_pc[:, :3], *dyn_points_in_target))
    if pc_range is not None:
        assert len(pc_range) == 6
        if not isinstance(pc_range, np.ndarray):
            pc_range = np.array(pc_range)
        in_range_mask = np.all((prop_pc[:, :3] >= pc_range[:3]) & (prop_pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
        return prop_pc[in_range_mask]
    else:
        return prop_pc


def get_boxes_4viz(nusc, sample_data_token):
    boxes = nusc.get_boxes(sample_data_token)  # global
    # a box's corners in its local frame
    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    corners = np.stack([xs, ys, zs], axis=1)
    out = []
    curr_from_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample_data_token))
    for box in boxes:
        if box.name.split('.')[0] not in DYNAMIC_CLASSES:
            continue
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['num_lidar_pts'] == 0:
            continue
        glob_from_box = tf(box.center, box.orientation)
        curr_from_box = curr_from_glob @ glob_from_box
        this_box_corners = corners * np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
        out.append(apply_tf(curr_from_box, this_box_corners))
    return out


def get_sweeps_token(nusc: NuScenes, curr_sd_token: str, n_sweeps: int, return_time_lag=True) -> list:
    ref_sd_rec = nusc.get('sample_data', curr_sd_token)
    ref_time = ref_sd_rec['timestamp'] * 1e-6
    sd_tokens_times = []
    for _ in range(n_sweeps):
        if curr_sd_token == '':
            break
        curr_sd = nusc.get('sample_data', curr_sd_token)
        sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6))
        # move to previous
        curr_sd_token = curr_sd['prev']

    # organize from PAST to PRESENCE
    sd_tokens_times.reverse()

    if return_time_lag:
        return sd_tokens_times
    else:
        return [token for token, _ in sd_tokens_times]
