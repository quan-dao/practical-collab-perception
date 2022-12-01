import numpy as np
import numpy.linalg as LA
from nuscenes.nuscenes import NuScenes
from _dev_space.tools_box import apply_tf, tf, get_sweeps_token, get_nuscenes_sensor_pose_in_global
from pcdet.datasets.nuscenes.nuscenes_utils import quaternion_yaw
from pyquaternion import Quaternion
from typing import List
from nuscenes.prediction import PredictHelper


def get_sample_data_point_cloud(nusc: NuScenes, sample_data_token: str, time_lag: float = None, sweep_idx: int = None) \
        -> np.ndarray:
    """
    Returns:
        pc: (N, 6) - (x, y, z, intensity, [time, sweep_idx])
    """
    pcfile = nusc.get_sample_data_path(sample_data_token)
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (N, 4) - (x, y, z, intensity)
    if time_lag is not None:
        assert sweep_idx is not None
        pc = np.pad(pc, pad_width=[(0, 0), (0, 2)], mode='constant', constant_values=0)  # (N, 6)
        pc[:, -2] = time_lag
        pc[:, -1] = sweep_idx
    return pc


def remove_ego_vehicle_points(points: np.ndarray, center_radius) -> np.ndarray:
    dist_xy = LA.norm(points[:, :2], axis=1)
    return points[dist_xy > center_radius]


def find_points_in_box(points: np.ndarray, target_from_box: np.ndarray, dxdydz: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Args:
        points: (N, 3 + C) - x, y, z in target frame
        target_from_box: (4, 4) - transformation from box frame to target frame
        dxdydz: box's size
        tolerance:
    """
    box_points = apply_tf(LA.inv(target_from_box), points[:, :3])  # (N, 3)
    mask_inside = np.all(np.abs(box_points / dxdydz) < (0.5 + tolerance), axis=1)  # (N,)
    return mask_inside


def inst_centric_get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int,
                            center_radius=2.0, in_box_tolerance=1e-2,
                            return_instances_last_box=False,
                            pointcloud_range=None,
                            predict_helper: PredictHelper = None, predict_horizon=3.0, predict_freq=2) -> dict:
    """
    Returns:
        {
            'points' (np.ndarray): (N, 7) - x, y, z, intensity, time-lag, sweep_idx, instance_idx
            'instances' (list): [[target_from_inst0@-10, ..., target_from_inst0@0]]
            'instances_sweep_indices' (list): [[inst0_sweep_idxFirst, ..., inst0_sweep_idxFinal]]
        }
        return_instances_last_box:
        pointcloud_range:
    """
    sample_rec = nusc.get('sample', sample_token)
    target_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, target_sd_token, n_sweeps, return_time_lag=True, return_sweep_idx=True)

    target_from_glob = LA.inv(get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))

    inst_token_2_index = dict()  # use this to access "instances"
    inst_idx = 0
    instances = list()  # for each instance, store list of poses
    instances_sweep_indices = list()  # for each instance, store list of sweep index
    instances_size = list()  # for each instance, store its sizes (dx, dy ,dz) == l, w, h
    inst_tk_2_sample_tk = dict()  # to store the sample_tk where an inst last appears
    all_points = []

    for sd_token, time_lag, s_idx in sd_tokens_times:
        glob_from_cur = get_nuscenes_sensor_pose_in_global(nusc, sd_token)
        cur_points = get_sample_data_point_cloud(nusc, sd_token, time_lag, s_idx)  # (N, 6), in "cur" frame
        cur_points = remove_ego_vehicle_points(cur_points, center_radius)

        # map to target
        cur_points[:, :3] = apply_tf(target_from_glob @ glob_from_cur, cur_points[:, :3])  # (N, 6) in target frame

        # pad points with instances index & augmented instance index
        cur_points = np.pad(cur_points, pad_width=[(0, 0), (0, 2)], constant_values=-1)

        boxes = nusc.get_boxes(sd_token)

        for b_idx, box in enumerate(boxes):
            if box.name.split('.')[0] not in ('vehicle'):  # 'human'
                continue

            anno_rec = nusc.get('sample_annotation', box.token)

            if anno_rec['num_lidar_pts'] < 1:
                continue

            # map box to target
            glob_from_box = tf(box.center, box.orientation)
            target_from_box = target_from_glob @ glob_from_box

            # store box's pose according to the instance which it belongs to
            inst_token = anno_rec['instance_token']
            if inst_token not in inst_token_2_index:
                # new instance
                inst_token_2_index[inst_token] = inst_idx
                inst_idx += 1
                instances.append([target_from_box])
                instances_sweep_indices.append([s_idx])
                # store size
                instances_size.append([box.wlh[1], box.wlh[0], box.wlh[2]])
            else:
                cur_instance_idx = inst_token_2_index[inst_token]
                instances[cur_instance_idx].append(target_from_box)
                instances_sweep_indices[cur_instance_idx].append(s_idx)

            inst_tk_2_sample_tk[inst_token] = anno_rec['sample_token']

            # set points' instance index
            mask_in = find_points_in_box(cur_points, target_from_box, np.array([box.wlh[1], box.wlh[0], box.wlh[2]]),
                                         in_box_tolerance)
            cur_points[mask_in, -2] = inst_token_2_index[inst_token]

            if np.any(mask_in):
                # this box has some points
                # --
                # Enlarge box to simulate False Positive foreground
                # ---
                noise_factor = np.random.uniform(in_box_tolerance, 0.35)
                large_mask_in = find_points_in_box(cur_points, target_from_box,
                                                   np.array([box.wlh[1], box.wlh[0], box.wlh[2]]), noise_factor)
                # set aug inst index
                cur_points[large_mask_in, -1] = inst_token_2_index[inst_token]

                # --
                # Drop points in box to simulate False Negative foreground
                # ---
                drop_prob = 0.5
                points_in_box_indices = np.arange(cur_points.shape[0])[large_mask_in]
                mask_keep_points = np.random.choice([0, 1], size=points_in_box_indices.shape[0],
                                                    p=[drop_prob, 1.0 - drop_prob])
                num_total = points_in_box_indices.shape[0]
                num_keep = mask_keep_points.sum()
                # if a box has points, it should still have points after dropping
                # => if drop too many, switch back a few to keep at least 2 points
                num_to_switch = min(2, num_total) - num_keep
                if num_to_switch > 0:
                    drop_indices = np.arange(num_total)[mask_keep_points == 0]  # drop_indices of mask_keep_points
                    mask_keep_points[drop_indices[:num_to_switch]] = 1

                # drop points by setting their aug inst idx to -1
                cur_points[points_in_box_indices[mask_keep_points == 0], -1] = -1

        all_points.append(cur_points)

    all_points = np.concatenate(all_points, axis=0)

    # merge instances & instances_sweep_indices
    instances_tf = np.zeros((len(instances), n_sweeps, 4, 4))  # rigid tf that map fg points to their correct position
    for inst_idx in range(len(instances)):
        inst_poses = instances[inst_idx]  # list
        inst_sweep_ids = instances_sweep_indices[inst_idx]  # list
        for sw_i, pose in zip(inst_sweep_ids, inst_poses):
            instances_tf[inst_idx, sw_i] = inst_poses[-1] @ LA.inv(pose)

    out = {
        'points': all_points,
        'instances_tf': instances_tf,
        # 'instances': instances,
        # 'instances_sweep_indices': instances_sweep_indices
    }

    if return_instances_last_box:
        assert pointcloud_range is not None
        if not isinstance(pointcloud_range, np.ndarray):
            pointcloud_range = np.array(pointcloud_range)
        instances_last_box = np.zeros((len(instances), 10))
        # 10 == c_x, c_y, c_z, d_x, d_y, d_z, yaw, dummy_vx, dummy_vy, instance_index
        for _idx, (_size, _poses) in enumerate(zip(instances_size, instances)):
            # find the pose that has center inside pointclud range & is closest to the target time step
            # if couldn't find any, take the 1st pose (i.e. the furthest into the past)
            chosen_pose_idx = 0
            for pose_idx in range(-1, -len(_poses) - 1, -1):
                if np.all(np.logical_and(_poses[pose_idx][:3, -1] >= pointcloud_range[:3],
                                         _poses[pose_idx][:3, -1] < pointcloud_range[3:] -1e-3)):
                    chosen_pose_idx = pose_idx
                    break
            yaw = np.arctan2(_poses[chosen_pose_idx][1, 0], _poses[chosen_pose_idx][0, 0])
            instances_last_box[_idx, :3] = _poses[chosen_pose_idx][:3, -1]
            instances_last_box[_idx, 3: 6] = np.array(_size)
            instances_last_box[_idx, 6] = yaw
            instances_last_box[_idx, 9] = _idx
        out['instances_last_box'] = instances_last_box

    if predict_helper is not None:
        # return instance's future position as well

        waypoints = []

        for inst_tk, inst_idx in inst_token_2_index.items():
            sequence = predict_helper.get_future_for_agent(inst_tk, inst_tk_2_sample_tk[inst_tk],
                                                           seconds=predict_horizon,
                                                           in_agent_frame=False, just_xy=False)
            # sequence is List[sample_annotation], len <=6 ( = 3s * 2Hz)

            current_pose = instances[inst_idx][-1]  # (4, 4) - target_from_cur_box
            poses = [current_pose]
            for future_anno in sequence:
                glob_from_future = tf(future_anno['translation'], future_anno['rotation'])
                target_from_future = target_from_glob @ glob_from_future  # (4, 4)
                poses.append(target_from_future)

            if len(poses) == 1:
                inst_waypoints = np.array([
                    current_pose[0, -1], current_pose[1, -1],
                    np.arctan2(current_pose[1, 0], current_pose[0, 0]),
                    0,  # waypts index  to indicate the time order
                    inst_idx
                ]).reshape(1, -1)
            else:
                inst_waypoints = []
                for p_idx in range(len(poses) - 1):
                    cur_pose = poses[p_idx]
                    next_pose = poses[p_idx + 1]

                    xs = np.interp(
                        np.arange(5),  # time step to interpolate - exclude the right-end point
                        [0, 5],  # time steps where value are known
                        [cur_pose[0, -1], next_pose[0, -1]]  # values @ t0, value @ t5
                    )  # (5,)

                    ys = np.interp(np.arange(5), [0, 5], [cur_pose[1, -1], next_pose[1, -1]])  # (5,)

                    qs = Quaternion.intermediates(
                        Quaternion(matrix=cur_pose), Quaternion(matrix=next_pose), n=4, include_endpoints=True
                    )  # (6,)
                    qs = list(qs)
                    qs = [quaternion_yaw(q) for q in qs[:-1]]  # (5,) - exclude the right-end point

                    waypts_idx = np.arange(5) + p_idx * 5

                    inst_waypoints.append(np.stack([xs, ys, qs, waypts_idx], axis=1))

                inst_waypoints = np.pad(np.concatenate(inst_waypoints, axis=0),
                                        pad_width=[(0, 0), (0, 1)], constant_values=inst_idx)

            waypoints.append(inst_waypoints)

        waypoints = np.concatenate(waypoints, axis=0)
        out['instance_future_waypoints'] = waypoints  # (N_waypoints, 5) - x, y, yaw, waypoints_idx, instance_idx

    return out

