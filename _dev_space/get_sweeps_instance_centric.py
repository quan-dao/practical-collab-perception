import numpy as np
import numpy.linalg as LA
from nuscenes.nuscenes import NuScenes
from _dev_space.tools_box import apply_tf, tf, get_sweeps_token, get_nuscenes_sensor_pose_in_global


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
                            center_radius=2.0, in_box_tolerance=1e-2) -> dict:
    """
    Returns:
        'points' (np.ndarray): (N, 7) - x, y, z, intensity, time-lag, sweep_idx, instance_idx
        'instances' (list): [[target_from_inst0@-10, ..., target_from_inst0@0]]
        'instances_sweep_indices' (list): [[inst0_sweep_idxFirst, ..., inst0_sweep_idxFinal]]
    """
    sample_rec = nusc.get('sample', sample_token)
    target_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, target_sd_token, n_sweeps, return_time_lag=True, return_sweep_idx=True)

    target_from_glob = LA.inv(get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))

    inst_token_2_index = dict()  # use this to access "instances"
    inst_idx = 0
    instances = list()  # for each instance, store list of poses
    instances_sweep_indices = list()  # for each instance, store list of sweep index
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

            if anno_rec['num_lidar_pts'] == 0:
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
            else:
                cur_instance_idx = inst_token_2_index[inst_token]
                instances[cur_instance_idx].append(target_from_box)
                instances_sweep_indices[cur_instance_idx].append(s_idx)

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

    return {
        'points': all_points,
        'instances_tf': instances_tf,
        # 'instances': instances,
        # 'instances_sweep_indices': instances_sweep_indices
    }




