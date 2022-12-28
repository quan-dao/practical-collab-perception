import numpy as np
import numpy.linalg as LA
from nuscenes.nuscenes import NuScenes
from _dev_space.tools_box import apply_tf, tf, get_sweeps_token, get_nuscenes_sensor_pose_in_global
from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection


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
                            center_radius=2.0, in_box_tolerance=5e-2,
                            return_instances_last_box=True,
                            point_cloud_range=None,
                            detection_classes=('car', 'pedestrian', 'bicycle'),
                            map_point_feat2idx: dict = None,
                            prob_drop_instance_pts=0.5) -> dict:
    """
    Returns:
        {
            'points' (np.ndarray): (N, 9) - x, y, z, intensity, time-lag, sweep_idx, instance_idx, aug_inst_idx, class_idx
            'instances' (list): [[target_from_inst0@-10, ..., target_from_inst0@0]]
            'instances_sweep_indices' (list): [[inst0_sweep_idxFirst, ..., inst0_sweep_idxFinal]]
        }
        return_instances_last_box:
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
    instances_name = list()  # for each instance, store its detection name
    inst_tk_2_sample_tk = dict()  # to store the sample_tk where an inst last appears
    inst_latest_anno_tk = list()
    all_points = []

    for sd_token, time_lag, s_idx in sd_tokens_times:
        glob_from_cur = get_nuscenes_sensor_pose_in_global(nusc, sd_token)
        cur_points = get_sample_data_point_cloud(nusc, sd_token, time_lag, s_idx)  # (N, 6), in "cur" frame
        cur_points = remove_ego_vehicle_points(cur_points, center_radius)

        # map to target
        cur_points[:, :3] = apply_tf(target_from_glob @ glob_from_cur, cur_points[:, :3])  # (N, 6) in target frame

        # pad points with instances index, augmented instance index & class index
        cur_points = np.pad(cur_points, pad_width=[(0, 0), (0, 3)], constant_values=-1)

        boxes = nusc.get_boxes(sd_token)

        for b_idx, box in enumerate(boxes):
            box_det_name = map_name_from_general_to_detection[box.name]
            if box_det_name not in detection_classes:
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
                # store name
                instances_name.append(box_det_name)
                # store anno token to be used later for calculating box's velocity
                inst_latest_anno_tk.append(anno_rec['token'])
            else:
                cur_instance_idx = inst_token_2_index[inst_token]
                instances[cur_instance_idx].append(target_from_box)
                instances_sweep_indices[cur_instance_idx].append(s_idx)
                # update anno token to be used later for calculating box's velocity
                inst_latest_anno_tk[cur_instance_idx] = anno_rec['token']

            inst_tk_2_sample_tk[inst_token] = anno_rec['sample_token']

            # set points' instance index
            mask_in = find_points_in_box(cur_points, target_from_box, np.array([box.wlh[1], box.wlh[0], box.wlh[2]]),
                                         in_box_tolerance)
            cur_points[mask_in, map_point_feat2idx['inst_idx']] = inst_token_2_index[inst_token]
            # set points' class index
            cur_points[mask_in, map_point_feat2idx['cls_idx']] = 1 + detection_classes.index(box_det_name)

            if np.any(mask_in):
                # this box has some points
                # --
                # Enlarge box to simulate False Positive foreground
                # ---
                noise_factor = np.random.uniform(in_box_tolerance, 0.35)
                large_mask_in = find_points_in_box(cur_points, target_from_box,
                                                   np.array([box.wlh[1], box.wlh[0], box.wlh[2]]), noise_factor)
                # set aug inst index
                cur_points[large_mask_in, map_point_feat2idx['aug_inst_idx']] = inst_token_2_index[inst_token]

                # --
                # Drop points in box to simulate False Negative foreground
                # ---
                points_in_box_indices = np.arange(cur_points.shape[0])[large_mask_in]
                mask_keep_points = np.random.choice([0, 1], size=points_in_box_indices.shape[0],
                                                    p=[prob_drop_instance_pts, 1.0 - prob_drop_instance_pts])
                num_total = points_in_box_indices.shape[0]
                num_keep = mask_keep_points.sum()
                # if a box has points, it should still have points after dropping
                # => if drop too many, switch back a few to keep at least 2 points
                num_to_switch = min(2, num_total) - num_keep
                if num_to_switch > 0:
                    drop_indices = np.arange(num_total)[mask_keep_points == 0]  # drop_indices of mask_keep_points
                    mask_keep_points[drop_indices[:num_to_switch]] = 1

                # drop points by setting their aug inst idx to -1
                cur_points[points_in_box_indices[mask_keep_points == 0], map_point_feat2idx['aug_inst_idx']] = -1

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
    }

    if return_instances_last_box:
        assert point_cloud_range is not None
        if not isinstance(point_cloud_range, np.ndarray):
            point_cloud_range = np.array(point_cloud_range)
        instances_last_box = np.zeros((len(instances), 9))
        # 10 == c_x, c_y, c_z, d_x, d_y, d_z, yaw, vx, vy
        # ------
        # NOTE: instances_last_box DON'T NEED TO BE CONSISTENT WITH instances_tf because aligner doesn't predict boxes
        # NOTE: instances_last_box & instances_tf are consistent here, but this consistency will be broken when
        # NOTE: boxes outside of range are removed
        # ------
        for _idx, (_size, _poses) in enumerate(zip(instances_size, instances)):
            # find the pose that has center inside point cloud range & is closest to the target time step
            # if couldn't find any, take the 1st pose (i.e. the furthest into the past)
            chosen_pose_idx = 0
            for pose_idx in range(-1, -len(_poses) - 1, -1):
                if np.all(np.logical_and(_poses[pose_idx][:3, -1] >= point_cloud_range[:3],
                                         _poses[pose_idx][:3, -1] < point_cloud_range[3:] - 1e-2)):
                    chosen_pose_idx = pose_idx
                    break
            yaw = np.arctan2(_poses[chosen_pose_idx][1, 0], _poses[chosen_pose_idx][0, 0])
            instances_last_box[_idx, :3] = _poses[chosen_pose_idx][:3, -1]
            instances_last_box[_idx, 3: 6] = np.array(_size)
            instances_last_box[_idx, 6] = yaw

            # instance velocity
            velo = nusc.box_velocity(inst_latest_anno_tk[_idx]).reshape(1, 3)  # - [vx, vy, vz] in global frame
            velo = apply_tf(target_from_glob, velo).reshape(3)[:2]  # [vx, vy] in target frame
            instances_last_box[_idx, 7: 9] = velo

        out['instances_last_box'] = instances_last_box
        out['instances_name'] = np.array(instances_name)

    return out

