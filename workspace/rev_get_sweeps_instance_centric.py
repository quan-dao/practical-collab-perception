import numpy as np
import numpy.linalg as LA
import torch
from nuscenes.nuscenes import NuScenes
from _dev_space.tools_box import apply_tf, tf, get_sweeps_token, get_nuscenes_sensor_pose_in_global
from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import functools
import operator


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


def revised_instance_centric_get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int,
                                        detection_classes: list = ['car', 'pedestrian', 'bicycle'], 
                                        **kwargs) -> dict:
    """
    
    Returns:
        out: {
            'points' (np.ndarray): (N, 5 [+ 2]) - x, y, z, intensity, time-lag, [sweep_idx, instance_idx]
            'instances_tf' (np.ndarray): (N_in, N_sw, 4, 4)
            'instances_box' (np.ndarray): (N_in, 7 [+ 2]) - x, y, z, dx, dy, dz, yaw, vx, vy
            'instances_name' (np.ndarray): (N_in,) - class of instance
        }
    """
    sample_rec = nusc.get('sample', sample_token)
    target_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, target_sd_token, n_sweeps, return_time_lag=True, return_sweep_idx=True)
    target_from_glob = LA.inv(get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))
    
    # -------------------------- #
    # ------- points ----------- #
    # -------------------------- #
    points = []

    for sd_token, time_lag, s_idx in sd_tokens_times:
        glob_from_cur = get_nuscenes_sensor_pose_in_global(nusc, sd_token)
        # get points in current frame
        cur_points = get_sample_data_point_cloud(nusc, sd_token, time_lag, s_idx)  # (N, 6) - x, y, z, intensity, time-lag, sweep_idx | in current  
        cur_points = remove_ego_vehicle_points(cur_points, kwargs.get('center_radius', 2.0))
        # map to target frame
        cur_points[:, :3] = apply_tf(target_from_glob @ glob_from_cur, cur_points[:, :3])  # (N, 6) | in target frame
        # store current points
        if cur_points.shape[0] > 0:
            points.append(cur_points)
    
    points = np.concatenate(points, axis=0)  # (N_pts, 6) - x, y, z, intensity, time-lag, sweep_idx | in target frame

    # ----------------------------- #
    # --------- instances --------- #
    # ----------------------------- # 
    inst_token_2_inst_idx = dict()
    inst_idx = 0
    inst_poses = list()  # for each instance, store list of poses in `target frame`
    inst_size = list()  # for each instance, store its first size -> to build last box
    inst_name = list()  # for each instance, store its first name -> to build last box's name
    inst_last_anno_tk = list()  # for each instance, store the token of its latest annotation for computing velocity

    for sd_token, _, s_idx in sd_tokens_times:
        boxes = nusc.get_boxes(sd_token)
        for box in boxes:
            box_det_name = map_name_from_general_to_detection[box.name]
            if box_det_name not in detection_classes:
                continue
            
            anno_rec = nusc.get('sample_annotation', box.token)
            if anno_rec['num_lidar_pts'] < 1:
                continue

            # map box to target
            glob_from_box = tf(box.center, box.orientation)
            target_from_box = target_from_glob @ glob_from_box

            # build-up instance's info
            inst_token = anno_rec['instance_token']
            if inst_token not in inst_token_2_inst_idx:  # new instance
                inst_token_2_inst_idx[inst_token] = inst_idx
                inst_idx += 1
                # init instance's info
                inst_poses.append([target_from_box])
                # ---
                inst_size.append([box.wlh[1], box.wlh[0], box.wlh[2]])
                inst_name.append(box_det_name)
                # ---
                inst_last_anno_tk.append(anno_rec['token'])
            else:  # previously saw instance
                instance_idx = inst_token_2_inst_idx[inst_token]
                inst_poses[instance_idx].append(target_from_box)
                # overwrite latest annotation token
                inst_last_anno_tk[instance_idx] = anno_rec['token']

    # compute instances_tf
    num_instances = len(inst_poses)

    if num_instances == 0:
        points = np.pad(points, pad_width=[(0, 0), (0, 1)], constant_values=-1)  # all background
        boxes = np.zeros((0, 9))
        boxes_name = np.array([])
        instances_tf = np.zeros((num_instances, n_sweeps, 4, 4))
        return {'points': points, 'instances_tf': instances_tf, 'gt_boxes': boxes, 'gt_names': boxes_name}

    instances_tf = np.zeros((num_instances, n_sweeps, 4, 4))
    for i_idx in range(num_instances):
        poses = np.stack(inst_poses[i_idx], axis=0)  # (N_act_sw, 4, 4) 
        instances_tf[i_idx, :poses.shape[0]] = np.einsum('ij, bjk -> bik', poses[-1], LA.inv(poses))

    # get instances' latest box
    inst_last_pose = np.stack([pose[-1] for pose in inst_poses], axis=0)  # (N_inst, 4, 4)
    boxes_yaw = np.arctan2(inst_last_pose[:, 1, 0], inst_last_pose[:, 0, 0])  # (N_inst,)
    boxes_velo = np.stack([nusc.box_velocity(anno_tk) for anno_tk in inst_last_anno_tk], axis=0)  # (N_inst, 3) - vx, vy, vz in global frame
    boxes_velo = boxes_velo @ target_from_glob[:3, :3].T   # in target frame
    boxes = np.concatenate([inst_last_pose[:, :3, -1], np.array(inst_size), boxes_yaw.reshape(-1, 1), boxes_velo[:, :2]], axis=1)  # (N_inst, 9)
    boxes_name = np.array(inst_name)

    # ----------------------------------------------------- #
    # --------- points to instances correspondent --------- #
    # ----------------------------------------------------- #
    inst_inst_idx = [[i] * len(poses) for i, poses in enumerate(inst_poses)]
    # flatten inst_inst_idx & inst_poses
    inst_poses = np.stack(functools.reduce(operator.iconcat, inst_poses, []), axis=0)  # (N_tot_poses, 4, 4)
    inst_inst_idx = np.array(functools.reduce(operator.iconcat, inst_inst_idx, [])).astype(int)  # (N_tot_poses,)
    
    # convert inst_poses to yaw
    inst_yaw = np.arctan2(inst_poses[:, 1, 0], inst_poses[:, 0, 0])  # (N_tot_poses)
    
    # assembly instances' boxes
    inst_size = np.array(inst_size)  # (N_inst, 3)
    inst_boxes = np.concatenate([inst_poses[:, :3, -1], inst_size[inst_inst_idx], inst_yaw.reshape(-1, 1)], axis=1)  # (N_tot_poses, 7)
    
    # establish points to inst_boxes correspondent
    box_ids_of_points = points_in_boxes_gpu(
        torch.from_numpy(points[:, :3]).unsqueeze(0).contiguous().float().cuda(),
        torch.from_numpy(inst_boxes).unsqueeze(0).contiguous().float().cuda(),
    ).long().squeeze(0).cpu().numpy()  # (N_pts,) to index into (N_tot_poses)

    # transfer boxes' inst_idx (inst_inst_idx) to points
    points_inst_idx = inst_inst_idx[box_ids_of_points]
    points_inst_idx[box_ids_of_points == -1] = -1  # background

    points = np.concatenate([points, points_inst_idx.reshape(-1, 1).astype(float)], axis=1)

    out = {'points': points, 'instances_tf': instances_tf, 'gt_boxes': boxes, 'gt_names': boxes_name}
    return out



