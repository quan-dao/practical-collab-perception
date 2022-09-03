import numpy as np
from sklearn.cluster import DBSCAN
from nuscenes.nuscenes import NuScenes
from typing import Tuple, List
from _dev_space.tools_box import get_nuscenes_pointcloud_in_target_frame, get_sweeps_token, check_list_to_numpy, \
    get_nuscenes_sweeps_partitioned_by_instances

from _dev_space.tools_box import apply_tf, get_nuscenes_sensor_pose_in_global, tf
import numpy.linalg as LA


def get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int, correct_dyna_pts=False, use_gt_fgr=False,
               center_radius=2., pc_range=None, bev_pix_size=None, dbscann_eps=7.0, dbscann_min_samples=5,
               threshold_velo_std_dev=4.50, return_points_offset=False, debug=False) -> List:
    """
    Accumulate n_sweeps to make a NuScenes pointcloud. By default, these sweeps are propagated to the timestamp of
    sample_token using Ego Motion Compensation (EMC). If correct_dyna_pts, foreground points are clustered across
    space, then each cluster is splited into groups based on timestamp. Older groups will be moved so that their
    centroids are coincide with the centroid of the most recent group.

    Args:
        nusc:
        sample_token: a keyframe
        n_sweeps:
        correct_dyna_pts:
        use_gt_fgr:
        center_radius: to filter points too close to LiDAR
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_pix_size (float):
        dbscann_eps:
        dbscann_min_samples:
        threshold_velo_std_dev:
        return_points_offset:
        debug: if True, return both EMC-only & correct sweeps
    Return:
         sweeps: (N, 3+C) - in sample's LiDAR frame
         mask_fgr: (N,) - bool | True for foreground points
    """
    if not use_gt_fgr:
        raise NotImplementedError

    sample = nusc.get('sample', sample_token)
    ref_sd_token = sample['data']['LIDAR_TOP']

    sd_tokens_times = get_sweeps_token(nusc, ref_sd_token, n_sweeps, return_time_lag=True)
    sweeps, mask_fgr = [], []
    for sd_token, sd_time_lag in sd_tokens_times:
        pc, m_fgr = get_nuscenes_pointcloud_in_target_frame(nusc, sd_token, ref_sd_token, return_foreground_mask=True,
                                                            center_radius=center_radius, pc_time=sd_time_lag)
        sweeps.append(pc)
        mask_fgr.append(m_fgr)
    sweeps = np.vstack(sweeps)
    mask_fgr = np.concatenate(mask_fgr, axis=0)

    # get points in pc_range
    pc_range = check_list_to_numpy(pc_range)
    mask_in_range = np.all((sweeps[:, :3] >= pc_range[:3]) & (sweeps[:, :3] < (pc_range[3:] - 1e-3)), axis=1)
    sweeps, mask_fgr = sweeps[mask_in_range], mask_fgr[mask_in_range]

    if not correct_dyna_pts:
        return [sweeps, mask_fgr]

    assert pc_range is not None and bev_pix_size is not None
    if debug:
        _emc_sweeps, _emc_mask_fgr = np.copy(sweeps), np.copy(mask_fgr)

    # Note: why set min_samples high (e.g., 5 pixels)? not moving obj -> less points -> treat them like static
    dbscanner = DBSCAN(eps=dbscann_eps, min_samples=dbscann_min_samples)

    # split sweeps into foreground & background
    background, foreground = sweeps[~mask_fgr], sweeps[mask_fgr]

    # correct foreground
    crt_fgr, not_crt, crt_fgr_offset = correct_points(foreground, dbscanner, pc_range, bev_pix_size,
                                                      threshold_velo_std_dev, return_points_offset)
    if not_crt.size > 0:
        background = np.vstack([background, not_crt])

    corrected_sweeps = np.vstack([background, crt_fgr]) if crt_fgr.size > 0 else background
    if return_points_offset:
        bgr_offset = np.zeros((background.shape[0], 2))
        sweeps_offset = np.vstack([bgr_offset, crt_fgr_offset]) if crt_fgr_offset.size > 0 else bgr_offset
        corrected_sweeps = np.concatenate([corrected_sweeps, sweeps_offset], axis=1)

    mask_corrected_sweeps_fgr = np.zeros(corrected_sweeps.shape[0], dtype=bool)
    mask_corrected_sweeps_fgr[background.shape[0]:] = True
    out = [corrected_sweeps, mask_corrected_sweeps_fgr]
    if debug:
        out.extend([_emc_sweeps, _emc_mask_fgr])
    return out


def correct_points(pts: np.ndarray, dbscanner, pc_range: np.ndarray, bev_pix_size: float,
                   threshold_velo_std_dev: float, return_offset_xy: bool) -> Tuple:
    """
    Points are clustered across space, then each cluster is splitted into groups based on timestamp.
    Older groups will be moved so that their centroids are coincide with the centroid of the most recent group.
    Returns:
        clusters (np.ndarray): (N_dyna, 3+C) - corrected points
        unclustered_pts (np.ndarray): (N_new_bgr, 3+C) - points that are not allocated to any clusters
    """
    if pts.shape[0] == 0:
        # empty foreground
        return np.array([]), np.array([]), np.array([])

    # get points' occupancy in BEV to save time doing clustering
    pts_pixel_coord = np.floor((pts[:, :2] - pc_range[:2]) / bev_pix_size).astype(int)
    bev_size = int((pc_range[3] - pc_range[0]) / bev_pix_size)
    pts_1d = pts_pixel_coord[:, 1] * bev_size + pts_pixel_coord[:, 0]
    occ_1d = np.unique(pts_1d)
    occ_2d = np.stack((occ_1d % bev_size, occ_1d // bev_size), axis=1)  # (N_occ, 2)

    # clustering occupied pixels in BEV
    dbscanner.fit(occ_2d)
    occ_labels = dbscanner.labels_  # (N_occ,)

    # iter clusters, split each cluster into groups using timestamp, offset older group to have the same
    # centroids as the most recent group
    mask_corrected_pts = np.zeros(pts.shape[0], dtype=bool)  # to keep track of pts that are corrected
    clusters, clusters_offset_xy = [], []
    for cluster_id in np.unique(occ_labels):
        if cluster_id == -1:
            # noise cluster -> skip
            continue
        # ---
        # find pts (in 3D) that are contained by this cluster (currently expressed in BEV)
        # ---
        occ_in_cluster = occ_2d[occ_labels == cluster_id]
        # smallest bounding rectangle (in BEV) of this cluster
        min_xy = np.amin(occ_in_cluster, axis=0)
        max_xy = np.amax(occ_in_cluster, axis=0)
        # find pts whose pixel coords fall inside the smallest bounding rectangle
        mask_pts_in_cluster = np.all((pts_pixel_coord >= min_xy) & (pts_pixel_coord <= max_xy), axis=1)

        # decide whether to correct cluster based on area
        cluster_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]) * (bev_pix_size ** 2)
        if cluster_area > 120:
            # too large -> contains more than 1 obj -> wrong cluster -> not correct this cluster
            continue

        # correct this cluster in a sequential manner
        pts_in_cluster = pts[mask_pts_in_cluster]  # (N_p, 3+C)
        pts_timestamp = pts_in_cluster[:, 4]
        unq_ts = np.unique(pts_timestamp).tolist()
        unq_ts.reverse()  # to keep the most recent timestamp at the last position
        window, window_offset_xy, velos = np.array([]), np.array([]), []
        for ts_idx in range(len(unq_ts)):
            cur_group = np.copy(pts_in_cluster[pts_timestamp == unq_ts[ts_idx]])
            if window.size == 0:
                window = cur_group
                window_offset_xy = np.zeros((cur_group.shape[0], 2))
                continue
            # calculate vector going from window's center toward cur_group's center
            win_to_cur = np.mean(cur_group[:, :2], axis=0) - np.mean(window[:, :2], axis=0)
            # calculate window's velocity
            delta_t = -unq_ts[ts_idx] + unq_ts[ts_idx - 1]
            velos.append(np.linalg.norm(win_to_cur) / delta_t)
            # correct window & merge with cur_group
            window[:, :2] += win_to_cur
            window_offset_xy += win_to_cur  # to keep track of how points have been moved
            window = np.vstack([window, cur_group])
            window_offset_xy = np.vstack([window_offset_xy, np.zeros((cur_group.shape[0], 2))])

        # decide whether to keep corrected cluster based on std dev of velocity
        if not velos:
            # empty velos cuz cluster only has 1 group (i.e. 1 timestamp)
            continue
        velo_var = np.mean(np.square(np.array(velos) - np.mean(velos)))
        if np.sqrt(velo_var) > threshold_velo_std_dev:
            # too large -> not keep corrected cluster
            continue
        else:
            # keep corrected cluster
            mask_corrected_pts = mask_corrected_pts | mask_pts_in_cluster
            clusters.append(window)
            clusters_offset_xy.append(window_offset_xy)

    # format output
    clusters = np.vstack(clusters) if clusters else np.array([])
    unclustered_pts = pts[~mask_corrected_pts]  # some pts are not corrected -> group them into a cluster
    if return_offset_xy:
        clusters_offset_xy = np.vstack(clusters_offset_xy) if clusters_offset_xy else np.array([])
        return clusters, unclustered_pts, clusters_offset_xy
    else:
        return clusters, unclustered_pts, np.array([])


def get_sweeps_for_distillation(nusc: NuScenes, sample_token: str, n_sweeps: int, pc_range=None,
                                bev_pix_size=None, dbscann_eps=7.0, dbscann_min_samples=5,
                                threshold_velo_std_dev=4.50, return_points_offset=False) -> dict:
    sweeps = get_nuscenes_sweeps_partitioned_by_instances(nusc, sample_token, n_sweeps, is_distill=True,
                                                          is_foreground_seg=False)
    # get points in pc_range
    pc_range = check_list_to_numpy(pc_range)
    mask_in_range = np.all((sweeps[:, :3] >= pc_range[:3]) & (sweeps[:, :3] < (pc_range[3:] - 1e-3)), axis=1)
    sweeps = sweeps[mask_in_range]
    indicator = sweeps[:, -1].astype(int)

    # correct foreground points that just underwent EMC
    dbscanner = DBSCAN(eps=dbscann_eps, min_samples=dbscann_min_samples)
    foreground = sweeps[indicator == 0]
    crt_fgr, not_crt, crt_fgr_offset = correct_points(foreground, dbscanner, pc_range, bev_pix_size,
                                                      threshold_velo_std_dev, return_points_offset)
    if not_crt.size > 0:
        not_crt[:, -1] = -2  # to indicate "kind-of" background (which will be used by student)

    out = {
        'background': sweeps[indicator == -1],
        'corrected_fgr': crt_fgr,  # indicator == 0
        'not_corrected_fgr': not_crt,  # indicator == -2
        'gt_corrected_fgr': sweeps[indicator == 1]
    }
    if return_points_offset:
        out['corrected_fgr_offset'] = crt_fgr_offset
    return out


def get_sweeps_for_foreground_seg(nusc: NuScenes, sample_token: str, n_sweeps: int) -> dict:
    sweeps = get_nuscenes_sweeps_partitioned_by_instances(nusc, sample_token, n_sweeps, is_distill=False,
                                                          is_foreground_seg=True)
    n_original_instances = int(np.max(sweeps[:, -1])) + 1
    return {'points': sweeps, 'n_original_instances': n_original_instances}


def e2e_crt_get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int) -> dict:
    """
    Returns:
        {
            'points': (N, 8) - xyz, intensity, time, target_offset_xy, indicator
            'num_original_instances': (int)
        }
    """
    sample_rec = nusc.get('sample', sample_token)
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, ref_sd_token, n_sweeps, return_time_lag=True)

    # build {'instance_token': pose @ curr ego} to map foreground points to target frame in an oracle way
    ref_from_glob = LA.inv(get_nuscenes_sensor_pose_in_global(nusc, ref_sd_token))
    inst_token_to_pose = dict()
    for sd_token, _ in sd_tokens_times[::-1]:
        # iterate sd_token_times backward to populate inst_token_to_pose
        # -> value of inst_token_to_pose the most recent pose in target frame of any intances
        boxes = nusc.get_boxes(sd_token)
        for box in boxes:
            if box.name.split('.')[0] not in ('human', 'vehicle'):
                continue
            anno_rec = nusc.get('sample_annotation', box.token)
            if anno_rec['instance_token'] not in inst_token_to_pose:
                glob_from_box = tf(box.center, box.orientation)
                inst_token_to_pose[anno_rec['instance_token']] = ref_from_glob @ glob_from_box

    # to compute label for offset toward cluster center
    # -> keep track of which foreground points belong to same cluster
    inst_token_to_index = dict(zip(inst_token_to_pose.keys(), range(len(inst_token_to_pose))))

    # process sweeps to get points
    points = [e2e_crt_process_1sweep(nusc, sd_token, ref_sd_token, timelag, inst_token_to_pose, inst_token_to_index)
              for sd_token, timelag in sd_tokens_times]
    points = np.vstack(points)
    return {'points': points, 'num_original_instances': len(inst_token_to_pose)}


def e2e_crt_process_1sweep(nusc: NuScenes, curr_sd_token: str, target_sd_token: str, sweep_timelag: float,
                           inst_token_to_pose: dict, inst_token_to_index: dict,
                           box_tol=1e-2, center_radius=2.) -> np.ndarray:
    """
    Returns:
        points: (N, 8) - xyz, intensity, time, target_offset_xy, indicator (-1: background, >= 0 foreground)
    """
    # read pointcloud from file
    pcfile = nusc.get_sample_data_path(curr_sd_token)
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (N, 4) - (x, y, z, intensity)
    pc = np.pad(pc, pad_width=[(0, 0), (0, 1)], mode='constant',
                constant_values=sweep_timelag)  # (N, 5) - (xyz, intensity, time)

    # remove ego points
    dist_to_sensor = LA.norm(pc[:, :2], axis=1)
    pc = pc[dist_to_sensor > center_radius]

    # map pc to global frame
    glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
    pc[:, :3] = apply_tf(glob_from_curr, pc[:, :3])

    # map pc to target frame
    target_from_glob = LA.inv(get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))
    xyz_wrt_target = apply_tf(target_from_glob, pc[:, :3])

    # ---
    # split pc to background & foreground
    # compute offset from foreground transmitted by EMC & "properly aligned" foreground
    # ---
    boxes = nusc.get_boxes(curr_sd_token)
    indicator = -np.ones(pc.shape[0])
    emc2oracle = np.zeros((pc.shape[0], 2))

    for bidx, box in enumerate(boxes):
        if box.name.split('.')[0] not in ('human', 'vehicle'):
            continue
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['num_lidar_pts'] == 0:
            continue

        # transform pc to box local frame
        box_from_glob = LA.inv(tf(box.center, box.orientation))
        xyz_wrt_box = apply_tf(box_from_glob, pc[:, :3])

        # find pts inside box
        mask_inside_box = np.all(
            np.abs(xyz_wrt_box / np.array([box.wlh[1], box.wlh[0], box.wlh[2]])) < (0.5 + box_tol),
            axis=1
        )
        # update indicator of points inside this box
        indicator[mask_inside_box] = inst_token_to_index[anno_rec['instance_token']]

        # transmit fgr to target frame using pose of box in target frame
        box_pts_emc = xyz_wrt_target[mask_inside_box]  # (N_bpts, 3)
        box_pts_oracle = apply_tf(inst_token_to_pose[anno_rec['instance_token']], xyz_wrt_box[mask_inside_box])  # (N_bpts, 3)
        emc2oracle[mask_inside_box] = box_pts_oracle[:, :2] - box_pts_emc[:, :2]

    # format output
    sweep = np.concatenate([
        xyz_wrt_target,
        pc[:, 3:],  # 2d vector: intensity, time
        emc2oracle,  # 2d vector: offset along X,Y- axis of target frame
        indicator[:, np.newaxis],  # -1 for background, >= 0 for foreground
    ], axis=1)  # (N, 8)
    return sweep




