import numpy as np
from sklearn.cluster import DBSCAN
from nuscenes.nuscenes import NuScenes
from typing import Tuple, List
from _dev_space.tools_box import get_nuscenes_pointcloud_in_target_frame, get_sweeps_token, check_list_to_numpy


def get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int, correct_dyna_pts=False, use_gt_fgr=False,
               center_radius=2., pc_range=None, bev_pix_size=None, dist_xy_near_threshold=None,
               dbscann_eps=(2.0, 7.0), dbscann_min_samples=(5, 5), threshold_velo_std_dev=4.50,  # 4.5
               debug=False) -> List:
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
        dist_xy_near_threshold (float): to partition pointcloud into near & far points
        dbscann_eps: eps of (near, far)
        dbscann_min_samples: n_samples of (near, far)
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

    assert pc_range is not None and bev_pix_size is not None and dist_xy_near_threshold is not None
    if debug:
        _emc_sweeps, _emc_mask_fgr = np.copy(sweeps), np.copy(mask_fgr)

    # Note: why set min_samples high (e.g., 5 pixels)? not moving obj -> less points -> treat them like static
    dbscanner = DBSCAN(eps=7, min_samples=5)

    # split sweeps into foreground & background
    background, foreground = sweeps[~mask_fgr], sweeps[mask_fgr]

    # correct foreground
    crt_fgr, not_crt = correct_points(foreground, dbscanner, pc_range, bev_pix_size, threshold_velo_std_dev)
    if not_crt.size > 0:
        background = np.vstack([background, not_crt])

    corrected_sweeps = np.vstack([background, crt_fgr]) if crt_fgr.size > 0 else background

    mask_corrected_sweeps_fgr = np.zeros(corrected_sweeps.shape[0], dtype=bool)
    mask_corrected_sweeps_fgr[background.shape[0]:] = True
    out = [corrected_sweeps, mask_corrected_sweeps_fgr]
    if debug:
        out.extend([_emc_sweeps, _emc_mask_fgr])
    return out


def correct_points(pts: np.ndarray, dbscanner, pc_range: np.ndarray, bev_pix_size: float,
                   threshold_velo_std_dev: float) -> Tuple:
    """
    Points are clustered across space, then each cluster is splitted into groups based on timestamp.
    Older groups will be moved so that their centroids are coincide with the centroid of the most recent group.
    Returns:
        clusters (np.ndarray): (N_dyna, 3+C) - corrected points
        unclustered_pts (np.ndarray): (N_new_bgr, 3+C) - points that are not allocated to any clusters
    """
    if pts.shape[0] == 0:
        # empty foreground
        return np.array([]), np.array([])

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
    clusters = []
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

        # TODO: decide whether to correct cluster based on area
        cluster_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]) * (bev_pix_size ** 2)
        if cluster_area > 120:
            # too large -> contains more than 1 obj -> wrong cluster -> not correct this cluster
            continue

        # correct this cluster in a sequential manner
        pts_in_cluster = pts[mask_pts_in_cluster]  # (N_p, 3+C)
        pts_timestamp = pts_in_cluster[:, 4]
        unq_ts = np.unique(pts_timestamp).tolist()
        unq_ts.reverse()  # to keep the most recent timestamp at the last position
        window, velos = np.array([]), []
        for ts_idx in range(len(unq_ts)):
            cur_group = np.copy(pts_in_cluster[pts_timestamp == unq_ts[ts_idx]])
            if window.size == 0:
                window = cur_group
                continue
            # calculate vector going from window's center toward cur_group's center
            win_to_cur = np.mean(cur_group[:, :2], axis=0) - np.mean(window[:, :2], axis=0)
            # calculate window's velocity
            delta_t = -unq_ts[ts_idx] + unq_ts[ts_idx - 1]
            velos.append(np.linalg.norm(win_to_cur) / delta_t)
            # correct window & merge with cur_group
            window[:, :2] += win_to_cur
            window = np.vstack([window, cur_group])

        # TODO: decide whether to keep corrected cluster based on std dev of velocity
        velo_var = np.mean(np.square(np.array(velos) - np.mean(velos)))
        if np.sqrt(velo_var) > threshold_velo_std_dev:
            # too large -> not keep corrected cluster
            continue
        else:
            # keep corrected cluster
            mask_corrected_pts = mask_corrected_pts | mask_pts_in_cluster
            clusters.append(window)

    # format output
    clusters = np.vstack(clusters) if clusters else np.array([])
    unclustered_pts = pts[~mask_corrected_pts]  # some pts are not corrected -> group them into a cluster
    return clusters, unclustered_pts
