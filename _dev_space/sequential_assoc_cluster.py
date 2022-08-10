from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import pprint

from tools_box import get_nuscenes_pointcloud, show_pointcloud, \
    get_nuscenes_sensor_pose_in_global, get_boxes_4viz, apply_tf, get_nuscenes_pointcloud_in_target_frame
from get_clean_pointcloud import get_merge_pointcloud


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


def main(debug):
    NUM_SWEEPS = 10
    pp = pprint.PrettyPrinter(indent=4, compact=True)
    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    scene = nusc.scene[0]
    target_sample_idx = 30
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    # get sample_data tokens
    sample = nusc.get('sample', sample_token)
    curr_sd_token = sample['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, curr_sd_token, n_sweeps=NUM_SWEEPS)
    pp.pprint(sd_tokens_times)

    # ====================
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    max_moving_dist = 1.0  # assume max velocity = 72km/h
    dbscaner = DBSCAN(eps=0.5, min_samples=3)
    # ====================

    prev_sd_token, prev_time = sd_tokens_times[0]
    prev_pc, prev_fgr_mask = get_nuscenes_pointcloud(nusc, prev_sd_token, return_foreground_mask=True, pc_time=prev_time)

    for j in range(1, NUM_SWEEPS):
        curr_sd_token, curr_time = sd_tokens_times[j]
        curr_pc, curr_fgr_mask = get_nuscenes_pointcloud(nusc, curr_sd_token, return_foreground_mask=True,
                                                         pc_time=curr_time)
        # ego motion compensation for prev_pc
        # this maps prev_pc to curr_sd 's LIDAR frame
        glob_from_prev = get_nuscenes_sensor_pose_in_global(nusc, prev_sd_token)
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
        curr_from_prev = np.linalg.inv(glob_from_curr) @ glob_from_prev
        prev_pc[:, :3] = apply_tf(curr_from_prev, prev_pc[:, :3])

        # merge background
        merge_bgr = np.vstack([prev_pc[~prev_fgr_mask], curr_pc[~curr_fgr_mask]])

        # ---
        # merge foreground & correct them
        # ---
        merge_fgr = np.vstack([prev_pc[prev_fgr_mask], curr_pc[curr_fgr_mask]])
        n_prev_fgr = prev_fgr_mask.astype(int).sum()

        dbscaner.fit(merge_fgr[:, :2])
        cluster_ids = np.unique(dbscaner.labels_)

        if debug:
            # show before correction
            print('===================')
            print(f"sd index {j} | showing BEFORE")
            viz_merge_pc = np.vstack([merge_bgr, merge_fgr])
            viz_pc_colors = np.zeros((viz_merge_pc.shape[0], 3))
            n_bgr = merge_bgr.shape[0]
            viz_pc_colors[n_bgr: n_bgr + n_prev_fgr, 2] = 1  # blue for prev
            viz_pc_colors[n_bgr + n_prev_fgr:, 0] = 1  # red for curr
            viz_boxes = get_boxes_4viz(nusc, curr_sd_token)
            show_pointcloud(viz_merge_pc[:, :3], viz_boxes, viz_pc_colors)
            print('---')
            print(f"sd index {j} | show clustering result")
            unique_labels, indices, counts = np.unique(dbscaner.labels_, return_inverse=True, return_counts=True)
            clusters_mean = np.zeros((unique_labels.shape[0], 3))
            np.add.at(clusters_mean, indices, merge_fgr[:, :3])
            clusters_mean = clusters_mean / counts.reshape(-1, 1)
            clusters_color = np.array([plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, unique_labels.shape[0])])
            merge_fgr_colors = clusters_color[indices]
            show_pointcloud(
                viz_merge_pc[:, :3], viz_boxes,
                pc_colors=np.vstack([np.zeros((n_bgr, 3)), merge_fgr_colors])
            )
            if j == 3:
                print(f"sd index {j} | show clustering result 2d")
                fig, ax = plt.subplots()
                noise = dbscaner.labels_ == -1
                ax.scatter(merge_fgr[~noise, 0], merge_fgr[~noise, 1], c=merge_fgr_colors[~noise])
                for i in range(clusters_mean.shape[0]):
                    if unique_labels[i] == -1:
                        continue
                    ax.annotate(unique_labels[i], (clusters_mean[i, 0], clusters_mean[i, 1]))
                plt.show()
                print('---')

        _prev_mask, _curr_mask = np.zeros(merge_fgr.shape[0], dtype=bool), np.zeros(merge_fgr.shape[0], dtype=bool)
        _prev_mask[:n_prev_fgr] = True
        _curr_mask[n_prev_fgr:] = True

        for cl_id in cluster_ids:
            cluster_mask = dbscaner.labels_ == cl_id
            prev_xy = merge_fgr[cluster_mask & _prev_mask, :2]
            curr_xy = merge_fgr[cluster_mask & _curr_mask, :2]
            if prev_xy.shape[0] > 0 and curr_xy.shape[0] > 0:
                offset_xy = np.mean(curr_xy, axis=0) - np.mean(prev_xy, axis=0)
                if np.square(offset_xy).sum() > max_moving_dist ** 2:
                    continue
                # apply offset_xy to prev_fgr of this cluster
                merge_fgr[cluster_mask & _prev_mask, :2] = merge_fgr[cluster_mask & _prev_mask, :2] + offset_xy

        # concatenate merge_bgr & merge_fgr, then overwrite prev_pc & prev_fgr_mask
        prev_pc = np.vstack([merge_bgr, merge_fgr])
        prev_fgr_mask = np.zeros(prev_pc.shape[0], dtype=bool)
        prev_fgr_mask[merge_bgr.shape[0]:] = True
        prev_sd_token = curr_sd_token

        if debug:
            # show after correction
            print(f"sd index {j} | showing After")
            show_pointcloud(prev_pc[:, :3], viz_boxes, viz_pc_colors)
            print('---')

    show_pointcloud(prev_pc[:, :3], get_boxes_4viz(nusc, curr_sd_token))


def cluster_and_move(nusc: NuScenes, sd_tokens_times: list, dbscanner, max_moving_dist=1.0) -> np.ndarray:
    """
    Get sweeps and sequentially propagate them (1 frame at a time) to the current sample_data. The propagation from 1
    sweep in the past to its subsequence sweep
        - partition past sweep (using annos in past sweep's LIDAR frame)
        - ego motion compensation the past sweep (w.r.t the subsequence sweep)
        - partition subsequence sweep (using annos in subsequence sweep's LIDAR frame)
        - cluster {past foreground, subsequence foreground} using DBSCAN
        - for each cluster
        -     id past points & subsequence points
        -     move past points such that their center is matched with the center of subsequence points

    Returns:
        (N_total, 3 + C)
    """
    assert len(sd_tokens_times) > 0
    prev_sd_token, prev_time = sd_tokens_times[0]
    prev_pc, prev_fgr_mask = get_nuscenes_pointcloud(nusc, prev_sd_token, return_foreground_mask=True,
                                                     pc_time=prev_time)

    for sd_idx in range(1, len(sd_tokens_times)):
        curr_sd_token, curr_time = sd_tokens_times[sd_idx]
        curr_pc, curr_fgr_mask = get_nuscenes_pointcloud(nusc, curr_sd_token, return_foreground_mask=True,
                                                         pc_time=curr_time)
        # ego motion compensation for prev_pc
        # this maps prev_pc to curr_sd 's LIDAR frame
        glob_from_prev = get_nuscenes_sensor_pose_in_global(nusc, prev_sd_token)
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
        curr_from_prev = np.linalg.inv(glob_from_curr) @ glob_from_prev
        prev_pc[:, :3] = apply_tf(curr_from_prev, prev_pc[:, :3])

        # merge background
        merge_bgr = np.vstack([prev_pc[~prev_fgr_mask], curr_pc[~curr_fgr_mask]])

        # ---
        # merge foreground & correct them
        # ---
        if not (np.any(prev_fgr_mask) and np.any(curr_fgr_mask)):
            prev_pc = merge_bgr
            prev_fgr_mask = np.zeros(prev_pc.shape[0], dtype=bool)
            prev_sd_token = curr_sd_token
            continue

        merge_fgr = np.vstack([prev_pc[prev_fgr_mask], curr_pc[curr_fgr_mask]])
        n_prev_fgr = prev_fgr_mask.astype(int).sum()

        # clustering
        dbscanner.fit(merge_fgr[:, :2])
        cluster_ids = np.unique(dbscanner.labels_)

        # for each cluster:
        # id prev points & current points
        # move prev points such that their center is matched with the center of current points
        _prev_mask, _curr_mask = np.zeros(merge_fgr.shape[0], dtype=bool), np.zeros(merge_fgr.shape[0], dtype=bool)
        _prev_mask[:n_prev_fgr] = True
        _curr_mask[n_prev_fgr:] = True
        for cl_id in cluster_ids:
            cluster_mask = dbscanner.labels_ == cl_id
            prev_xy = merge_fgr[cluster_mask & _prev_mask, :2]
            curr_xy = merge_fgr[cluster_mask & _curr_mask, :2]
            if prev_xy.shape[0] > 0 and curr_xy.shape[0] > 0:
                offset_xy = np.mean(curr_xy, axis=0) - np.mean(prev_xy, axis=0)
                if np.square(offset_xy).sum() > max_moving_dist ** 2:
                    # invalid offset probably from a wrong match, not do anything
                    continue
                # apply offset_xy to prev_fgr of this cluster
                merge_fgr[cluster_mask & _prev_mask, :2] = merge_fgr[cluster_mask & _prev_mask, :2] + offset_xy

        # concatenate merge_bgr & merge_fgr, then overwrite prev_pc & prev_fgr_mask & prev_sd_token
        prev_pc = np.vstack([merge_bgr, merge_fgr])
        prev_fgr_mask = np.zeros(prev_pc.shape[0], dtype=bool)
        prev_fgr_mask[merge_bgr.shape[0]:] = True
        prev_sd_token = curr_sd_token

    return prev_pc


def cluster_and_move_vs_emc_only_vs_clean_by_anno(scene_idx, target_sample_idx):
    NUM_SWEEPS = 10
    pp = pprint.PrettyPrinter(indent=4, compact=True)
    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    # get sample_data tokens
    sample = nusc.get('sample', sample_token)
    nusc.render_sample_data(sample['data']['CAM_FRONT'])
    plt.show()
    curr_sd_token = sample['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, curr_sd_token, n_sweeps=NUM_SWEEPS)
    pp.pprint(sd_tokens_times)

    pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    # ========================
    # get pointcloud using annotations
    # ========================
    ref_sd_token = sd_tokens_times[-1][0]
    viz_boxes = get_boxes_4viz(nusc, ref_sd_token)

    pcs = get_merge_pointcloud(nusc, sample_token, num_samples=1, clean_using_annos=True)

    mask_un_cleaned = (pcs[:, -1] == 0) | (pcs[:, -1] == 1)
    pcs_uncleaned = pcs[mask_un_cleaned]
    in_range_mask = np.all((pcs_uncleaned[:, :3] >= pc_range[:3]) & (pcs_uncleaned[:, :3] < pc_range[3:] - 1e-3), axis=1)
    pcs_uncleaned = pcs_uncleaned[in_range_mask]
    print(f"scene{scene_idx}_sample{target_sample_idx}_{NUM_SWEEPS}sweeps_emc_only")
    show_pointcloud(pcs_uncleaned[:, :3], viz_boxes)
    print('---')

    mask_cleaned = (pcs[:, -1] == 0) | (pcs[:, -1] == 2)
    pcs_cleaned = pcs[mask_cleaned]
    in_range_mask = np.all((pcs_cleaned[:, :3] >= pc_range[:3]) & (pcs_cleaned[:, :3] < pc_range[3:] - 1e-3), axis=1)
    pcs_cleaned = pcs_cleaned[in_range_mask]
    print(f"scene{scene_idx}_sample{target_sample_idx}_{NUM_SWEEPS}sweeps_annos")
    show_pointcloud(pcs_cleaned[:, :3], viz_boxes)
    print('---')

    # ========================
    # get pointcloud using cluster-and-move
    # ========================
    dbscanner = DBSCAN(eps=0.5, min_samples=3)  # 0.5, 3
    cam_pcs = cluster_and_move(nusc, sd_tokens_times, dbscanner)
    in_range_mask = np.all((cam_pcs[:, :3] >= pc_range[:3]) & (cam_pcs[:, :3] < pc_range[3:] - 1e-3), axis=1)
    cam_pcs = cam_pcs[in_range_mask]
    print(f"scene{scene_idx}_sample{target_sample_idx}_{NUM_SWEEPS}sweeps_cam")
    show_pointcloud(cam_pcs[:, :3], viz_boxes)
    print('---')


if __name__ == '__main__':
    # main(debug=False)
    # cluster_and_move_vs_emc_only()
    cluster_and_move_vs_emc_only_vs_clean_by_anno(scene_idx=1, target_sample_idx=25)


