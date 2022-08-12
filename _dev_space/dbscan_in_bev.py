from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import pprint
from copy import deepcopy
from tools_box import get_nuscenes_pointcloud, show_pointcloud, get_nuscenes_sensor_pose_in_global, \
    get_boxes_4viz, apply_tf, get_sweeps_token, get_nuscenes_pointcloud_in_target_frame

NUM_SWEEPS = 10
pp = pprint.PrettyPrinter(indent=4, compact=True)
nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
PC_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
MAX_MOVING_DIST = 1.0
THRESHOLD_INTER_CLUSTER_DIST = 1.0  # 0.


def show_cluster_on_image(xy, labels, ax, marker='o', anno_prefix=''):
    assert len(xy.shape) == 2 and len(labels.shape) == 1
    assert xy.shape[0] == labels.shape[0]
    unq_labels, indices, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centroid = np.zeros((unq_labels.shape[0], 2))
    np.add.at(centroid, indices, xy)
    centroid = centroid / counts.reshape(-1, 1)
    colors_palette = np.array([plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, unq_labels.shape[0])])
    xy_colors = colors_palette[indices]
    mask_valid = labels > -1
    ax.scatter(xy[mask_valid, 0], xy[mask_valid, 1], c=xy_colors[mask_valid], marker=marker)
    for i in range(centroid.shape[0]):
        if unq_labels[i] == -1:
            continue
        ax.annotate(f"{anno_prefix}{unq_labels[i]}", tuple(centroid[i].tolist()))


def main(scene_idx, target_sample_idx):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    sample_rec = nusc.get('sample', sample_token)
    curr_sd_token = sample_rec['data']['LIDAR_TOP']
    curr_sd_rec = nusc.get('sample_data', curr_sd_token)
    prev_sd_token = curr_sd_rec['prev']

    sample_rec = nusc.get('sample', sample_token)
    curr_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, curr_sd_token, n_sweeps=NUM_SWEEPS, return_time_lag=True)
    ref_sd_token = sd_tokens_times[-1][0]

    # ========================================
    dbscanner = DBSCAN(eps=3.0, min_samples=5)  # not moving obj -> less points -> treat them like static
    vox_size = 0.2
    size = int(np.round((PC_RANGE[3] - PC_RANGE[0]) / vox_size))
    # =========================================

    pcs, mask_fgrs = [], []
    for sd_token, timelag in sd_tokens_times:
        pc, fgr = get_nuscenes_pointcloud_in_target_frame(nusc, sd_token, ref_sd_token, return_foreground_mask=True,
                                                          pc_time=timelag)
        pcs.append(pc)
        mask_fgrs.append(fgr)

    pcs = np.vstack(pcs)
    mask_fgrs = np.concatenate(mask_fgrs)
    in_range = np.all((pcs[:, :3] > PC_RANGE[:3]) & (pcs[:, :3] < (PC_RANGE[3:] - 1e-3)), axis=1)
    pcs = pcs[in_range]
    mask_fgrs = mask_fgrs[in_range]

    pc_colors = np.zeros((pcs.shape[0], 3))
    pc_colors[mask_fgrs, 0] = 1
    show_pointcloud(pcs[:, :3], get_boxes_4viz(nusc, ref_sd_token), pc_colors)

    # split pcs into bgr & fgr
    pc_bgr = pcs[~mask_fgrs]  # (N_bgr, 3+C)
    pc_fgr = pcs[mask_fgrs]  # (N_fgr, 3+C)

    # compute BEV pixel coord
    pc_fgr_pixels = np.floor((pc_fgr[:, :2] - PC_RANGE[:2]) / vox_size).astype(int)
    pc_fgr_pixels_1d = pc_fgr_pixels[:, 1] * size + pc_fgr_pixels[:, 0]
    unq_pixels_1d = np.unique(pc_fgr_pixels_1d)
    unq_pixels_2d = np.stack((unq_pixels_1d % size, unq_pixels_1d // size), axis=1)

    # clustering
    dbscanner.fit(unq_pixels_2d[:, :2])
    pixels_labels = dbscanner.labels_  # (N_unq, )

    mask_fgr_valid = np.zeros(pc_fgr.shape[0], dtype=bool)
    dyna_clusters = []
    for unq_lb in np.unique(pixels_labels):
        if unq_lb == -1:
            continue

        unq_pixels_2d_in_cluster = unq_pixels_2d[pixels_labels == unq_lb]
        # smallest bounding rectangle
        min_xy = np.amin(unq_pixels_2d_in_cluster, axis=0)
        max_xy = np.amax(unq_pixels_2d_in_cluster, axis=0)
        # identify fgr points that fall inside bounding rectangle using their pixel coords
        mask_fgr_in_cluster = np.all((pc_fgr_pixels >= min_xy) & (pc_fgr_pixels <= max_xy), axis=1)
        mask_fgr_valid = mask_fgr_valid | mask_fgr_in_cluster

        # extract points inside cluster from pc_fgr
        cluster_points = pc_fgr[mask_fgr_in_cluster]
        # pc_fgr = pc_fgr[~mask_fgr_in_cluster]

        # split cluster's points using timestamp
        points_ts = cluster_points[:, 4]
        unq_ts = np.unique(points_ts).tolist()
        unq_ts.reverse()
        final_center = np.mean(cluster_points[points_ts == 0, :2], axis=0)
        for ts in unq_ts:
            if ts == 0:
                break
            mask_group = points_ts == ts
            curr_center = np.mean(cluster_points[mask_group, :2], axis=0)
            offset = final_center - curr_center
            cluster_points[mask_group, :2] += offset

        dyna_clusters.append(cluster_points)

    dyna_clusters = np.vstack(dyna_clusters)

    if not np.all(mask_fgr_valid):
        pc_bgr = np.vstack([pc_bgr, pc_fgr[~mask_fgr_valid]])

    corrected_pc = np.vstack([pc_bgr, dyna_clusters])
    pc_colors = np.zeros((corrected_pc.shape[0], 3))
    pc_colors[pc_bgr.shape[0]:, 0] = 1
    show_pointcloud(corrected_pc[:, :3], get_boxes_4viz(nusc, ref_sd_token), pc_colors)
    return



    # =======================================
    # fig, ax = plt.subplots(1, 2)
    # for aix in range(len(ax)):
    #     ax[aix].grid()
    #     ax[aix].set_aspect('equal')
    #     ax[aix].set_xlim([0, size])
    #     ax[aix].set_ylim([0, size])
    # show_cluster_on_image(fgr_bev_xy, fgr_labels, ax[0], marker='o', anno_prefix='pr')
    # ax[1].scatter(pc_fgr_pixels[:, 0], pc_fgr_pixels[:, 1], marker='o')
    # # show_cluster_on_image(fgr_pixels, prev_fg_mask[prev_fg_mask > -1].astype(int), ax[1], marker='o', anno_prefix='pr')
    # plt.show()
    #
    # chosen_cluster = input('enter chosen cluster: ')
    # # TODO: find pts inside a cluster
    # pix_in_cluster = fgr_bev_xy[fgr_labels == int(chosen_cluster)]
    # min_xy = np.amin(pix_in_cluster, axis=0)
    # max_xy = np.amax(pix_in_cluster, axis=0)
    # in_cluster = np.all((pc_fgr_pixels >= min_xy) & (pc_fgr_pixels < max_xy), axis=1)
    # pts_in_cluster = np.hstack((pc_fgr_pixels[in_cluster], pc_fgr[in_cluster, -1].reshape(-1, 1)))
    #
    # unq_ts = np.unique(pts_in_cluster[:, -1]).tolist()
    # unq_ts.reverse()
    # print('unq_ts: ', unq_ts)
    # sub_clusters = [pts_in_cluster[pts_in_cluster[:, -1] == ts] for ts in unq_ts]
    # sub_clusters_center = [np.copy(np.mean(each[:, :2], axis=0)) for each in sub_clusters]
    #
    # _old_sub_clusters = deepcopy(sub_clusters)
    #
    # # sequential offset
    # for sub_clt_idx in range(len(sub_clusters) - 1):
    #     curr_center = np.mean(sub_clusters[sub_clt_idx][:, :2], axis=0)
    #     # next_center = np.mean(sub_clusters[sub_clt_idx + 1][:, :2], axis=0)
    #     next_center = sub_clusters_center[sub_clt_idx + 1]
    #     offset = next_center - curr_center
    #     sub_clusters[sub_clt_idx][:, :2] += offset
    #     sub_clusters[sub_clt_idx + 1] = np.vstack([sub_clusters[sub_clt_idx + 1], sub_clusters[sub_clt_idx]])
    #
    # colors_palette = np.array([plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, len(sub_clusters))])
    # fig2, ax2 = plt.subplots(1, 3)
    # for aix in range(len(ax2)):
    #     ax2[aix].grid()
    #     ax2[aix].set_aspect('equal')
    #     ax2[aix].set_xlim([min_xy[0] - 2, max_xy[0] + 2])
    #     ax2[aix].set_ylim([min_xy[1] - 2, max_xy[1] + 2])
    # for idx, old_sub_clt, sub_clt in zip(range(len(sub_clusters)), _old_sub_clusters, sub_clusters):
    #     ax2[0].scatter(old_sub_clt[:, 0], old_sub_clt[:, 1], c=np.tile(colors_palette[[idx]], (old_sub_clt.shape[0], 1)),
    #                    marker='o')
    #     ax2[1].scatter([sub_clusters_center[idx][0]], [sub_clusters_center[idx][1]], marker='^',
    #                    c=colors_palette[[idx]])
    # ax2[2].scatter(sub_clt[:, 0], sub_clt[:, 1],
    #                c=np.tile(colors_palette[[idx]], (sub_clt.shape[0], 1)),
    #                marker='o')
    #
    # plt.show()
    # np.save(f'cluster_{chosen_cluster}.npy', pts_in_cluster)


if __name__ == '__main__':
    main(1, 25)
    main(0, 10)
    main(2, 15)
    main(3, 20)  # cars too close to each other
    main(4, 20)
    main(5, 20)
    main(6, 20)
    main(7, 20)
    main(8, 20)
    main(9, 20)



