from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import pprint
from tools_box import get_nuscenes_pointcloud, show_pointcloud, get_nuscenes_sensor_pose_in_global, \
    get_boxes_4viz, apply_tf, get_sweeps_token

NUM_SWEEPS = 10
pp = pprint.PrettyPrinter(indent=4, compact=True)
nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
PC_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
MAX_MOVING_DIST = 1.0
THRESHOLD_INTER_CLUSTER_DIST = 1.0  # 0.75


def compute_cluster_centroids(xy, labels, return_unq_labels=False):
    unq_labels, indices, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centroid = np.zeros((unq_labels.shape[0], 2))
    np.add.at(centroid, indices, xy)
    centroid = centroid / counts.reshape(-1, 1)
    if return_unq_labels:
        return centroid, unq_labels
    else:
        return centroid


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


def greedy_matching(cost_matrix, cost_threshold):
    num_rows, num_cols = cost_matrix.shape
    cost_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(cost_1d)
    index_2d = np.stack((index_1d // num_cols, index_1d % num_cols), axis=1)

    # init matching result
    cols_for_rows = [-1 for i in range(num_rows)]  # num_rows * [-1]
    rows_for_cols = [-1 for i in range(num_cols)]  # num_cols * [-1]

    # match
    assoc_rows, assoc_cols = [], []
    for row_id, col_id in index_2d:
        if cost_matrix[row_id, col_id] >= cost_threshold:
            break
        if cols_for_rows[row_id] == -1 and rows_for_cols[col_id] == -1:
            cols_for_rows[row_id] = col_id
            rows_for_cols[col_id] = row_id
            # log associated row & col
            assoc_rows.append(row_id)
            assoc_cols.append(col_id)

    matched_pair = list(zip(assoc_rows, assoc_cols))
    unassoc_rows = list(set([i for i in range(num_rows)]) - set(assoc_rows))
    unassoc_cols = list(set([j for j in range(num_cols)]) - set(assoc_cols))
    return matched_pair, unassoc_rows, unassoc_cols


class PointsCluster:
    cluster_id = 0
    alpha = 0.70  # coefficient for moving avg to compute velocity | OK: 0.7

    def __init__(self, points: np.ndarray):
        self.cid = PointsCluster.cluster_id
        PointsCluster.cluster_id += 1

        self.points = points
        self.offset_mag = 0

    def merge_(self, associated_cluster):
        offset = associated_cluster.get_centroid() - self.get_centroid()
        self.update_velo_and_move_(offset)
        self.add_points_(associated_cluster.points)

    def update_velo_and_move_(self, offset: np.ndarray):
        # measured_velo = offset / PointsCluster.delta_t
        measured_mag = np.linalg.norm(offset)
        # if self.offset_mag > 0:
        self.offset_mag = PointsCluster.alpha * measured_mag + (1. - PointsCluster.alpha) * self.offset_mag
        # else:
        #     self.offset_mag = measured_mag
        self.points[:, :2] += (self.offset_mag * offset / measured_mag)
        # if np.linalg.norm(offset) > 0.2:
        #     self.points[:, :2] += offset

    def apply_tf_(self, tf: np.ndarray):
        self.points[:, :3] = apply_tf(tf, self.points[:, :3])

    def add_points_(self, new_points: np.ndarray):
        self.points = np.vstack([self.points, new_points])

    def get_centroid(self, return_xy_only=True):
        if return_xy_only:
            return np.mean(self.points[:, :2], axis=0)
        else:
            # return x, y, z
            return np.mean(self.points[:, :3], axis=0)

    @staticmethod
    def build_cost_matrix(prev_clusters, curr_clusters, return_sqr_dist=True) -> np.ndarray:
        prev_centroids = np.stack([each.get_centroid() for each in prev_clusters], axis=0)
        curr_centroids = np.stack([each.get_centroid() for each in curr_clusters], axis=0)
        cost_matrix = np.square(prev_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :]).sum(axis=2)
        if not return_sqr_dist:
            cost_matrix = np.sqrt(cost_matrix)
        return cost_matrix

    @staticmethod
    def cluster_pointcloud(pc, dbscanner) -> list:
        dbscanner.fit(pc[:, :2])
        pc_labels = dbscanner.labels_  # (N_prev_fg,)
        clusters = []
        unq_labels = np.unique(pc_labels)
        for lb in unq_labels:
            if lb == -1:  # noise
                continue
            clusters.append(PointsCluster(pc[pc_labels == lb]))
        return clusters

    @staticmethod
    def cluster_pointcloud_using_gt(pc, gt_cluster_labels) -> list:
        clusters = []
        unq_labels = np.unique(gt_cluster_labels)
        for lb in unq_labels:
            if lb == -1:  # noise
                continue
            clusters.append(PointsCluster(pc[gt_cluster_labels == lb]))
        return clusters


def _main(scene_idx=1, target_sample_idx=25):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    sample_rec = nusc.get('sample', sample_token)
    curr_sd_token = sample_rec['data']['LIDAR_TOP']
    curr_sd_rec = nusc.get('sample_data', curr_sd_token)
    prev_sd_token = curr_sd_rec['prev']

    # ========================================
    dbscanner = DBSCAN(eps=0.5, min_samples=3)

    # cluster prev_pc
    prev_pc, prev_fg_mask = get_nuscenes_pointcloud(nusc, prev_sd_token, return_foreground_mask=True)
    # prev_colors = np.zeros((prev_pc.shape[0], 3))
    # prev_colors[prev_fg_mask, 2] = 1
    # show_pointcloud(prev_pc[:, :3], get_boxes_4viz(nusc, prev_sd_token), prev_colors)

    # map prev_pc to curr_sd frame
    glob_from_prev = get_nuscenes_sensor_pose_in_global(nusc, prev_sd_token)
    glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
    curr_from_prev = np.linalg.inv(glob_from_curr) @ glob_from_prev
    prev_pc[:, :3] = apply_tf(curr_from_prev, prev_pc[:, :3])
    # cluster
    prev_fg = prev_pc[prev_fg_mask]
    # dbscanner.fit(prev_fg[:, :2])
    # prev_fg_labels = dbscanner.labels_  # (N_prev_fg,)
    # prev_centroids, prev_unq_labels = compute_cluster_centroids(prev_fg[:, :2], prev_fg_labels, return_unq_labels=True)
    prev_clusters = PointsCluster.cluster_pointcloud(prev_fg, dbscanner)

    # cluster curr_pc
    curr_pc, curr_fg_mask = get_nuscenes_pointcloud(nusc, curr_sd_token, return_foreground_mask=True)
    # curr_colors = np.zeros((curr_pc.shape[0], 3))
    # curr_colors[curr_fg_mask, 2] = 1
    # show_pointcloud(curr_pc[:, :3], get_boxes_4viz(nusc, curr_sd_token), curr_colors)

    curr_fg = curr_pc[curr_fg_mask]
    # dbscanner.fit(curr_fg[:, :2])
    # curr_fg_labels = dbscanner.labels_  # (N_prev_fg,)
    # curr_centroids, curr_unq_labels = compute_cluster_centroids(curr_fg[:, :2], curr_fg_labels, return_unq_labels=True)
    curr_clusters = PointsCluster.cluster_pointcloud(curr_fg, dbscanner)

    # show prev clusters & curr clusters on 2D
    # fig, ax = plt.subplots()
    # ax.grid()
    # show_cluster_on_image(prev_fg[:, :2], prev_fg_labels, ax, marker='o', anno_prefix='pr')
    # show_cluster_on_image(curr_fg[:, :2], curr_fg_labels, ax, marker='^', anno_prefix='cr')
    # plt.show()

    # cluster-to-cluster association
    # build cost mastrix
    # cost_mat = np.square(prev_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :]).sum(axis=2)
    cost_mat = PointsCluster.build_cost_matrix(prev_clusters, curr_clusters)
    # assoc
    matched, unassoc_prev, unassoc_curr = greedy_matching(cost_mat, THRESHOLD_INTER_CLUSTER_DIST ** 2)
    print(matched)
    print(f'unassoc_prev:\n{unassoc_prev}')
    print(f'unassoc_curr:\n{unassoc_curr}')

    # merge mathced clusters
    for pidx, cidx in matched:
        prev_clusters[pidx].merge_(curr_clusters[cidx])

    # add unassoc_curr to prev_clusters
    prev_clusters.extend([curr_clusters[cidx] for cidx in unassoc_curr])

    # show corrected pointcloud
    stat_pc = np.vstack([prev_pc[~prev_fg_mask], curr_pc[~curr_fg_mask]])
    # stat_color = np.zeros([stat_pc.shape[0], 3])
    # dyna_pc = np.vstack([prev_fg, curr_fg])
    dyna_pc = np.vstack([each.points for each in prev_clusters])
    # dyna_color = np.zeros([dyna_pc.shape[0], 3])
    # dyna_color[:prev_fg.shape[0], 2] = 1  # blue - prev
    # dyna_color[prev_fg.shape[0]:, 0] = 1  # red - curr
    show_pointcloud(
        np.vstack([stat_pc, dyna_pc])[:, :3],
        get_boxes_4viz(nusc, curr_sd_token),
        # np.vstack([stat_color, dyna_color])
    )


def main(scene_idx=1, target_sample_idx=25):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    sample_rec = nusc.get('sample', sample_token)
    curr_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens = get_sweeps_token(nusc, curr_sd_token, n_sweeps=NUM_SWEEPS, return_time_lag=False)

    dbscanner = DBSCAN(eps=1.350, min_samples=3)  # 0.5

    # init
    prev_pc, prev_fg_mask = get_nuscenes_pointcloud(nusc, sd_tokens[0], return_foreground_mask=True)
    prev_clusters = PointsCluster.cluster_pointcloud_using_gt(prev_pc, prev_fg_mask)
    stat_pc = prev_pc[prev_fg_mask == -1]

    # ---
    # main loop
    # ---
    for sd_idx in range(1, len(sd_tokens)):
        # map stat_pc & prev_clusters to curr_sd's frame
        glob_from_prev = get_nuscenes_sensor_pose_in_global(nusc, sd_tokens[sd_idx - 1])
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, sd_tokens[sd_idx])
        curr_from_prev = np.linalg.inv(glob_from_curr) @ glob_from_prev
        stat_pc[:, :3] = apply_tf(curr_from_prev, stat_pc[:, :3])
        for cluster in prev_clusters:
            cluster.apply_tf_(curr_from_prev)

        # get current pc & cluster it
        curr_pc, curr_fg_mask = get_nuscenes_pointcloud(nusc, sd_tokens[sd_idx], return_foreground_mask=True)
        curr_clusters = PointsCluster.cluster_pointcloud_using_gt(curr_pc, curr_fg_mask)

        # merge static pc
        stat_pc = np.vstack([stat_pc, curr_pc[curr_fg_mask == -1]])

        # associate prev_cluster with curr_clusters
        cost_mat = PointsCluster.build_cost_matrix(prev_clusters, curr_clusters)
        matched, _, unassoc_curr = greedy_matching(cost_mat, THRESHOLD_INTER_CLUSTER_DIST ** 2)
        # merge matched clusters
        for pidx, cidx in matched:
            prev_clusters[pidx].merge_(curr_clusters[cidx])
        # add unassoc_curr to prev_clusters
        prev_clusters.extend([curr_clusters[cidx] for cidx in unassoc_curr])

        # -
        # viz
        # -
        # print(f"showing frame {sd_idx}")
        # dyna_pc = np.vstack([each.points for each in prev_clusters])
        # _pc = np.vstack([stat_pc, dyna_pc])[:, :3]
        # _in_range = np.all((_pc >= PC_RANGE[:3]) & (_pc < (PC_RANGE[3:] - 1e-2)), axis=1)
        # show_pointcloud(_pc[_in_range, :3], get_boxes_4viz(nusc, sd_tokens[sd_idx]))
        # print('---')

    dyna_pc = np.vstack([each.points for each in prev_clusters])
    _pc = np.vstack([stat_pc, dyna_pc])[:, :3]
    _in_range = np.all((_pc >= PC_RANGE[:3]) & (_pc < (PC_RANGE[3:] - 1e-2)), axis=1)
    show_pointcloud(_pc[_in_range, :3], get_boxes_4viz(nusc, curr_sd_token))


if __name__ == '__main__':
    main(0, 25)
