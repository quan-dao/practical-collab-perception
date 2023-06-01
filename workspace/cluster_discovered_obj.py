import numpy as np
import hdbscan
import pickle
from pathlib import Path
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

from workspace.o3d_visualization import PointsPainter
from workspace.nuscenes_temporal_utils import apply_se3_


def main(num_sweeps: int = 15, 
         pc_range: np.array = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])):
    database_root = Path(f'../data/nuscenes/v1.0-mini/discovered_database_{num_sweeps}sweeps')
    trajs_path = list(database_root.glob('*.pkl'))
    trajs_path.sort()
    print(f'found {len(trajs_path)} trajectories')

    grid_size_meters = pc_range[3:] - pc_range[:3]

    trajs_descriptor = list()
    invalid_trajs_idx = list()
    for idx, path_to_traj in enumerate(trajs_path):
        with open(path_to_traj, 'rb') as f:
            traj_info = pickle.load(f)
        
        # build higher dimension descriptor
        # variance of num points
        points_in_glob = traj_info['points_in_glob']
        _, num_points_per_sweep = np.unique(points_in_glob[:, -2].astype(int), return_counts=True)
        var_num_points = np.var(num_points_per_sweep / num_points_per_sweep.sum())

        # covariance of points coord in box's local frame - normalized by box'size
        boxes_in_glob = traj_info['boxes_in_glob']
        if np.any(boxes_in_glob[0, 3: 6] < 1e-1):
            invalid_trajs_idx.append(idx)
            continue
        cos, sin = np.cos(boxes_in_glob[:, 6]), np.sin(boxes_in_glob[:, 6])
        zeros, ones = np.zeros(boxes_in_glob.shape[0]), np.ones(boxes_in_glob.shape[0])
        batch_glob_se3_boxes = np.stack([
            cos,    -sin,       zeros,      boxes_in_glob[:, 0],
            sin,     cos,       zeros,      boxes_in_glob[:, 1],
            zeros,   zeros,     ones,       boxes_in_glob[:, 2],
            zeros,   zeros,     zeros,      ones
        ], axis=1).reshape(-1, 4, 4)
        batch_boxes_se3_glob = np.linalg.inv(batch_glob_se3_boxes)  # (N_valid_sw, 4, 4)
        
        boxes_sweep_idx = boxes_in_glob[:, -1].astype(int)
        assert np.unique(boxes_sweep_idx).shape[0] == boxes_sweep_idx.shape[0]
        assert num_sweeps >= boxes_sweep_idx.shape[0]  # assert N_sw >= N_valid_sw 

        pad_batch_boxes_se3_glob = np.tile(np.eye(4).reshape(1, 4, 4), (num_sweeps,1, 1))
        pad_batch_boxes_se3_glob[boxes_sweep_idx] = batch_boxes_se3_glob  # (N_sw, 4, 4)  ! N_sw >= N_valid_sw 

        points_sweep_idx = points_in_glob[:, -2].astype(int)  # (N_pts,)
        perpoint_box_se3_glob = pad_batch_boxes_se3_glob[points_sweep_idx]  # (N_pts, 4, 4)
        points_in_box = np.einsum('bik, bk -> bi', perpoint_box_se3_glob[:, :3, :3], points_in_glob[:, :3]) + perpoint_box_se3_glob[:, :3, -1]
        points_in_box = 2.0 * points_in_box / boxes_in_glob[0, 3: 6]
        
        cov_coord3d = np.cov(points_in_box.T)
        cc, rr = np.meshgrid(np.arange(cov_coord3d.shape[1]), np.arange(cov_coord3d.shape[0]))
        mask_above_diag = cc >= rr
        cov_coord3d_idp = cov_coord3d[rr[mask_above_diag], cc[mask_above_diag]]

        # dimension
        dx, dy, dz = traj_info['boxes_in_glob'][0, 3: 6]
        area_xy = dx * dy / (grid_size_meters[0] * grid_size_meters[1])
        
        # total distance travel
        travelled_dist = np.linalg.norm(boxes_in_glob[1:, :2] - boxes_in_glob[:-1, :2], axis=1).sum() \
            / np.linalg.norm(grid_size_meters[:2])

        desriptor = np.array([var_num_points.item(), *cov_coord3d_idp.tolist(), area_xy, dz / grid_size_meters[2], travelled_dist])


        trajs_descriptor.append(desriptor)

    trajs_descriptor = np.stack(trajs_descriptor, axis=0)  # (N_trajs, C)
    
    print(f'encounter {len(invalid_trajs_idx)} invalid trajs')
    # remove invalid trajs
    for _idx in reversed(invalid_trajs_idx):
        del trajs_path[_idx]
    print(f'num_valid trajs: {len(trajs_path)}')

    scaled_trajs_descriptor = StandardScaler().fit_transform(trajs_descriptor)

    # trajs_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(scaled_trajs_descriptor)

    reducer = umap.UMAP(n_components=3)  # 2 -> 10 clusters; 3-> 9 clusters; 4-> 9 clusters
    trajs_embedded = reducer.fit_transform(scaled_trajs_descriptor)

    # fig, ax = plt.subplots()
    # ax.scatter(trajs_embedded[:, 0], trajs_embedded[:, 1])

    clusterer = hdbscan.HDBSCAN(algorithm='best',
                                # leaf_size=100,
                                metric='euclidean', min_cluster_size=50, min_samples=None)
    clusterer.fit(trajs_embedded)
    trajs_label = clusterer.labels_
    unq_labels, counts = np.unique(trajs_label, return_counts=True)
    print('unq_lalbes:\n', unq_labels)
    print('counts:\n', counts)

    clusters_color = matplotlib.cm.rainbow(np.linspace(0.1, 1, unq_labels.shape[0]))[:, :3]
    trajs_color = clusters_color[trajs_label]
    trajs_color[trajs_label == -1] = 0.  # outlier

    # fig, ax = plt.subplots()
    # ax.scatter(trajs_descriptor[:, 0] * trajs_descriptor[:, 1], trajs_descriptor[:, 2], c=trajs_color)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajs_embedded[:, 0], trajs_embedded[:, 1], c=trajs_color)
    plt.show()

    print('========================')
    indices_trajs_path = np.arange(len(trajs_path))
    trajs_prob = clusterer.probabilities_
    for label in unq_labels:
        # if label != 4:
        #     continue
        print(f'showing examplar of cluster {label}')
        mask_cluster = trajs_label == label
        max_prob_idx = np.argsort(-trajs_prob[mask_cluster])

        for _i in range(1):
            max_prob_path = trajs_path[indices_trajs_path[mask_cluster][max_prob_idx[_i]]]
            with open(max_prob_path, 'rb') as f:
                traj_info = pickle.load(f)

            points = traj_info['points_in_glob']
            boxes = traj_info['boxes_in_glob']
            lidar_se3_glob = np.linalg.inv(traj_info['glob_se3_lidar'])
            apply_se3_(lidar_se3_glob, points_=points, boxes_=boxes)

            painter = PointsPainter(points[:, :3], boxes[:, :7])
            painter.show()




if __name__ == '__main__':
    main()
