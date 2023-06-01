import numpy as np
import hdbscan
import pickle
from pathlib import Path
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

from workspace.o3d_visualization import PointsPainter
from workspace.nuscenes_temporal_utils import apply_se3_, map_points_on_traj_to_local_frame


np.random.seed(666)

def main(num_sweeps: int = 15, 
         pc_range: np.array = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])):
    database_root = Path(f'../data/nuscenes/v1.0-mini/discovered_database_{num_sweeps}sweeps')
    trajs_path = list(database_root.glob('*.pkl'))
    trajs_path.sort()
    trajs_path = np.array(trajs_path)
    print(f'found {trajs_path.shape[0]} trajectories')

    grid_size_meters = pc_range[3:] - pc_range[:3]

    trajs_descriptor, trajs_static_descriptor = list(), list()
    valid_trajs_idx = list()
    for idx, path_to_traj in enumerate(trajs_path):
        with open(path_to_traj, 'rb') as f:
            traj_info = pickle.load(f)
        
        boxes_in_glob = traj_info['boxes_in_glob']

        if np.any(boxes_in_glob[0, 3: 6] < 1e-1):
            continue
        
        valid_trajs_idx.append(idx)

        # build higher dimension descriptor
        # variance of num points
        points_in_glob = traj_info['points_in_glob']
        _, num_points_per_sweep = np.unique(points_in_glob[:, -2].astype(int), return_counts=True)
        var_num_points = np.var(num_points_per_sweep / num_points_per_sweep.sum())

        # covariance of points coord in box's local frame - normalized by box'size
        points_in_box = map_points_on_traj_to_local_frame(points_in_glob, boxes_in_glob, num_sweeps)
        points_in_box = 2.0 * points_in_box / boxes_in_glob[0, 3: 6]
        
        cov_coord3d = np.cov(points_in_box.T)
        cc, rr = np.meshgrid(np.arange(cov_coord3d.shape[1]), np.arange(cov_coord3d.shape[0]))
        mask_above_diag = cc >= rr
        cov_coord3d_idp = cov_coord3d[rr[mask_above_diag], cc[mask_above_diag]]

        # dimension
        dx, dy, dz = boxes_in_glob[0, 3: 6]
        area_xy = dx * dy / (grid_size_meters[0] * grid_size_meters[1])
        
        # total distance travel
        travelled_dist = np.linalg.norm(boxes_in_glob[1:, :2] - boxes_in_glob[:-1, :2], axis=1).sum() \
            / np.linalg.norm(grid_size_meters[:2])

        # assemble traj's descriptor
        descriptor = np.array([var_num_points.item(), *cov_coord3d_idp.tolist(), area_xy, dz / grid_size_meters[2], travelled_dist])
        static_descriptor = np.array([*cov_coord3d_idp.tolist(), area_xy, dz / grid_size_meters[2]])
        # store
        trajs_descriptor.append(descriptor)
        trajs_static_descriptor.append(static_descriptor)

    trajs_descriptor = np.stack(trajs_descriptor, axis=0)  # (N_trajs, C)
    trajs_static_descriptor = np.stack(trajs_static_descriptor, axis=0)  # (N_trajs, C_static)
    valid_trajs_idx = np.array(valid_trajs_idx)
    
    # keep only valid trajs
    print(f'encounter {trajs_path.shape[0] - valid_trajs_idx.shape[0]} invalid trajs')
    print(f'num_valid trajs: {valid_trajs_idx.shape[0]}')
    trajs_path = trajs_path[valid_trajs_idx]

    # umap full
    reducer = umap.UMAP(n_components=3, random_state=78)  # 2 -> 10 clusters; 3-> 9 clusters; 4-> 9 clusters
    scaled_trajs_descriptor = StandardScaler().fit_transform(trajs_descriptor)
    trajs_embedded = reducer.fit_transform(scaled_trajs_descriptor)

    # umap static
    reducer_static = umap.UMAP(n_components=3, random_state=78)
    scaled_trajs_static_descriptor = StandardScaler().fit_transform(trajs_static_descriptor)
    reducer_static.fit(scaled_trajs_static_descriptor)
    with open(f'artifact/umap_static_{num_sweeps}sweeps_v1.0-mini.pkl', 'wb') as f:
        pickle.dump(reducer_static, f)

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

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajs_embedded[:, 0], trajs_embedded[:, 1], c=trajs_color)
    plt.show()

    print('========================')
    indices_trajs_path = np.arange(trajs_path.shape[0])
    trajs_prob = clusterer.probabilities_
    for label, num_trajs_in_cluster in zip(unq_labels, counts):
        if label == -1:
            continue
        # if label != 4:
        #     continue
        print(f'showing examplar of cluster {label} | size {num_trajs_in_cluster}')
        mask_cluster = trajs_label == label
        ids_by_prob = np.argsort(-trajs_prob[mask_cluster])  # (N_traj_in_cluster,)
        indices_cluster_trajs_path = indices_trajs_path[mask_cluster]  # (N_traj_in_cluster,)

        for _i in range(10):
            max_prob_path = trajs_path[indices_cluster_trajs_path[ids_by_prob[_i]]]
            with open(max_prob_path, 'rb') as f:
                traj_info = pickle.load(f)

            points_in_lidar, boxes_in_lidar = apply_se3_(np.linalg.inv(traj_info['glob_se3_lidar']),
                                                         points_=traj_info['points_in_glob'], 
                                                         boxes_=traj_info['boxes_in_glob'],
                                                         return_transformed=True)
            # ---
            # for viz purpose, display points in box coordinate too
            points_in_box = map_points_on_traj_to_local_frame(traj_info['points_in_glob'], traj_info['boxes_in_glob'], num_sweeps)
            
            # map points_in_box to lidar-frame using last box's coord
            last_box_in_lidar = boxes_in_lidar[[-1]]
            _c, _s = np.cos(last_box_in_lidar[0, 6]), np.sin(last_box_in_lidar[0, 6])
            lidar_se3_last_box = np.array([
                [_c,    -_s,    0.,     last_box_in_lidar[0, 0]],
                [_s,    _c,     0.,     last_box_in_lidar[0, 1]],
                [0.,    0.,     1.,     last_box_in_lidar[0, 2]],
                [0.,    0.,     0.,     1.]
            ])
            
            corrected_points_in_lidar = apply_se3_(lidar_se3_last_box, 
                                                   points_=points_in_box, 
                                                   return_transformed=True)
            
            # translate the whole thing by 10 meters along box's heading to make viz clearer
            tf = np.eye(4)
            tf[:2, -1] = 10. * np.array([np.cos(boxes_in_lidar[-1, 6]), np.sin(boxes_in_lidar[-1, 6])])
            apply_se3_(tf, points_=corrected_points_in_lidar, boxes_=last_box_in_lidar)

            painter = PointsPainter(
                np.concatenate([points_in_lidar[:, :3], corrected_points_in_lidar[:, :3]]),
                np.concatenate([boxes_in_lidar[:, :7], last_box_in_lidar[:, :7]])
            )
            painter.show()

        # make cluster info {cluster embedd, cluster file path}
        # cls_name = input('>>> what class is this: ')
        # if cls_name not in ('car', 'ped'):
        #     continue

        members_path = [trajs_path[_i] for _i in indices_cluster_trajs_path]
        cluster_top_members_static_desc = scaled_trajs_static_descriptor[mask_cluster][ids_by_prob[:30]]
        cluster_top_members_static_embed = reducer_static.transform(cluster_top_members_static_desc)
        
        cluster_info = {
            'members_path': members_path,
            'cluster_top_members_static_embed': cluster_top_members_static_embed
        }
        with open(f'artifact/cluster_info_{label}_{num_sweeps}sweeps.pkl', 'wb') as f:
            pickle.dump(cluster_info, f)



if __name__ == '__main__':
    main()
