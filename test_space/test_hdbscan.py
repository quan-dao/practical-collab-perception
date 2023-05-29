import numpy as np
import hdbscan
import matplotlib.cm as cm
from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary
from workspace.uda_tools_box import remove_ground, init_ground_segmenter


def main():
    class_names = ['car',]
    num_sweeps = 30
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
        training=True,
        batch_size=2,
        version='v1.0-mini',
        debug_dataset=True,
        MAX_SWEEPS=num_sweeps
    )
    segmenter = init_ground_segmenter(th_dist=0.3)

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None)
    

    # ---------------
    data_dict = dataset[10]  
    points = data_dict['points']  # (N, 3 + C) - x, y, z, C-channel
    
    # ========================================================
    points = remove_ground(points, segmenter)
    print('points: ', points.shape)  # NOTE: lost time-stamp here!  

    clusterer.fit(points[:, :2])
    labels = clusterer.labels_.copy()
    print('points: ', points.shape)
    print('labels: ', labels.shape)
    unq_labels, inv_unq_lables, counts = np.unique(labels, return_inverse=True, return_counts=True)
    
    # ========================================================
    for lb in unq_labels:
        if lb not in [59, 44, 31]:
            continue
        print('showing cluster: ', lb)
        mask_this_cluster = labels == lb
        points_color = color_points_binary(mask_this_cluster.astype(int))
        painter = PointsPainter(points[:, :3])
        painter.show(xyz_color=points_color)

        np.save(f'artifact/hdbscan_dataset10_cluster{lb}.npy', points[mask_this_cluster])

    return 
    # TODO: visualize 20 largest clusters
    num_cluster_to_viz = 25
    unq_labels = unq_labels[np.argsort(-counts)]
    
    offset = 3
    unq_labels = unq_labels[offset * num_cluster_to_viz : (offset + 1)* num_cluster_to_viz]
    
    clusters_color = cm.rainbow(np.linspace(0, 1, num_cluster_to_viz))[:, :3]
    points_color = np.zeros((points.shape[0], 3))
    for idx, lb in enumerate(unq_labels):
        if lb == -1:
            continue
        mask_this_cluster = labels == lb
        points_color[mask_this_cluster] = clusters_color[idx]

    painter = PointsPainter(points[:, :3])
    painter.show(xyz_color=points_color)


if __name__ == '__main__':
    main()

