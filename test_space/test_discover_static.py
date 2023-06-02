import numpy as np
import torch
import hdbscan
from pathlib import Path
from pprint import pprint
import pickle
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.cm
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from test_space.tools import build_dataset_for_testing
from workspace.uda_tools_box import remove_ground, init_ground_segmenter, BoxFinder
from workspace.o3d_visualization import PointsPainter
from workspace.nuscenes_temporal_utils import apply_se3_
from workspace.traj_discovery import TrajectoryProcessor


NUM_SWEEPS = 15
class_names = ['car',]
dataset, dataloader = build_dataset_for_testing(
    '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
    training=True,
    batch_size=2,
    version='v1.0-mini',
    debug_dataset=True,
    MAX_SWEEPS=NUM_SWEEPS
)

ground_segmenter = init_ground_segmenter(th_dist=0.2)

clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.,
                            leaf_size=100,
                            metric='euclidean', min_cluster_size=30, min_samples=None)

traj_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.,
                                 metric='euclidean', min_cluster_size=10, min_samples=None)

box_finder = BoxFinder(return_in_form='box_openpcdet', return_theta_star=True)

disco_traj_root = Path('../data/nuscenes/v1.0-mini/discovered_database_15sweeps')
disco_traj_clusters_info_root = Path('../workspace/artifact/good/')


def load_discovered_trajs(sample_token: str, disco_database_root: Path, return_in_lidar_frame: bool = True) -> np.ndarray:
    discovered_trajs_path = list(disco_database_root.glob(f'{sample_token}_label*'))
    
    disco_boxes = list()
    for traj_path in discovered_trajs_path:
        with open(traj_path, 'rb') as f:
            traj_info = pickle.load(f)
        if np.any(traj_info['boxes_in_glob'][0, 3: 6] < 1e-1) or np.any(traj_info['boxes_in_glob'][0, 3: 6] > 7.):
            continue
        if return_in_lidar_frame:
            boxes_in_liar = apply_se3_(np.linalg.inv(traj_info['glob_se3_lidar']), 
                                       boxes_=traj_info['boxes_in_glob'], 
                                       return_transformed=True)  # (N_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx    
            disco_boxes.append(boxes_in_liar)
        else:
            disco_boxes.append(traj_info['boxes_in_glob'])

    disco_boxes = np.concatenate(disco_boxes, axis=0)
    return disco_boxes


def load_trajs_static_embedding(traj_clusters_info_root: Path) -> List[np.ndarray]:
    classes_name = ['car', 'car+cyc+motor', 'ped']
    trajs_info_path = [traj_clusters_info_root / Path(f'cluster_info_{name}_15sweeps.pkl') for name in classes_name]
    
    clusters_top_embeddings = list()
    for idx, info_path in enumerate(trajs_info_path):
        with open(info_path, 'rb') as f:
            traj_info = pickle.load(f)
        
        clusters_top_embeddings.append(
            np.pad(traj_info['cluster_top_members_static_embed'], pad_width=[(0, 0), (0, 1)], constant_values=idx)
        )
    
    clusters_top_embeddings = np.concatenate(clusters_top_embeddings)
    return clusters_top_embeddings


def filter_points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Remove points that are inside 1 of the boxes
    """
    points_box_index = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3]).float(),
        torch.from_numpy(boxes[:, :7]).float(),
    ).numpy()  # (N_disco_boxes, N_pts)
    mask_points_in_boxes = (points_box_index > 0).any(axis=0)  # (N_pts,)
    
    return points[np.logical_not(mask_points_in_boxes)]


def main(sample_idx: int):
    data_dict = dataset[sample_idx]  
    points = data_dict['points']  # (N_pts, 3 + C) - x, y, z, C-channel

    # load all traj found in this sample
    sample_token = data_dict['metadata']['token']
    disco_boxes = load_discovered_trajs(sample_token, disco_traj_root, 
                                        return_in_lidar_frame=True)  # (N_disco_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
    
    # remove ground
    points, ground_pts = remove_ground(points, ground_segmenter, return_ground_points=True)
    tree_ground = KDTree(ground_pts[:, :3])  # to query for ground height given a 3d coord

    # remove points in disco_boxes
    points = filter_points_in_boxes(points, disco_boxes)

    # cluster remaining points
    clusterer.fit(points[:, :3])
    points_label = clusterer.labels_.copy()
    unq_labels = np.unique(points_label)

    # load traj_clusters' embedding
    traj_clusters_top_embeddings = load_trajs_static_embedding(disco_traj_clusters_info_root)
    traj_clusterer.fit(traj_clusters_top_embeddings[:, :3])
    traj_clusterer.generate_prediction_data()

    # load embedding computer
    with open(disco_traj_clusters_info_root / Path('umap_static_15sweeps_v1.0-mini.pkl'), 'rb') as f:
        reducer = pickle.load(f)

    TrajectoryProcessor.setup_class_attribute(num_sweeps=NUM_SWEEPS, debug=True, look_for_static=True)
    all_static_traj_boxes, all_static_traj_embedding = list(), list()
    all_static_traj_boxes_last = list()
    for label in tqdm(unq_labels, total=unq_labels.shape[0]):
        if label == -1:
            continue
            
        traj = TrajectoryProcessor()
        traj(points[points_label == label], None, None, box_finder, tree_ground, ground_pts)
        if traj.info is None:
            # invalid traj -> skip
            continue

        traj_boxes = traj.info['boxes_in_lidar']
        # check recovered boxes' dimension
        if np.logical_or(traj_boxes[0, 3: 6] < 0.1, traj_boxes[0, 3: 6] > 7.).any():
            # invalid dimension -> skip
            continue

        # compute descriptor to id class
        traj_embedding = traj.build_descriptor(use_static_attribute_only=True)

        # store
        all_static_traj_boxes.append(np.pad(traj_boxes, pad_width=[(0, 0), (0, 1)], constant_values=label))
        all_static_traj_boxes_last.append(traj_boxes[-1])
        all_static_traj_embedding.append(traj_embedding)
    
    all_static_traj_boxes = np.concatenate(all_static_traj_boxes)
    all_static_traj_embedding = np.stack(all_static_traj_embedding, axis=0)
    all_static_traj_boxes_last = np.stack(all_static_traj_boxes_last, axis=0)
    all_static_traj_embedding = reducer.transform(StandardScaler().fit_transform(all_static_traj_embedding))
    
    print('assign cls for static traj')
    all_static_traj_labels, _ = hdbscan.approximate_predict(traj_clusterer, all_static_traj_embedding)
    
    _unq_labels, _counts = np.unique(all_static_traj_labels, return_counts=True)
    print('_unq_labels: ', _unq_labels)
    print('_counts: ', _counts)

    def viz():
        print('showing filtered points & disco_boxes')
        painter = PointsPainter(points[:, :3], boxes=disco_boxes)
        painter.show()
        print('---')
        
        print('showing traj_clusters embedding')
        traj_clusters_color = np.array(['r', 'g', 'b'])
        traj_embeddings_color = traj_clusters_color[traj_clusterer.labels_]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(traj_clusters_top_embeddings[:, 0], 
                   traj_clusters_top_embeddings[:, 1], 
                   traj_clusters_top_embeddings[:, 2], 
                   c=traj_embeddings_color)
        
        ax.scatter(all_static_traj_embedding[:, 0], 
                   all_static_traj_embedding[:, 1], 
                   all_static_traj_embedding[:, 2], 
                   'x', c='k')
        plt.show()
        print('---')

        print('showing clustered points & static boxes')
        clusters_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_labels.shape[0]))[:, :3]
        points_color = clusters_color[points_label]
        boxes_color = clusters_color[all_static_traj_boxes[:, -1].astype(int)]
        painter = PointsPainter(points[:, :3], all_static_traj_boxes[:, :7])
        painter.show(points_color, boxes_color)
        print('---')

        print('showing clustered points & static boxes & their classes')
        traj_clusters_color = np.eye(3)
        boxes_color = traj_clusters_color[all_static_traj_labels]
        boxes_color[all_static_traj_labels == -1] = 0.
        painter = PointsPainter(points[:, :3], all_static_traj_boxes_last[:, :7])
        painter.show(points_color, boxes_color)
        print('---')

    viz()




if __name__ == '__main__':
    main(sample_idx=46)
    # 46: parked car
    # 10: has 3 moving cars
