import numpy as np
import hdbscan
import matplotlib.cm
from sklearn.neighbors import KDTree
from tqdm import tqdm

from test_space.tools import build_dataset_for_testing
from workspace.uda_tools_box import remove_ground, init_ground_segmenter, BoxFinder
from workspace.traj_discovery import TrajectoryProcessor
from workspace.o3d_visualization import PointsPainter


def main(sample_idx: int, 
         num_sweeps: int,
         show_last: bool):
    
    class_names = ['car',]
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
        training=True,
        batch_size=2,
        version='v1.0-mini',
        debug_dataset=True,
        MAX_SWEEPS=num_sweeps
    )
    ground_segmenter = init_ground_segmenter(th_dist=0.2)

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=30, min_samples=None)
    
    box_finder = BoxFinder(return_in_form='box_openpcdet', return_theta_star=True)
    
    def main_():
        TrajectoryProcessor.setup_class_attribute(num_sweeps=num_sweeps, debug=True)
        
        data_dict = dataset[sample_idx]  
        points = data_dict['points']  # (N, 3 + C) - x, y, z, C-channel

        # remove ground
        points, ground_pts = remove_ground(points, ground_segmenter, return_ground_points=True)
        tree_ground = KDTree(ground_pts[:, :3])  # to query for ground height given a 3d coord

        # cluster
        clusterer.fit(points[:, :3])
        points_label = clusterer.labels_.copy()
        
        unq_labels = np.unique(points_label)

        all_traj_boxes = list()
        for label in tqdm(unq_labels, total=unq_labels.shape[0]):
            if label == -1:
                # label of cluster representing outlier
                continue
            
            traj = TrajectoryProcessor()    
            traj(points[points_label == label], 
                 None, None, 
                 box_finder, tree_ground, ground_pts)
            
            if traj.info is None:
                # invalid trajectory
                continue
            
            all_traj_boxes.append(np.pad(traj.info['boxes_in_lidar'], pad_width=[(0, 0), (0, 1)], constant_values=label))
        
        # =================================================
        # =================================================
        assert len(all_traj_boxes) > 0
        all_traj_boxes = np.concatenate(all_traj_boxes)

        if show_last:
            mask_show_pts = points[:, -2].astype(int) == num_sweeps - 1 
            mask_show_boxes = all_traj_boxes[:, -2].astype(int) == num_sweeps - 1
        else:
            mask_show_pts = np.ones(points.shape[0], dtype=bool)
            mask_show_boxes = np.ones(all_traj_boxes.shape[0], dtype=bool)
        
        labels_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_labels.shape[0]))[:, :3]
        points_color = labels_color[points_label]
        points_color[points_label == -1] = 0.0
        boxes_color = labels_color[all_traj_boxes[:, -1].astype(int)] 

        painter = PointsPainter(points[mask_show_pts, :3], all_traj_boxes[mask_show_boxes])
        painter.show(xyz_color=points_color[mask_show_pts], boxes_color=boxes_color[mask_show_boxes])

    main_()


if __name__ == '__main__':
    main(sample_idx=110,
         num_sweeps=15,
         show_last=False)

    # NOTE:
    # sample 5: left of Lidar -> seeing only the back of the car -> bad box fitting
    # sample 46: missing park cars | small cluster in the middle of a car -> perhaps failure in clustering

