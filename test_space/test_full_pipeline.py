import numpy as np
import hdbscan
import matplotlib.cm
from pathlib import Path
from sklearn.neighbors import KDTree
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

from test_space.tools import build_dataset_for_testing

from workspace.uda_tools_box import remove_ground, init_ground_segmenter, BoxFinder, PolynomialRegression
from workspace.o3d_visualization import PointsPainter
from workspace.box_fusion_utils import kde_fusion


def main():
    # init
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

    ground_segmenter = init_ground_segmenter(th_dist=0.3)

    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None)
    
    box_finder = BoxFinder(return_in_form='box_openpcdet', return_theta_star=True)

    ransac = RANSACRegressor(PolynomialRegression(degree=2), 
                             min_samples=5,
                             random_state=0)

     # ---------------
    data_dict = dataset[110]  
    points = data_dict['points']  # (N, 3 + C) - x, y, z, C-channel

    # =================================================
    # =================================================

    # ground segmentation
    points, ground_pts = remove_ground(points, ground_segmenter, return_ground_points=True)
    
    tree_ground = KDTree(ground_pts[:, :3])  # to query for ground height given a 3d coord

    # clustering
    clusterer.fit(points[:, :2])
    points_label = clusterer.labels_.copy()
    unq_labels, inv_unq_labels = np.unique(points_label, return_inverse=True)

    # cluter iteration
    # uda_database_root = Path('artifact/uda_db')
    # if not uda_database_root.exists():
    #     uda_database_root.mkdir(parents=True, exist_ok=True)

    all_traj_boxes = list()
    for label in tqdm(unq_labels, total=unq_labels.shape[0]):
        if label == -1:
            # label of cluster representing outlier
            continue

        this_points = points[points_label == label]  # (N_in_cluster, 3 + C)
        
        this_points_sweep_idx = this_points[:, -2].astype(int)  # (N_in_cluster,)
        unq_sweep_idx, num_points_per_sweep = np.unique(this_points_sweep_idx, return_counts=True)

        # filter: contains points from a single sweep
        if unq_sweep_idx.shape[0] == 1:
            continue
        
        # if label in (44, 59):
        #     painter = PointsPainter(this_points[:, :3])
        #     painter.show()
        #     print('hold')

        traj_boxes = list()
        encounter_invalid_box = False
        init_theta, prev_num_points = None, -1
        for sweep_idx in unq_sweep_idx:
            sweep_points = this_points[this_points_sweep_idx == sweep_idx]  # (N_in_sw, 3 + C)

            box_bev, mean_z, theta_star = box_finder.fit(sweep_points, init_theta=init_theta)
            # box_bev: [x, y, dx, dy, heading]
            
            # update init_theta
            if sweep_points.shape[0] > prev_num_points:
                init_theta = theta_star
                prev_num_points = sweep_points.shape[0]

            # get box's height & its z-coord of its center
            perspective_center = np.pad(box_bev[:2], pad_width=[(0, 1)], constant_values=mean_z)
            
            dist_to_neighbor, neighbor_ids = tree_ground.query(perspective_center.reshape(1, -1), k=3, return_distance=True)
            weights = 1.0 / np.clip(dist_to_neighbor.reshape(-1), a_min=1e-3, a_max=None)
            ground_height_at_center = np.sum(ground_pts[neighbor_ids.reshape(-1), 2] * weights) / np.sum(weights)
            
            # TODO: get ground heigh for every point
            dist_to_neighbor, neighbor_ids = tree_ground.query(sweep_points[:, :3], k=3, return_distance=True)
            #  dist_to_neighbor (N, k)
            weights = 1.0 / np.clip(dist_to_neighbor, a_min=1e-3, a_max=None)  # (N, k)
            weights = weights / weights.sum(axis=1).reshape(-1, 1)  # (N, k)

            ground_height = ground_pts[neighbor_ids.reshape(-1), 2].reshape(sweep_points.shape[0], -1)  # (N * k,) -> (N, k)
            ground_height = np.sum(ground_height * weights, axis=1)

            box_height = np.max(sweep_points[:, 2] - ground_height)
            center_z = ground_height_at_center + 0.5 * box_height

            # assembly box
            box = np.array([box_bev[0], box_bev[1], center_z, box_bev[2], box_bev[3], box_height, box_bev[4]])

            # filter: box's volume
            box_volume = box[3] * box[4] * box[5]
            if box_volume < 0.15 or box_volume > 120.:  # lower threshold as 0.15 to include pedestrian
                encounter_invalid_box = True
                break
            else:
                traj_boxes.append(box)
            
        if encounter_invalid_box or len(traj_boxes) == 0:
            continue
        else:
            # valid trajectory 
            traj_boxes = np.stack(traj_boxes, axis=0)
            traj_boxes = np.pad(traj_boxes, pad_width=[(0, 0), (0, 1)], constant_values=label)  # [x, y, z, dx, dy, dz, yaw, cluster_id]
            
            # aggregate boxes'size using KDE
            __traj_boxes = np.pad(traj_boxes[:, :7], pad_width=[(0, 0), (0, 2)], constant_values=0.)
            __traj_boxes[:, -1] = num_points_per_sweep.astype(float) / float(this_points.shape[0])
            fused_box = kde_fusion(__traj_boxes, src_weights=__traj_boxes[:, -1])
            traj_boxes[:, 3: 6] = fused_box[3: 6]

            # refind boxes' location on XY-plane using RANSAC
            ransac.fit(np.expand_dims(traj_boxes[:, 0], axis=1), traj_boxes[:, 1])
            traj_boxes[:, 1] = ransac.predict(traj_boxes[:, 0].reshape(-1, 1))
            
            # TODO: think of a way to get better init of heading

            all_traj_boxes.append(traj_boxes)
    
    # =================================================
    # =================================================
    assert len(all_traj_boxes) > 0
    all_traj_boxes = np.concatenate(all_traj_boxes)
    painter = PointsPainter(points[:, :3], all_traj_boxes)
    
    labels_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_labels.shape[0]))[:, :3]
    points_color = labels_color[points_label]
    points_color[points_label == -1] = 0.0
    boxes_color = labels_color[all_traj_boxes[:, -1].astype(int)] 

    painter.show(xyz_color=points_color, boxes_color=boxes_color)



if __name__ == '__main__':
    main()

