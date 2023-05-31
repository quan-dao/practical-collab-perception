import numpy as np
import hdbscan
import matplotlib.cm
from pathlib import Path
from sklearn.neighbors import KDTree
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

from test_space.tools import build_dataset_for_testing

from workspace.uda_tools_box import remove_ground, init_ground_segmenter, BoxFinder, PolynomialRegression
from workspace.o3d_visualization import PointsPainter
from workspace.box_fusion_utils import kde_fusion


def main(sample_idx: int, num_sweeps: int, show_last: bool, correct_dyn_pts: bool):
    # init
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

    min_samples_traj_estim = 3
    huber = HuberRegressor(epsilon=1.75)

     # ---------------
    data_dict = dataset[sample_idx]  
    points = data_dict['points']  # (N, 3 + C) - x, y, z, C-channel

    # =================================================
    # =================================================

    # ground segmentation
    points, ground_pts = remove_ground(points, ground_segmenter, return_ground_points=True)
    points_sweep_idx = points[:, -2].astype(int)

    tree_ground = KDTree(ground_pts[:, :3])  # to query for ground height given a 3d coord

    # clustering
    clusterer.fit(points[:, :2])
    points_label = clusterer.labels_.copy()
    unq_labels = np.unique(points_label)

    # cluter iteration
    # uda_database_root = Path('artifact/uda_db')
    # if not uda_database_root.exists():
    #     uda_database_root.mkdir(parents=True, exist_ok=True)

    all_traj_boxes = list()
    for label in tqdm(unq_labels, total=unq_labels.shape[0]):
        if label == -1:
            # label of cluster representing outlier
            continue
        
        mask_this = points_label == label
        this_points = points[mask_this]  # (N_in_cluster, 3 + C)
        
        this_points_sweep_idx = this_points[:, -2].astype(int)  # (N_in_cluster,)
        unq_sweep_idx = np.unique(this_points_sweep_idx)

        # filter: contains points from a single sweep
        if unq_sweep_idx.shape[0] < min_samples_traj_estim:
            continue
        
        # if label in (44, 59):
        #     painter = PointsPainter(this_points[:, :3])
        #     painter.show()
        #     print('hold')

        # get rough heading estimation
        pts_1st_group = this_points[this_points_sweep_idx == unq_sweep_idx[0], :2].mean(axis=0)
        pts_last_group = this_points[this_points_sweep_idx == unq_sweep_idx[-1], :2].mean(axis=0)
        rough_est_heading = pts_last_group - pts_1st_group
        rough_est_heading /= np.linalg.norm(rough_est_heading)

        traj_boxes, num_points_per_box = list(), list()
        encounter_invalid_box = False
        init_theta, prev_num_points = None, -1
        for sweep_idx in unq_sweep_idx:
            sweep_points = this_points[this_points_sweep_idx == sweep_idx]  # (N_in_sw, 3 + C)

            box_bev, mean_z, theta_star = box_finder.fit(sweep_points, rough_est_heading, init_theta=init_theta)
            # box_bev: [x, y, dx, dy, heading]
            
            # update init_theta
            if sweep_points.shape[0] > prev_num_points:
                init_theta = theta_star
                prev_num_points = sweep_points.shape[0]

            # get ground heigh for every point in inside the box
            dist_to_neighbor, neighbor_ids = tree_ground.query(sweep_points[:, :3], k=3, return_distance=True)
            #  dist_to_neighbor (N, k)
            # neighbor_ids: (N, k)
            weights = 1.0 / np.clip(dist_to_neighbor, a_min=1e-3, a_max=None)  # (N, k)
            weights = weights / weights.sum(axis=1).reshape(-1, 1)  # (N, k)

            ground_height = ground_pts[neighbor_ids.reshape(-1), 2].reshape(sweep_points.shape[0], -1)  # (N * k,) -> (N, k)
            ground_height = np.sum(ground_height * weights, axis=1)  # (N,)

            box_height = np.max(sweep_points[:, 2] - ground_height)
            center_z = np.mean(ground_height) + 0.5 * box_height

            # assembly box
            box = np.array([box_bev[0], box_bev[1], center_z, box_bev[2], box_bev[3], box_height, box_bev[4], sweep_idx])

            # filter: box's dim
            too_large_dim = np.any(box[3: 6] > 20)
            if too_large_dim or sweep_points.shape[0] < 5:  # lower threshold as 0.15 to include pedestrian
                continue
            else:
                traj_boxes.append(box)
                num_points_per_box.append(sweep_points.shape[0])
            
        if encounter_invalid_box or len(traj_boxes) == 0:
            continue
        else:
            # valid trajectory 
            traj_boxes = np.stack(traj_boxes, axis=0)

            traj_boxes = np.pad(traj_boxes, pad_width=[(0, 0), (0, 1)], constant_values=label)  # [x, y, z, dx, dy, dz, yaw, sweep_idx, cluster_id]
            
            # aggregate boxes'size using KDE
            num_points_per_box = np.array(num_points_per_box)
            __traj_boxes = np.pad(traj_boxes[:, :7], pad_width=[(0, 0), (0, 2)], constant_values=0.)
            __traj_boxes[:, -1] = num_points_per_box.astype(float) / float(num_points_per_box.sum())
            fused_box = kde_fusion(__traj_boxes, src_weights=__traj_boxes[:, -1])

            # filter: dynamic trajectory
            first_to_last_translation = np.linalg.norm(traj_boxes[-1, :2] - traj_boxes[0, :2])
            disp_threshold = 0.35 if fused_box[3] * fused_box[4] < 0.5 else 2.0
            if first_to_last_translation < disp_threshold:
                # -> static traj => skip
                continue

            # filter: box's dimension
            if np.logical_or(fused_box[3: 6] < 0.1, fused_box[3: 6] > 10).any():
                # valid box => skip
                continue

            traj_boxes[:, 3: 6] = fused_box[3: 6]
            traj_boxes[:, 6] = fused_box[6]

            if correct_dyn_pts:
                last_box = traj_boxes[-1]
                cos, sin = np.cos(last_box[6]), np.sin(last_box[6])
                last_box_pose = np.array([
                    [cos,   -sin,   0,      last_box[0]],
                    [sin,   cos,    0,      last_box[1]],
                    [0,     0,      1,      last_box[2]],
                    [0,     0,      0,      1]
                ])
                traj_sweep_idx = traj_boxes[:, -2].astype(int)
                for sweep_idx in np.unique(traj_sweep_idx):
                    mask_this_sweep = np.logical_and(mask_this, points_sweep_idx == sweep_idx)
                    
                    this_box = traj_boxes[traj_sweep_idx == sweep_idx].reshape(-1)
                    cos, sin = np.cos(this_box[6]), np.sin(this_box[6])
                    sweep_box_pose = np.array([
                        [cos,   -sin,   0,      this_box[0]],
                        [sin,   cos,    0,      this_box[1]],
                        [0,     0,      1,      this_box[2]],
                        [0,     0,      0,      1]
                    ])
                    tf = last_box_pose @ np.linalg.inv(sweep_box_pose)

                    points[mask_this_sweep, :3] = (tf[:3, :3] @ np.expand_dims(points[mask_this_sweep, :3], axis=-1)).reshape(-1, 3) + tf[:3, -1]
                    

            # store
            all_traj_boxes.append(traj_boxes)
    
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

    if correct_dyn_pts:
        # show last box & all points
        mask_show_pts = np.ones(points.shape[0], dtype=bool)
        mask_show_boxes = all_traj_boxes[:, -2].astype(int) == num_sweeps - 1
    
    labels_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_labels.shape[0]))[:, :3]
    points_color = labels_color[points_label]
    points_color[points_label == -1] = 0.0
    boxes_color = labels_color[all_traj_boxes[:, -1].astype(int)] 

    painter = PointsPainter(points[mask_show_pts, :3], all_traj_boxes[mask_show_boxes])
    painter.show(xyz_color=points_color[mask_show_pts], boxes_color=boxes_color[mask_show_boxes])



if __name__ == '__main__':
    main(sample_idx=10,  # 10 110 200
         num_sweeps=15,
         show_last=True,
         correct_dyn_pts=True)

