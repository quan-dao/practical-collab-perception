import numpy as np
from pathlib import Path
from sklearn.neighbors import KDTree
import hdbscan
from typing import Tuple
import pickle

from workspace.uda_tools_box import BoxFinder
from workspace.box_fusion_utils import kde_fusion
from workspace.nuscenes_temporal_utils import apply_se3_


class TrajectoryProcessor(object):
    num_sweeps: int = -1
    box_finder: BoxFinder = None
    tree_ground: KDTree = None
    ground_points: np.ndarray = None
    min_num_sweeps_in_traj, num_ground_neighbors = None, None
    filter_max_dim, filter_min_points_in_sweep = None, None
    threshold_area_small_obj = None
    threshold_displacement_small_obj, threshold_displacement_large_obj = None, None

    @staticmethod
    def setup_class_attribute(num_sweeps: int,
                              min_num_sweeps_in_traj: int = 3, 
                              num_ground_neighbors: int =3,
                              filter_min_points_in_sweep: int = 5,
                              filter_max_dim: float = 10.,
                              threshold_area_small_obj: float = 0.5,
                              threshold_displacement_small_obj: float = 0.35,
                              threshold_displacement_large_obj: float = 2.0,
                              debug: bool = False):
        TrajectoryProcessor.num_sweeps = num_sweeps
        TrajectoryProcessor.min_num_sweeps_in_traj = min_num_sweeps_in_traj
        TrajectoryProcessor.num_ground_neighbors = num_ground_neighbors
        TrajectoryProcessor.filter_max_dim = filter_max_dim
        TrajectoryProcessor.filter_min_points_in_sweep = filter_min_points_in_sweep
        TrajectoryProcessor.threshold_area_small_obj = threshold_area_small_obj
        TrajectoryProcessor.threshold_displacement_small_obj = threshold_displacement_small_obj
        TrajectoryProcessor.threshold_displacement_large_obj = threshold_displacement_large_obj
        TrajectoryProcessor.debug = debug

    def __init__(self) -> None:
        self.info = None

    def __call__(self, 
                 points: np.ndarray, 
                 glob_se3_lidar: np.ndarray,
                 save_to_path: Path,
                 box_finder: BoxFinder, tree_ground: KDTree, ground_points: np.ndarray) -> None:
        """
        Args:
            points: (N, 3 + C) - x, y, z, intensity, time-lag, [sweep_idx, instance_idx (always = -1 in UDA setting)]
            glob_se3_lidar: (4, 4) - tf from lidar frame to glob frame
            box_finder: to fit a rectangle to a set of points
            tree_ground: for querying k-nearest ground points
            ground_points: (N_ground_pts, 3) - x, y, z
            
        """
        assert self.num_sweeps > 0, "invoke TrajectoryProcessor.setup_class_attribute() first"
        
        points_sweep_idx = points[:, -2].astype(int)
        unq_sweep_idx = np.unique(points_sweep_idx)
        
        self.flag_long_enough = unq_sweep_idx.shape[0] >= self.min_num_sweeps_in_traj
        if not self.flag_long_enough:
            return  # skip the following processing
        
        rough_est_heading = self.get_rough_heading_estimation(points, points_sweep_idx, unq_sweep_idx)

        # -------------------------------
        # iterate sweep index to discover boxes
        # -------------------------------
        iter_init_theta, iter_prev_num_points = None, -1
        traj_boxes = list()
        num_points_in_boxes = list()  # to use as confident in box fusion

        for sweep_idx in unq_sweep_idx:
            sweep_points = points[points_sweep_idx == sweep_idx]
            if sweep_points.shape[0] < self.filter_min_points_in_sweep:
                # this sweep has too few points -> skip
                continue

            # fit a box to points in the current sweep
            box_bev, theta_star = box_finder.fit(sweep_points, rough_est_heading, iter_init_theta)
            
            # get box param along z-axis
            box_height, center_z = self.compute_box_height_and_center_z_coord(sweep_points, tree_ground, ground_points)

            # assemble box3d
            box = np.array([box_bev[0], box_bev[1], center_z, 
                            box_bev[2], box_bev[3], box_height,  # dx, dy, dz
                            box_bev[4],  # heading
                            sweep_idx])
            if np.logical_or(box[3: 6] < 0, box[3: 6] > self.filter_max_dim).any():
                # box has spurious dim -> skip
                continue
            
            # valid box -> store
            traj_boxes.append(box)
            num_points_in_boxes.append(sweep_points.shape[0])

            # update iterating vars
            if sweep_points.shape[0] > iter_prev_num_points:
                iter_init_theta = theta_star
                iter_prev_num_points = sweep_points.shape[0]
        
        # -------------------------------
        # refine discoverd boxes
        # -------------------------------
        if len(traj_boxes) < 2:
            # nonempty traj, but doesn't have enough valid boxes
            return
        
        traj_boxes = np.stack(traj_boxes, axis=0)  # (N_valid_sweeps, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
        self.refine_boxes(traj_boxes, np.array(num_points_in_boxes))

        # check if this traj is actually dynamic
        area_xy = traj_boxes[0, 3] * traj_boxes[0, 4]
        threshold_disp = self.threshold_displacement_small_obj if area_xy < self.threshold_area_small_obj else \
              self.threshold_displacement_large_obj
        first_to_last_translation = np.linalg.norm(traj_boxes[-1, :2] - traj_boxes[0, :2])
        if first_to_last_translation < threshold_disp:
            return

        # -------------------------------
        # build traj info
        # -------------------------------
        pts = points.copy()
        
        # filter points belong to invalid sweep (i.e. sweep doesn't have valid box)
        valid_sweep_ids = traj_boxes[:, -1].astype(int)
        mask_pts_valid_sweep = np.any(points_sweep_idx.reshape(-1, 1) == valid_sweep_ids.reshape(1, -1), 
                                      axis=1)  # (N_pts, 1) == (1, N_valid_sw) -> (N_pts,)
        pts = pts[mask_pts_valid_sweep]

        # map pts & traj_boxes to global frame
        if not self.debug:
            apply_se3_(glob_se3_lidar, points_=pts, boxes_=traj_boxes)
            
            self.info = {
                'points_in_glob': pts,  # (N, 3 + C) - x, y, z, intensity, time-lag, [sweep_idx, instance_idx (always = -1 in UDA setting)]
                'boxes_in_glob': traj_boxes,  # (N_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
                'glob_se3_lidar': glob_se3_lidar
            }
            self.pickle(save_to_path)
        else:
            # debugging -> keep pts & box in lidar coord
            self.info = {
                'points_in_lidar': pts, 'boxes_in_lidar': traj_boxes, 'total_translation': first_to_last_translation
            }


    @staticmethod
    def get_rough_heading_estimation(points: np.ndarray, points_sweep_idx: np.ndarray, unq_sweep_idx: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (N, 3 + C) - x, y, z, intensity, time-lag, [sweep_idx, instance_idx (always = -1 in UDA setting)]
            points_sweep_idx: (N,) - sweep index of every point
            unq_sweep_idx: (N_unq_sw) - array of unique sweep index (sorted in ascending order)

        Returns:
            rough_est_heading: (2,) - x, y  | heading direction
        """
        first_centroid_xy = points[points_sweep_idx == unq_sweep_idx[0], :2].mean(axis=0)
        last_centroid_xy = points[points_sweep_idx == unq_sweep_idx[-1], :2].mean(axis=0)
        rough_est_heading = last_centroid_xy - first_centroid_xy
        rough_est_heading /= np.linalg.norm(rough_est_heading)
        return rough_est_heading
    
    def compute_box_height_and_center_z_coord(self, sweep_points: np.ndarray, tree_ground: KDTree, ground_pts: np.ndarray) -> Tuple[float]:
        """
        Args:
            sweep_points: (N_pts_in_sweep, 3 + C) - - x, y, z, intensity, time-lag, [sweep_idx, instance_idx]
            tree_ground:
            ground_pts: (N_gr, 3)

        Returns:
            box_height:
            center_z:
        """
        # compute ground height for points in this sweep
        dist_to_neighbor, neighbor_ids = tree_ground.query(sweep_points[:, :3], 
                                                           k=self.num_ground_neighbors, 
                                                           return_distance=True)  # (N_p, k), (N_p, k)
        
        weights = 1.0 / np.clip(dist_to_neighbor, a_min=1e-3, a_max=None)  # (N_p, k)
        weights = weights / weights.sum(axis=1).reshape(-1, 1)  # (N_p, k)
        
        pts_ground_height = ground_pts[neighbor_ids.reshape(-1), 2].reshape(sweep_points.shape[0], -1)  # (N_p * k,) -> (N_p, k)
        pts_ground_height = np.sum(pts_ground_height * weights, axis=1)  # (N,)

        # get box height as max points_z - their ground height
        box_height = np.max(sweep_points[:, 2] - pts_ground_height)

        # get center's z-coord
        center_z = np.mean(pts_ground_height) + 0.5 * box_height

        return box_height, center_z

    @staticmethod
    def refine_boxes(traj_boxes_: np.ndarray, num_points_in_boxes: np.ndarray) -> None:
        """
        Args:
            traj_boxes: (N_valid_sweeps, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
            num_points_in_boxes: (N_valid_sweeps,)
        """
        # use KDE to aggregate boxes' size & heading
        __traj_boxes = np.pad(traj_boxes_[:, :7], pad_width=[(0, 0), (0, 2)], constant_values=0.)
        __traj_boxes[:, -1] = num_points_in_boxes.astype(float) / float(num_points_in_boxes.sum())
        fused_box = kde_fusion(__traj_boxes, src_weights=__traj_boxes[:, -1])

        # overwrites traj_boxes' size & heading
        traj_boxes_[:, 3: 7] = fused_box[3: 7]

    def pickle(self, save_to_path: Path) -> None:
        with open(save_to_path, 'wb') as f:
            pickle.dump(self.info, f)


