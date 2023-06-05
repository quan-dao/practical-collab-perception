import numpy as np
import torch
from sklearn.neighbors import KDTree, LocalOutlierFactor
from pathlib import Path
from typing import Tuple, List
import pickle

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

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
    point_cloud_range = None
    look_for_static = None
    hold_pickle = False
    debug = False

    @staticmethod
    def setup_class_attribute(num_sweeps: int,
                              min_num_sweeps_in_traj: int = 3, 
                              num_ground_neighbors: int =3,
                              filter_min_points_in_sweep: int = 5,
                              filter_max_dim: float = 7.,
                              threshold_area_small_obj: float = 0.5,
                              threshold_displacement_small_obj: float = 0.35,
                              threshold_displacement_large_obj: float = 2.0,
                              point_cloud_range: np.ndarray = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                              look_for_static: bool = False,
                              hold_pickle: bool = False,
                              debug: bool = False):
        TrajectoryProcessor.num_sweeps = num_sweeps
        TrajectoryProcessor.min_num_sweeps_in_traj = min_num_sweeps_in_traj
        TrajectoryProcessor.num_ground_neighbors = num_ground_neighbors
        TrajectoryProcessor.filter_max_dim = filter_max_dim
        TrajectoryProcessor.filter_min_points_in_sweep = filter_min_points_in_sweep
        TrajectoryProcessor.threshold_area_small_obj = threshold_area_small_obj
        TrajectoryProcessor.threshold_displacement_small_obj = threshold_displacement_small_obj
        TrajectoryProcessor.threshold_displacement_large_obj = threshold_displacement_large_obj
        TrajectoryProcessor.point_cloud_range = point_cloud_range
        TrajectoryProcessor.look_for_static = look_for_static
        TrajectoryProcessor.hold_pickle = hold_pickle
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

        # remove outlier
        classifier = LocalOutlierFactor(n_neighbors=10)
        mask_inliers = classifier.fit_predict(points[:, :3])
        points = points[mask_inliers > 0]
        
        points_sweep_idx = points[:, -2].astype(int)
        unq_sweep_idx = np.unique(points_sweep_idx)
        
        self.flag_long_enough = unq_sweep_idx.shape[0] >= self.min_num_sweeps_in_traj
        if not self.flag_long_enough:
            return  # skip the following processing
        
        rough_est_heading = self.get_rough_heading_estimation(points, points_sweep_idx, unq_sweep_idx)

        # -------------------------------
        # iterate sweep index to discover boxes
        # -------------------------------
        traj_boxes = list()
        num_points_in_boxes = list()  # to use as confident in box fusion

        for sweep_idx in unq_sweep_idx:
            sweep_points = points[points_sweep_idx == sweep_idx]
            if sweep_points.shape[0] < self.filter_min_points_in_sweep:
                # this sweep has too few points -> skip
                continue

            # fit a box to points in the current sweep
            box_bev, theta_star = box_finder.fit(sweep_points, rough_est_heading, None)
            
            # get box param along z-axis
            box_height, center_z = self.compute_box_height_and_center_z_coord(sweep_points, tree_ground, ground_points)

            # assemble box3d
            box = np.array([box_bev[0], box_bev[1], center_z, 
                            box_bev[2], box_bev[3], box_height,  # dx, dy, dz
                            box_bev[4],  # heading
                            sweep_idx])
            if np.logical_or(box[3: 6] < 0.1, box[3: 6] > self.filter_max_dim).any():
                # box has spurious dim -> skip
                continue
            
            # valid box -> store
            traj_boxes.append(box)
            num_points_in_boxes.append(sweep_points.shape[0])
        
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
        if self.look_for_static:
            threshold_disp = 0.
        else:
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
            if not self.hold_pickle:
                # not holding pickle -> pickle right the way
                self.pickle(save_to_path)
        else:
            # debugging -> keep pts & box in lidar coord
            self.info = {
                'points_in_lidar': pts, 
                'boxes_in_lidar': traj_boxes
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

    def build_descriptor(self, use_static_attribute_only: bool = False) -> np.ndarray:
        boxes = self.info['boxes_in_lidar'] if self.debug else self.info['boxes_in_glob']
        # normalized dimension
        grid_size_meters = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        dx, dy, dz = boxes[0, 3: 6] / grid_size_meters
        
        # assemble descriptor
        if use_static_attribute_only:
            descriptor = np.array([dx, dy, dz])
        else:  
            # -> use dynamic attributes also
            # total distance travel
            travelled_dist = np.linalg.norm(boxes[1:, :2] - boxes[:-1, :2], axis=1).sum() / np.linalg.norm(grid_size_meters[:2])

            descriptor = np.array([dx, dy, dz, travelled_dist])

        return descriptor


def load_discovered_trajs(sample_token: str, disco_database_root: Path, return_in_lidar_frame: bool = True) -> np.ndarray:
    """
    Load discovered dynamic trajs of a sampple

    Args:
        sample_token:
        disco_database_root: where to search for precomputed boxes of dynamic trajs
        return_in_lidar_frame: if True return boxes in LiDAR frame of the sample_token, else return in Global frame

    Returns:
        discovered_boxes: (N_disco_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx

    """
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


def load_trajs_static_embedding(traj_clusters_info_root: Path, classes_name: list = ['car', 'ped']) -> List[np.ndarray]:
    """
    Load pre-computed embeddings of top-k members of each cluster of trajs. "Static" means embeddings are computed solely based on static attributes which are dx, dy, dz in this case.

    Args:
        traj_clusters_info_root: where to look for pre-computed static embedding
        classes_name: names of clusters

    Return:
        clusters_top_embeddings: List[ (N_top_k, C) ]. 
    """
    trajs_info_path = [traj_clusters_info_root / Path(f'cluster_info_{name}_15sweeps.pkl') for name in classes_name]
    
    clusters_top_embeddings = list()
    for idx, info_path in enumerate(trajs_info_path):
        with open(info_path, 'rb') as f:
            traj_info = pickle.load(f)
        
        clusters_top_embeddings.append(
            np.pad(traj_info['cluster_top_members_static_embed'], pad_width=[(0, 0), (0, 1)], constant_values=idx)
        )
    
    return clusters_top_embeddings


def filter_points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Remove points that are inside 1 of the boxes. Use for removing points inside 1 or several trajs described by a set of boxes
    
    Args:
        points: (N_tot, 3 + C) - x, y, z, C-channel
        boxes: (N_boxes, 7 + C) - x, y, z, dx, dy, dz, yaw, C-channel

    Return:
        points_outside: (N_outside, 3 + C)
    """
    points_box_index = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3]).float(),
        torch.from_numpy(boxes[:, :7]).float(),
    ).numpy()  # (N_disco_boxes, N_pts)
    mask_points_in_boxes = (points_box_index > 0).any(axis=0)  # (N_pts,)
    
    return points[np.logical_not(mask_points_in_boxes)]


def organize_dyn_trajs(disco_dyn_info_root: Path, num_sweeps: int = 15, classes_name: list = ['car', 'ped']) -> None:
    for cls in classes_name:
        dict_sample_token2dyn = dict()
        with open(disco_dyn_info_root / Path(f"cluster_info_{cls}_{num_sweeps}sweeps.pkl"), 'rb') as f:
            cluster_info = pickle.load(f)
            
        for mem_path in cluster_info['members_path']:
            sample_token = str(mem_path.parts[-1]).split('_')[0]
            if sample_token not in dict_sample_token2dyn:
                dict_sample_token2dyn[sample_token] = [mem_path,]
            else:
                dict_sample_token2dyn[sample_token].append(mem_path)

        # sort dict_sample_token2dyn[sample_token]
        for k in dict_sample_token2dyn.keys():
            dict_sample_token2dyn[k].sort()

        with open(disco_dyn_info_root / Path(f"{cls}_sample_token2dyn.pkl"), 'wb') as f:
            pickle.dump(dict_sample_token2dyn, f)


def organize_static_trajs(disco_stat_root: Path, disco_stat_info_root: Path, classes_name: list = ['car', 'ped']) -> None:
    all_paths = disco_stat_root.glob('*.pkl')
    meta_dict = dict([(cls_name, dict()) for cls_name in classes_name])
    for _path in all_paths:
        # print(_path)
        # info['token']_label{idx_sta_traj}_{car or ped}.pkl
        filename_components = str(_path.parts[-1]).split('_')
        sample_token = filename_components[0]
        cls_of_traj = filename_components[-1].split('.')[0]
        
        if sample_token not in meta_dict[cls_of_traj]:
            meta_dict[cls_of_traj][sample_token] = [_path,]
        else:
            meta_dict[cls_of_traj][sample_token].append(_path)

    for cls in classes_name:
        for k in meta_dict[cls].keys():
            meta_dict[cls][k].sort()

        with open(disco_stat_info_root / Path(f"stat_{cls}_sample_token2dyn.pkl"), 'wb') as f:
            pickle.dump(meta_dict[cls], f)


def load_disco_traj_for_1sample(sample_token: str, car_sample_token2dyn: dict, ped_sample_token2dyn: dict, return_in_lidar_frame: bool = True) -> np.ndarray:
    """
    Args:
        sample_token:
        car_sample_token2dyn:
        ped_sample_token2dyn

    Returns:
        disco_dyn_boxes: (N, 10) - box-7, sweep_idx, inst_idx, cls_idx (0: car, 1: ped)
    """

    disco_dyn_boxes = list()
    offset_idx_traj = 0

    for cls_idx, dict_sample_token2dyn in enumerate([car_sample_token2dyn, ped_sample_token2dyn]):
        if sample_token not in dict_sample_token2dyn:
            continue
    
        for idx_traj, traj_path in enumerate(dict_sample_token2dyn[sample_token]):
            with open(traj_path, 'rb') as f:
                traj_info = pickle.load(f)
            
            if return_in_lidar_frame:
                boxes = apply_se3_(np.linalg.inv(traj_info['glob_se3_lidar']), 
                                   boxes_=traj_info['boxes_in_glob'], 
                                   return_transformed=True)  # (N_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx    
            else:
                boxes = traj_info['boxes_in_glob']  # (N_boxes, 8) - x, y, z, dx, dy, dz, heading, sweep_idx
            
            # append boxes with instance_idx, class_idx
            boxes = np.pad(boxes, pad_width=[(0, 0), (0, 2)], constant_values=0)
            boxes[:, -2] = idx_traj + offset_idx_traj
            boxes[:, -1] = cls_idx

            disco_dyn_boxes.append(boxes)

        # make sure ped trajs don't have the same traj idx as car
        offset_idx_traj = idx_traj + 1        

    if len(disco_dyn_boxes) == 0:
        return np.zeros([0, 10])  # box-7, sweep_idx, inst_idx, cls_idx (0: car, 1: ped)
    else:
        disco_dyn_boxes = np.concatenate(disco_dyn_boxes, axis=0)
        return disco_dyn_boxes  # (N, 10) - box-7, sweep_idx, inst_idx, cls_idx (0: car, 1: ped)
