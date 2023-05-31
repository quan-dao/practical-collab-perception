import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, '/home/user/Desktop/python_ws/patchwork-plusplus/build/python_wrapper')
import pypatchworkpp


def remove_ground(points: np.ndarray, segmenter, return_ground_points: bool = False) -> np.ndarray:
    """
    Args:
        points: (N, 4 + C) - x, y, z, reflectant
        segmenter: PathWork++
        return_ground_points: if True, return ground points (N_gr, 3) - x, y, z

    Returns:
        nonground_points: (M, 4 + C) - x, y, z, reflectant
        ground_points: (N_gr, 3) - x, y, z
    """
    # assert points.shape[1] == 4, f"{points.shape[1]} != 4"
    segmenter.estimateGround(points)
    nonground_points = segmenter.getNonground()  # (M, 3)
    
    # ---
    # recover features of nonground_points
    tree = KDTree(points[:, :3])
    dist_to_neighbor, neighbor_ids = tree.query(nonground_points, k=1, return_distance=True)
    # neighbor_ids: (M, 1)
    # dist: (M, 1)
    
    # remove outlier outlier
    mask_inlier = dist_to_neighbor.reshape(-1) < 0.001  
    neighbor_ids = neighbor_ids[mask_inlier].reshape(-1)
    
    # extract nonground_points with feature from points
    nonground_points = points[neighbor_ids]

    if return_ground_points:
        ground_points = segmenter.getGround()
        return nonground_points, ground_points
    else:
        return nonground_points


def init_ground_segmenter(th_dist: float = None):
    params = pypatchworkpp.Parameters()
    params.sensor_height = 0.0
    if th_dist is not None:
        params.th_dist = th_dist
    
    segmenter = pypatchworkpp.patchworkpp(params)
    return segmenter


class BoxFinder(object):
    def __init__(self, 
                 criterion: str = 'closeness', 
                 return_in_form: str = 'edges_homogeneous_coord',
                 return_fitness: bool = False,
                 return_theta_star=True):
        
        assert criterion in ('area', 'closeness'), f"{criterion} is unknown"
        assert return_in_form in ('edges_homogeneous_coord', 'box_openpcdet')
        
        self.d0 = 0.01  # to be used for closeness covariance
        self.angle_resolution = np.deg2rad(3.0)
        self.cost_fnc = self.__getattribute__(f"criterion_{criterion}")
        self.return_in_form = return_in_form
        self.return_fitness = return_fitness
        self.return_theta_star = return_theta_star

    def fit(self, points: np.ndarray, rough_est_heading: np.ndarray, init_theta=None):
        assert len(points.shape) == 2
        assert points.shape[1] >= 3, "need xyz"
        
        xy = points[:, : 2]  # (N_pts, 2)
        
        if init_theta is None:
            thetas = np.arange(start=0., stop=np.pi/2.0, step=self.angle_resolution)
        else:
            if init_theta < 0:
                init_theta *= -1
            if init_theta > np.pi/2.0:
                init_theta -= np.pi/2.0
            thetas = np.arange(start=max(0., init_theta - np.deg2rad(20.)), 
                               stop=min(init_theta + np.deg2rad(20.), np.pi/2.0), 
                               step=np.deg2rad(1.5))

        cos, sin = np.cos(thetas), np.sin(thetas)
        e1 = np.stack([cos, sin], axis=1)  # (N_theta, 2)
        e2 = np.stack([-sin, cos], axis=1)  # (N_theta, 2)
        
        C1 = e1 @ xy.T  # (N_theta, N_pts)
        C2 = e2 @ xy.T  # (N_theta, N_pts)
        
        fitness = self.criterion_closeness(C1, C2)  # (N_theta,)
        
        _idx_max_fitness = np.argmax(fitness)
        theta_star = thetas[_idx_max_fitness]

        cos, sin = np.cos(theta_star), np.sin(theta_star)
        C1_star = xy @ np.array([cos, sin])
        C2_star = xy @ np.array([-sin, cos])

        a1, b1, c1 = cos, sin, np.min(C1_star)
        a2, b2, c2 = -sin, cos, np.min(C2_star)
        a3, b3, c3 = cos, sin, np.max(C1_star)    
        a4, b4, c4 = -sin, cos, np.max(C2_star)
    
        edges =  np.array([
            [a1, b1, c1],  # ax + by = c
            [a2, b2, c2],
            [a3, b3, c3],
            [a4, b4, c4]
        ])
        # ax + by - c = 0
        edges[:, -1] *= -1
        
        if self.return_in_form == 'edges_homogeneous_coord': 
            out = [edges,]
        else:
            box_bev = self.cvt_4edges_to_box(edges, rough_est_heading)
            out = [box_bev,]
        
        if self.return_fitness:
            out.append(fitness[_idx_max_fitness])

        if self.return_theta_star:
            out.append(theta_star)

        if len(out) == 1:
            return out[0]
        else:
            return out
        
    def criterion_closeness(self, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
        """
        Args:
            C1: (N_thetas, N_pts)
            C2: (N_thetas, N_pts)

        Returns:
            cost: (N_thetas,)
        """
        c1_max, c1_min = C1.max(axis=1), C1.min(axis=1)  # (N_thetas,), (N_thetas,)
        c2_max, c2_min = C2.max(axis=1), C2.min(axis=1)  # (N_thetas,), (N_thetas,)

        c1_max_diff = np.abs(c1_max.reshape(-1, 1) - C1)  # (N_thetas, N_pts)
        c1_min_diff = np.abs(C1 - c1_min.reshape(-1, 1))  # (N_thetas, N_pts)
        D1 = np.min(np.stack([c1_max_diff, c1_min_diff], axis=2), axis=2)  # (N_thetas, N_pts)

        c2_max_diff = np.abs(c2_max.reshape(-1, 1) - C2)  # (N_thetas, N_pts)
        c2_min_diff = np.abs(C2 - c2_min.reshape(-1, 1))  # (N_thetas, N_pts)
        D2 = np.min(np.stack([c2_max_diff, c2_min_diff], axis=2), axis=2)  # (N_thetas, N_pts)

        min_D1_D2 = np.clip(np.minimum(D1, D2), a_min=self.d0, a_max=None)  # (N_thetas, N_pts)
        fitness = np.sum(1.0 / min_D1_D2, axis=1)  # (N_thetas,)
        return fitness
 
    def cvt_4edges_to_box(self, edges_homo: np.ndarray, rough_est_heading: np.ndarray):
        """
        Convert 4-edges to [x, y, dx, dy, yaw] & compute mean z-coord of points in box
        Args:
            edges: (4, 3) - ax + by - c = 0
            points: (N, 3 + C) - x, y, z, C-channel

        Returns:
            box_bev: (5,) - x, y, dx, dy, yaw
        """
        rect = edges_homo
        vers = np.stack([np.cross(rect[_i], rect[_j]) for _i, _j in zip(range(4), [1, 2, 3, 0])], 
                        axis=0)  # (4, 3)
        vers /= vers[:, [-1]]  # normalize homogeneous coord to get regular coord

        center_xy = np.mean(vers[:, :2], axis=0)
        
        dim03 = np.linalg.norm(vers[3] - vers[0])
        dim01 = np.linalg.norm(vers[1] - vers[0])

        if dim03 > dim01:
            dx, dy = dim03, dim01
            perspective_heading_dir = vers[3] - vers[0]
        else:
            dx, dy = dim01, dim03
            perspective_heading_dir = vers[1] - vers[0]

        # two possible headinds
        heading0 = np.arctan2(perspective_heading_dir[1], perspective_heading_dir[0])
        heading1 = heading0 + np.pi

        # find the one that is clsoer the rough estimationg of heading
        _prod0 = np.array([np.cos(heading0), np.sin(heading0)]) @ rough_est_heading
        _prod1 = np.array([np.cos(heading1), np.sin(heading1)]) @ rough_est_heading
        if _prod0 > _prod1:
            heading = heading0
        else:
            heading = heading1

        box_bev = [*center_xy.tolist(), dx, dy, heading]
        return box_bev
