import numpy as np
from sklearn.neighbors import KDTree
import sys
sys.path.insert(0, '/home/user/Desktop/python_ws/patchwork-plusplus/build/python_wrapper')
import pypatchworkpp


def remove_ground(points: np.ndarray, segmenter) -> np.ndarray:
    """
    Args:
        points: (N, 4 + C) - x, y, z, reflectant
    Returns:
        ground_mask: (N,) - True if points are ground points
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
    out = points[neighbor_ids]
    return out


def init_ground_segmenter(th_dist: float = None):
    params = pypatchworkpp.Parameters()
    params.sensor_height = 0.0
    if th_dist is not None:
        params.th_dist = th_dist
    
    segmenter = pypatchworkpp.patchworkpp(params)
    return segmenter


class BoxFinder(object):
    def __init__(self, criterion: str = 'closeness', return_edges_in_homogeneous_coord: bool = True):
        assert criterion in ('area', 'closeness'), f"{criterion} is unknown"
        self.d0 = 0.01  # to be used for closeness covariance
        self.angle_resolution = 0.1
        self.cost_fnc = self.__getattribute__(f"criterion_{criterion}")
        self.return_edges_in_homogeneous_coord = return_edges_in_homogeneous_coord

    def fit(self, points: np.ndarray):
        assert len(points.shape) == 2
        assert points.shape[1] >= 2
        xy = points[:, : 2]
        queue = list()
        thetas = np.arange(start=0., stop=np.pi/2.0, step=self.angle_resolution)
        for _theta in thetas:
            cos, sin = np.cos(_theta), np.sin(_theta)
            e1 = np.array([cos, sin])
            e2 = np.array([-sin, cos])

            C1 = xy @ e1  # (N,)
            C2 = xy @ e2  # (N,)

            q = self.cost_fnc(C1, C2)
            queue.append(q)
        
        queue = np.array(queue)
        theta_star = thetas[np.argmax(queue)]

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

        if self.return_edges_in_homogeneous_coord:
            # ax + by - c = 0
            edges[:, -1] *= -1  

        return edges

    def criterion_area(self, C1: np.ndarray, C2: np.ndarray):
        assert len(C1.shape) == len(C2.shape) == 1
        assert C1.shape == C2.shape
        c1_max, c1_min = C1.max(), C1.min()
        c2_max, c2_min = C2.max(), C2.min()
        cost = -(c1_max - c1_min) * (c2_max - c2_min)
        return cost


    def criterion_closeness(self, C1: np.ndarray, C2: np.ndarray, d0: float = 0.01):
        assert len(C1.shape) == len(C2.shape) == 1
        assert C1.shape == C2.shape
        c1_max, c1_min = C1.max(), C1.min()
        c2_max, c2_min = C2.max(), C2.min()

        c1_max_diff = np.abs(c1_max - C1) # (N,)
        c1_min_diff = np.abs(C1 - c1_min)  # (N,)
        D1 = np.min(np.stack([c1_max_diff, c1_min_diff], axis=1), axis=1)

        c2_max_diff = np.abs(c2_max - C2) # (N,)
        c2_min_diff = np.abs(C2 - c2_min)  # (N,)
        D2 = np.min(np.stack([c2_max_diff, c2_min_diff], axis=1), axis=1)

        cost = 0
        for idx in range(D1.shape[0]):
            d = max([min(D1[idx], D2[idx]), d0])
            cost = cost + 1.0 / d

        return cost
    
    # TODO: convert 4-edges to [x, y, yaw] & find a way to have z
