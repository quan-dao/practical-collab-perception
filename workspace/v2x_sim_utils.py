import numpy as np
import torch
from nuscenes import NuScenes
from typing import Set, Tuple, Dict

from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from workspace.nuscenes_temporal_utils import apply_se3_, get_nuscenes_sensor_pose_in_global, get_one_pointcloud


def get_annos_of_1lidar(nusc: NuScenes, 
                        sample_data_token: str, 
                        classes_of_interest: Set[str] = set(['car', 'pedestrian'])) -> Tuple[np.ndarray]:
    """
    Returns:
        boxes_in_lidar: (N, 7) - box-7
        boxes_name: (N,) - str
        inst_tokens: (N,) - array of str, each is an instance token
    """
    boxes_in_glob, boxes_name, inst_tokens = list(), list(), list()
    boxes = nusc.get_boxes(sample_data_token)
    for box in boxes:
        box_name = map_name_from_general_to_detection[box.name]
        if box_name not in classes_of_interest:
            continue
        anno_record = nusc.get('sample_annotation', box.token)

        box_in_glob = np.zeros(7)  
        box_in_glob[:3] = box.center
        box_in_glob[3: 6] = box.wlh[[1, 0, 2]]

        box_in_glob[6] = np.arctan2(box.orientation.rotation_matrix[1, 0], 
                                    box.orientation.rotation_matrix[0, 0])
        # store output
        boxes_in_glob.append(box_in_glob)
        boxes_name.append(box_name)
        inst_tokens.append(anno_record['instance_token'])

    if len(box_in_glob) > 0:
        boxes_in_glob = np.stack(boxes_in_glob, axis=0)
        boxes_name = np.array(boxes_name)
        inst_tokens = np.array(inst_tokens)
    else:
        boxes_in_glob = np.zeros((0, 7))
        boxes_name, inst_tokens = np.array([], dtype=str), np.array([], dtype=str)
    
    glob_se3_lidar = get_nuscenes_sensor_pose_in_global(nusc, sample_data_token)
    boxes_in_lidar = apply_se3_(np.linalg.inv(glob_se3_lidar), 
                                boxes_=boxes_in_glob, 
                                return_transformed=True)
    
    return boxes_in_lidar, boxes_name, inst_tokens


def find_nonempty_boxes(points: np.ndarray, boxes: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Args:
        points: (N_pts, 3 + C) - point-3
        boxes: (N_box, 7 + C) - box-7
        use_gpu:

    Returns:
        mask_nonempty_boxes: (N_box,) - True if non empty
        num_points_in_box: (N_box, )
        box_idx_of_points: (N_pts,) - int, index of box that points are inside
    """
    if use_gpu:
        box_idx_of_points = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(points[:, :3]).float().unsqueeze(0).float().cuda(),
            torch.from_numpy(boxes[:, :7]).float().unsqueeze(0).float().cuda(),
        ).long().squeeze(0).cpu().numpy()  # (N_pts,) to index into (N_b)
        # box_idx_of_points == -1 means background points

        unq_box_idx_of_points, counts = np.unique(box_idx_of_points, return_counts=True)

        # exclude index -1 (outlier)
        _mask = unq_box_idx_of_points > -1
        unq_box_idx_of_points = unq_box_idx_of_points[_mask]
        counts = counts[_mask]

        mask_nonempty_boxes = np.zeros(boxes.shape[0], dtype=bool)
        mask_nonempty_boxes[unq_box_idx_of_points] = True

        num_points_in_boxes = np.zeros(boxes.shape[0], dtype=int)
        num_points_in_boxes[unq_box_idx_of_points] = counts
    else:
        mat_boxes_have_pts = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, :3]).float(), 
            torch.from_numpy(boxes[:, :7]).float()
        ).numpy()  # (N_b, N_pts) | box_idx_of_points[i, j] == 1 if points[j] in boxes
        mat_boxes_have_pts = mat_boxes_have_pts > 0
        mask_nonempty_boxes = np.any(mat_boxes_have_pts, axis=1)
        num_points_in_boxes = mat_boxes_have_pts.astype(int).sum(axis=1)

        # convert box_idx_of_points to (N_pts,) - int, index of box that points are inside
        box_idx_of_points = (mat_boxes_have_pts.T * np.arange(boxes.shape[0])).sum(axis=1)
        # (N_pts, N_b) * (N_b,), then sum(axis=1) -> (N_pts)
        # mark background points
        box_idx_of_points[np.all(mat_boxes_have_pts <= 0, axis=0)] = -1
        # make sure there is no points inside 2 boxes -> mark background those are weird
        box_idx_of_points[box_idx_of_points >= boxes.shape[0]] = -1
  
    # sanity test
    assert np.all((num_points_in_boxes > 0) == mask_nonempty_boxes)

    return mask_nonempty_boxes, num_points_in_boxes, box_idx_of_points


def get_points_and_boxes_of_1lidar(nusc: NuScenes, 
                                   sample_data_token: str, 
                                   classes_of_interest: Set[str] = set(['car', 'pedestrian']),
                                   points_in_boxes_by_gpu: bool = False,
                                   return_nonempty_boxes: bool = True,
                                   threshold_boxes_by_points: int = None) -> Dict[str, np.ndarray]:
    """
    Returns:
        points_in_lidar: (N_pts, 4 + 1) - point-4 (x, y, z, intensity)
        box_idx_of_points: (N_pts,)
        boxes_in_lidar: (N_box, 7)
        boxes_name: (N_box,) - str
        inst_tokens: (N_box,) - array of str, each is an instance token
    """
    points_in_lidar = get_one_pointcloud(nusc, sample_data_token)

    boxes_in_lidar, boxes_name, inst_tokens = get_annos_of_1lidar(nusc, sample_data_token, classes_of_interest)

    mask_nonempty_boxes, num_points_in_boxes, box_idx_of_points = find_nonempty_boxes(points_in_lidar, boxes_in_lidar, points_in_boxes_by_gpu)

    if return_nonempty_boxes or threshold_boxes_by_points is not None:
        mask = mask_nonempty_boxes if threshold_boxes_by_points is None else num_points_in_boxes > threshold_boxes_by_points

        boxes_in_lidar, boxes_name, inst_tokens = boxes_in_lidar[mask], boxes_name[mask], inst_tokens[mask]
        num_points_in_boxes = num_points_in_boxes[mask]

        pad_kept_boxes_ids = -np.ones(mask.shape[0], dtype=int)  # (N_box_all,)
        pad_kept_boxes_ids[mask] = np.arange(boxes_in_lidar.shape[0])  # (N_box_kept,)
        box_idx_of_points = pad_kept_boxes_ids[box_idx_of_points]

    output = {
        'points_in_lidar': points_in_lidar,  # (N_pts, 4 + 1) - point-4 (x, y, z, intensity)
        'box_idx_of_points': box_idx_of_points,  # (N_pts,)
        'boxes_in_lidar': boxes_in_lidar,  # (N_box, 7)
        'boxes_name': boxes_name,  # (N_box,) 
        'inst_tokens': inst_tokens,  # (N_box,)
    }
    return output
