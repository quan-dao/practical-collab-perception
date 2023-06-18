import numpy as np
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion
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
        anno_tokens: (N,) - array of str, each is an annotation token
    """
    boxes_in_glob, boxes_name, inst_tokens, anno_tokens = list(), list(), list(), list()
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
        anno_tokens.append(box.token)

    if len(box_in_glob) > 0:
        boxes_in_glob = np.stack(boxes_in_glob, axis=0)
        boxes_name = np.array(boxes_name)
        inst_tokens = np.array(inst_tokens)
        anno_tokens = np.array(anno_tokens)
    else:
        boxes_in_glob = np.zeros((0, 7))
        boxes_name, inst_tokens, anno_tokens = np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str)
    
    glob_se3_lidar = get_nuscenes_sensor_pose_in_global(nusc, sample_data_token)
    boxes_in_lidar = apply_se3_(np.linalg.inv(glob_se3_lidar), 
                                boxes_=boxes_in_glob, 
                                return_transformed=True)
    
    return boxes_in_lidar, boxes_name, inst_tokens, anno_tokens


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
        points_in_lidar: (N_pts, 4) - point-4 (x, y, z, intensity)
        box_idx_of_points: (N_pts,)
        boxes_in_lidar: (N_box, 7)
        boxes_name: (N_box,) - str
        inst_tokens: (N_box,) - array of str, each is an instance token
    """
    points_in_lidar = get_one_pointcloud(nusc, sample_data_token)

    boxes_in_lidar, boxes_name, inst_tokens, anno_tokens = get_annos_of_1lidar(nusc, sample_data_token, classes_of_interest)

    mask_nonempty_boxes, num_points_in_boxes, box_idx_of_points = find_nonempty_boxes(points_in_lidar, boxes_in_lidar, points_in_boxes_by_gpu)

    if return_nonempty_boxes or threshold_boxes_by_points is not None:
        mask = mask_nonempty_boxes if threshold_boxes_by_points is None else num_points_in_boxes > threshold_boxes_by_points

        boxes_in_lidar, boxes_name, inst_tokens, anno_tokens = boxes_in_lidar[mask], boxes_name[mask], inst_tokens[mask], anno_tokens[mask]
        num_points_in_boxes = num_points_in_boxes[mask]

        pad_kept_boxes_ids = -np.ones(mask.shape[0], dtype=int)  # (N_box_all,)
        pad_kept_boxes_ids[mask] = np.arange(boxes_in_lidar.shape[0])  # (N_box_kept,)
        box_idx_of_points = pad_kept_boxes_ids[box_idx_of_points]

    output = {
        'points_in_lidar': points_in_lidar,  # (N_pts, 4) - point-4 (x, y, z, intensity)
        'box_idx_of_points': box_idx_of_points,  # (N_pts,)
        'boxes_in_lidar': boxes_in_lidar,  # (N_box, 7)
        'boxes_name': boxes_name,  # (N_box,) 
        'inst_tokens': inst_tokens,  # (N_box,)
        'anno_tokens': anno_tokens,  # (N_box,)
        'num_points_in_boxes': num_points_in_boxes,  # (N_box,)
    }
    return output


def get_historical_boxes_1instance(nusc: NuScenes,
                                   current_sample_data_tk: str, 
                                   current_box: np.ndarray, 
                                   current_anno_tk: str, 
                                   instance_idx: int,
                                   num_historical_boxes: int = 10) -> np.ndarray:
    """
    Returns:
        interp_boxes: (num_historical_boxes + 1, 7 + 2) - box-7, sweep_idx, inst_idx | in LiDAR
            num_historical_boxes + 1 +1 because this includes current_box too
    """
    def _qua2yaw(q: Quaternion) -> float:
        return np.arctan2(q.rotation_matrix[1, 0], q.rotation_matrix[0, 0])
    
    assert current_box.shape == (7,), f"{current_box.shape} is invalid"
    anno_record = nusc.get('sample_annotation', current_anno_tk)

    num_boxes_total = num_historical_boxes + 1  
    if anno_record['prev'] == '':
        # have no prev 
        return np.tile(current_box.reshape(1, -1), (num_boxes_total, 1))
    
    prev_box = nusc.get_box(anno_record['prev'])
    # map box to lidar frame
    glob_se3_prev = np.eye(4)
    glob_se3_prev[:3, :3] = prev_box.orientation.rotation_matrix
    glob_se3_prev[:3, -1] = prev_box.center

    glob_se3_lidar = get_nuscenes_sensor_pose_in_global(nusc, current_sample_data_tk)
    lidar_se3_prev = np.linalg.inv(glob_se3_lidar) @ glob_se3_prev
    
    # interpolate x, y, z
    timesteps = np.linspace(0., 1., num_boxes_total)  # include the current
    centers = np.stack([np.interp(timesteps,
                                  [timesteps[0], timesteps[-1]],
                                  [lidar_se3_prev[_i, -1], current_box[_i]]) for _i in range(3)], axis=1)
    
    # interpolate heading
    prev_quaternion = Quaternion(matrix=lidar_se3_prev[:3, :3])
    current_quaternion = Quaternion(axis=[0, 0, 1], angle=current_box[6])
    headings = np.array([_qua2yaw(Quaternion.slerp(prev_quaternion, current_quaternion, ts)) 
                         for ts in timesteps])

    # assemble output
    dxdydz = np.tile(current_box[3: 6].reshape(1, -1), (num_boxes_total, 1))
    interp_boxes = np.concatenate([centers, 
                                  dxdydz, 
                                  headings.reshape(-1, 1),
                                  np.arange(num_boxes_total).reshape(-1, 1),  # sweep_idx
                                  np.zeros((num_boxes_total, 1)) + instance_idx],  # inst_idx
                                  axis=1)

    return interp_boxes


def get_pseudo_sweeps_of_1lidar(nusc: NuScenes, 
                                sample_data_token: str, 
                                num_historical_sweeps: int = 10,
                                classes_of_interest: Set[str] = set(['car', 'pedestrian']),
                                points_in_boxes_by_gpu: bool = False,
                                return_nonempty_boxes: bool = True,
                                threshold_boxes_by_points: int = None) -> Dict[str, np.ndarray]:
    """
    Get "merged" point cloud of a LiDAR

    Returns:
        points: (N_pts, 5 + 2) - point-5 (x, y, z, intensity, time-lag), [sweep_idx, inst_idx] | in LiDAR
        gt_boxes: (N_box, 7), current & historical boxes | in LiDAR
        gt_name: (N_box,) - str | in LiDAR
        instance_tf: ()
    """
    # NOTE: num_sweeps = num_historical_sweeps + 1  (+ 1 is for the current sweep)
    sweep_indices = np.arange(num_historical_sweeps + 1)  # [0, 1, ..., 10]
    timelag_of_sweep_indices = 1.0 - np.linspace(0., 1., sweep_indices.shape[0])  # the more recent, the least time-lag

    lidar_info = get_points_and_boxes_of_1lidar(nusc, sample_data_token, 
                                                classes_of_interest, 
                                                points_in_boxes_by_gpu, 
                                                return_nonempty_boxes, threshold_boxes_by_points)
    
    points = lidar_info['points_in_lidar']  # (N_pts, 4) - x, y, z, intensity
    box_idx_of_points = lidar_info['box_idx_of_points']  # (N_pts,)

    # process background
    backgr = points[box_idx_of_points < 0]  # (N_bg, 4) - x, y, z, intensity
    backgr = np.pad(backgr, pad_width=[(0, 0), (0, 3)], constant_values=0.)  # (N_bg, 7) - point-5, [sweep_idx, inst_idx]
    backgr[:, 4] = timelag_of_sweep_indices[sweep_indices[-1]]  # NOTE: all background come from the most recent sweep
    backgr[:, -2] = float(sweep_indices[-1])  # sweep_idx
    backgr[:, -1] = -1.  # instance_idx

    # assemble gt_boxes & gt_names
    gt_boxes = lidar_info['boxes_in_lidar']  # (N_box, 7)
    gt_names = lidar_info['boxes_name']  # (N_box,)

    # simulate historical sweep by pushing points of each box backward
    sim_points, instances_tf = list(), list()
    for inst_idx, anno_token in enumerate(lidar_info['anno_tokens']):
        # extract foreground points of this box
        pts_of_box = points[box_idx_of_points == inst_idx]  # (N_pts, 4) - x, y, z, intensity | in LiDAR

        # map to box coord
        lidar_se3_box = np.eye(4)
        lidar_se3_box[:3, :3] = Quaternion(axis=[0, 0, 1], angle=gt_boxes[inst_idx, 6]).rotation_matrix
        lidar_se3_box[:3, -1] = gt_boxes[inst_idx, :3]
        apply_se3_(np.linalg.inv(lidar_se3_box), points_=pts_of_box)  # (N_pts, 4) - x, y, z, intensity | in box frame

        # get box's hitorical poses & compute points's coord accordingly
        histo_boxes = get_historical_boxes_1instance(nusc, sample_data_token, gt_boxes[inst_idx], anno_token, 
                                                     inst_idx, num_historical_sweeps)  # (N_sweep, 7 + 2) - box-7, sweep_idx, inst_idx
        
        cos, sin = np.cos(histo_boxes[:, 6]), np.sin(histo_boxes[:, 6])
        zs, os = np.zeros(histo_boxes.shape[0]), np.ones(histo_boxes.shape[0])
        batch_lidar_se3_histo_boxes = np.stack([
            cos,    -sin,       zs,     histo_boxes[:, 0],
            sin,     cos,       zs,     histo_boxes[:, 1],
            zs,      zs,        os,     histo_boxes[:, 2],
            zs,      zs,        zs,     os
        ], axis=1).reshape(-1, 4, 4)  # (N_sweep, 4, 4)
        
        batch_pts = np.tile(pts_of_box[np.newaxis], (histo_boxes.shape[0], 1, 1))  # (N_sweep, N_pts, 4) - x, y, z, intensity
        # rotate
        batch_pts[:, :, :3] = np.einsum('bpij, bpj -> bpi', 
                                        batch_lidar_se3_histo_boxes[:, np.newaxis, :3, :3], 
                                        batch_pts[:, :, :3])
        # translate
        batch_pts[:, :, :3] += batch_lidar_se3_histo_boxes[:, np.newaxis, :3, -1]

        # assemble new points
        batch_timelags = np.tile(timelag_of_sweep_indices.reshape(-1, 1, 1), (1, pts_of_box.shape[0], 1))  # (N_sweep, N_pts, 1)
        batch_sweep_idx = np.tile(sweep_indices.reshape(-1, 1, 1), (1, pts_of_box.shape[0], 1))  # (N_sweep, N_pts, 1)
        batch_inst_idx = np.zeros_like(batch_sweep_idx) + inst_idx  # (N_sweep, N_pts, 1)
        batch_pts = np.concatenate([batch_pts, batch_timelags, batch_sweep_idx, batch_inst_idx], 
                                   axis=-1)  # (N_sweep, N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
        
        # instance's correction tf
        _inst_tf = np.einsum('ij, bjk -> bik', 
                             batch_lidar_se3_histo_boxes[-1], 
                             np.linalg.inv(batch_lidar_se3_histo_boxes))  # (N_sweep, 4, 4)

        # store
        sim_points.append(batch_pts.reshape(-1, 7))
        instances_tf.append(_inst_tf[np.newaxis])

    # stick sim_points & background
    if len(sim_points) > 0:
        sim_points = np.concatenate(sim_points)
        points = np.concatenate([backgr, sim_points])
    else:
        points = backgr
    
    if len(instances_tf) > 0: 
        instances_tf = np.concatenate(instances_tf, axis=0)  # (N_inst, N_sweep, 4, 4)
    else:
        instances_tf = np.zeros((0, 11, 4, 4))


    out = {'points': points,  # (N_pts_tot, 5 + 2) - point-5, sweep_idx, inst_idx
           'gt_boxes': gt_boxes,  # (N_inst, 7)
           'gt_names': gt_names,  # (N_inst,)
           'instances_tf': instances_tf,
           }
    return out


def correction_numpy(points: np.ndarray, instances_tf: np.ndarray):
    """
    Args:
        points: (N, 5 + 2) - point-5, sweep_idx, instance_idx
        instances_tf: (N_inst, N_sweep, 3, 4)
    Returns:
        points_new_xyz: (N, 3)
    """
    # merge sweep_idx & instance_idx
    n_sweeps = instances_tf.shape[1]
    points_merge_idx = points[:, -1].astype(int) * n_sweeps + points[:, -2].astype(int)  # (N,)
    _tf = instances_tf.reshape((-1, instances_tf.shape[-2], 4))
    _tf = _tf[points_merge_idx]  # (N, 3, 4)

    # apply transformation
    points_new_xyz = np.matmul(_tf[:, :3, :3], points[:, :3, np.newaxis]) + _tf[:, :3, [-1]]
    return points_new_xyz.squeeze(axis=-1)
