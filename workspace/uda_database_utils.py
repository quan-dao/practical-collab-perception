import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
from typing import List, Dict, FrozenSet
import pickle
from nuscenes import NuScenes
from lyft_dataset_sdk.lyftdataset import LyftDataset
from tqdm import tqdm
from pprint import pprint

from workspace.nuscenes_temporal_utils import *
from workspace.downsample_utils import compute_angles, beam_label_gpu


def get_points_on_trajectory(nusc: NuScenes, instance_token: str, compute_points_beam_idx: bool = False) -> List[Dict]:
    """
    out = [
        {'sample_token': '',
         'points': np.array([])},

        {'sample_token': '',
         'points': np.array([])}
    ]
    """
    instance = nusc.get('instance', instance_token)
    anno_tk = instance['first_annotation_token']
    out = list()
    while anno_tk != '':
        sample_anno = nusc.get('sample_annotation', anno_tk)
        sample = nusc.get('sample', sample_anno['sample_token'])
        lidar_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample['data']['LIDAR_TOP']))

        # box
        box_in_glob = np.array([
            *sample_anno['translation'],
            sample_anno['size'][1], sample_anno['size'][0], sample_anno['size'][2],
            quaternion_to_yaw(Quaternion(sample_anno['rotation']))
        ])  # (7,) - [x, y, z, dx, dy, dz, yaw]

        box_in_lidar = np.copy(box_in_glob).reshape(1, -1)
        apply_se3_(lidar_se3_glob, boxes_=box_in_lidar)
        
        # ---------------------------------
        # points
        pcd = get_one_pointcloud(nusc, sample['data']['LIDAR_TOP'])  # (N, 4) in LiDAR

        if compute_points_beam_idx:
            pts_theta, pts_phi = compute_angles(pcd)
            pts_label, centroids = beam_label_gpu(pts_theta, beam=40, use_cuda=False)  # Lyft: 40 | (N_pts, ), (N_beams,)
            beam_ids = np.argsort(centroids)  # (N_beam,)
            beams2pts = pts_label[:, np.newaxis].astype(int) == beam_ids[np.newaxis, :]  # (N_pts, N_beam)
            pts_beam_idx = np.sum(beams2pts.astype(int) * np.arange(beam_ids.shape[0]), axis=1)  # (N_pts,)
        
        # get points in box
        lidar_se3_box = lidar_se3_glob @ make_se3(box_in_glob[:3], yaw=box_in_glob[6])
        apply_se3_(np.linalg.inv(lidar_se3_box), points_=pcd)
        mask_in_box = np.all(np.abs(pcd[:, :3] / box_in_glob[3: 6]) < (0.5 + 2.5e-2), axis=1)  # (N,)
        
        # store points & box
        _out = {
            'sample_tk': sample_anno['sample_token'],
            'points': pcd[mask_in_box],  # (N, 4) - x, y, z, intensity | in box
            'box_in_glob': box_in_glob,  # (7,) - [x, y, z, dx, dy, dz, yaw]
            'box_in_lidar': box_in_lidar.reshape(-1),  # (7,) - [x, y, z, dx, dy, dz, yaw]
        }
        if compute_points_beam_idx:
            _out['points_beam_idx'] =  pts_beam_idx[mask_in_box],  # (N,)
        
        out.append(_out)

        # move to next
        anno_tk = sample_anno['next']
    
    return out


def process_1scene(nusc: NuScenes, 
                   scene_token: str, 
                   database_root: Path, 
                   src_domain_name: str = 'lyft',
                   classes_of_interest=set(('car', 'pedestrian', 'bicycle'))) -> Dict[str, int]:
    scene = nusc.get('scene', scene_token)
    seen_instances_token = set()
    sample_tk = scene['first_sample_token']

    num_trajs = dict()
    for cls_ in classes_of_interest:
        num_trajs[cls_] = 0

    while sample_tk != '':
        sample = nusc.get('sample', sample_tk)
        for anno_tk in sample['anns']:
            sample_anno = nusc.get('sample_annotation', anno_tk)
            if src_domain_name == 'lyft':
                det_name = sample_anno['category_name']
            else:
                raise NotImplementedError
            
            # filter by det name
            if det_name not in classes_of_interest:
                continue
            
            # filter by instance_token
            if sample_anno['instance_token'] in seen_instances_token:
                continue

            # get points on this traj & save to disk
            seen_instances_token.add(sample_anno['instance_token'])
            traj_info = get_points_on_trajectory(nusc, sample_anno['instance_token'])
            with open(database_root / f"{det_name}" / f"{sample_anno['instance_token']}.pkl", 'wb') as f:
                pickle.dump(traj_info, f)
            
            num_trajs[det_name] += 1

        # move to next
        sample_tk = sample['next']

    return num_trajs


def create_database(data_root: Path, classes_of_interest: FrozenSet, src_domain_name: str = 'lyft') -> None:
    """
    Args:
        data_root: for example Path('/home/user/dataset/lyft')

    """
    if src_domain_name != 'lyft':
        raise NotImplementedError
    
    lyft = LyftDataset(data_path=data_root / 'trainval', 
                       json_path=data_root / 'trainval' / 'data', 
                       verbose=True)
    
    database_root = data_root / 'gt_boxes_database_lyft'
    if not database_root.is_dir():
        database_root.mkdir(parents=True, exist_ok=True)

    for cls_ in classes_of_interest:
        cls_dir = database_root / cls_
        if not cls_dir.is_dir():
            cls_dir.mkdir(parents=True, exist_ok=True)
    
    num_trajs = dict()
    for cls_ in classes_of_interest:
        num_trajs[cls_] = 0

    for scene in tqdm(lyft.scene, total=len(lyft.scene)):
        scene_num_trajs = process_1scene(lyft, scene['token'], database_root, classes_of_interest=classes_of_interest)
        
        # update number of traj per cls
        for k, v in scene_num_trajs.items():
            num_trajs[k] += v
    
    print(f'finish making gt database in src domain {src_domain_name}')
    print('number of traj per class:')
    pprint(num_trajs)

    return


def load_1traj(path_traj: Path, 
               traj_index: int,
               num_sweeps_in_target: int, 
               src_frequency: float, 
               pc_range: np.ndarray,
               noise_rotation: float,
               target_frequency: float = 20., 
               beam_ratio: int = None):
    
    with open(path_traj, 'rb') as f:
        traj_info = pickle.load(f)
    traj_len = len(traj_info)
    
    num_sweeps_in_src = int(np.ceil((num_sweeps_in_target / target_frequency) * src_frequency))

    start_idx = np.random.randint(low=0, high=max(traj_len - num_sweeps_in_src, 0))
    end_idx = min(start_idx + num_sweeps_in_src, traj_len)
    
    points, boxes, mask_keep_points = list(), list(), list()
    for idx in range(start_idx, end_idx):
        info = traj_info[idx]
        
        box_in_glob = info['box_in_glob']  # in glob
        
        # ----
        # points | in box -> in glob
        pts = info['points']  # in box
        if pts.shape[0] == 0:
            continue
        glob_se3_box = make_se3(box_in_glob[:3], yaw=box_in_glob[6])
        apply_se3_(glob_se3_box, points_=pts)

        # ---
        # downsample based on points' beam idx
        if 'points_beam_idx' in info and beam_ratio is not None:
            points_beam_idx = info['points_beam_idx']
            mask_keep = (points_beam_idx % beam_ratio) == 0  # (N,)
        else:
            mask_keep = np.ones(pts.shape[0], dtype=bool)

        # add timelag, time-idx, instance_idx to pts
        pts = np.pad(pts, pad_width=[(0, 0), (0, 3)], constant_values=-1)
        pts[:, -3] = float(idx - start_idx) / src_frequency   # time-lag
        pts[:, -2] = np.floor(pts[:, -3] * target_frequency)  # sweep idx
        pts[:, -1] = traj_index

        # add time-idx, instance_idx to box
        box_in_glob = np.pad(box_in_glob, pad_width=[(0, 2)], constant_values=0)
        box_in_glob[-2] = pts[-1, -2]  # take sweep_idx of the last point
        box_in_glob[-1] = traj_index

        points.append(pts)
        boxes.append(box_in_glob.reshape(1, -1))
        mask_keep_points.append(mask_keep)

    if len(points) == 0:
        return np.zeros((0, 7)), np.zeros((0, 9)), np.zeros(0)

    points = np.concatenate(points, axis=0)  # (N_pts, 5 + 2) - x, y, z, intensity, timelag, [sweep_idx, inst_idx] | in glob
    boxes = np.concatenate(boxes, axis=0)  # (N_box, 7 + 2) - x, y, z, dx, dy, dz, yaw, [sweep_idx, inst_idx] | in glob 
    mask_keep_points = np.concatenate(mask_keep_points)

    glob_se3_last_box = make_se3(boxes[-1, :3], yaw=boxes[-1, 6])
    # map points and boxes to last_box
    apply_se3_(np.linalg.inv(glob_se3_last_box), points_=points, boxes_=boxes)

    # add random local rotation here to points, last box, & instance_tf -> check augmentor_utils.py/global_rotation
    noise_rotation = np.pi / 3.
    cos_r, sin_r = np.cos(noise_rotation), np.sin(noise_rotation)
    tf = np.array([
        [cos_r,     -sin_r,     0.,     0.],
        [sin_r,     cos_r,      0.,     0.],
        [0.,        0.,         1.,     0.],
        [0.,        0.,         0.,     1.],
    ])
    apply_se3_(tf, points_=points, boxes_=boxes)

    last_box_in_lidar = traj_info[-1]['box_in_lidar']
    last_box_in_lidar[6] += noise_rotation

    # map points and boxes from last_box to lidar
    lidar_se3_last_box = make_se3(last_box_in_lidar[:3], yaw=last_box_in_lidar[6])
    apply_se3_(lidar_se3_last_box, points_=points, boxes_=boxes)

    last_box_in_range = np.logical_and(last_box_in_lidar[:2] > pc_range[:2],
                                       last_box_in_lidar[:2] < pc_range[3: 5]).all()
    if not last_box_in_range:
        transl = (pc_range[3] - np.linalg.norm(last_box_in_lidar[:2])) * \
            np.array([last_box_in_lidar[0], last_box_in_lidar[1], 0.]) / np.linalg.norm(last_box_in_lidar[:2])
        tf = np.eye(4)
        tf[:3, -1] = transl
        apply_se3_(tf, points_=points, boxes_=boxes)

    return points, boxes, mask_keep_points
