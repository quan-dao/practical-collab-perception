import numpy as np
import numpy.linalg as LA
from pathlib import Path
from typing import List, Dict, Tuple, Union
from av2.utils.io import read_feather
from av2.geometry.geometry import quat_to_mat, mat_to_quat
from av2.map.map_api import ArgoverseStaticMap
import pandas as pd


def get_timestamp_from_feather_file(filename: Path, return_int: bool = False) -> int:
    timestamp_ns = filename.parts[-1].split('.')[0]
    if return_int:
        return int(timestamp_ns)
    return timestamp_ns


def get_log_name_from_sensor_file(filename: Path) -> str:
    sensor_idx = filename.parts.index('sensors')
    log_name = filename.parts[sensor_idx - 1]
    return log_name


def apply_SE3(tf: np.ndarray, points_: np.ndarray, inplace: bool = True) -> None:
    if not inplace:
        return points_[:, :3] @ tf[:3, :3].T + tf[:3, -1]
    points_[:, :3] = points_[:, :3] @ tf[:3, :3].T + tf[:3, -1]


def yaw_from_rotz(rot_mat: np.ndarray) -> np.ndarray:    
    """
    Args:
        rot_mat: (N, 4, 4) - batch of rotation matrices representing rotation around z-axis
    
    Returns:
        (N,): yaw angle
    """
    return np.arctan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])


class AV2Parser(object):
    av2_cls = ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'BICYCLE', 'LARGE_VEHICLE',
               'WHEELED_DEVICE', 'BUS', 'BOX_TRUCK', 'SIGN', 'TRUCK', 'MOTORCYCLE', 'BICYCLIST', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'MOTORCYCLIST',
               'DOG', 'SCHOOL_BUS', 'WHEELED_RIDER', 'STROLLER', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_SIGN', 'WHEELCHAIR',
               'RAILED_VEHICLE', 'OFFICIAL_SIGNALER', 'TRAFFIC_LIGHT_TRAILER', 'ANIMAL']
    def __init__(self, log_info: Dict, current_lidar_timestamp_ns: int, num_sweeps: int, sweep_stride: int, detection_cls: List[str]):
        """
        Args:
            log_info:
                {
                    'lidar_timestamp_ns': lidar_timestamp_ns.tolist(), 
                    'anno_file': log_dir / 'annotations.feather',
                    'lidar_dir': lidar_dir,
                    'map_dir': log_dir / 'map',
                    'log_dir': log_dir
                }
            current_lidar_timestamp_ns:
            num_sweeps: number of lidar in a sweep
            sweep_stride: gap between 2 lidar in a sweep
        """
        self.log_info = log_info
        self.current_ts_ns = current_lidar_timestamp_ns
        self.num_sweeps = num_sweeps
        self.sweep_stride = sweep_stride
        self.detection_cls = np.array(detection_cls)
    
    def get_files_of_lidar_sequence(self) -> List[str]:
        """
        Given a lidar file, get N previous lidar files for point concatenation

        Returns:
            (N): str - path to N previous lidar files, order from past to present
        """
        current_idx_in_log = self.log_info['lidar_timestamp_ns'].index(self.current_ts_ns)

        seq_idx_in_log = []
        for i in range(self.num_sweeps):
            idx_in_log = current_idx_in_log - i * self.sweep_stride 
            if idx_in_log < 0:
                seq_idx_in_log.append(0)
                break
            else:
                seq_idx_in_log.append(idx_in_log)
        seq_idx_in_log.reverse()

        sequence_files = [self.log_info['lidar_dir'] / f"{self.log_info['lidar_timestamp_ns'][idx]}.feather" for idx in seq_idx_in_log]
        return sequence_files
    
    def parse_av_annotation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            ego_SE3_annos: (N, 4, 4) - annos' pose in egovehicle frame
            annos_size: (N, 3) - dx, dy, dz
            annos_id: (N,) - str | unique string representing each object traj
            annos_timestamp_ns: (N,)
            annos_category: (N,) - str (e.g., BUS, REGULAR_VEHICLE)
        """
        annos_df = read_feather(self.log_info['anno_file'])
        annos = annos_df.loc[:, ['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 
                                 'track_uuid', 'timestamp_ns']].to_numpy()  # annotations of the entire log
        
        # annos' category (to make gt_names)
        annos_category = annos_df.loc[:, 'category'].to_numpy()

        # remove boxes by names to save time establishing points to boxes corr
        mask_detected_categoy = np.any(annos_category.reshape(-1, 1) == self.detection_cls.reshape(1, -1), axis=1)
        annos, annos_category = annos[mask_detected_categoy], annos_category[mask_detected_categoy]

        if annos.shape[0] == 0:
            return np.zeros((0, 4, 4)), np.zeros((0, 3)), np.array([]), np.array([]), np.array([])
        
        # construct annos' pose in egovehicle frame
        annos_rot_mat = quat_to_mat(annos[:, 6: 10])  # (N, 3, 3)
        ego_SE3_annos = np.pad(annos_rot_mat, pad_width=[(0, 0), (0, 1), (0, 1)], constant_values=0.0)  # (N, 4, 4)
        ego_SE3_annos[:, :3, -1] = annos[:, :3]
        ego_SE3_annos[:, 3, -1] = 1.0
        
        # annos' size
        annos_size = annos[:, 3: 6]  # (N, 3)

        # annos' track_uuid
        annos_id = annos[:, -2]  # (N,)

        # annos' timestamp_ns
        annos_timestamp_ns = annos[:, -1].astype(int)

        return ego_SE3_annos, annos_size, annos_id, annos_timestamp_ns, annos_category
    
    def get_sweep_info(self) -> Dict:
        """
        Get info of sweep id by the current_lidar_timestamp
        """
        out = dict()
        out['sweep_files'] = self.get_files_of_lidar_sequence()
        out['ego_SE3_annos'], out['annos_size'], out['annos_id'], out['annos_timestamp_ns'], out['annos_category'] = self.parse_av_annotation()
        return out
    
    @staticmethod
    def read_av_lidar_file(lidar_file: str, timestamp_ns: int, use_offset_ns: bool = False) -> np.ndarray:
        """
        Args:
            lidar_file: 
            timestamp_ns: timestamp of the lidar sweep == timestamp at the _beginning_ of the sweep
            use_offset_ns: if True, individual points' timestamp = timestamp_ns + its offset_ns, else = timetstamp_ns
        
        Returns:
            (N, 5) - x, y, z, intensity, time_s
        """
        lidar_df = read_feather(lidar_file)
        sweep = lidar_df.loc[:, ['x', 'y', 'z', 'intensity', 'offset_ns']].to_numpy()
        # normalize intensity
        sweep[:, 3] /= 255.0

        if use_offset_ns:
            sweep[:, -1] += timestamp_ns
        else:
            sweep[:, -1] = timestamp_ns
        # convert timestamp from nanosecond to second
        sweep[:, -1] *= 1e-9

        return sweep.astype(np.float32)
    

class AV2MapHelper(object):
    def __init__(self, map_dir: Path, city_SE3_src: np.ndarray):
        self.avm = ArgoverseStaticMap.from_map_dir(map_dir.resolve(), build_raster=True)
        self.city_SE3_src = city_SE3_src
        self.src_SE3_city = LA.inv(self.city_SE3_src)
    
    def find_points_on_ground(self, points: np.ndarray) -> np.ndarray:
        """
        Given a set of points, find which poins are on the ground

        Args:
            points: (N, 3 + C) - x, y, z, C features (xyz in "src" frame)

        Returns:
            mask_on_ground: (N,) - 1 if a point on the ground, 0 otherwise
        """
        # map points from src to city frame
        points_city = apply_SE3(self.city_SE3_src, points, inplace=False)  # (N, 3) - x, y, z in city frame
        
        # query map for points' drivable are status
        mask_on_ground = self.avm.get_ground_points_boolean(points_city)

        return mask_on_ground.astype(float)
    
    def compensate_ground_height(self, points: np.ndarray, in_place: bool = True) -> Union[None, np.ndarray]:
        """
        Pull points to the ground by subtracting their z-coord for ground height at the same location

        Args:
            points: (N, 3 + C) - x, y, z, C features (xyz in "src" frame)
            in_place: overwrite points' xyz with compensated xyz (in src frame) if True, else return compensated xyz (in src frame)
        
        Returns:
            new_xyz: (N, 3) in src frame
        """
        points_city = apply_SE3(self.city_SE3_src, points, inplace=False)  # (N, 3) - x, y, z in city frame

        # query map for ground height
        points_z = self.avm.raster_ground_height_layer.get_ground_height_at_xy(points_city)  # (N,) - ground height in city frame

        # subtract points_city's z-coord for their ground height
        points_city[:, 2] -= points_z

        # map grounded points_city back to src frame
        new_xyz = apply_SE3(self.src_SE3_city, points_city, inplace=False)

        if in_place:
            points[:, :3] = new_xyz
        else:
            return new_xyz

    def uncompensate_ground_height(self, points: np.ndarray, in_place: bool = True) -> Union[None, np.ndarray]:
        """
        Inverse function of compensate_ground_height

        Args:
            points: (N, 3 + C) - x, y, z, C features (xyz in "src" frame)
            in_place: overwrite points' xyz with compensated xyz (in src frame) if True, else return compensated xyz (in src frame)
        
        Returns:
            new_xyz: (N, 3) in src frame
        """
        points_city = apply_SE3(self.city_SE3_src, points, inplace=False)  # (N, 3) - x, y, z in city frame
        
        # query map for ground height
        points_z = self.avm.raster_ground_height_layer.get_ground_height_at_xy(points_city)  # (N,) - ground height in city frame

        # recover points' actual z-coordinate in city frame
        points_city[:, 2] += points_z

        # map points_city back to src frame
        points_src = apply_SE3(self.src_SE3_city, points_city, in_place=False)

        if in_place:
            points[:, :3] = points_src
        else:
            return points_src        

    def find_points_on_drivable_area(self, points: np.ndarray) -> np.ndarray:
        """
        Given a set of points, find which poins are on the drivable area

        Args:
            points: (N, 3 + C) - x, y, z, C features (xyz in "src" frame)

        Returns:
            mask_on_drivable_area: (N,) - 1 if a point on the drivable_area, 0 otherwise
        """
        # map points from src to city frame
        points_city = apply_SE3(self.city_SE3_src, points, inplace=False)  # (N, 3) - x, y, z in city frame
        
        # query map for points' drivable are status
        mask_on_drivable_area = self.avm.raster_drivable_area_layer.get_raster_values_at_coords(points_city, fill_value=0.0)

        return mask_on_drivable_area.astype(float)
    

def transform_det_annos_to_av2_feather(det_annos: List[Dict], detection_cls: np.ndarray) -> pd.DataFrame: 
    """
    Args:
        det_annos: {
            frame_id: LiDAR file
            metadata: {
                log_name (str):
                lidar_timestamp_ns (int): to be used as the timestamp of a detection
                num_sweeps (int):
            }
            boxes_lidar (np.ndarray): (N, 7) - x, y, z, dx, dy, dz, yaw
            score (np.ndarray): (N,)
            pred_labels (np.ndarray): (N,) - int in [1, N_cls]
        }
        detection_cls: (str) to be used for converting pred_labels to category
    
    Returns:
        detection_df: pandas dataframe according to AV2 format
    """
    # init output
    out = dict()
    fields = ('tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'score', 'log_id', 'timestamp_ns', 'category')
    for f in fields:
        out[f] = list()

    # populate output with info from det_annos
    for anno in det_annos:
        # boxes' translation
        for coord_idx, coord in enumerate(fields[:3]):
            out[coord].append(anno['boxes_lidar'][:, coord_idx])
        
        # boxes' size
        for dim_idx, dim in enumerate(fields[3: 6]):
            out[dim].append(anno['boxes_lidar'][:, 3 + dim_idx])

        # orientation: convert yaw to quaternion
        num_boxes = anno['boxes_lidar'].shape[0]
        cos, sin = np.cos(anno['boxes_lidar'][:, 6]), np.sin(anno['boxes_lidar'][:, 6])
        zeros, ones = np.zeros(num_boxes), np.ones(num_boxes)
        mat = np.stack([cos,    -sin,   zeros,
                        sin,    cos,    zeros,
                        zeros,  zeros,  ones], axis=1).reshape(num_boxes, 3, 3)
        quat = mat_to_quat(mat)  # (N, 4)
        for q_idx, q in enumerate(('qw', 'qx', 'qy', 'qz')):
            out[q].append(quat[:, q_idx])

        # score
        out['score'].append(anno['score'])

        # log_id
        out['log_id'].append(np.tile(np.array([anno['metadata']['log_name']]), num_boxes))

        # timestamp_ns
        out['timestamp_ns'].append(np.tile(np.array([str(anno['metadata']['lidar_timestamp_ns'])]), num_boxes))

        # category
        out['category'].append(detection_cls[anno['pred_labels'].astype(int) - 1])

    return pd.DataFrame(data=out)

