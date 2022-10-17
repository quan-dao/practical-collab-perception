from pyquaternion import Quaternion
import numpy as np
import torch
from torch_scatter import scatter_mean
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


DYNAMIC_CLASSES = ('vehicle', 'human')  # to get ground truth for training CFG
CENTER_RADIUS = 2.


def show_pointcloud(xyz, boxes=None, pc_colors=None, fgr_mask=None, fgr_offset=None, boxes_color=None):
    """
    Visualize pointcloud & annotations
    Args:
        xyz (np.ndarray): (N, 3)
        boxes (list): list of boxes, each box is denoted by coordinates of its 8 vertices - np.ndarray (8, 3)
        pc_colors (np.ndarray): (N, 3) - r, g, b
    """
    def create_cube(vers, box_color):
        # vers: (8, 3)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]
        if box_color is None:
            colors = [[1, 0, 0] for _ in range(len(lines))]  # red
        else:
            colors = [box_color for _ in range(len(lines))]
        cube = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        cube.colors = o3d.utility.Vector3dVector(colors)
        return cube

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if fgr_mask is not None and pc_colors is None:
        pc_colors = np.zeros((xyz.shape[0], 3))
        pc_colors[fgr_mask, 0] = 1

    offset_lines = []
    if fgr_offset is not None:
        assert fgr_mask is not None
        if fgr_offset.shape[1] == 2:
            fgr_offset = np.concatenate((fgr_offset, np.zeros((fgr_offset.shape[0], 1))), axis=1)
        final = xyz[fgr_mask]
        initial = final - fgr_offset[fgr_mask]
        for fidx in range(final.shape[0]):
            ln = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack([initial[[fidx]], final[[fidx]]])),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            ln.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]))
            offset_lines.append(ln)

        offset_lines = offset_lines[::8]

    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    if boxes is not None:
        o3d_cubes = [create_cube(box, boxes_color[b_idx] if boxes_color is not None else None)
                     for b_idx, box in enumerate(boxes)]
        o3d.visualization.draw_geometries([pcd, *o3d_cubes, *offset_lines])
    else:
        o3d.visualization.draw_geometries([pcd, *offset_lines])


def tf(translation, rotation):
    """
    Build transformation matrix
    """
    if not isinstance(rotation, Quaternion):
        assert isinstance(rotation, list) or isinstance(rotation, np.ndarray), f"{type(rotation)} is not supported"
        rotation = Quaternion(rotation)
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rotation.rotation_matrix
    tf_mat[:3, 3] = translation
    return tf_mat


def apply_tf(tf: np.ndarray, points: np.ndarray):
    assert points.shape[1] >= 3
    assert tf.shape == (4, 4)
    points_homo = np.pad(points[:, :3], pad_width=[(0, 0), (0, 1)], mode='constant', constant_values=1.)
    points_homo = tf @ points_homo.T  # (4, N)
    points[:, :3] = points_homo[:3, :].T
    return points


def get_nuscenes_pointcloud(nusc: NuScenes, sample_data_token: str, center_radius=CENTER_RADIUS, ground_height=None,
                            return_foreground_mask=False, dyna_cls=DYNAMIC_CLASSES, box_tol=1e-2,
                            pc_time=None, return_gt_clusters=False) -> tuple:
    pcfile = nusc.get_sample_data_path(sample_data_token)
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)
    if pc_time is not None:
        assert isinstance(pc_time, float)
        pc = np.concatenate([pc, np.tile(np.array([[pc_time]]), (pc.shape[0], 1))], axis=1)
    # remove ego points
    too_close_mask = np.square(pc[:, :2]).sum(axis=1) < center_radius**2
    pc = pc[np.logical_not(too_close_mask)]
    # remove ground using lidar height
    if ground_height is not None:
        assert isinstance(ground_height, float)
        above_ground_mask = pc[:, 2] > ground_height
        pc = pc[above_ground_mask]

    if return_foreground_mask:
        boxes = nusc.get_boxes(sample_data_token)  # in global
        mask_foreground = -np.ones(pc.shape[0], dtype=int)
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, sample_data_token)
        for bidx, box in enumerate(boxes):
            if box.name.split('.')[0] not in dyna_cls:
                continue
            anno_rec = nusc.get('sample_annotation', box.token)
            if anno_rec['num_lidar_pts'] == 0:
                continue
            # map pc to box's local frame
            box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
            pts_wrt_box = apply_tf(box_from_glob @ glob_from_curr, pc[:, :3])
            mask_inside_box = np.all(
                np.abs(pts_wrt_box / np.array([box.wlh[1], box.wlh[0], box.wlh[2]])) < (0.5 + box_tol),
                axis=1
            )
            mask_foreground[mask_inside_box] = bidx
        if not return_gt_clusters:
            mask_foreground = mask_foreground > -1
        return pc, mask_foreground
    else:
        return pc, None


def get_nuscenes_pointcloud_in_target_frame(nusc: NuScenes, curr_sd_token: str, target_sd_token: str = None,
                                            center_radius=CENTER_RADIUS, ground_height: float = None,
                                            return_foreground_mask=False,
                                            pc_range: list = None, pc_time=None):
    """
    Get pointcloud and map its points to the target frame represented by target_sd_token

    Args:
        nusc:
        curr_sd_token: sample data from which pointcloud is obtained
        target_sd_token: sample data w.r.t pointcloud is expressed
        center_radius: to filter ego vehicle points
        ground_height: to filter points on & below ground
        return_foreground_mask:
        pc_range:
        pc_time: time lag w.r.t reference sample data
    Returns:
        (np.ndarray): (N, 3+C)
    """
    pc, mask_foreground = get_nuscenes_pointcloud(nusc, curr_sd_token, center_radius, ground_height,
                                                  return_foreground_mask, pc_time=pc_time)
    if target_sd_token is not None:
        glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
        glob_from_target = get_nuscenes_sensor_pose_in_global(nusc, target_sd_token)
        target_from_curr = np.linalg.inv(glob_from_target) @ glob_from_curr
        pc[:, :3] = apply_tf(target_from_curr, pc[:, :3])

    if pc_range is not None:
        assert len(pc_range) == 6
        if not isinstance(pc_range, np.ndarray):
            pc_range = np.array(pc_range)
        in_range_mask = np.all((pc[:, :3] >= pc_range[:3]) & (pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
        pc = pc[in_range_mask]
        if mask_foreground is not None:
            mask_foreground = mask_foreground[in_range_mask]

    return pc, mask_foreground


def get_nuscenes_sensor_pose_in_ego_vehicle(nusc: NuScenes, curr_sd_token: str):
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_cs_rec = nusc.get('calibrated_sensor', curr_rec['calibrated_sensor_token'])
    ego_from_curr = tf(curr_cs_rec['translation'], curr_cs_rec['rotation'])
    return ego_from_curr


def get_nuscenes_sensor_pose_in_global(nusc: NuScenes, curr_sd_token: str):
    ego_from_curr = get_nuscenes_sensor_pose_in_ego_vehicle(nusc, curr_sd_token)
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_ego_rec = nusc.get('ego_pose', curr_rec['ego_pose_token'])
    glob_from_ego = tf(curr_ego_rec['translation'], curr_ego_rec['rotation'])
    glob_from_curr = glob_from_ego @ ego_from_curr
    return glob_from_curr


def get_pointcloud_propagated_by_annotations(nusc, curr_sd_token: str, target_sd_token: str,
                                             dyna_cls=DYNAMIC_CLASSES, center_radius=CENTER_RADIUS,
                                             pc_range=None, ground_height=None,
                                             tol=1e-2):
    curr_pc = get_nuscenes_pointcloud(nusc, curr_sd_token, center_radius, ground_height)  # in curr_sd frame
    glob_from_curr = get_nuscenes_sensor_pose_in_global(nusc, curr_sd_token)
    curr_pc[:, :3] = apply_tf(glob_from_curr, curr_pc[:, :3])  # in global

    curr_boxes = nusc.get_boxes(curr_sd_token)  # in global
    # partition curr_pc
    curr_annos = {}  # instance_token: points inside
    for box in curr_boxes:
        if box.name.split('.')[0] not in dyna_cls:
            continue
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['num_lidar_pts'] == 0:
            continue
        box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
        pts_wrt_box = apply_tf(box_from_glob, curr_pc[:, :3])
        in_box_mask = np.all(np.abs(pts_wrt_box / np.array([box.wlh[1], box.wlh[0], box.wlh[2]])) < (0.5 + tol), axis=1)
        # store points inside the box
        curr_annos[anno_rec['instance_token']] = pts_wrt_box[in_box_mask]  # in box
        # remove points inside the box from pointcloud
        curr_pc = curr_pc[~in_box_mask]  # in global

    # propagate to target_sd frame
    glob_from_target = get_nuscenes_sensor_pose_in_global(nusc, target_sd_token)
    dyn_points_in_target = []
    for box in nusc.get_boxes(target_sd_token):
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['instance_token'] in curr_annos:
            box_from_glob = np.linalg.inv(tf(box.center, box.orientation))
            box_from_target = box_from_glob @ glob_from_target
            target_from_box = np.linalg.inv(box_from_target)
            dyn_points_in_target.append(apply_tf(target_from_box, curr_annos[anno_rec['instance_token']]))

    curr_pc[:, :3] = apply_tf(np.linalg.inv(glob_from_target), curr_pc[:, :3])
    prop_pc = np.vstack((curr_pc[:, :3], *dyn_points_in_target))
    if pc_range is not None:
        assert len(pc_range) == 6
        if not isinstance(pc_range, np.ndarray):
            pc_range = np.array(pc_range)
        in_range_mask = np.all((prop_pc[:, :3] >= pc_range[:3]) & (prop_pc[:, :3] < pc_range[3:] - 1e-3), axis=1)
        return prop_pc[in_range_mask]
    else:
        return prop_pc


def get_boxes_4viz(nusc, sample_data_token):
    boxes = nusc.get_boxes(sample_data_token)  # global
    # a box's corners in its local frame
    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    corners = np.stack([xs, ys, zs], axis=1)
    out = []
    curr_from_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample_data_token))
    for box in boxes:
        if box.name.split('.')[0] not in DYNAMIC_CLASSES:
            continue
        anno_rec = nusc.get('sample_annotation', box.token)
        if anno_rec['num_lidar_pts'] == 0:
            continue
        glob_from_box = tf(box.center, box.orientation)
        curr_from_box = curr_from_glob @ glob_from_box
        this_box_corners = corners * np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
        out.append(apply_tf(curr_from_box, this_box_corners))
    return out


def get_sweeps_token(nusc: NuScenes, curr_sd_token: str, n_sweeps: int, return_time_lag=True) -> list:
    ref_sd_rec = nusc.get('sample_data', curr_sd_token)
    ref_time = ref_sd_rec['timestamp'] * 1e-6
    sd_tokens_times = []
    for s_idx in range(n_sweeps):
        curr_sd = nusc.get('sample_data', curr_sd_token)
        sd_tokens_times.append((curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6, n_sweeps - 1 - s_idx))
        # s_idx: the higher, the closer to the current
        # move to previous
        if curr_sd['prev'] != '':
            curr_sd_token = curr_sd['prev']

    # organize from PAST to PRESENCE
    sd_tokens_times.reverse()

    if return_time_lag:
        return sd_tokens_times
    else:
        return [token for token, _ in sd_tokens_times]


def check_list_to_numpy(ls):
    return np.array(ls) if isinstance(ls, list) else ls


def project_pointcloud_to_image(nusc: NuScenes, sweep_token: str, cam_token: str, sweep: np.ndarray):
    """
    Args:
        nusc:
        sweep_token:
        cam_token:
        sweep: (N, 3+C) in lidar frame
    Returns:
        sweep_pix_coord: (N, 2) pixel coordinate
        sweep_in_img: (N,) - bool | True for points in image
    """
    # map sweep to camera frame
    glob_from_lidar = get_nuscenes_sensor_pose_in_global(nusc, sweep_token)
    glob_from_cam = get_nuscenes_sensor_pose_in_global(nusc, cam_token)
    cam_from_lidar = np.linalg.inv(glob_from_cam) @ glob_from_lidar
    sweep_cam_coord = apply_tf(cam_from_lidar, sweep[:, :3])  # (N, 3)
    sweep_in_img = sweep_cam_coord[:, 2] > 1.  # depth > 1.

    # project
    cam_rec = nusc.get('sample_data', cam_token)
    cam_cs_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
    sweep_pix_coord = view_points(sweep_cam_coord.T, np.array(cam_cs_rec['camera_intrinsic']), normalize=True)  # (3, N)
    sweep_pix_coord = sweep_pix_coord[:2].T  # (N, 2)

    # in side img
    cam_size = np.array([cam_rec['width'], cam_rec['height']])
    sweep_in_img = np.all((sweep_pix_coord >= 1.) & (sweep_pix_coord < (cam_size - 1)), axis=1) & sweep_in_img

    return sweep_pix_coord, sweep_in_img


def get_target_sample_token(nusc, scene_idx: int, target_sample_idx: int):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    for _ in range(target_sample_idx - 1):
        sample_rec = nusc.get('sample', sample_token)
        sample_token = sample_rec['next']

    return sample_token


def compute_bev_coord(pts: np.ndarray, pc_range: np.ndarray, bev_pix_size: float, pts_feat=None):
    """
    Compute occupancy in BEV -> output has less element than input
    Returns:
        occ_2d (np.ndarray): (N_occ, 2) - pix_x, pix_y
        occ_feat (np.ndarray): (N_occ, N_feat)
    """
    # find in range
    mask_in_range = np.all((pts[:, :3] >= pc_range[:3]) & (pts[:, :3] < (pc_range[3:] - 1e-3)), axis=1)

    bev_size = np.floor((pc_range[3: 5] - pc_range[0: 2]) / bev_pix_size).astype(int)  # [width, height]

    pts_pixel_coord = np.floor((pts[mask_in_range, :2] - pc_range[:2]) / bev_pix_size).astype(int)

    pts_1d = pts_pixel_coord[:, 1] * bev_size[0] + pts_pixel_coord[:, 0]
    occ_1d, inv_indices, counts = np.unique(pts_1d, return_inverse=True, return_counts=True)

    if pts_feat is not None:
        assert len(pts_feat.shape) == 2, f"pts_feat must have shape of (N_pts, N_feat)"
        pts_feat = pts_feat[mask_in_range]
        pix_feat = np.zeros((occ_1d.shape[0], pts_feat.shape[1]))
        np.add.at(pix_feat, inv_indices, pts_feat)
        pix_feat /= counts.reshape(-1, 1)
    else:
        pix_feat = None

    occ_2d = np.stack((occ_1d % bev_size[0], occ_1d // bev_size[0]), axis=1)  # (N_occ, 2)
    return occ_2d, pix_feat


def pad_points_with_scalar(points: np.ndarray, scalar: float) -> np.ndarray:
    padded_val = np.tile([[scalar]], (points.shape[0], 1))
    return np.concatenate([points, padded_val], axis=1)


def compute_bev_coord_torch(points: torch.Tensor, pc_range: torch.Tensor, pix_size: float, pts_feat=None):
    """
    Args:
        points: (N, 1 + 3 + C) - batch_idx, X, Y, Z, C features
        pc_range: (6) - [x_min, y_min, z_min, x_max, y_max, z_max]
        pix_size:
        pts_feat: (N, f)
    """
    mask_in_range = torch.all((points[:, 1: 4] >= pc_range[:3]) & (points[:, 1: 4] < (pc_range[3:] - 1e-3)), dim=1)
    bev_size = torch.floor((pc_range[3: 5] - pc_range[0: 2]) / pix_size).int()  # (s_x, s_y) (i.e. width, height)
    pts_pix_coord = torch.floor((points[:, 1: 3] - pc_range[:2]) / pix_size).int()  # (N, 2)- pix_x, pix_y
    # convert to 1D coord
    pts_1d = points[mask_in_range, 0] * (bev_size[0] * bev_size[1]) + \
             pts_pix_coord[mask_in_range, 1] * bev_size[0] + \
             pts_pix_coord[mask_in_range, 0]
    unq_pts_1d, inv_indices = torch.unique(pts_1d, return_inverse=True, sorted=False)

    # convert 1D coord back to 2D coord
    unq_pts_2d = torch.stack([
        unq_pts_1d // (bev_size[0] * bev_size[1]),  # batch_idx
        unq_pts_1d % bev_size[0],  # pix_x
        (unq_pts_1d % (bev_size[0] * bev_size[1])) // bev_size[0]  # pix_y
    ], dim=1).long()

    # compute feat of unq_pts_2d (if applicable)
    if pts_feat is not None:
        assert len(pts_feat.shape) == 2
        feat_unq_pts_2d = scatter_mean(pts_feat[mask_in_range], inv_indices, dim=0)
    else:
        feat_unq_pts_2d = None
    return unq_pts_2d, feat_unq_pts_2d
