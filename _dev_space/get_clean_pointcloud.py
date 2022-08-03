import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from typing import List, Tuple
import matplotlib.pyplot as plt
import open3d as o3d


def show_pointcloud(xyz, boxes=None, pc_colors=None):
    """
    Visualize pointcloud & annotations
    Args:
        xyz (np.ndarray): (N, 3)
        boxes (list): list of boxes, each box is denoted by coordinates of its 8 vertices - np.ndarray (8, 3)
        pc_colors (np.ndarray): (N, 3) - r, g, b
    """
    def create_cube(vers):
        # vers: (8, 3)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]
        colors = [[1, 0, 0] for i in range(len(lines))]  # red
        cube = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        cube.colors = o3d.utility.Vector3dVector(colors)
        return cube

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    if boxes is not None:
        o3d_cubes = [create_cube(box) for box in boxes]
        o3d.visualization.draw_geometries([pcd, *o3d_cubes])
    else:
        o3d.visualization.draw_geometries([pcd])


def get_lidar_and_sweeps_tokens(nusc: NuScenes, sample_token: str, max_sweeps=10) -> List[str]:
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['LIDAR_TOP']
    curr_sd_rec = nusc.get('sample_data', sample_data_token)
    lidar_and_sweeps = [sample_data_token]  # sample's lidar first, sweeps follow
    while len(lidar_and_sweeps) < max_sweeps:
        if curr_sd_rec['prev'] == '':
            lidar_and_sweeps.append(curr_sd_rec['token'])
        else:
            curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
            lidar_and_sweeps.append(curr_sd_rec['token'])
    # check if the last sweep is keyframe, remove it if True
    last_sd_rec = nusc.get('sample_data', lidar_and_sweeps[-1])
    if last_sd_rec['is_key_frame']:
        lidar_and_sweeps.pop()
    return lidar_and_sweeps


def get_pointclouds_sequence_token(nusc: NuScenes, sample_token: str, num_samples=5) -> List[str]:
    cur_lips = get_lidar_and_sweeps_tokens(nusc, sample_token)
    out = cur_lips
    cur_sample = nusc.get('sample', sample_token)
    for _ in range(num_samples - 1):
        if cur_sample['prev'] == '':
            cur_lips = get_lidar_and_sweeps_tokens(nusc, cur_sample['token'])
        else:
            cur_sample = nusc.get('sample', cur_sample['prev'])
            cur_lips = get_lidar_and_sweeps_tokens(nusc, cur_sample['token'])
        out.extend(cur_lips)
    out.reverse()
    return out


def apply_transform_to_points(tf: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    assert tf.shape == (4, 4)
    assert xyz.shape == (xyz.shape[0], 3)
    xyz_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)  # (N, 4)
    xyz_homo = tf @ xyz_homo.T  # (4, N)
    return xyz_homo[:3, :].T


def partition_pointcloud(pc_in_glo: np.ndarray, box: Box, tol=1e-2) -> Tuple:
    # map pointcloud to box
    glo_from_box = transform_matrix(box.center, box.orientation)
    box_from_glo = np.linalg.inv(glo_from_box)
    xyz_in_box = apply_transform_to_points(box_from_glo, pc_in_glo[:, :3])  # (N, 3)
    # split
    lwh = np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
    mask_inside = np.all((np.abs(xyz_in_box) / lwh) < (0.5 + tol), axis=1)

    pts_in_box = np.concatenate([xyz_in_box[mask_inside], pc_in_glo[mask_inside, 3:]], axis=1)  # (N_in, 3+C)
    return pts_in_box, pc_in_glo[np.logical_not(mask_inside)]


def get_merge_pointcloud(nusc: NuScenes, sample_token: str, num_samples=5, dyna_classes=('vehicle', 'human'),
                         debug=False, center_radius=2., clean_using_annos=False) -> np.ndarray:
    sd_tokens = get_pointclouds_sequence_token(nusc, sample_token, num_samples)
    ref_rec = nusc.get('sample_data', sd_tokens[-1])

    ref_cs_rec = nusc.get('calibrated_sensor', ref_rec['calibrated_sensor_token'])
    car_from_ref = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']))

    ref_car_rec = nusc.get('ego_pose', ref_rec['ego_pose_token'])
    glo_from_car = transform_matrix(ref_car_rec['translation'], Quaternion(ref_car_rec['rotation']))

    ref_from_glo = np.linalg.inv(glo_from_car @ car_from_ref)
    ref_time = ref_rec['timestamp'] * 1e-6

    annos = dict()
    static_pc = []
    for _sd_token in sd_tokens:
        lidar_path = nusc.get_sample_data_path(_sd_token)
        pc = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)
        # remove ego points
        mask = ~((np.abs(pc[:, 0]) < center_radius) & (np.abs(pc[:, 1]) < center_radius))
        pc = pc[mask]
        # get time lag
        sd_rec = nusc.get('sample_data', _sd_token)
        pc_times = np.tile(np.array([[ref_time - sd_rec['timestamp'] * 1e-6]]), (pc.shape[0], 1))
        pc = np.concatenate([pc, pc_times], axis=1)  # (x, y, z, intensity, times)

        # ===
        # transform matrix
        # ===
        cur_sd_rec = nusc.get('sample_data', _sd_token)
        cur_cs_rec = nusc.get('calibrated_sensor', cur_sd_rec['calibrated_sensor_token'])
        car_from_cur = transform_matrix(cur_cs_rec['translation'], Quaternion(cur_cs_rec['rotation']))

        cur_car_rec = nusc.get('ego_pose', cur_sd_rec['ego_pose_token'])
        glo_from_car = transform_matrix(cur_car_rec['translation'], Quaternion(cur_car_rec['rotation']))

        glo_from_cur = glo_from_car @ car_from_cur
        pc[:, :3] = apply_transform_to_points(glo_from_cur, pc[:, :3])  # pc is now in GLOBAL frame

        # ===
        # partition pc into static & dynamic
        # ===
        if clean_using_annos:
            boxes = nusc.get_boxes(_sd_token)
            # remove nondynamic & empty boxes
            mask_nonempty = []
            inst_tokens = []
            for _bi, box in enumerate(boxes):
                anno_rec = nusc.get('sample_annotation', box.token)
                inst_tokens.append(anno_rec['instance_token'])
                if box.name.split('.')[0] not in dyna_classes:
                    continue
                num_pts = anno_rec['num_lidar_pts'] + anno_rec['num_radar_pts']
                if num_pts > 0:
                    mask_nonempty.append(_bi)
            boxes = [boxes[_bi] for _bi in mask_nonempty]
            inst_tokens = [inst_tokens[_bi] for _bi in mask_nonempty]
            # iterate boxes, accumulate points of annotation
            for _inst_token, box in zip(inst_tokens, boxes):
                pts_in_box, pc = partition_pointcloud(pc, box)
                if _inst_token in annos:
                    annos[_inst_token].append(pts_in_box)
                else:
                    annos[_inst_token] = [pts_in_box]
        # store static pc
        static_pc.append(pc)  # in GLOBAL frame

    # map static pc to ref frame
    static_pc = np.vstack(static_pc)
    static_pc[:, :3] = apply_transform_to_points(ref_from_glo, static_pc[:, :3])

    # get points in boxes that are visible @ the sample_token (so-called dynamic pc)
    dyna_pc = []
    if clean_using_annos:
        sample = nusc.get('sample', sample_token)
        for ann_token in sample['anns']:
            anno_rec = nusc.get('sample_annotation', ann_token)
            if anno_rec['instance_token'] in annos:
                pts_in_box = np.vstack(annos[anno_rec['instance_token']])  # (N_in_box, 3+C)
                # map pts_in_box from box's frame to ref frame
                glo_from_box = transform_matrix(anno_rec['translation'], Quaternion(anno_rec['rotation']))
                pts_in_box[:, :3] = apply_transform_to_points(ref_from_glo @ glo_from_box, pts_in_box[:, :3])
                dyna_pc.append(pts_in_box)

    if dyna_pc:
        dyna_pc = np.vstack(dyna_pc)
        merge_pc = np.vstack([static_pc, dyna_pc])
    else:
        merge_pc = static_pc

    if not debug:
        return merge_pc
    else:
        mask_dyn = np.zeros(merge_pc.shape[0], dtype=bool)
        mask_dyn[static_pc.shape[0]:] = 1
        return merge_pc, mask_dyn


if __name__ == '__main__':
    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', version='v1.0-mini', verbose=False)
    sample = nusc.sample[10]
    ref_sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ref_time = ref_sd_rec['timestamp'] * 1e-6

    # seq_tokens = get_lidar_and_sweeps_tokens(nusc, sample['token'])
    seq_tokens = get_pointclouds_sequence_token(nusc, sample['token'], num_samples=5)
    kf_tokens = []
    for i, sd_token in enumerate(seq_tokens):
        sd_rec = nusc.get('sample_data', sd_token)
        print(f"{i} | time diff: {ref_time - sd_rec['timestamp'] * 1e-6} | is_kf: {sd_rec['is_key_frame']}")
        if sd_rec['is_key_frame']:
            kf_tokens.append(sd_token)

    # for kf_t in kf_tokens:
    #     sd_rec = nusc.get('sample_data', kf_t)
    #     _sample_rec = nusc.get('sample', sd_rec['sample_token'])
    #     nusc.render_sample_data(_sample_rec['data']['CAM_FRONT'])
    #     plt.show()

    merge_pc, mask_dync = get_merge_pointcloud(nusc, sample['token'], debug=True)
    pc_colors = np.zeros((merge_pc.shape[0], 3))
    pc_colors[mask_dync, 2] = 1  # blue for dyna
    show_pointcloud(merge_pc[:, :3], None, pc_colors=None)



