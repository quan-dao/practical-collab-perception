import numpy as np
from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary

import octomap

import sys
sys.path.insert(0, '/home/user/Desktop/python_ws/patchwork-plusplus/build/python_wrapper')
import pypatchworkpp


def xyz2key(xyz: np.ndarray, point_cloud_range: np.ndarray, resolution: float):
    vox_xyz = np.floor((xyz - point_cloud_range[:3]) / resolution).astype(int) 

    grid_size = np.floor((point_cloud_range[3:] - point_cloud_range[:3]) / resolution).astype(int)
    area_xy = grid_size[0] * grid_size[1]
    merged_vox_xyz = vox_xyz[:, 2] * area_xy + vox_xyz[:, 1] * grid_size[0] + vox_xyz[:, 0]
    return merged_vox_xyz


def key2xyz(keys: np.ndarray, point_cloud_range: np.ndarray, resolution: float):
    grid_size = np.floor((point_cloud_range[3:] - point_cloud_range[:3]) / resolution).astype(int)
    area_xy = grid_size[0] * grid_size[1]
    vox_xyz = np.stack([
        keys // area_xy,
        (keys % area_xy) // grid_size[0],
        keys % grid_size[0],
    ], axis=1)[:, ::-1]
    xyz = vox_xyz * resolution + point_cloud_range[:3]
    return xyz


def main(show_ground_segmented_pointcloud: bool, show_occupied: bool):
    class_names = ['car',]
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
        training=True,
        batch_size=2,
        version='v1.0-mini',
        debug_dataset=True
    )
    batch_dict = dataset[10]
    print_dict(batch_dict, 'batch_dict')

    params = pypatchworkpp.Parameters()
    ground_segmenter = pypatchworkpp.patchworkpp(params)
    
    octomap_resolution = 0.2

    point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    points = batch_dict['points']
    lidar_coords = batch_dict['metadata']['lidar_coords']
    # print(f'lidar_coords {lidar_coords.shape}:\n', lidar_coords, '\n')


    # filter pointcloud by range
    mask_in_range = np.logical_and(points[:, :3] > point_cloud_range[:3], points[:, :3] < point_cloud_range[3:]).all(axis=1)
    points = points[mask_in_range]

    points_sweep_idx = points[:, -2].astype(int)

    dict_voxels_state = dict()

    for sweep_index in range(5):
        current_points = points[points_sweep_idx == 0, :4]
        ground_segmenter.estimateGround(current_points)
        current_points = ground_segmenter.getNonground()

        if show_ground_segmented_pointcloud:
            print(f'sweep {sweep_index} | showing ground')
            ground_pts = ground_segmenter.getGround()
            pts = np.concatenate([ground_pts, current_points])
            pts_color = np.zeros((pts.shape[0], 3))
            pts_color[:ground_pts.shape[0], 0] = 1.0  # red - ground
            pts_color[ground_pts.shape[0]:, 1] = 1.0  # green - non ground

            painter = PointsPainter(pts[:, :3])
            painter.show(pts_color)

        octree = octomap.OcTree(octomap_resolution)
        octree.insertPointCloud(
            pointcloud=current_points[:, :3].astype(float),  # NOTE: lost reflectant here
            origin=lidar_coords[lidar_coords[:, -1].astype(int) == sweep_index, :3].astype(float).reshape(-1),
            maxrange=-1
        )
        
        occ_coords, empty_coords = octree.extractPointCloud()
        
        if show_occupied:
            print(f'sweep {sweep_index} | showing occupied')
            painter = PointsPainter(occ_coords[:, :3])
            painter.show()


        # update voxels state
        keys_occ = xyz2key(occ_coords, point_cloud_range, octomap_resolution)
        for k_occ in keys_occ:
            if k_occ in dict_voxels_state:
                dict_voxels_state[k_occ]['refl'] += 1
            else:
                dict_voxels_state[k_occ] = {
                    'refl': 1,
                    'trans': 0,
                }

        keys_empt = xyz2key(empty_coords, point_cloud_range, octomap_resolution)
        for k_empt in keys_empt:
            if k_empt in dict_voxels_state:
                dict_voxels_state[k_empt]['trans'] += 1
            else:
                dict_voxels_state[k_empt] = {
                    'refl': 0,
                    'trans': 1,
                }

    # integrate evidential grid
    eR_theta = 0.6
    eT_theta = 0.2
    conflict_keys, conflict_evid = [], []
    for vox_key, vox_count in dict_voxels_state.items():
        m, n = vox_count['refl'], vox_count['trans']
        evidence_occ  = (1.0 - eR_theta ** m) * (eT_theta ** n)
        evidence_free = (1.0 - eT_theta ** n) * (eR_theta ** m)
        evidence_conf = 1.0 - evidence_occ - evidence_free

        if evidence_conf > 0.3:
            conflict_keys.append(vox_key.item())
            conflict_evid.append(evidence_conf)

    conflict_keys = np.array(conflict_keys)
    evidence_conf = np.array(evidence_conf)
    print(f"conflict_evid: {np.min(conflict_evid)} | {np.max(conflict_evid)}")

    # convert conflict_keys to float
    conf_coords = key2xyz(conflict_keys, point_cloud_range, octomap_resolution)
    print(f'all | showing conflict')


    coords = np.concatenate([conf_coords, occ_coords], axis=0)
    colors = np.zeros((coords.shape[0], 3))
    colors[:conf_coords.shape[0], 0] = 1.  # conf == red
    painter = PointsPainter(coords[:, :3])
    painter.show(xyz_color=colors)



if __name__ == '__main__':
    main(show_ground_segmented_pointcloud=False,
         show_occupied=False)

