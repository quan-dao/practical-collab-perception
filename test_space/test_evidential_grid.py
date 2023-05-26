import numpy as np
from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary

import octomap

import sys
sys.path.insert(0, '/home/user/Desktop/python_ws/patchwork-plusplus/build/python_wrapper')
import pypatchworkpp


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

    # filter pointcloud by range
    mask_in_range = np.logical_and(points[:, :3] > point_cloud_range[:3], points[:, :3] < point_cloud_range[3:]).all(axis=1)
    points = points[mask_in_range]

    points_sweep_idx = points[:, -2].astype(int)

    dict_voxels_state = dict()

    for sweep_index in range(10):
        current_points = points[points_sweep_idx == 0, :3]
        ground_segmenter.estimateGround(current_points[:, :4])
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
            pointcloud=current_points.astype(float),
            origin=lidar_coords[lidar_coords[:, -1].astype(int) == sweep_index, :3].astype(float),
            maxrange=-1
        )

        if show_occupied:
            pass

        occ_coords, empty_coords = octree.extractPointCloud()
        # TODO: convert occ & empty coords to merge int coord


if __name__ == '__main__':
    main(show_ground_segmented_pointcloud=False)

