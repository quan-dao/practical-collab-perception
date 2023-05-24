import numpy as np
from nuscenes import NuScenes
import open3d as o3d
import matplotlib.pyplot as plt

from rev_get_sweeps_instance_centric import revised_instance_centric_get_sweeps
from _dev_space.tools_box import show_pointcloud


def make_bev(points, pc_range, voxel_size):
    bev_size_xy = np.floor((pc_range[3: 5] - pc_range[:2]) / voxel_size).astype(int)
    
    pixels = np.floor((points[:, :2] - pc_range[:2]) / voxel_size).astype(int)  # (N_pix, 2)
    pixels_merge = pixels[:, 1] * bev_size_xy[0] + pixels[:, 0]
    pixels_unique = np.unique(pixels_merge)
    pixels = np.stack([pixels_merge % bev_size_xy[0], pixels_merge // bev_size_xy[0]], axis=1)

    bev_img = np.zeros((bev_size_xy[1], bev_size_xy[0]))
    bev_img[pixels[:, 1], pixels[:, 0]] = 1
    return bev_img


nusc = NuScenes(dataroot='/home/user/dataset/nuscenes', version='v1.0-mini', verbose=False)
scene = nusc.scene[0]

sample_token = scene['first_sample_token']

for _ in range(5):
    sample_rec = nusc.get('sample', sample_token)
    sample_token = sample_rec['next']


points = revised_instance_centric_get_sweeps(nusc, sample_token, n_sweeps=10, return_points_only=True)
pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
mask_inrange = np.logical_and(points[:, :3] > pc_range[:3], points[:, :3] < (pc_range[3:] - 1e-3)).all(axis=1)
points = points[mask_inrange]
print("points.shape: ", points.shape)
show_pointcloud(points[:, :3])

# TODO: hard vox -> BEV
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)
voxels = voxel_grid.get_voxels()
print('num voxels: ', len(voxels))

voxels_center = []
for vox in voxels:
    center = voxel_grid.get_voxel_center_coordinate(vox.grid_index)
    voxels_center.append(center.reshape(1, 3))

voxels_center = np.concatenate(voxels_center)
np.random.shuffle(voxels_center)
voxels_center = voxels_center[:8000]
print('voxels_center: ', voxels_center.shape)
# show_pointcloud(voxels_center[:, :3])

hard_bev_img = make_bev(voxels_center, pc_range, voxel_size=0.2)

dyn_bev_img = make_bev(points[:, :3], pc_range, voxel_size=0.2)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(hard_bev_img[::-1], cmap='gray')
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(dyn_bev_img[::-1], cmap='gray')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.show()
