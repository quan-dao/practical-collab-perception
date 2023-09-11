import numpy as np
import torch
import torch.nn.functional as F
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from einops import rearrange
import cv2

from pcdet.datasets.nuscenes.nuscenes_temporal_utils import get_one_pointcloud, get_nuscenes_sensor_pose_in_global, apply_se3_
from workspace.o3d_visualization import PointsPainter


def points_to_bev(points: np.ndarray, pc_range: np.ndarray, pix_size: float):
    mask_in_range = np.logical_and(points[:, :3] > pc_range[:3], points[:, :3] < pc_range[3:] - 1e-3).all(axis=1)
    pix_xy = np.floor((points[mask_in_range, :2] - pc_range[:2]) / pix_size).astype(int)

    img_size = np.floor((pc_range[3: 5] - pc_range[:2]) / pix_size).astype(int)
    bev_img = np.zeros((img_size[1], img_size[0]))

    merge_pix_xy = pix_xy[:, 1] * bev_img.shape[1] + pix_xy[:, 0]
    unq_merge_pix = np.unique(merge_pix_xy)

    unq_pix_y = unq_merge_pix // bev_img.shape[1]
    unq_pix_x = unq_merge_pix % bev_img.shape[1]
    bev_img[unq_pix_y, unq_pix_x] = 1.0
    return bev_img
    

def transform_bev_img(dst_se3_src: torch.Tensor, bev_in_src:torch.Tensor, pc_range_min: float, pix_size: float) -> torch.Tensor:
    assert len(bev_in_src.shape) == 3, "expect shape of bev_in_src == (C, H, W) "
    rot = dst_se3_src[:2, :2]
    t = dst_se3_src[:2, [-1]]
    t_pix_norm = 2.0 * ((t - pc_range_min) / pix_size) / bev_in_src.shape[1] - 1.0
    theta = torch.cat([rot.T, -torch.matmul(rot.T, t_pix_norm)], dim=1)  # (2, 3)

    # pad theta and bev_in_src with batch dimension
    theta = rearrange(theta, 'C1 C2 -> 1 C1 C2', C1=2, C2=3)
    bev_in_src = rearrange(bev_in_src, 'C H W -> 1 C H W')
    
    # warp
    grid = F.affine_grid(theta, bev_in_src.size())
    bev_in_target = F.grid_sample(bev_in_src, grid, mode='nearest')
    bev_in_target = rearrange(bev_in_target, '1 C H W -> C H W')
    return bev_in_target


def aug_rot(img: np.ndarray, angle: float):
    rot_matrix = cv2.getRotationMatrix2D((img.shape[2] / 2, img.shape[1] / 2), np.rad2deg(-angle), 1)
    img = cv2.warpAffine(rearrange(img, 'C H W -> H W C'), rot_matrix, img.shape[1:], cv2.INTER_NEAREST)
    # rotated = rearrange(img, 'H W C -> C H W')
    return img


def main():
    nusc = NuScenes(dataroot='/home/user/dataset/v2x-sim', version='v2.0-mini', verbose=False)
    
    pc_range = np.array([-51.2, -51.2, -8.0, 51.2, 51.2, 0.0])
    pix_size = .2

    scene = nusc.scene[0]
    sample_tk = scene['first_sample_token']

    sample = nusc.get('sample', sample_tk)
    rsu = sample['data']['LIDAR_TOP_id_0']
    ego = sample['data']['LIDAR_TOP_id_1']
    cav2 = sample['data']['LIDAR_TOP_id_5']

    glob_se3_ego = get_nuscenes_sensor_pose_in_global(nusc, ego)
    glob_se3_rsu = get_nuscenes_sensor_pose_in_global(nusc, rsu)
    glob_se3_cav2 = get_nuscenes_sensor_pose_in_global(nusc, cav2)
    ego_se3_rsu = np.linalg.inv(glob_se3_ego) @ glob_se3_rsu
    ego_se3_cav2 = np.linalg.inv(glob_se3_ego) @ glob_se3_cav2

    print('rsu pix coord: ', (ego_se3_rsu[:2, -1] - pc_range[0]) / pix_size)
    print('cav pix coord: ', (ego_se3_cav2[:2, -1] - pc_range[0]) / pix_size)
    print('cav ori:\n', ego_se3_cav2[:3, :3])

    ego_pc = get_one_pointcloud(nusc, ego)
    rsu_pc = get_one_pointcloud(nusc, rsu)
    cav2_pc = get_one_pointcloud(nusc, cav2)
    painter = PointsPainter(cav2_pc[:, :3])
    painter.show()
    
    # ====================================
    # ====================================
    # make ego bev
    ego_bev = points_to_bev(ego_pc, pc_range, pix_size)

    rsu_bev = points_to_bev(rsu_pc, pc_range, pix_size)
    rsu_bev = torch.from_numpy(rsu_bev).float().unsqueeze(0)  # (1, H, W)
    rsu_bev = transform_bev_img(torch.from_numpy(ego_se3_rsu).float(), rsu_bev, pc_range[0], pix_size)

    cav2_bev = points_to_bev(cav2_pc, pc_range, pix_size)
    cav2_bev = torch.from_numpy(cav2_bev).float().unsqueeze(0)  # (1, H, W)
    cav2_bev = transform_bev_img(torch.from_numpy(ego_se3_cav2).float(), cav2_bev, pc_range[0], pix_size)

    # ====================================
    # apply rotation pi/3
    angle = np.pi/3.
    rot_ego_bev = aug_rot(ego_bev.reshape(1, 512, 512), angle)
    rot_rsu_bev = aug_rot(rsu_bev.numpy(), angle)
    rot_cav2_bev = aug_rot(cav2_bev.numpy(), angle)

    # ----
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(ego_bev, cmap='gray', origin='lower')
    ax[0, 1].imshow(rsu_bev[0], cmap='gray', origin='lower')
    ax[0, 2].imshow(cav2_bev[0], cmap='gray', origin='lower')

    ax[1, 0].imshow(rot_ego_bev, cmap='gray', origin='lower')
    ax[1, 1].imshow(rot_rsu_bev, cmap='gray', origin='lower')
    ax[1, 2].imshow(rot_cav2_bev, cmap='gray', origin='lower')
    plt.show()

    # ====================================
    # ====================================
    apply_se3_(ego_se3_rsu, points_=rsu_pc)
    apply_se3_(ego_se3_cav2, points_=cav2_pc)
    painter = PointsPainter(np.concatenate([ego_pc[:, :3], rsu_pc[:, :3], cav2_pc[:, :3]]))
    painter.show()


if __name__ == '__main__':
    main()
