import numpy as np
from nuscenes import NuScenes
import pickle

from workspace.o3d_visualization import print_dict, PointsPainter
from workspace.nuscenes_temporal_utils import apply_se3_, get_nuscenes_sensor_pose_in_global, get_one_pointcloud
# from workspace.v2x_sim_utils import get_one_pointcloud


def main(chosen_batch_idx: int):
    with open('artifact/v2x_sim_batch_dict_ego.pkl', 'rb') as f:
        batch_dict = pickle.load(f)
    
    points = batch_dict['points']
    gt_boxes = batch_dict['gt_boxes']
    metadata = batch_dict['metadata']

    # -----------
    points = points[points[:, 0].astype(int) == chosen_batch_idx, 1:]
    gt_boxes = gt_boxes[chosen_batch_idx, :, :7]
    metadata = metadata[chosen_batch_idx]
    
    # find foreground points & modar_points
    mask_foreground = (points[:, 5: 8] > 0.1).any(axis=1)
    mask_modar = points[:, -3] > 0
    mask_original = np.logical_not(np.logical_or(mask_foreground, mask_modar))

    print('showing original')
    painter = PointsPainter(points[mask_original, :3], gt_boxes)
    painter.show()

    print('showing original + exchange')
    painter = PointsPainter(points[:, :3], gt_boxes)
    points_color = np.zeros((points.shape[0], 3))
    points_color[mask_foreground, 0] = 1
    points_color[mask_modar, 2] = 1
    painter.show(points_color)

    


if __name__ == '__main__':
    main(chosen_batch_idx=1)
