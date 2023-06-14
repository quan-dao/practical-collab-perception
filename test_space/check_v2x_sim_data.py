import numpy as np
from nuscenes import NuScenes
from pprint import pprint
import matplotlib.cm

from workspace.v2x_sim_utils import get_points_and_boxes_of_1lidar
from workspace.o3d_visualization import PointsPainter


def main(classes_of_interest: set):
    nusc = NuScenes(dataroot='/home/user/dataset/v2x-sim', version='v2.0-mini', verbose=False)
    scene = nusc.scene[0]
    
    sample_tk = scene['first_sample_token']
    for _ in range(2):
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']
    
    sample = nusc.get('sample', sample_tk)
    anno = nusc.get('sample_annotation', sample['anns'][0])
    print('anno')    
    pprint(anno)
    print('---')
    
    lidar0_tk = sample['data']['LIDAR_TOP_id_0']
    lidar0_info = get_points_and_boxes_of_1lidar(nusc, lidar0_tk, threshold_boxes_by_points=5)
    points = lidar0_info['points_in_lidar']
    print('points: ', points.shape)
    box_idx_of_points = lidar0_info['box_idx_of_points']

    boxes_in_lidar = lidar0_info['boxes_in_lidar']
    print('boxes_in_lidar: ', boxes_in_lidar.shape)

    painter = PointsPainter(points[:, :3], boxes_in_lidar)
    boxes_color = matplotlib.cm.rainbow(np.linspace(0, 1, boxes_in_lidar.shape[0]))[:, :3]
    points_color = boxes_color[box_idx_of_points]
    points_color[box_idx_of_points < 0] = 0.
    painter.show(points_color, boxes_color)


if __name__ == '__main__':
    classes_of_interest = set(['car', 'pedestrian'])
    main(classes_of_interest=classes_of_interest)
