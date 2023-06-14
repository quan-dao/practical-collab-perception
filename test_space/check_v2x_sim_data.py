import numpy as np
from nuscenes import NuScenes
from pprint import pprint
import matplotlib.cm

from workspace.v2x_sim_utils import get_points_and_boxes_of_1lidar, get_historical_boxes_1instance
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

    anno_tokens = lidar0_info['anno_tokens']
    print(anno_tokens)
    _idx = 5
    interp_boxes = get_historical_boxes_1instance(nusc, lidar0_tk, boxes_in_lidar[_idx], anno_tokens[_idx], 0)
    print('interp_boxes:\n', interp_boxes[:, :3])

    painter = PointsPainter(points[:, :3], np.concatenate([boxes_in_lidar, interp_boxes[:, :7]]))
    boxes_color = matplotlib.cm.rainbow(np.linspace(0, 1, boxes_in_lidar.shape[0]))[:, :3]
    points_color = boxes_color[box_idx_of_points]
    points_color[box_idx_of_points < 0] = 0.

    boxes_color = np.concatenate([boxes_color, np.tile(np.array([[0., 0., 0.]]), (interp_boxes.shape[0], 1))])
    painter.show(points_color, boxes_color)


if __name__ == '__main__':
    classes_of_interest = set(['car', 'pedestrian'])
    main(classes_of_interest=classes_of_interest)
