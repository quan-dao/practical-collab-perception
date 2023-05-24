import numpy as np
import torch

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary
from test_space.utils_4testing_corrector import correct_points


def main():
    class_names = ['car',]
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, training=True,
        version='v1.0-mini'
    )
    batch_dict = dataset[10]
    
    print_dict(batch_dict, 'batch_dict')

    points = batch_dict['points']
    gt_boxes = batch_dict['gt_boxes']

    print('showing original')
    painter = PointsPainter(xyz=points[:, :3], boxes=gt_boxes[:, :7])
    points_color = color_points_binary(points[:, -1] > -1)
    painter.show(xyz_color=points_color, boxes_velo=gt_boxes[:, -2:])

    # TODO: test correction
    print('showing corrected')
    points = np.pad(points, pad_width=[(0, 0), (1, 0)], constant_values=0.)  # pad points with bath_idx
    instances_tf = batch_dict['instances_tf'][np.newaxis, :, :, :3]
    points_corrected = correct_points(points, instances_tf)
    painter = PointsPainter(xyz=points_corrected[:, 1: 4], boxes=gt_boxes[:, :7])
    points_color = color_points_binary(points_corrected[:, -1] > -1)
    painter.show(xyz_color=points_color)

    # TODO: bug in velo

if __name__ == '__main__':
    main()

