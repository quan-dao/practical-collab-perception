import numpy as np
import torch

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary
from test_space.utils_4testing_corrector import correct_points


def main(test_dataset: bool, test_dataloader: bool, chosen_batch_idx: int = 0):
    class_names = ['car', 'ped']
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
        training=True,
        batch_size=2,
        version='v1.0-mini',
        debug_dataset=True
    )

    if test_dataset:
        batch_dict = dataset[110]
        
        print_dict(batch_dict, 'batch_dict')

        points = batch_dict['points']
        gt_boxes = batch_dict['gt_boxes']
        # gt_boxes = batch_dict['metadata']['disco_boxes']

        print('showing original')
        painter = PointsPainter(xyz=points[:, :3], boxes=gt_boxes[:, :7])
        points_color = color_points_binary(points[:, -1] > -1)
        painter.show(xyz_color=points_color)

    if test_dataloader:
        iter_dataloader = iter(dataloader)
        for _ in range(3):
            batch_dict = next(iter_dataloader)

        print_dict(batch_dict, 'batch_dict')

        points = batch_dict['points']
        gt_boxes = batch_dict['gt_boxes']

        print('showing original')
        cur_points = points[points[:, 0].astype(int) == chosen_batch_idx]
        cur_boxes = gt_boxes[chosen_batch_idx]  # (N_inst, 10)
        cur_points_nusc = cur_points[cur_points[:, -1] < 0]
        painter = PointsPainter(xyz=cur_points_nusc[:, 1: 4])
        painter.show()

        
        print('cur_points: ', cur_points.shape)
        painter = PointsPainter(xyz=cur_points[:, 1: 4], boxes=cur_boxes[:, :7])
        points_color = color_points_binary(cur_points[:, -1] > -1)
        painter.show(xyz_color=points_color, boxes_velo=cur_boxes[:, -3: -1])

        # test correction
        # print('showing corrected')
        # instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)
        # points_corrected = correct_points(points, instances_tf)
        # cur_points_corrected = points_corrected[points[:, 0].astype(int) == chosen_batch_idx]

        # print('cur_points_corrected: ', cur_points_corrected.shape)
        # painter = PointsPainter(xyz=cur_points_corrected[:, 1: 4], boxes=cur_boxes[:, :7])
        # points_color = color_points_binary(cur_points_corrected[:, -1] > -1)
        # painter.show(xyz_color=points_color)


if __name__ == '__main__':
    main(test_dataset=True, 
         test_dataloader=False,
         chosen_batch_idx=1)

