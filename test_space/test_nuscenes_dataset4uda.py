import numpy as np
import torch
import argparse

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary
from test_space.utils_4testing_corrector import correct_points


def main(test_dataset: bool,
         dataset_sample_idx: int = 10,
         test_dataloader: bool = False, chosen_batch_idx: int = 0):
    class_names = ['car', 'ped']
    dataset, dataloader = build_dataset_for_testing(
        '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
        training=True,
        batch_size=2,
        version='v1.0-mini',
        debug_dataset=True
    )

    if test_dataset:
        batch_dict = dataset[dataset_sample_idx]
        
        print_dict(batch_dict, 'batch_dict')

        points = batch_dict['points']
        gt_boxes = batch_dict['gt_boxes']
        # gt_boxes = batch_dict['metadata']['disco_boxes']

        print('showing original')
        painter = PointsPainter(xyz=points[:, :3], boxes=gt_boxes[:, :7])
        points_color = color_points_binary(points[:, -1] > -1)

        classes_color = np.array([
            [1., 0., 0.],  # red - car  
            [0., 0., 1.],  # blue - ped
        ])
        boxes_color = classes_color[gt_boxes[:, -1].astype(int) - 1]

        painter.show(xyz_color=points_color,  boxes_color=boxes_color)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--test_dataset', type=int, default=1)
    parser.add_argument('--dataset_sample_idx', type=int, default=10)

    parser.add_argument('--test_dataloader', type=int, default=0)
    parser.add_argument('--chosen_batch_idx', type=int, default=1)
    args = parser.parse_args()

    main(test_dataset=args.test_dataset == 1,
         dataset_sample_idx=args.dataset_sample_idx,
         test_dataloader=args.test_dataloader == 1,
         chosen_batch_idx=args.chosen_batch_idx)

