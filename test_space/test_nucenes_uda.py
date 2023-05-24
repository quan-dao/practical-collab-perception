import numpy as np
import torch

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict


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

    painter = PointsPainter(xyz=points[:, :3], boxes=gt_boxes[:, :7])
    painter.show()


if __name__ == '__main__':
    main()

