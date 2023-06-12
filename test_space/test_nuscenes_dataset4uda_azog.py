import numpy as np
import torch
import argparse
import pickle

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict, color_points_binary
from test_space.utils_4testing_corrector import correct_points


def main(test_dataset: bool,
         dataset_sample_idx: int = 10,
         test_dataloader: bool = False, 
         batch_size: int = 2, 
         save_batch_dict: bool = False,
         load_batch_dict_from_path: str = '',
         use_nuscenes_dataset_4self_training: bool = False):
    class_names = ['car', 'ped']
    if not use_nuscenes_dataset_4self_training:
        cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
    else:
        cfg_file = '../tools/cfgs/dataset_configs/nuscenes_dataset_4self_training.yaml'
    
    if test_dataset:
        dataset, dataloader = build_dataset_for_testing(
            cfg_file, class_names, 
            training=True,
            batch_size=batch_size,
            version='v1.0-mini',
            debug_dataset=True
        )
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
        if load_batch_dict_from_path == '':
            dataset, dataloader = build_dataset_for_testing(
                cfg_file, class_names, 
                training=True,
                batch_size=batch_size,
                version='v1.0-trainval',
                debug_dataset=True
            )
            iter_dataloader = iter(dataloader)
            for _ in range(3):
                batch_dict = next(iter_dataloader)
        else:
            with open(load_batch_dict_from_path, 'rb') as f:
                batch_dict = pickle.load(f)

        print_dict(batch_dict, 'batch_dict')
        if save_batch_dict:
            filename = 'artifact/nuscenes_disco_batch_dict.pkl' if not use_nuscenes_dataset_4self_training else 'artifact/nuscenes_self_training_batch_dict.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(batch_dict, f)
        else:
            points = batch_dict['points']
            gt_boxes = batch_dict['gt_boxes']

            for chosen_batch_idx in range(batch_dict['batch_size'] if 'batch_size' in batch_dict else batch_size):
                print(f'sample {chosen_batch_idx} of batch_dict')
                cur_points = points[points[:, 0].astype(int) == chosen_batch_idx]
                # cur_boxes = gt_boxes[chosen_batch_idx]  # (N_inst, 8) - box-7, class_idx
                cur_boxes = batch_dict['metadata'][chosen_batch_idx]['pseudo_boxes']
                print(f'sample {chosen_batch_idx} | pseudo_boxes: {cur_boxes.shape}')
                
                painter = PointsPainter(xyz=cur_points[:, 1: 4], boxes=cur_boxes[:, :7])
                points_color = color_points_binary(cur_points[:, -1] > -1)
                classes_color = np.array([
                    [1., 0., 0.],  # red - car  
                    [0., 0., 1.],  # blue - ped
                ])
                boxes_color = classes_color[cur_boxes[:, -1].astype(int) - 1]
                painter.show(points_color, boxes_color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--test_dataset', type=int, default=1)
    parser.add_argument('--dataset_sample_idx', type=int, default=10)

    parser.add_argument('--test_dataloader', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_batch_dict', type=int, default=0)
    parser.add_argument('--load_batch_dict_from_path', type=str, default='')
    parser.add_argument('--use_nuscenes_dataset_4self_training', type=int, default=0)
    args = parser.parse_args()

    main(test_dataset=args.test_dataset == 1,
         dataset_sample_idx=args.dataset_sample_idx,
         test_dataloader=args.test_dataloader == 1,
         batch_size=args.batch_size,
         save_batch_dict=args.save_batch_dict == 1,
         load_batch_dict_from_path=args.load_batch_dict_from_path,
         use_nuscenes_dataset_4self_training=args.use_nuscenes_dataset_4self_training == 1)

