import numpy as np
import argparse
import matplotlib.cm
import pickle

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict
from workspace.v2x_sim_utils import correction_numpy


classes_color = np.array([
    [1., 0., 0.],
    [0., 0., 1.]
])


def _dataset(dataset, sample_idx: int):
    data_dict = dataset[sample_idx]
    print_dict(data_dict, 'data_dict')

    points = data_dict['points']  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
    gt_boxes = data_dict['gt_boxes']  # (N_gt, 7 + 1) - box-7, class_idx
    instances_tf = data_dict['instances_tf']  # (N_inst, N_sweep, 4, 4)
    
    # NOTE: N_inst >= N_gt because some gt are outside of range

    print('showing original')
    painter = PointsPainter(points[:, :3], gt_boxes[:, :7])
    boxes_color = classes_color[gt_boxes[:, -1].astype(int) - 1]
    instances_color = matplotlib.cm.rainbow(np.linspace(0, 1, instances_tf.shape[0]))[:, :3]
    points_color = instances_color[points[:, -1].astype(int)] * (points[:, -1] > -1).astype(float).reshape(-1, 1)
    painter.show(points_color, boxes_color)

    print('showing correction')
    corrected_xyz = correction_numpy(points, instances_tf)
    painter = PointsPainter(corrected_xyz, gt_boxes[:, :7])
    painter.show(points_color, boxes_color)


def _dataloader(dataloader, target_batch_idx: int, chosen_batch_idx: int, save_batch_dict: bool, dataset_type: str):
    if save_batch_dict:
        iter_dataloader = iter(dataloader)
        batch_dict = None
        for _ in range(target_batch_idx + 1):
            batch_dict = next(iter_dataloader)
        
        with open(f'artifact/v2x_sim_batch_dict_{dataset_type}.pkl', 'wb') as f:
            pickle.dump(batch_dict, f)
        
        print(f'batch dict saved in artifact/v2x_sim_batch_dict_{dataset_type}.pkl')
        return
    else:
        with open(f'artifact/v2x_sim_batch_dict_{dataset_type}.pkl', 'rb') as f:
            batch_dict = pickle.load(f)
    
    print_dict(batch_dict, 'batch_dict')

    points = batch_dict['points']  # (N_pts, 1 + 5 + 2) - batch_idx, point-5, sweep_idx, inst_idx
    gt_boxes = batch_dict['gt_boxes']  # (B, N_gt_max, 7 + 1) - box-7, class_idx
    instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)

    points = points[points[:, 0].astype(int) == chosen_batch_idx]
    gt_boxes = gt_boxes[chosen_batch_idx]
    instances_tf = instances_tf[chosen_batch_idx]

    print('showing original')
    painter = PointsPainter(points[:, 1: 4], gt_boxes[:, :7])
    boxes_color = classes_color[gt_boxes[:, -1].astype(int) - 1]
    instances_color = matplotlib.cm.rainbow(np.linspace(0, 1, instances_tf.shape[0]))[:, :3]
    points_color = instances_color[points[:, -1].astype(int)] * (points[:, -1] > -1).astype(float).reshape(-1, 1)
    painter.show(points_color, boxes_color)

    print('showing correction')
    corrected_xyz = correction_numpy(points[:, 1:], instances_tf)
    painter = PointsPainter(corrected_xyz, gt_boxes[:, :7])
    painter.show(points_color, boxes_color)



def main(test_dataset: bool = False,
         dataset_sample_idx: int = 5,
         test_dataloader: bool = False,
         target_batch_idx: int = 3,
         chosen_batch_idx: int = 1,
         debug_dataset: bool = False,
         dataset_type: str = 'rsu',
         save_batch_dict: bool = True,
         version: str ='v2.0-mini'):
    np.random.seed(666)
    assert dataset_type in ('rsu', 'car', 'ego')
    cfg_file = f'../tools/cfgs/dataset_configs/v2x_sim_dataset_{dataset_type}.yaml'
    dataset, dataloader = build_dataset_for_testing(
        cfg_file, ['car', 'pedestrian'], debug_dataset=debug_dataset, version=version, batch_size=2, training=True)
    if test_dataset:
        _dataset(dataset, dataset_sample_idx)

    if test_dataloader:
        _dataloader(dataloader, target_batch_idx, chosen_batch_idx, save_batch_dict, dataset_type)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--test_dataset', type=int, default=0)
    parser.add_argument('--dataset_sample_idx', type=int, default=5)
    parser.add_argument('--test_dataloader', type=int, default=0)
    parser.add_argument('--target_batch_idx', type=int, default=3)
    parser.add_argument('--chosen_batch_idx', type=int, default=1)
    parser.add_argument('--debug_dataset', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='rsu')
    parser.add_argument('--save_batch_dict', type=int, default=0)
    parser.add_argument('--version', type=str, default='v2.0-mini')
    args = parser.parse_args()
    main(args.test_dataset==1,
         args.dataset_sample_idx,
         args.test_dataloader==1,
         args.target_batch_idx,
         args.chosen_batch_idx,
         args.debug_dataset,
         args.dataset_type,
         args.save_batch_dict==1,
         args.version)

    # python test_v2x_dataset.py --test_dataloader 1 --debug_dataset 1 --dataset_type ego --save_batch_dict 1 --version v2.0-trainval
