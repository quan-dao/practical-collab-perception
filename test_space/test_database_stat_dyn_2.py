import numpy as np
from pathlib import Path
import pickle
from pprint import pprint

from test_space.tools import build_dataset_for_testing
from workspace.o3d_visualization import PointsPainter, print_dict
from workspace.traj_discovery import TrajectoriesManager


NUM_SWEEPS = 15
class_names = ['car',]
dataset, dataloader = build_dataset_for_testing(
    '../tools/cfgs/dataset_configs/nuscenes_dataset.yaml', class_names, 
    training=True,
    batch_size=2,
    version='v1.0-mini',
    debug_dataset=True,
    MAX_SWEEPS=NUM_SWEEPS
)

traj_manager = TrajectoriesManager(
    info_root=Path('../workspace/artifact/rev1/'),
    static_database_root=Path('../data/nuscenes/v1.0-mini/rev1_discovered_static_database_15sweeps'),
    classes_name=['car', 'ped'],
    num_sweeps=NUM_SWEEPS,
)


def main(sample_idx: int, num_sweeps=NUM_SWEEPS):
    data_dict = dataset[sample_idx]
    sample_token = data_dict['metadata']['token']
    points = data_dict['points']  # (N_pts, 3 + C) - x, y, z, C-channel

    # load dynmamic boxes
    boxes_dyn = traj_manager.load_disco_traj_for_1sample(sample_token, is_dyna=True, return_in_lidar_frame=True)
    print('boxes_dyn: ', boxes_dyn.shape)
    print(boxes_dyn[:5])
    print('---')

    # load static boxes
    boxes_stat = traj_manager.load_disco_traj_for_1sample(sample_token, is_dyna=False, return_in_lidar_frame=True)
    print('boxes_stat: ', boxes_stat.shape)
    print(boxes_stat[:5])
    print('---')

    def viz():
        classes_color = np.array([
            [1, 0, 0],  # red - car
            [0, 0, 1]  # blue - ped
        ])
        print('showing dynamic boxes')
        painter = PointsPainter(points[:, :3], boxes_dyn[:, :7])
        boxes_color = classes_color[boxes_dyn[:, -1].astype(int)]
        painter.show(boxes_color=boxes_color)

        print('showing static boxes')
        painter = PointsPainter(points[:, :3], boxes_stat[:, :7])
        boxes_color = classes_color[boxes_stat[:, -1].astype(int)]
        painter.show(boxes_color=boxes_color)


    viz()


if __name__ == '__main__':
    main(sample_idx=115)

