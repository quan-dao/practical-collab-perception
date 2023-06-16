import numpy as np
from pprint import pprint
import argparse

from test_space.tools import build_dataset_for_testing


def main(version: str, test_car: bool):
    np.random.seed(666)
    if not test_car:
        cfg_file = '../tools/cfgs/dataset_configs/v2x_sim_dataset_rsu.yaml'
    else:
        cfg_file = '../tools/cfgs/dataset_configs/v2x_sim_dataset_car.yaml'
    dataset, dataloader = build_dataset_for_testing(
        cfg_file, ['car', 'pedestrian'], debug_dataset=False, version=version, batch_size=2, training=False)
    
    # build det_annos
    det_annos = []
    for info in dataset.infos:
        anno = {
            'metadata': {'lidar_token': info['lidar_token']},
            'boxes_lidar': info['gt_boxes'],
            'score': np.ones(info['gt_boxes'].shape[0], dtype=float),
            'pred_labels': np.array([1 if name == 'car' else 2 for name in info['gt_names']]),
            'name': info['gt_names']
        }

        det_annos.append(anno)

    result_str, result_dict = dataset.nuscenes_eval(det_annos, '', output_path='./artifact/')
    pprint(result_str)
    print('---')
    pprint(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='blah')
    parser.add_argument('--version', type=str, default='v2.0-mini')
    parser.add_argument('--test_car', type=int, default=0)
    args = parser.parse_args()
    main(args.version, args.test_car)
