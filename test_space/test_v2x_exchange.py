import numpy as np
import pickle
import argparse

from workspace.o3d_visualization import print_dict, PointsPainter


def test_ego(chosen_batch_idx: int):
    with open('artifact/v2x_sim_batch_dict_ego.pkl', 'rb') as f:
        batch_dict = pickle.load(f)
    
    points = batch_dict['points']
    gt_boxes = batch_dict['gt_boxes']
    metadata = batch_dict['metadata']

    # -----------
    points = points[points[:, 0].astype(int) == chosen_batch_idx, 1:]
    gt_boxes = gt_boxes[chosen_batch_idx, :, :7]
    metadata = metadata[chosen_batch_idx]
    print(metadata['exchange'])

    # filter gt_boxes by range
    mask_display = np.linalg.norm(gt_boxes[:, :2], axis=1) > 51.2
    gt_boxes = gt_boxes[mask_display]

    # find foreground points & modar_points
    mask_foreground = (points[:, 11: 14] > 0.1).any(axis=1)
    mask_modar = points[:, 10] > 0
    mask_original = np.logical_not(np.logical_or(mask_foreground, mask_modar))

    frames = metadata['exchange_coord'][0].reshape(-1, 3)

    print('showing original')
    painter = PointsPainter(points[mask_original, :3], gt_boxes)
    painter.show()

    print('showing original + exchange')
    
    points_color = np.zeros((points.shape[0], 3))
    points_color[mask_foreground, 0] = 1
    points_color[mask_modar, 2] = 1

    frames_color = np.zeros_like(frames)
    frames_color[:, 0] = 0.5
    frames_color[:, 2] = 0.5

    painter = PointsPainter(np.concatenate([points[:, :3], frames]), gt_boxes)
    painter.show(np.concatenate([points_color, frames_color]))


def test_ego_early(chosen_batch_idx: int):
    with open('artifact/v2x_sim_batch_dict_ego_early.pkl', 'rb') as f:
        batch_dict = pickle.load(f)
    
    points = batch_dict['points']
    gt_boxes = batch_dict['gt_boxes']
    metadata = batch_dict['metadata']

    # -----------
    points = points[points[:, 0].astype(int) == chosen_batch_idx, 1:]
    print('points: ', points.shape)
    print('num original: ', metadata[chosen_batch_idx]['num_original'])
    print_dict(metadata[chosen_batch_idx]['exchange'], 'exchange')

    gt_boxes = gt_boxes[chosen_batch_idx, :, :7]
    metadata = metadata[chosen_batch_idx]

    painter = PointsPainter(points[:, :3], gt_boxes[:, :7])
    painter.show()


def main(dataset_type: str, chosen_batch_idx):
    assert dataset_type in ('ego', 'ego_early')
    if dataset_type == 'ego':
        test_ego(chosen_batch_idx)
    else:
        test_ego_early(chosen_batch_idx)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--chosen_batch_idx', type=int, default=0)
    args = parser.parse_args()
    main(args.dataset_type, args.chosen_batch_idx)
