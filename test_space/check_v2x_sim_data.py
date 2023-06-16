import numpy as np
from nuscenes import NuScenes
from pprint import pprint
import matplotlib.cm

# from workspace.v2x_sim_utils import get_pseudo_sweeps_of_1lidar, correction_numpy
from workspace.o3d_visualization import PointsPainter


def main(num_sweeps, classes_of_interest: set):
    nusc = NuScenes(dataroot='/home/user/dataset/v2x-sim', version='v2.0-mini', verbose=False)
    scene = nusc.scene[0]
    
    sample_tk = scene['first_sample_token']
    for _ in range(2):
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']
    
    sample = nusc.get('sample', sample_tk)
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP_id_1'])
    pprint(sample_data)
    return
    
    out = get_pseudo_sweeps_of_1lidar(nusc, sample['data']['LIDAR_TOP_id_0'], num_sweeps, 
                                      classes_of_interest, 
                                      threshold_boxes_by_points=5)
    points = out['points']  # (N_pts, 5 + 2) - point-5, sweep_idx, inst_idx
    print('points: ', points.shape)

    gt_boxes = out['gt_boxes']  # (N_inst, 7)
    print('gt_boxes: ', gt_boxes.shape)

    # original
    painter = PointsPainter(points[:, :3], gt_boxes)
    painter.show()

    painter = PointsPainter(points[points[:, -2].astype(int) == num_sweeps, :3], gt_boxes)
    painter.show()

    # corrected
    instances_tf = out['instances_tf']
    corrected_xyz = correction_numpy(points, instances_tf)
    painter = PointsPainter(corrected_xyz, gt_boxes)
    painter.show()


if __name__ == '__main__':
    classes_of_interest = set(['car', 'pedestrian'])
    num_sweeps = 10
    main(num_sweeps=num_sweeps, classes_of_interest=classes_of_interest)
