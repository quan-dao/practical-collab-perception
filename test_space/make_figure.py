import numpy as np
from nuscenes import NuScenes
import matplotlib.cm
import pickle

from workspace.rev_get_sweeps_instance_centric import revised_instance_centric_get_sweeps
from workspace.o3d_visualization import PointsPainter


nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
scene = nusc.scene[2]
sample_tk = scene['first_sample_token']
for _ in range(10):
    sample = nusc.get('sample', sample_tk)
    sample_tk = sample['next']


def make_fig_accum_shadow():
    info = revised_instance_centric_get_sweeps(nusc, sample_tk, 10, detection_classes=['car', 'pedestrian', 'bicylce', 'bus', 'truck', 'trailer'])
    with open('artifact/make_fig_accum_shadow.pkl', 'wb')as f:
        pickle.dump(info, f)
    return
    points = info['points']
    boxes = info['inst_boxes']
    boxes_name = info['gt_names']
    
    print('points: ', points.shape)
    print('boxes: ', boxes.shape)
    print('boxes_name: ', boxes_name[:5])

    # =============
    painter = PointsPainter(points[:, :3], boxes=boxes[:, :7])
    # color poins by timestamp
    points_sweep_idx = points[:, -2].astype(int)
    unq_sweep_idx = np.unique(points_sweep_idx)
    sweeps_colors = matplotlib.cm.rainbow(np.linspace(0., 1, unq_sweep_idx.shape[0]))[:, :3]
    # points_colors = sweeps_colors[points_sweep_idx]# * (points[:, -1] > -1).reshape(-1, 1)

    points_inst_idx = points[:, -1].astype(int)
    unq_inst_idx = np.unique(points_inst_idx)
    instances_colors = matplotlib.cm.rainbow(np.linspace(0., 1, unq_inst_idx.max() + 1))[:, :3] 
    points_colors = instances_colors[points_inst_idx] * (points_inst_idx > -1).astype(float).reshape(-1, 1)
    
    painter.show(points_colors)


if __name__ == '__main__':
    make_fig_accum_shadow()
