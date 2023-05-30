import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from sklearn.decomposition import PCA

from workspace.uda_tools_box import BoxFinder
from workspace.box_fusion_utils import kde_fusion


def main():
    points = np.load('artifact/hdbscan_dataset10_cluster31.npy')
    print('points: ', points.shape)
    
    points_sweep_idx = points[:, -2].astype(int)
    unq_sweep_idx  = np.unique(points_sweep_idx)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    sweeps_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_sweep_idx.shape[0]))[:, :3]

    box_finder = BoxFinder(return_in_form='box_openpcdet')

    traj_boxes = []
    for _idx, sweep_idx in enumerate(unq_sweep_idx):    
        mask_this_sweep = points_sweep_idx == sweep_idx
        box_bev, mean_z = box_finder.fit(points[mask_this_sweep])
        # assembly
        x, y, dx, dy, heading = box_bev
        box3d = np.array([x, y, mean_z, dx, dy, np.max(points[mask_this_sweep, :2]), heading, 1, mask_this_sweep.sum() / float(points.shape[0])])
        traj_boxes.append(box3d)

    traj_boxes = np.stack(traj_boxes, axis=0)
    fused_box = kde_fusion(traj_boxes, src_weights=traj_boxes[:, -1])

    print('traj_boxes:\n', traj_boxes[np.argsort(-traj_boxes[:, -1]), 3:])
    print('---')
    print(fused_box[3: 6])
    

if __name__ == '__main__':
    main()

