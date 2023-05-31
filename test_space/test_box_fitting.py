import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from sklearn.decomposition import PCA

from workspace.uda_tools_box import BoxFinder
from workspace.box_fusion_utils import kde_fusion


def main():
    points = np.load('artifact/hdbscan_dataset110_cluster22.npy')
    print('points: ', points.shape)
    print(points[:5])
    
    points_sweep_idx = points[:, -2].astype(int)
    unq_sweep_idx  = np.unique(points_sweep_idx)

    rough_est_heading = (points[points_sweep_idx == unq_sweep_idx[0], :2]).mean(axis=0) - (points[points_sweep_idx == unq_sweep_idx[-1], :2]).mean(axis=0)
    rough_est_heading /= np.linalg.norm(rough_est_heading)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    sweeps_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_sweep_idx.shape[0]))[:, :3]

    box_finder = BoxFinder(return_fitness=True)

    fitnesses, num_points = [], []
    for _idx, sweep_idx in enumerate(unq_sweep_idx):
    
        mask_this_sweep = points_sweep_idx == sweep_idx
        
        rect, fitness, theta_star = box_finder.fit(points[mask_this_sweep], rough_est_heading)
        print(f'sweep {sweep_idx}: fitness = {fitness} | num points: {mask_this_sweep.sum()}')
        fitnesses.append(fitness)
        num_points.append(mask_this_sweep.sum())

        p01 = np.cross(rect[0], rect[1])
        p12 = np.cross(rect[1], rect[2])
        p23 = np.cross(rect[2], rect[3])
        p30 = np.cross(rect[3], rect[0])
        vers = np.stack([p01, p12, p23, p30], axis=0)
        vers /= vers[:, [-1]]

        ax.scatter(points[mask_this_sweep, 0], points[mask_this_sweep, 1], color=sweeps_color[_idx])
        ax.plot(vers[[0, 1], 0], vers[[0, 1], 1], color=sweeps_color[_idx])
        ax.plot(vers[[1, 2], 0], vers[[1, 2], 1], color=sweeps_color[_idx])
        ax.plot(vers[[2, 3], 0], vers[[2, 3], 1], color=sweeps_color[_idx])
        ax.plot(vers[[3, 0], 0], vers[[3, 0], 1], color=sweeps_color[_idx])
        # ax.plot(vers[[0, 1], 0], vers[[0, 1], 1], color='r')
        # ax.plot(vers[[1, 2], 0], vers[[1, 2], 1], color='g')
        # ax.plot(vers[[2, 3], 0], vers[[2, 3], 1], color='b')
        # ax.plot(vers[[3, 0], 0], vers[[3, 0], 1], color='k')
        # break

    plt.show()
    

if __name__ == '__main__':
    main()

