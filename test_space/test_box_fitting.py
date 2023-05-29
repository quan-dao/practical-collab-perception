import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

from workspace.uda_tools_box import BoxFinder


def main():
    points = np.load('artifact/hdbscan_dataset10_cluster59.npy')
    print('points: ', points.shape)
    print(points[:5])
    
    points_sweep_idx = points[:, -2].astype(int)
    unq_sweep_idx  = np.unique(points_sweep_idx)

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    sweeps_color = matplotlib.cm.rainbow(np.linspace(0, 1, unq_sweep_idx.shape[0]))[:, :3]


    box_finder = BoxFinder(return_edges_in_homogeneous_coord=True)

    for _idx, sweep_idx in enumerate(unq_sweep_idx):
    
        mask_this_sweep = points_sweep_idx == sweep_idx
        
        rect = box_finder.fit(points[mask_this_sweep])

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
    
    plt.show()
    

if __name__ == '__main__':
    main()

