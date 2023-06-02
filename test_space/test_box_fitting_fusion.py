import numpy as np
import matplotlib.cm
from sklearn.neighbors import LocalOutlierFactor

from workspace.uda_tools_box import BoxFinder
from workspace.o3d_visualization import PointsPainter
from workspace.traj_discovery import TrajectoryProcessor


def main(sample_idx: int, cluster_idx: int):
    points = np.load(f'artifact/hdbscan_dataset{sample_idx}_cluster{cluster_idx}.npy')
    print('points: ', points.shape)
    
    # remove outlier
    classifier = LocalOutlierFactor(n_neighbors=10)
    mask_inliers = classifier.fit_predict(points[:, :3])
    points = points[mask_inliers > 0]
    
    TrajectoryProcessor.setup_class_attribute(num_sweeps=15, look_for_static=True, debug=True)
    box_finder = BoxFinder(return_in_form='box_openpcdet', return_theta_star=True)

    traj = TrajectoryProcessor()
    traj(points, None, None, box_finder, None, None)

    def viz():
        painter = PointsPainter(points[:, :3])
        points_color = np.zeros((points.shape[0], 3))
        points_color[mask_inliers > 0, 0] = 1.0
        painter.show(points_color)
    
    viz()


if __name__ == '__main__':
    main(sample_idx=10,
         cluster_idx=31)
    
    # 10 - 31 -> has outliers far away from the main traj

