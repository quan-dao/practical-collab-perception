import numpy as np
import pickle
import argparse
from pathlib import Path

from workspace.o3d_visualization import PointsPainter


def main(cluster_idx: int):
    clusters_samples_root = Path('./artifact/clusters_samples')
    samples_path = list(clusters_samples_root.glob(f"cluster{cluster_idx}_traj*.pkl"))
    samples_path.sort()

    for path_ in samples_path:
        print(f"showing {path_.parts[-1]}")
        with open(path_, 'rb') as f:
            info = pickle.load(f)
        
        painter = PointsPainter(
            np.concatenate([info['points_in_lidar'], info['corrected_points_in_lidar']]),
            np.concatenate([info['boxes_in_lidar'], info['last_box_in_lidar']])
        )
        painter.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='blah')
    parser.add_argument('--cluster_idx', type=int)

    args = parser.parse_args()

    main(cluster_idx=args.cluster_idx)


