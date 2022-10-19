import numpy as np
import torch


def correction_numpy(points: np.ndarray, instances_tf: np.ndarray):
    """
    Args:
        points: (N, 7) - x, y, z, instensity, time-lag, sweep_idx, instance_idx
        instances_tf: (N_inst, N_sweep, 3, 4)
    Returns:
         points_new_xyz: (N, 3)
    """
    # merge sweep_idx & instance_idx
    n_sweeps = instances_tf.shape[1]
    points_merge_idx = points[:, -1].astype(int) * n_sweeps + points[:, -2].astype(int)  # (N,)
    _tf = instances_tf.reshape((-1, 3, 4))
    _tf = _tf[points_merge_idx]  # (N, 3, 4)

    # apply transformation
    points_new_xyz = np.matmul(_tf[:, :3, :3], points[:, :3, np.newaxis]) + _tf[:, :3, [-1]]
    return points_new_xyz.squeeze(axis=-1)
