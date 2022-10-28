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
    points_merge_idx = points[:, -2].astype(int) * n_sweeps + points[:, -3].astype(int)  # (N,)
    _tf = instances_tf.reshape((-1, 3, 4))
    _tf = _tf[points_merge_idx]  # (N, 3, 4)

    # apply transformation
    points_new_xyz = np.matmul(_tf[:, :3, :3], points[:, :3, np.newaxis]) + _tf[:, :3, [-1]]
    return points_new_xyz.squeeze(axis=-1)


def quat2mat(quat):
    """
    convert quaternion to rotation matrix ([x, y, z, w] to follow scipy
    :param quat: (B, 4) four quaternion of rotation
    :return: rotation matrix [B, 3, 3]
    """
    # norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    # w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

