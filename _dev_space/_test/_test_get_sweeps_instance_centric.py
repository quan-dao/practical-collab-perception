from nuscenes.nuscenes import NuScenes
from _dev_space.tools_box import show_pointcloud, get_nuscenes_sensor_pose_in_global, tf, apply_tf
from _dev_space.get_sweeps_instance_centric import inst_centric_get_sweeps
from _dev_space.viz_tools import viz_boxes
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from pcdet.datasets.nuscenes.nuscenes_utils import quaternion_yaw
from einops import rearrange


nusc = NuScenes(dataroot='../../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=True)
scene = nusc.scene[0]
sample_token = scene['first_sample_token']
for _ in range(5):
    sample_rec = nusc.get('sample', sample_token)
    sample_token = sample_rec['next']

sample_rec = nusc.get('sample', sample_token)

show_cam_front = False
if show_cam_front:
    nusc.render_sample_data(sample_rec['data']['CAM_FRONT'])
    plt.show()

out = inst_centric_get_sweeps(nusc, sample_token, n_sweeps=10)

points = out['points']
print('points: ', points.shape)

# convert nusc.boxes to [x, y, z, dx, dy, dz, yaw, cls]
boxes = nusc.get_boxes(sample_rec['data']['LIDAR_TOP'])  # in global frame
target_from_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, sample_rec['data']['LIDAR_TOP']))
_boxes = []
for box in boxes:
    glob_from_box = tf(box.center, box.orientation)
    target_from_box = target_from_glob @ glob_from_box
    yaw = quaternion_yaw(Quaternion(matrix=target_from_box[:3, :3]))
    _boxes.append([*target_from_box[:3, -1].tolist(), box.wlh[1], box.wlh[0], box.wlh[2], yaw])
_boxes = np.array(_boxes)
print(f"_boxes: {_boxes.shape}")

_boxes = viz_boxes(_boxes)
show_pointcloud(points[:, :3], _boxes, fgr_mask=points[:, -1] > -1)

# -----------------------
# batch correction
points_instance_idx = points[:, -1].astype(int)
bg_points = points[points_instance_idx == -1]
fg_points = points[points_instance_idx > -1]

instances_tf = out['instances_tf']  # (N_inst, N_sweeps, 4, 4)
instances_tf = instances_tf.reshape(-1, 4, 4)  # (N_inst * N_sweeps, 4, 4)
n_sweeps = 10

fg_merge_ids = fg_points[:, -1].astype(int) * n_sweeps + fg_points[:, -2].astype(int)  # (N_fg, 3 + C)
instances_traj_4fg = instances_tf[fg_merge_ids]  # (N_fg, 4, 4)

fg_points_xyz1 = np.pad(fg_points[:, :3], pad_width=[(0, 0), (0, 1)], constant_values=1.)  # (N_fg, 4)
fg_points_xyz1 = np.einsum('BCD,BD -> BC', instances_traj_4fg, fg_points_xyz1)  # (N_fg, 4)
fg_points[:, :3] = fg_points_xyz1[:, :3]  # (N_fg, 3)

_points = np.concatenate([bg_points, fg_points], axis=0)
show_pointcloud(_points[:, :3], _boxes, fgr_mask=_points[:, -1] > -1)
