import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from pcdet.datasets.nuscenes.nuscenes_utils import quaternion_yaw


def print_record(rec, rec_type=None):
    print(f'--- {rec_type}' if rec_type is not None else '---')
    for k, v in rec.items():
        print(f"{k}: {v}")
    print('---\n')


def get_sample_location(sample_tk):
    sample_rec = nusc.get('sample', sample_tk)
    scene = nusc.get('scene', sample_rec['scene_token'])
    log = nusc.get('log', scene['log_token'])
    return log['location']


nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
nusc_map = NuScenesMap(dataroot='../data/nuscenes/v1.0-mini', map_name='singapore-onenorth')

scene = nusc.scene[0]
print_record(scene, 'scene')

sample_tk = scene['first_sample_token']
for _ in range(5):
    sample_rec = nusc.get('sample', sample_tk)
    sample_tk = sample_rec['next']

sample_rec = nusc.get('sample', sample_tk)
print_record(sample_rec, 'sample')

loc = get_sample_location(sample_tk)
print(f"sample loc: {loc}")

# TODO: get ego pose of LiDAR
sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
print_record(ego_pose, 'ego_pose')

# TODO: render driveable are in global frame & ego-pose frame
ego_loc = ego_pose['translation']
ego_yaw = quaternion_yaw(Quaternion(*ego_pose['rotation']))
print(ego_yaw)

layer_names = ['drivable_area', 'walkway']
patch_box = (ego_loc[0], ego_loc[1], 51.2*2, 51.2*2)
canvas_size = (512, 512)  # (height, width) of BEV image

map_mask = nusc_map.get_map_mask(patch_box, 0, layer_names, canvas_size)  # (n_layers, H, W)
print('map_mask: ', map_mask.shape)

map_mask_local = nusc_map.get_map_mask(patch_box, np.rad2deg(ego_yaw), layer_names, canvas_size)  # (n_layers, H, W)

fig, ax = plt.subplots(1, 2)
ax[0].set_title("driveable_area @ global")
ax[0].imshow(map_mask[0], cmap='gray')
ax[0].arrow(256, 256, 15 * np.cos(ego_yaw), 15 * np.sin(ego_yaw), color='r', width=7.)

ax[1].set_title("driveable_area @ ego")
ax[1].imshow(map_mask_local[0], cmap='gray')
ax[1].arrow(256, 256, 15, 0, color='r', width=7.)

_fig, _ax = plt.subplots()
nusc.render_sample_data(sample_rec['data']['CAM_FRONT'], ax=_ax)

plt.show()
