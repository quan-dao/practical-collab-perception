from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from tools_box import get_sweeps_token, get_nuscenes_pointcloud_partitioned_by_instances, show_pointcloud, \
    get_boxes_4viz, get_target_sample_token, compute_bev_coord


NUM_SWEEPS = 10
PC_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
VOXEL_SIZE = 0.2


def main(scene_idx=0, target_sample_idx=10):
    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    sample_token = get_target_sample_token(nusc, scene_idx, target_sample_idx)
    sample_rec = nusc.get('sample', sample_token)
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc, ref_sd_token, NUM_SWEEPS, return_time_lag=True)

    background, instances = [], dict()
    for sd_token, sd_time in sd_tokens_times:
        curr_bgr, curr_instances = get_nuscenes_pointcloud_partitioned_by_instances(nusc, sd_token, ref_sd_token,
                                                                                    sd_time)
        background.append(curr_bgr)
        for inst_token, inst_pts in curr_instances.items():
            if inst_token not in instances:
                instances[inst_token] = [inst_pts]
            else:
                instances[inst_token].append(inst_pts)

    background = np.vstack(background)
    fgr, fgr_feat = [], []
    for idx, inst_tk in enumerate(instances.keys()):
        instances[inst_tk] = np.vstack(instances[inst_tk])
        offset_to_center = np.mean(instances[inst_tk][:, :2], axis=0) - instances[inst_tk][:, :2]
        fgr_feat.append(offset_to_center)
        fgr.append(instances[inst_tk])

    # compute bev
    bev_size = np.floor((PC_RANGE[3: 5] - PC_RANGE[0: 2]) / VOXEL_SIZE).astype(int)
    bev_bgr, _ = compute_bev_coord(background, PC_RANGE, VOXEL_SIZE)  # LABEL

    fgr = np.vstack(fgr)
    fgr_feat = np.vstack(fgr_feat)
    bev_fgr, bev_fgr_feat = compute_bev_coord(fgr, PC_RANGE, VOXEL_SIZE, pts_feat=fgr_feat)  # LABEL

    # ---
    # viz 2D
    # ---
    bev_img = np.zeros((bev_size[1], bev_size[0]))
    bev_img[bev_bgr[:, 1], bev_bgr[:, 0]] = 1

    bev_img_fgr = np.zeros((bev_size[1], bev_size[0]))
    bev_img_fgr[bev_fgr[:, 1], bev_fgr[:, 0]] = 1

    bev_mix = np.zeros((bev_size[1], bev_size[0]))
    bev_mix[bev_bgr[:, 1], bev_bgr[:, 0]] = 0.5
    bev_mix[bev_fgr[:, 1], bev_fgr[:, 0]] = 1

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(bev_img)

    ax[1].imshow(bev_img_fgr)
    for fidx in range(bev_fgr.shape[0]):
        ax[1].arrow(bev_fgr[fidx, 0], bev_fgr[fidx, 1], bev_fgr_feat[fidx, 0], bev_fgr_feat[fidx, 1])

    ax[2].imshow(bev_mix, cmap='gray')

    for title, _ax in zip(['background', 'foreground', 'mix'], ax):
        _ax.set_title(title)
    plt.show()

    # ---
    # viz 3D
    # ---
    instance_tokens = list(instances.keys())
    color_palette = np.array([plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, len(instance_tokens))])
    foreground, foreground_colors = [], []
    for inst_idx, inst_token in enumerate(instance_tokens):
        foreground.append(instances[inst_token])
        inst_color = np.tile(color_palette[inst_idx].reshape(1, -3), (instances[inst_token].shape[0], 1))
        foreground_colors.append(inst_color)

    foreground = np.vstack(foreground)
    foreground_colors = np.vstack(foreground_colors)

    show_pointcloud(np.vstack([background, foreground])[:, :3], get_boxes_4viz(nusc, ref_sd_token),
                    np.vstack([np.zeros((background.shape[0], 3)), foreground_colors]))


if __name__ == '__main__':
    main()
