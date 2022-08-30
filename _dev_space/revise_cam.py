import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from pcdet.models.detectors.detector3d_template import Detector3DTemplate

from sklearn.cluster import DBSCAN
from _dev_space.bev_segmentation_utils import sigmoid


class PointCloudCorrector(Detector3DTemplate):
    def __init__(self):
        self.ckpt = './from_idris/ckpt/bev_seg_focal_fullnusc_ep5.pth'
        self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        self.voxel_size = np.array([0.2, 0.2, 8.0])
        cls_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                     'pedestrian', 'traffic_cone']
        self.model_cfg = edict({
            'NAME': 'BEVSegmentation',
            'DEBUG': True,
            'VFE': {
                'NAME': 'DynPillarVFE',
                'WITH_DISTANCE': False,
                'USE_ABSLOTE_XYZ': True,
                'USE_NORM': True,
                'NUM_FILTERS': [64, 64]
            },
            'MAP_TO_BEV': {
                'NAME': 'PointPillarScatter',
                'NUM_BEV_FEATURES': 64
            },
            'BACKBONE_2D': {
                'NAME': 'PoseResNet',
                'NUM_FILTERS': [64, 128, 256],
                'LAYERS_SIZE': [2, 2, 2],
                'HEAD_CONV': 64,
                'BEV_IMG_STRIDE': 2,
                'FOREGROUND_SEG_LOSS_WEIGHTS': [1.0, 0.1],
            }
        })
        self.dataset = edict({
            'class_names': cls_names,
            'point_cloud_range': self.point_cloud_range,
            'voxel_size': self.voxel_size,
            'depth_downsample_factor': None,
            'point_feature_encoder': {'num_point_features': 5}
        })
        self.dataset.grid_size = np.round((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(int)
        super().__init__(self.model_cfg, len(cls_names), self.dataset)
        self.module_list = self.build_networks()
        self.eval()
        self.load_weights()
        self.cuda()

        # ---
        # Cluster-and-move
        # ---
        self.scanner = DBSCAN(eps=1.0, min_samples=3)
        bev_img_size = self.dataset.grid_size[:2] / self.model_cfg.BACKBONE_2D.BEV_IMG_STRIDE  # size_x, size_y
        self.xx, self.yy = np.meshgrid(np.arange(bev_img_size[0]), np.arange(bev_img_size[1]))
        self.fgr_prob_threshold = 0.5
        self.velo_std_dev_threshold = 4.5

    def load_weights(self):
        checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
        if self.model_cfg.get('DEBUG', False):
            print(f'Distillation | ==> Loading parameters from checkpoint {self.ckpt} to CPU')
        model_state_dict = checkpoint['model_state']
        state_dict, update_model_state = self._load_state_dict(model_state_dict, strict=False)

        if self.model_cfg.get('DEBUG', False):
            for key in state_dict:
                if key not in update_model_state:
                    print('Distillation | Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
            print('Distillation | ==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    @torch.no_grad()
    def forward(self, batch_dict):
        self.eval()

        # hold out points sampled from database
        points = batch_dict['points']  # (N_tot, 7) - batch_idx, xyz, intensity, time, indicator
        # indicator: -1=original bgr | 0=original fgr | >=1 = sampled (from database) fgr
        print(f'DEBUGGING | num original points: {points.shape[0]}')

        mask_sampled_points = points[:, -1].int() > 0
        sampled_points = points[mask_sampled_points]  # (N_sampled, 7)
        points = points[torch.logical_not(mask_sampled_points)]  # (N, 7)
        # overwrite batch_dict's 'points' with original 'points' only
        batch_dict['points'] = points

        # ---
        # invoke forward pass of BEVSegmentation net
        # ---
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # ---
        # cluster foreground segmented BEV images
        # ---
        pred_dict = batch_dict['bev_pred_dict']
        pred_cls = sigmoid(pred_dict['bev_cls_pred'].contiguous())  # (B, 1, 256, 256) | stride (w.r.t input of BEVSeg net) = 2
        pred_reg = pred_dict['bev_reg_pred'].contiguous()  # (B, 2, 256, 256)

        # move to cpu
        pred_cls = pred_cls.cpu().numpy()
        pred_reg = pred_reg.cpu().numpy()
        points = points.cpu().numpy()  # (N, 7) - batch_idx, xyz, intensity, time, indicator
        points_batch_idx = points[:, 0].astype(int)

        # ---
        # clustering & correcting
        # ---
        pred_cls = pred_cls > self.fgr_prob_threshold  # bool - (B, 1, 256, 256)
        batch_crt_points = []
        for b_idx in range(batch_dict['batch_size']):
            cur_points = points[points_batch_idx == b_idx, 1:]  # xyz, intensity, time, indicator

            mask_pred_fgr = pred_cls[b_idx, 0] & (np.linalg.norm(pred_reg[b_idx, :], axis=0) < 10)  # (256, 256)
            if not np.any(mask_pred_fgr):
                batch_crt_points.append(np.pad(cur_points, [(0, 0), (1, 0)], mode='constant', constant_values=b_idx))
                continue

            # clustering
            fgr_xy = np.stack([self.xx[mask_pred_fgr], self.yy[mask_pred_fgr]], axis=1)  # (N_fgr, 2)
            vector_fgr2centroid = pred_reg[b_idx, :, mask_pred_fgr]  # (2, N_fgr)
            # print(f'DEBUGGING | pred_reg: {pred_reg.shape}')
            # print(f'DEBUGGING | mask_pred_fgr: {mask_pred_fgr.shape}')
            # print(f'DEBUGGING | fgr_xy: {fgr_xy.shape}')
            # print(f'DEBUGGING | vector_fgr2centroid: {vector_fgr2centroid.shape}')
            moved_fgr = fgr_xy + vector_fgr2centroid  # (N_fgr, 2)
            self.scanner.fit(moved_fgr)

            # correcting
            crt_cur_points, offset_cur_points = self.correct_points3d(
                cur_points, self.scanner.labels_, fgr_xy,
                pc_range=self.point_cloud_range,
                bev_pix_size=self.voxel_size[0] * self.model_cfg.BACKBONE_2D.BEV_IMG_STRIDE,
                threshold_velo_std_dev=self.velo_std_dev_threshold
            )

            batch_crt_points.append(np.pad(crt_cur_points, [(0, 0), (1, 0)], mode='constant', constant_values=b_idx))

        # organize output:
        # put sampled_points back to batch_dict after correction
        batch_crt_points = torch.from_numpy(np.vstack(batch_crt_points)).cuda()
        batch_dict['points'] = torch.cat([batch_crt_points, sampled_points], dim=0) if sampled_points.shape[0] > 0 \
            else batch_crt_points
        print(f"DEBUGGING | num points after correction: {batch_dict['points'].shape[0]}")
        return batch_dict

    @staticmethod
    def correct_points3d(points3d: np.ndarray, bev_fgr_labels: np.ndarray, bev_fgr_xy: np.ndarray,
                         pc_range: np.ndarray, bev_pix_size: float, threshold_velo_std_dev: float):
        assert points3d.shape[1] == 6, f'current shape {points3d.shape}, remember of exclude batch_idx'
        assert len(bev_fgr_labels.shape) == 1, bev_fgr_labels.shape[0] == bev_fgr_xy.shape[0]
        assert len(bev_fgr_xy.shape) == 2 and bev_fgr_xy.shape[1] == 2

        pts_pixel_coord = np.floor((points3d[:, :2] - pc_range[:2]) / bev_pix_size).astype(int)

        mask_corrected_pts = np.zeros(points3d.shape[0], dtype=bool)  # to keep track of pts that are corrected
        clusters, clusters_offset_xy = [], []
        all_cluster_ids = np.unique(bev_fgr_labels)
        for cluster_id in all_cluster_ids:
            if cluster_id == -1:
                # noise cluster -> skip
                continue

            # ---
            # find pts (in 3D) that are contained by this cluster (currently expressed in BEV)
            # ---
            fgr_in_cluster = bev_fgr_xy[bev_fgr_labels == cluster_id]
            # smallest bounding rectangle (in BEV) of this cluster
            min_xy = np.amin(fgr_in_cluster, axis=0)
            max_xy = np.amax(fgr_in_cluster, axis=0)
            # find pts whose pixel coords fall inside the smallest bounding rectangle
            mask_pts_in_cluster = np.all((pts_pixel_coord >= min_xy) & (pts_pixel_coord <= max_xy), axis=1)

            # decide whether to correct cluster based on area
            cluster_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]) * (bev_pix_size ** 2)
            if cluster_area > 60:  # original area threshold is 120 when bev stride = 1, now bev stride = 2 -> reduce
                # too large -> contains more than 1 obj -> wrong cluster -> not correct this cluster
                continue

            # correct this cluster in a sequential manner
            pts_in_cluster = points3d[mask_pts_in_cluster]  # (N_p, 3+C)
            pts_timestamp = pts_in_cluster[:, 4]
            unq_ts = np.unique(pts_timestamp).tolist()
            unq_ts.reverse()  # to keep the most recent timestamp at the last position
            window, window_offset_xy, velos = np.array([]), np.array([]), []
            for ts_idx in range(len(unq_ts)):
                cur_group = np.copy(pts_in_cluster[pts_timestamp == unq_ts[ts_idx]])
                if window.size == 0:
                    window = cur_group
                    window_offset_xy = np.zeros((cur_group.shape[0], 2))
                    continue
                # calculate vector going from window's center toward cur_group's center
                win_to_cur = np.mean(cur_group[:, :2], axis=0) - np.mean(window[:, :2], axis=0)
                # calculate window's velocity
                delta_t = -unq_ts[ts_idx] + unq_ts[ts_idx - 1]
                velos.append(np.linalg.norm(win_to_cur) / delta_t)
                # correct window & merge with cur_group
                window[:, :2] += win_to_cur
                window_offset_xy += win_to_cur  # to keep track of how points have been moved
                window = np.vstack([window, cur_group])
                window_offset_xy = np.vstack([window_offset_xy, np.zeros((cur_group.shape[0], 2))])

            # decide whether to keep corrected cluster based on std dev of velocity
            if not velos:
                # empty velos cuz cluster only has 1 group (i.e. 1 timestamp)
                continue

            velo_var = np.mean(np.square(np.array(velos) - np.mean(velos)))
            if np.sqrt(velo_var) > threshold_velo_std_dev:
                # too large -> not keep corrected cluster
                continue
            else:
                # keep corrected cluster
                mask_corrected_pts = mask_corrected_pts | mask_pts_in_cluster
                clusters.append(window)
                clusters_offset_xy.append(window_offset_xy)

        # format output
        out_pts3d = np.vstack((*clusters, points3d[~mask_corrected_pts]))
        n_not_corrected_pts = np.logical_not(mask_corrected_pts).astype(int).sum()
        out_offset_xy = np.vstack((*clusters_offset_xy, np.zeros((n_not_corrected_pts, 2))))

        return out_pts3d, out_offset_xy
