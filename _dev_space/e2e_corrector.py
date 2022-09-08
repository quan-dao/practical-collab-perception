import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from pcdet.models.backbones_3d import vfe
from pcdet.models.backbones_2d import map_to_bev
from pcdet.models import backbones_2d
from pcdet.utils.spconv_utils import find_all_spconv_keys

from sklearn.cluster import DBSCAN
from _dev_space.bev_segmentation_utils import sigmoid


class PointCloudCorrectorE2E(nn.Module):
    def __init__(self, bev_seg_net_ckpt, return_offset=True, return_cluster_encoding=False):
        """
        Args:
            bev_seg_net_ckpt: path to BEVSegmetation checkpoint
            return_offset: return offset_x,_y predicted by BEVSegmentation
            return_cluster_encoding: TODO: cluster points based on offset to_mean predicted by BEVSegmentation & encode
                cluster index using sine wave (Attention is all you need-style)
        """
        if return_cluster_encoding:
            raise NotImplementedError
        self.ckpt = bev_seg_net_ckpt
        self.return_offset = return_offset
        self.return_cluster_encoding = return_cluster_encoding
        self.fgr_prob_threshold = 0.5
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
                'NUM_FILTERS': [64, 64],
                'NUM_RAW_POINT_FEATURES': 5
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
                'NUM_UP_FILTERS': [128, 64, 64],
                'BEV_IMG_STRIDE': 2,
                'LOSS_WEIGHTS': [1.0, 1.0, 1.0, 1.0, 1.0],
                'CRT_NUM_BINS': 40,
                'CRT_DIR_NUM_BINS': 80,
                'CRT_MAX_MAGNITUDE': 15.0,
            }
        })
        self.n_raw_pts_feat = self.model_cfg.VFE.NUM_RAW_POINT_FEATURES
        self.bev_img_pix_size = self.voxel_size[0] * self.model_cfg.BACKBONE_2D.BEV_IMG_STRIDE
        self._prefix_crt_cls2mag = \
            self.model_cfg.BACKBONE_2D.CRT_MAX_MAGNITUDE / \
            float(self.model_cfg.BACKBONE_2D.CRT_NUM_BINS * (self.model_cfg.BACKBONE_2D.CRT_NUM_BINS + 1))

        self.dataset = edict({
            'class_names': cls_names,
            'point_cloud_range': self.point_cloud_range,
            'voxel_size': self.voxel_size,
            'depth_downsample_factor': None,
            'point_feature_encoder': {'num_point_features': 5}
        })
        self.dataset.grid_size = np.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(int)

        # ---
        # build BEVSegmentation
        super().__init__()
        self.module_topology = ['vfe', 'map_to_bev_module', 'backbone_2d']
        self.module_list = self.build_networks()
        self.eval()
        self.load_weights()
        self.cuda()

    def load_weights(self):
        checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
        if self.model_cfg.get('DEBUG', False):
            print(f'BEVSegmentation | ==> Loading parameters from checkpoint {self.ckpt} to CPU')
        model_state_dict = checkpoint['model_state']
        state_dict, update_model_state = self._load_state_dict(model_state_dict, strict=False)

        if self.model_cfg.get('DEBUG', False):
            for key in state_dict:
                if key not in update_model_state:
                    print('BEVSegmentation | Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
            print('BEVSegmentation | ==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    @torch.no_grad()
    def forward(self, batch_dict):
        self.eval()
        # separate original points and points sampled from database
        pts_indicator = batch_dict['points'][:, -1].int()  # (N_tot,)
        pts_batch_idx = batch_dict['points'][:, 0].long()  # (N_tot,)
        mask_added_pts = pts_indicator.new_zeros(pts_indicator.shape[0]).bool()  # (N_tot,)
        for b_i in range(batch_dict['batch_size']):
            mask_batch = pts_batch_idx == b_i  # (N_tot,)
            mask_added_pts[mask_batch] = pts_indicator[mask_batch] > batch_dict['metadata'][b_i]['n_original_instances']

        pts_added = batch_dict['points'][mask_added_pts]  # (N_added, C)
        if self.return_offset:
            pts_added = torch.cat([
                pts_added[:, :1 + self.n_raw_pts_feat], pts_added.new_zeros(pts_added.shape[0], 2),
                pts_added[:, 1 + self.n_raw_pts_feat:]
            ], dim=1).contiguous()

        batch_dict['points'] = batch_dict['points'][torch.logical_not(mask_added_pts)]  # (N_original, C)
        pts_batch_idx = pts_batch_idx[torch.logical_not(mask_added_pts)]  # (N_ori)

        # ---
        # invoke forward pass of BEVSegmentation net
        # ---
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # ==================================
        # correction
        # ==================================
        bev_cls = sigmoid(batch_dict['bev_pred_dict']['pred_cls'])  # (B, 1, H, W)
        bev_crt_cls = batch_dict['bev_pred_dict']['pred_crt_cls']  # (B, D+1, H, W)
        bev_crt_dir_cls = batch_dict['bev_pred_dict']['pred_crt_dir_cls']  # (B, D_+1, H, W)
        bev_crt_dir_res = batch_dict['bev_pred_dict']['pred_crt_dir_res']  # (B, 1, H, W)

        # compute pixel coordinates in BEV image of points
        pts_pix_coord = torch.floor(
            (batch_dict['points'][:, 1: 3] - self.point_cloud_range[0]) / self.bev_img_pix_size).long()  # (N_ori, 2)

        # ---
        # find fgr points
        # ---
        pts_fgr_prob = bev_cls[pts_batch_idx, 0, pts_pix_coord[:, 1], pts_pix_coord[:, 0]]  # (N_ori,)
        pts_fgr_mask = pts_fgr_prob > self.fgr_prob_threshold  # (N_ori,)

        # ---
        # correction magnitude
        # ---
        pts_crt_prob = bev_crt_cls[pts_batch_idx, :, pts_pix_coord[:, 1], pts_pix_coord[:, 0]]  # (D+1, N_ori)
        pts_crt_cls = torch.argmax(pts_crt_prob, dim=1).float()  # (N_ori)
        pts_crt_magnitude = self._prefix_crt_cls2mag * pts_crt_cls * (pts_crt_cls + 1)  # (N_ori)
        # zero-out invalid magnitude (i.e. cls == CRT_NUM_BINS)
        pts_crt_magnitude = pts_crt_magnitude * (pts_crt_cls < self.model_cfg.BACKBONE_2D.CRT_NUM_BINS).float()

        # ---
        # correction direction
        # ---
        pts_crt_dir_prob = bev_crt_dir_cls[pts_batch_idx, :, pts_pix_coord[:, 1], pts_pix_coord[:, 0]]  # (D_+1, N_ori)
        pts_crt_dir_cls = torch.argmax(pts_crt_dir_prob, dim=1).float()  # (N_ori)
        pts_crt_angle = (2 * np.pi - 1e-3) * pts_crt_dir_cls / self.model_cfg.BACKBONE_2D.CRT_DIR_NUM_BINS

        pts_crt_res = bev_crt_dir_res[pts_batch_idx, 0, pts_pix_coord[:, 1], pts_pix_coord[:, 0]]  # (N_ori,)
        pts_crt_angle = pts_crt_angle + pts_crt_res

        # ---
        # apply correction to fgr points only
        # ---
        pts_crt = pts_crt_magnitude.unsqueeze(1) * torch.stack([torch.cos(pts_crt_angle), torch.sin(pts_crt_angle)], dim=1)  # (N_ori, 2)
        # zero-out correction for background
        pts_crt = pts_crt * pts_fgr_mask.float().unsqueeze(1)  # (N_ori, 2)

        batch_dict['points'][:, 1: 3] = batch_dict['points'][:, 1: 3] + pts_crt
        if self.return_offset:
            batch_dict['points'] = torch.cat([
                batch_dict['points'][:, :(1 + self.n_raw_pts_feat)], pts_crt, batch_dict['points'][:, (1 + self.n_raw_pts_feat):]],
                dim=1).contiguous()

        # format output
        if torch.any(mask_added_pts):
            batch_dict['points'] = torch.cat([batch_dict['points'], pts_added], dim=0).contiguous()

        return batch_dict

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
