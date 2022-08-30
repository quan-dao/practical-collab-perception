import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from pcdet.models.detectors.detector3d_template import Detector3DTemplate


class PointCloudCorrector(Detector3DTemplate):
    def __init__(self):
        self.ckpt = './from_idris/ckpt/bev_seg_focal_fullnusc_ep5.pth'
        point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        voxel_size = np.array([0.2, 0.2, 8.0])
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
            'point_cloud_range': point_cloud_range,
            'voxel_size': voxel_size,
            'depth_downsample_factor': None,
            'point_feature_encoder': {'num_point_features': 5}
        })
        self.dataset.grid_size = np.round((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)
        super().__init__(self.model_cfg, len(cls_names), self.dataset)
        self.module_list = self.build_networks()
        self.eval()
        self.load_weights()
        self.cuda()

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
        # ---
        # invoke forward pass of
        # ---
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        return batch_dict




