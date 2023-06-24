import torch
import torch.nn as nn
from easydict import EasyDict as edict
from typing import Dict
import os
import logging

from pcdet.models.backbones_3d import vfe
from pcdet.models.backbones_2d import map_to_bev
from pcdet.models import backbones_2d
from pcdet.utils.spconv_utils import find_all_spconv_keys


class BEVMaker(nn.Module):
    def __init__(self, model_cfg: edict, num_class: int, dataset, logger=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names

        self.module_topology = ['vfe', 'map_to_bev_module', 'backbone_2d']

        self.module_list = self.build_networks()
        self.maker_type = model_cfg.MAKER_TYPE

        # load weights
        if logger is None:
            logger = logging.getLogger()
        self.load_params_from_file(model_cfg.CKPT, logger, to_cpu=True)

        # freeze weights
        for param in self.parameters():
            param.requires_grad = False

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
            input_channels=model_info_dict.get('num_bev_features', None)
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

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('[TEACHER] ==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            print('[TEACHER] ==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('[TEACHER] Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('[TEACHER] ==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    @torch.no_grad()
    def forward_rsu_car(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.eval()
        
        points = batch_dict['points']  # (N, 1 + 5 + 1) - batch_idx, point-5, agent-idex  # TODO: organize dataset accordingly
        points_agent_idx = points[:, -1].long()
        
        unq_anget_idx = torch.unique(points_agent_idx).cpu().numpy()
        batch_dict['bev_img'] = dict()

        for agent_idx in unq_anget_idx:
            if agent_idx == 1:  
                # ego vehicle
                continue
            
            if self.maker_type == 'rsu' and agent_idx != 0:
                continue

            agent_points = points[points_agent_idx == agent_idx]

            agent_batch_dict = {'points': agent_points}
            for cur_module in self.module_list:
                agent_batch_dict = cur_module(agent_batch_dict)

            # clean up
            keys = list(agent_batch_dict.keys())
            for k in keys:
                if k not in ('spatial_features_2d',):
                    agent_batch_dict.pop(k)

            # save agent_spatial_features_2d for distillation
            batch_dict['bev_img'][agent_idx] = agent_batch_dict['spatial_features_2d']

        return batch_dict
    
    @torch.no_grad()
    def forward_early(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.eval()
        
        points = batch_dict['points']  # (N, 1 + 5 + 1) - batch_idx, point-5, agent-idex  # TODO: organize dataset accordingly
        points_agent_idx = points[:, -1].long()    
        # TODO
        return batch_dict
    
    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.maker_type != 'early':
            batch_dict = self.forward_rsu_car(batch_dict)
        else:
            batch_dict = self.forward_early(batch_dict)
        return batch_dict