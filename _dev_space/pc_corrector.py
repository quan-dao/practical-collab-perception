import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_min, scatter_mean
from einops import rearrange
from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from _dev_space.tail_cutter_utils import bilinear_interpolate_torch, eval_binary_segmentation
from typing import List


class PointCloudCorrector(nn.Module):
    def __init__(self, model_cfg, num_bev_features, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_sweeps = model_cfg.NUM_SWEEPS
        self.voxel_size = torch.tensor(voxel_size).float().cuda()  # [vox_x, vox_y, vox_z]
        self.bev_image_stride = model_cfg.BEV_IMAGE_STRIDE
        self.point_cloud_range = torch.tensor(point_cloud_range).float().cuda()

        num_pts_raw_feat = 1 + model_cfg.NUM_RAW_POINT_FEATURES  # batch_idx, x, y, z, intensity, time
        idx_offset = model_cfg.POINT_FEAT_INDEX_OFFSET_FROM_RAW_FEAT
        self.map_point_feat2idx = {
            'sweep_idx': num_pts_raw_feat + idx_offset.SWEEP_INDEX,
            'inst_idx': num_pts_raw_feat + idx_offset.INSTANCE_INDEX,
            'aug_inst_idx': num_pts_raw_feat + idx_offset.AUGMENTED_INSTANCE_INDEX,
            # 'cls_idx': num_pts_raw_feat + idx_offset.CLASS_INDEX,  # NOT USED!
        }

        self.forward_return_dict = dict()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                num_bev_features, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        # -------
        # Points Head
        # -------
        num_points_features = self.model_cfg.SHARED_CONV_CHANNEL
        self.points_seg = self._make_mlp(num_points_features, 3, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 3 cls:= bg, static fg, dynamic fg

        self.points_reg = self._make_mlp(num_points_features, 3, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 3:= offset_x, offset_y, offset_z

        self.points_embedding = self._make_mlp(num_points_features, 2, model_cfg.get('POINT_HEAD_MID_CHANNELS', None))
        # 2:= offset_toward_instance_center_x|_y

        # -------
        # Instances Head
        # -------
        self.local_shape_encoder = self._make_mlp(3, num_points_features,
                                                  model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))

        self.local_tf_decoder = self._make_mlp(2 * num_points_features + 6, 7,
                                               model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))
        # in := local_feat | global_feat | local_centroid | target_local_centroid
        # 7 := 3 (translation vector)  + 4 (quaternion)

        self.instance_motion_seg = self._make_mlp(num_points_features + 6, 1,
                                                  model_cfg.get('INSTANCE_HEAD_MID_CHANNELS', None))
        # in := global_feat | init_local_centroid | target_local_centroid

        # -------
        # Loss
        # -------
        self.loss_points_seg = CELovaszLoss(num_classes=3)
        self.loss_instance_mos = CELovaszLoss(num_classes=2)
        self.eval_segmentation = model_cfg.get('EVAL_SEGMENTATION_WHILE_TRAINING', False)  # TODO: multi class P/R

    @staticmethod
    def _make_mlp(in_c: int, out_c: int, mid_c: List = None, use_drop_out=False):
        if mid_c is None:
            mid_c = []
        channels = [in_c] + mid_c + [out_c]
        layers = []
        for c_idx in range(1, len(channels)):
            c_in = channels[c_idx - 1]
            c_out = channels[c_idx]
            is_last = c_idx == len(channels) - 1
            layers.append(nn.Linear(c_in, c_out, bias=is_last))
            if not is_last:
                layers.append(nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(True))
                if c_idx == len(channels) - 2 and use_drop_out:
                    layers.append(nn.Dropout(p=0.5))

        return nn.Sequential(*layers)

    def build_meta(self, foreground: torch.Tensor, max_num_instances_in_batch: int):
        fg_batch_inst_sw = (foreground[:, 0].long() * max_num_instances_in_batch
                            + foreground[:, self.map_point_feat2idx['inst_idx']].long()) * self.num_sweeps \
                           + foreground[:, self.map_point_feat2idx['sweep_idx']].long()
        # group foreground to locals
        locals_bis, locals2fg = torch.unique(fg_batch_inst_sw, sorted=True, return_inverse=True)

        # groups local to instance
        locals_batch_inst = torch.div(locals_bis, self.num_sweeps, rounding_mode='floor')
        instance_bi, inst2locals = torch.unique(locals_batch_inst, sorted=True, return_inverse=True)

        # find target local for each instance
        locals_sweep = locals_bis - locals_batch_inst * self.num_sweeps  # (N_local,)
        instance_max_sweep, indices_locals_max_sweep = scatter_max(locals_sweep, inst2locals, dim=0)
        # indices_locals_max_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        # find init local for each instance
        instance_min_sweep, indices_locals_min_sweep = scatter_min(locals_sweep, inst2locals, dim=0)
        # indices_locals_min_sweep: (N_inst,) - indices of locals that give max_sweep for each instance

        meta = {'locals2fg': locals2fg,
                'inst2locals': inst2locals,
                'indices_locals_max_sweep': indices_locals_max_sweep,
                'indices_locals_min_sweep': indices_locals_min_sweep}
        return meta

    def forward(self, batch_dict: dict):
        assert batch_dict['gt_boxes'].shape[1] == batch_dict['instances_tf'].shape[1], \
            f"{batch_dict['gt_boxes'].shape[1]} != {batch_dict['instances_tf'].shape[1]}"
        self.forward_return_dict = {}

        points = batch_dict['points']
        points_batch_idx = points[:, 0].long()
        num_points = points.shape[0]

        spatial_features_2d = batch_dict['spatial_features_2d']
        bev_img = self.shared_conv(spatial_features_2d)  # (B, num_pts_feat, H, W)

        points_bev_coord = (points[:, 1: 3] - self.point_cloud_range[:2]) / (self.voxel_size[:2] * self.bev_image_stride)
        points_feat = points.new_zeros(num_points, bev_img.shape[1])
        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            cur_points_feat = bilinear_interpolate_torch(_img, points_bev_coord[batch_mask, 0],
                                                         points_bev_coord[batch_mask, 1])
            points_feat[batch_mask] = cur_points_feat

        # -------
        # invoke Point Heads
        # -------
        points_cls_logit = self.points_seg(points_feat)  # (N, 3)
        points_offset = self.points_reg(points_feat)  # (N, 3)
        points_embedding = self.points_embedding(points_feat)  # (N, 2)

        prediction_dict = {
            'points_cls_logit': points_cls_logit,
            'points_offset': points_offset,
            'points_embedding': points_embedding
        }

        if self.training:
            # -------
            # instance stuff
            # -------
            fg_mask = points[:, self.map_point_feat2idx['inst_idx']] > -1  # all 10 classes
            fg = points[fg_mask]
            fg_feat = points_feat[fg_mask]  # (N_fg, C)

            meta = self.build_meta(fg, batch_dict['gt_boxes'].shape[1])

            # compute locals' shape encoding
            locals_centroid = scatter_mean(fg[:, 1: 4], meta['locals2fg'], dim=1)  # (N_local, 3)
            centered_fg = fg[:, 1: 4] - locals_centroid[meta['locals2fg']]  # (N_fg, 3)
            locals_shape_encoding = scatter_max(self.local_shape_encoder(centered_fg), meta['locals2fg'],
                                                dim=1)[0]  # (N_local, C)

            # compute locals feat
            locals_feat = scatter_max(fg_feat, meta['locals2fg'], dim=1)[0]  # (N_local, C)
            locals_feat = locals_feat + locals_shape_encoding  # (N_local, C)

            # compute globals stuff
            globals_feat = scatter_max(locals_feat, meta['inst2locals'], dim=1)[0]  # (N_global, C)
            globals_init_local_center = locals_centroid[meta['indices_locals_min_sweep']]  # (N_global, 3)
            globals_target_local_center = locals_centroid[meta['indices_locals_max_sweep']]  # (N_global, 3)

            # invoke instance head
            locals_feat = torch.cat((locals_feat,
                                     globals_feat[meta['inst2locals']],
                                     locals_centroid,
                                     globals_target_local_center[meta['inst2locals']]), dim=1)  # (N_local, 2*C + 6)
            locals_tf = self.local_tf_decoder(locals_feat)  # (N_local, 7)

            inst_feat = torch.cat((globals_feat, globals_init_local_center, globals_target_local_center), dim=1)
            inst_mos = self.instance_motion_seg(inst_feat)  # (N_global, 1)

            prediction_dict.update({
                'locals_tf': locals_tf,
                'inst_mos': inst_mos
            })

            # invoke assign target
            target_dict = self.assign_target(batch_dict, meta)

            # save prediction & target for computing loss
            self.forward_return_dict.update({
                'prediction': prediction_dict,
                'target': target_dict
            })

        else:
            pass  # TODO: apply points_offset on dynamic foreground points

    def assign_target(self, batch_dict, meta):
        pass





