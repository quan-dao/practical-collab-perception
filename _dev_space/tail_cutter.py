import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from _dev_space.tail_cutter_utils import *


class PointAligner(nn.Module):
    def __init__(self, cfg, num_bev_features: int, voxel_size: list, point_cloud_range: list, class_names: list):
        super().__init__()
        self.cfg = cfg
        self.num_sweeps = cfg.NUM_SWEEPS
        self.return_corrected_pc = cfg.RETURN_CORRECTED_POINT_CLOUD
        self.thresh_motion_prob = cfg.THRESHOLD_MOTION_RPOB
        self.thresh_foreground_prob = cfg.THRESHOLD_FOREGROUND_RPOB
        self.voxel_size = voxel_size  # [vox_x, vox_y, vox_z]
        self.bev_image_stride = cfg.BEV_IMAGE_STRIDE
        self.point_cloud_range = point_cloud_range
        self.vehicle_class_indices = tuple([1 + class_names.index(cls_name) for cls_name in cfg.VEHICLE_CLASSES])

        num_pts_raw_feat = 1 + cfg.NUM_RAW_POINT_FEATURES  # batch_idx, x, y, z, intensity, time
        idx_offset = cfg.POINT_FEAT_INDEX_OFFSET_FROM_RAW_FEAT
        self.map_point_feat2idx = {
            'sweep_idx': num_pts_raw_feat + idx_offset.SWEEP_INDEX,
            'inst_idx': num_pts_raw_feat + idx_offset.INSTANCE_INDEX,
            'aug_inst_idx': num_pts_raw_feat + idx_offset.AUGMENTED_INSTANCE_INDEX,
            'cls_idx': num_pts_raw_feat + idx_offset.CLASS_INDEX,
        }

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                num_bev_features, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        # point heads
        num_points_features = self.model_cfg.SHARED_CONV_CHANNEL
        self.point_fg_seg = self._make_mlp(num_points_features, 1, cfg.get('HEAD_MID_CHANNELS', None))
        self.point_inst_assoc = self._make_mlp(num_points_features, 2, cfg.get('HEAD_MID_CHANNELS', None))

        # ---
        self.inst_global_mlp = self._make_mlp(num_points_features, cfg.INSTANCE_OUT_CHANNELS,
                                              cfg.get('INSTANCE_MID_CHANNELS', None))
        self.inst_local_mlp = self._make_mlp(3, cfg.INSTANCE_OUT_CHANNELS, cfg.get('INSTANCE_MID_CHANNELS', None))
        # input channels == 3 because: x - \bar{x}, y - \bar{y}, z - \bar{z}; \bar{} == center

        # ----
        # instance heads
        self.inst_motion_seg = self._make_mlp(cfg.INSTANCE_OUT_CHANNELS, 1,
                                              cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)

        self.inst_local_transl = self._make_mlp(6 + 3 * cfg.INSTANCE_OUT_CHANNELS, 3,
                                                cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # out == 3 for 3 components of translation vector

        self.inst_local_rot = self._make_mlp(6 + 3 * cfg.INSTANCE_OUT_CHANNELS, 4,
                                             cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # out == 4 for 4 components of quaternion
        fill_fc_weights(self.inst_local_rot)

        # ---
        self.forward_return_dict = dict()

        # loss func
        self.loss_weights = cfg.LOSS.LOSS_WEIGHTS
        self.eval_segmentation = cfg.LOSS.get('EVAL_SEGMENTATION_WHILE_TRAINING', False)
        self.seg_loss = CELovaszLoss(num_classes=2)  # binary classification for both fg seg & motion seg

        # TODO: use PCAccumulation cluster.py
        self.clusterer = DBSCAN(eps=cfg.CLUSTER.EPS, min_samples=cfg.CLUSTER.MIN_POINTS, metric='precomputed')

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

    def forward(self, batch_dict: dict, **kwargs):
        """
        Args:
            batch_dict:
                'points': (N, 6 + C) - batch_idx, x, y, z, intensity, time, [...]
        """
        # sanity check
        assert batch_dict['gt_boxes'].shape[1] == batch_dict['instances_tf'].shape[1], \
            f"{batch_dict['gt_boxes'].shape[1]} != {batch_dict['instances_tf'].shape[1]}"

        points = batch_dict['points']
        num_points = points.shape[0]

        self.forward_return_dict = {'prediction': {}}  # clear previous output

        spatial_features_2d = batch_dict['spatial_features_2d']
        bev_img = self.shared_conv(spatial_features_2d)  # (B, num_pts_feat, H, W)

        # interpolate points feature from bev_img
        points_batch_idx = points[:, 0].long()

        points_bev_coord = points.new_zeros(num_points, 2)
        points_bev_coord[:, 0] = \
            (points[:, 1] - self.point_cloud_range[0]) / (self.voxel_size[0] * self.bev_image_stride)
        points_bev_coord[:, 1] = \
            (points[:, 2] - self.point_cloud_range[1]) / (self.voxel_size[1] * self.bev_image_stride)

        points_feat = bev_img.new_zeros(num_points, bev_img.shape[1])
        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            cur_points_feat = bilinear_interpolate_torch(_img,
                                                         points_bev_coord[batch_mask, 0],
                                                         points_bev_coord[batch_mask, 1])
            points_feat[batch_mask] = cur_points_feat

        # invoke point heads
        pred_points_fg = self.point_fg_seg(points_feat)  # (N, 1)
        pred_points_inst_assoc = self.point_inst_assoc(points_feat)  # (N, 2) - x-,y-offset to mean of points in instance
        self.add_to_forward_return_dict_('fg_logits', pred_points_fg, 'prediction')
        self.add_to_forward_return_dict_('offset_to_centers', pred_points_inst_assoc, 'prediction')

        # ---------------------------------------------------------------
        # INSTANCE stuff
        # ---------------------------------------------------------------

        if self.training:
            gt_boxes_cls_idx = rearrange(batch_dict['gt_boxes'][:, :, -1].long(), 'B N_i -> (B N_i)')

            max_num_inst = batch_dict['gt_boxes'].shape[1]
            # max_num_inst is far bigger than number of instances of vehicle classes because instances_tf includes
            # tf of all 10 classes (meaning it includes pedestrian, barrier, traffic cone)

            points_merge_batch_aug_inst_idx = (points_batch_idx * max_num_inst
                                               + points[:, self.map_point_feat2idx['aug_inst_idx']].long())
            # Note: here all points, including those have aug_inst_idx == -1, are in play

            points_aug_cls_idx = gt_boxes_cls_idx[points_merge_batch_aug_inst_idx]
            # zero-out cls_idx of points whose aug_inst_idx == -1
            # Note: obj classes have index starting from 1 (ending at 10)
            points_aug_cls_idx[points[:, self.map_point_feat2idx['aug_inst_idx']].long() == -1] = 0

            # find points tagged with vehicle classes
            mask_fg = points_aug_cls_idx.new_zeros(num_points).bool()
            for vehicle_cls_idx in self.vehicle_class_indices:
                mask_fg = torch.logical_or(mask_fg, points_aug_cls_idx == vehicle_cls_idx)

            fg = batch_dict['points'][mask_fg]
            # (N_fg, 10) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_inst_idx, cls_idx

            fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)

            meta = self.build_meta_dict(fg, max_num_inst, self.map_point_feat2idx['aug_inst_idx'])

        else:
            raise NotImplementedError
            if self.thresh_foreground_prob > 0:
                mask_fg = rearrange(sigmoid(pred_points_fg), 'N 1 -> N') > self.thresh_foreground_prob
                mask_motion = rearrange(sigmoid(pred_points_mos), 'N 1 -> N') > self.thresh_motion_prob
                mask_fg = torch.logical_and(mask_fg, mask_motion)
                fg = batch_dict['points'][mask_fg]
                # (N_fg, 9) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_inst_idx
                # during testing, inst_idx & aug_inst_idx can be -1
                fg_batch_idx = points_batch_idx[mask_fg]  # (N_fg,)
                fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)
                fg_inst_idx = -fg.new_ones(fg.shape[0]).long()

                bg = batch_dict['points'][torch.logical_not(mask_fg)]
                # ==
                # DBSCAN to get instance_idx during inference
                # ==
                # apply inst_assoc
                fg_embed = fg[:, 1: 3] + pred_points_inst_assoc[mask_fg]  # (N_fg, 2)

                # invoke dbscan
                max_num_inst = 0
                for batch_idx in range(batch_dict['batch_size']):
                    mask_batch = fg_batch_idx == batch_idx
                    cur_fg_embed_numpy = fg_embed[mask_batch]  # (N_cur, 2)
                    if cur_fg_embed_numpy.shape[0] == 0 or cur_fg_embed_numpy.shape[0] > 15000:
                        # there is no fg for this data sample ==> no need for clustering
                        print(f"skipping | num detected fg: {cur_fg_embed_numpy.shape[0]} | "
                              f"num gt fg: {(batch_dict['points'][:, -2] > -1).long().sum()}")
                        continue
                    distances = rearrange(cur_fg_embed_numpy, 'N C -> N 1 C') - rearrange(cur_fg_embed_numpy, 'N C -> 1 N C')
                    distances = distances.square_().sum(dim=-1).sqrt_().detach().cpu().numpy()
                    self.clusterer.fit(distances)
                    fg_labels = self.clusterer.labels_  # (N_cur,)

                    # update fg_inst_idx
                    fg_inst_idx[mask_batch] = torch.from_numpy(fg_labels).long().to(fg.device)

                    # update max_num_inst
                    max_num_inst = max(max_num_inst, np.max(fg_labels))

                # remove noisy foreground (i.e. foreground that don't get assigned to any clusters)
                valid_fg = fg_inst_idx > -1
                if not torch.all(valid_fg):
                    # have some invalid fg -> add them to background
                    bg = torch.cat([bg, fg[torch.logical_not(valid_fg)]], dim=0)  # (N_bg, 9)
                fg = fg[valid_fg]  # (N_fg_valid, 9)
                fg[:, -1] = fg_inst_idx[valid_fg]  # overwrite fg's dummy instance index with DBSCAN result
                fg_feat = fg_feat[valid_fg]  # (N_fg_valid, C_bev)
                if max_num_inst > 0:
                    max_num_inst += 1  # because np.max(fg_labels) return the index (ranging from 0 to N_inst-1)
            else:
                # use for training detector, DBSCAN takes wait too long
                mask_fg = batch_dict['points'][:, -1].long() > -1  # aug instance_idx (here drop less heavily)
                fg = batch_dict['points'][mask_fg]
                # (N_fg, 9) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_inst_idx
                max_num_inst = batch_dict['instances_tf'].shape[1]
                fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)
                bg = batch_dict['points'][torch.logical_not(mask_fg)]

            if max_num_inst == 0:
                # not id any instance -> no need for correction
                # early return
                batch_dict['points'] = batch_dict['points'][:, :-3]
                return batch_dict

            meta = self.build_meta_dict(fg, max_num_inst, kwargs.get('use_augmented_instance_idx', True))

        # ------------
        # compute instance global feature
        fg_feat4glob = self.inst_global_mlp(fg_feat)  # (N_fg, C_inst)
        inst_global_feat = torch_scatter.scatter_max(fg_feat4glob, meta['inst_bi_inv_indices'], dim=0)[0]  # (N_inst, C_inst)

        # use inst_global_feat to predict motion stat
        pred_inst_motion_stat = self.inst_motion_seg(inst_global_feat)  # (N_inst, 1)
        self.add_to_forward_return_dict_('inst_motion_logits', pred_inst_motion_stat, 'prediction')

        # ------------
        # compute instance local shape encoding
        local_center = torch_scatter.scatter_mean(fg[:, 1: 4], meta['local_bisw_inv_indices'], dim=0)  # (N_local, 3)
        fg_centered_xyz = fg[:, 1: 4] - local_center[meta['local_bisw_inv_indices']]  # (N_fg, 3)
        fg_shape_enc = self.inst_local_mlp(fg_centered_xyz)  # (N_fg, C_inst)

        local_shape_enc = torch_scatter.scatter_max(fg_shape_enc, meta['local_bisw_inv_indices'], dim=0)[0]  # (N_local, C_inst)

        # ------------
        # compute instance shape encoding of the local @ target time step (i.e. local having the largest sweep idx)
        inst_target_center = local_center[meta['indices_local_to_inst_target']]  # (N_inst, 3)

        inst_target_center_shape = torch.cat(
            (inst_target_center, local_shape_enc[meta['indices_local_to_inst_target']]), dim=1)  # (N_inst, 3+C_inst)
        inst_global_feat = torch.cat((inst_global_feat, inst_target_center_shape), dim=1)  # (N_inst, 3+2*C_inst)

        # ------------------------------------
        # generate local_tf
        # ------------------------------------
        local_global_feat = inst_global_feat[meta['local_bi_in_inst_bi']]  # (N_local, 3+2*C_inst)

        # concatenate global feat with local shape enc to make local feat
        local_feat = torch.cat((local_global_feat, local_center, local_shape_enc), dim=1)  # (N_local, 6+3*C_inst)

        # use local_feat to predict local_tf of size (N_local, 3, 4)
        pred_local_transl = self.inst_local_transl(local_feat)  # (N_local, 3)
        self.add_to_forward_return_dict_('local_transl', pred_local_transl, 'prediction')

        pred_local_rot = self.inst_local_rot(local_feat)  # (N_local, 4)
        pred_local_rot = quat2mat(pred_local_rot)  # (N_local, 3, 3)
        self.add_to_forward_return_dict_('local_rot', pred_local_rot, 'prediction')

        # update forward_return_dict with prediction & target for computing loss
        if self.training:
            self.forward_return_dict['target'] = self.assign_target(batch_dict)
            target_meta = self.forward_return_dict['target']['meta']
            # below to ensure inst_ computed based on aug_inst_idx & inst_idx are consistent
            assert target_meta['inst_bi'].shape[0] == meta['inst_bi'].shape[0], \
                f"{target_meta['inst_bi'].shape[0]} == {meta['inst_bi'].shape[0]}"
            assert torch.all(target_meta['inst_bi'] == meta['inst_bi'])
        else:
            raise NotImplementedError
            _ = correct_point_cloud(
                fg,
                rearrange(sigmoid(pred_inst_motion_stat), 'N_inst 1 -> N_inst'),
                torch.cat([pred_dict['local_rot'], rearrange(pred_dict['local_transl'], 'N_local C -> N_local C 1')], dim=-1),
                meta
            )
            batch_dict['points'] = torch.cat([fg, bg], dim=0)

            mask_foreground = batch_dict['points'].new_zeros(batch_dict['points'].shape[0])
            mask_foreground[:fg.shape[0]] = 1.0

            if isinstance(sampled_points, torch.Tensor):
                batch_dict['points'] = torch.cat([batch_dict['points'], sampled_points], dim=0)
                mask_foreground = torch.cat([mask_foreground, sampled_points.new_ones(sampled_points.shape[0])], dim=0)

            # remove sweep_idx, inst_idx, aug_inst_idx
            batch_dict['points'] = batch_dict['points'][:, :-3]

            # store mask_foreground for debugging purpose
            batch_dict['debug_mask_foreground'] = mask_foreground

        return batch_dict

    @torch.no_grad()
    def assign_target(self, batch_dict):
        """
        Args:
            batch_dict:
                'points': (N, 8) - batch_idx, x, y, z, intensity, time, sweep_idx, instance_idx
                    * inst_indicator: -1 for background, >= 0 for foreground
                'instances_tf': (B, N_inst_max, N_sweeps, 3, 4)
        """
        points = batch_dict['points']
        num_points = points.shape[0]
        max_num_inst = batch_dict['gt_boxes'].shape[1]

        # -------------------------------------------------------
        # Point-wise target
        # use ORIGINAL instance index
        # -------------------------------------------------------

        # target foreground := points belong to vehicle class (use cls_idx to have the same effect as inst_idx)
        points_cls_idx = points[:, self.map_point_feat2idx['cls_idx']].long()
        mask_fg = points.new_zeros(num_points).bool()
        for vehicle_cls_idx in self.vehicle_class_indices:
            mask_fg = torch.logical_or(mask_fg, points_cls_idx == vehicle_cls_idx)
        target_fg = mask_fg.long()  # (N_pts,) - use original instance index for foreground seg - TODO: test by displaying

        # --------------
        # target of inst_assoc as offset toward mean of points inside each instance
        gt_boxes_xy = rearrange(batch_dict['gt_boxes'][:, :, :2].long(), 'B N_i C -> (B N_i) C')
        fg_merge_batch_inst_idx = (points[mask_fg, 0].long() * max_num_inst
                                   + points[mask_fg, self.map_point_feat2idx['inst_idx']].long())
        fg_boxes_xy = gt_boxes_xy[fg_merge_batch_inst_idx]  # (N_fg, 2)
        target_inst_assoc = fg_boxes_xy - points[mask_fg, 1: 3]  # (N_fg, 2) - TODO: test by displaying

        # -------------------------------------------------------
        # Instance-wise target
        # use inst_idx
        # -------------------------------------------------------
        instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)
        meta = self.build_meta_dict(points[mask_fg], max_num_inst, self.map_point_feat2idx['inst_idx'])

        # --------------
        # target motion
        inst_motion_stat = torch.linalg.norm(instances_tf[:, :, 0, :, -1], dim=-1) > 0.5  # translation more than 0.5m
        inst_motion_stat = rearrange(inst_motion_stat.long(), 'B N_inst_max -> (B N_inst_max)')
        inst_motion_stat = inst_motion_stat[meta['inst_bi']]  # (N_inst) - TODO: test by displaying

        # --------------
        # locals' transformation
        instances_tf = rearrange(instances_tf, 'B N_inst_max N_sweep C1 C2 -> (B N_inst_max N_sweep) C1 C2', C1=3, C2=4)
        local_tf = instances_tf[meta['local_bisw']]  # (N_local, 3, 4) - TODO: test by displaying oracle point cloud

        # format output
        target_dict = {
            'fg': target_fg,
            'inst_assoc': target_inst_assoc,
            'inst_motion_stat': inst_motion_stat,
            'local_tf': local_tf,
            'meta': meta
        }
        return target_dict

    def get_training_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = dict()

        pred_dict = self.forward_return_dict['prediction']
        target_dict = self.forward_return_dict['target']

        # -----------------------
        # Point-wise loss
        # -----------------------
        # ---
        # foreground seg
        fg_logits = pred_dict['fg_logits']  # (N_pts, 1)
        fg_target = target_dict['fg']  # (N_pts,)
        assert torch.all(torch.isfinite(fg_logits))
        loss_fg = self.seg_loss(fg_logits, fg_target, tb_dict, loss_name='fg') * self.loss_weights.FOREGROUND
        tb_dict['loss_fg'] = loss_fg.item()

        device = loss_fg.device

        # ---
        # instance assoc
        mask_fg = fg_target > 0
        inst_assoc = pred_dict['offset_to_centers']  # (N_pts, 2)
        if torch.any(mask_fg):
            inst_assoc_target = target_dict['inst_assoc']  # (N_fg, 2) ! target was already filtered by foreground mask
            loss_inst_assoc = nn.functional.smooth_l1_loss(inst_assoc[mask_fg], inst_assoc_target,
                                                           reduction='mean') * self.loss_weights.INSTANCE_ASSOC
        else:
            loss_inst_assoc = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['loss_inst_assoc'] = loss_inst_assoc.item()

        # -----------------------
        # Instance-wise loss
        # -----------------------
        meta = target_dict['meta']  # meta built based on inst_idx (not aug_inst_idx)

        # ---
        # motion seg
        inst_mos_logits = pred_dict['inst_motion_logits']  # (N_inst, 1)
        inst_mos_target = target_dict['inst_motion_stat']  # (N_inst,)
        if inst_mos_target.shape[0] > 0:
            loss_inst_mos = self.seg_loss(inst_mos_logits, inst_mos_target, tb_dict,
                                          loss_name='mos') * 2. * self.loss_weights.INSTANCE_MOTION_SEG
        else:
            loss_inst_mos = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['loss_inst_mos'] = loss_inst_mos.item()

        # ---
        # Local tf - regression loss | ONLY FOR LOCAL OF DYNAMIC INSTANCE
        local_transl = pred_dict['local_transl']  # (N_local, 3)
        local_rot_mat = pred_dict['local_rot']  # (N_local, 3, 3)

        local_tf_target = target_dict['local_tf']  # (N_local, 3, 4)

        assert torch.all(torch.isfinite(local_transl))
        assert torch.all(torch.isfinite(local_rot_mat))
        assert torch.all(torch.isfinite(local_tf_target))

        # which local are associated with dynamic instances
        local_bi_in_inst_bi = meta['local_bi_in_inst_bi']  # (N_local,)
        local_mos_target = inst_mos_target[local_bi_in_inst_bi]  # (N_local,)
        local_mos_mask = local_mos_target == 1  # (N_local,)

        if torch.any(local_mos_mask):
            # translation
            loss_local_transl = nn.functional.smooth_l1_loss(local_transl[local_mos_mask],
                                                             local_tf_target[local_mos_mask, :, -1],
                                                             reduction='mean') * 3. * self.loss_weights.LOCAL_TRANSLATION

            # rotation
            loss_local_rot = torch.linalg.norm(
                local_rot_mat[local_mos_mask] - local_tf_target[local_mos_mask, :, :3], dim=(1, 2), ord='fro'
            ).mean() * self.loss_weights.LOCAL_ROTATION
        else:
            loss_local_transl = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
            loss_local_rot = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['loss_local_transl'] = loss_local_transl.item()
        tb_dict['loss_local_rot'] = loss_local_rot.item()

        # ---
        # Local tf - reconstruction loss
        # get motion mask of foreground
        inst_bi_inv_indices = meta['inst_bi_inv_indices']  # (N_fg)
        fg_motion = inst_mos_target[inst_bi_inv_indices] == 1  # (N_fg)

        fg = batch_dict['points'][mask_fg]  # (N_fg)

        if torch.any(fg_motion):
            # extract dyn fg points
            dyn_fg = fg[fg_motion, 1: 4]  # (N_dyn, 3)

            # reconstruct with ground truth
            local_bisw_inv_indices = meta['local_bisw_inv_indices']
            gt_fg_tf = local_tf_target[local_bisw_inv_indices]  # (N_fg, 3, 4)
            gt_dyn_fg_tf = gt_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            gt_recon_dyn_fg = (torch.matmul(gt_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1)
                               + gt_dyn_fg_tf[:, :, -1])  # (N_dyn, 3)

            # reconstruct with prediction
            local_tf_pred = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = local_tf_pred[local_bisw_inv_indices]  # (N_fg, 3, 4)
            pred_dyn_fg_tf = pred_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            pred_recon_dyn_fg = (torch.matmul(pred_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1)
                                 + pred_dyn_fg_tf[:, :, -1])  # (N_dyn, 3)

            loss_recon = nn.functional.smooth_l1_loss(pred_recon_dyn_fg, gt_recon_dyn_fg,
                                                      reduction='mean') * 3.0 * self.loss_weights.RECONSTRUCTION
        else:
            loss_recon = torch.tensor(0.0, dtype=torch.float, requires_grad=True, device=device)
        tb_dict['loss_recon'] = loss_recon.item()

        # --------------
        # total loss
        # --------------
        loss = loss_fg + loss_inst_assoc + loss_inst_mos + loss_local_transl + loss_local_rot + loss_recon
        tb_dict['loss'] = loss.item()

        # eval foregound seg, motion seg during training
        if self.eval_segmentation:
            fg_precision, fg_recall = eval_binary_segmentation(fg_logits.detach().squeeze(-1), fg_target, False)
            tb_dict['fg_P'] = fg_precision
            tb_dict['fg_R'] = fg_recall

            mos_precision, mos_recall = eval_binary_segmentation(inst_mos_logits.detach().squeeze(-1),
                                                                 inst_mos_target, False)
            tb_dict['mos_P'] = mos_precision
            tb_dict['mos_R'] = mos_recall
        out = [loss, tb_dict]
        return out

    @torch.no_grad()
    def build_meta_dict(self, fg: torch.Tensor, max_num_instances: int, index_of_instance_idx: int) -> dict:
        """
        Args:
            fg: (N_fg, 10) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_idx, cls_idx
            max_num_instances:
            index_of_instance_idx: where is instance_index in point_feats
        """
        meta = {}

        fg_sweep_idx = fg[:, self.map_point_feat2idx['sweep_idx']].long()
        fg_batch_idx = fg[:, 0].long()
        fg_inst_idx = fg[:, index_of_instance_idx].long()

        # merge batch_idx & instance_idx
        fg_bi_idx = fg_batch_idx * max_num_instances + fg_inst_idx  # (N,)

        # merge batch_idx, instance_idx & sweep_idx
        fg_bisw_idx = fg_bi_idx * self.num_sweeps + fg_sweep_idx

        # group foreground points to instance
        inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
        # inst_bi: (N_inst,)
        # inst_bi_inv_indices: (N_fg,)

        # group foreground points to local group
        local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
        # local_bisw: (N_local,)

        meta.update({
            'max_num_inst': max_num_instances,
            'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices,
            'local_bisw': local_bisw, 'local_bisw_inv_indices': local_bisw_inv_indices
        })

        # -----------------------------------------------------------------------------
        # the following is to find the center of each instance's target local group
        # -----------------------------------------------------------------------------
        # get the max sweep_index of each instance
        inst_max_sweep_idx = torch_scatter.scatter_max(fg_sweep_idx, inst_bi_inv_indices)[0]  # (N_inst,)
        # get bisw_index of each instance's max sweep
        inst_target_bisw_idx = inst_bi * self.num_sweeps + inst_max_sweep_idx  # (N_inst,)
        # for each value in inst_target_bisw_idx find WHERE (i.e., index) it appear in local_bisw
        corr = local_bisw[:, None] == inst_target_bisw_idx[None, :]  # (N_local, N_inst)
        corr = corr.long() * torch.arange(local_bisw.shape[0]).unsqueeze(1).to(fg.device)
        meta['indices_local_to_inst_target'] = corr.sum(dim=0)  # (N_inst)

        # -----------------------------------------------------------------------------
        # the following is to establish correspondence between instances & locals
        # -----------------------------------------------------------------------------
        local_bi = local_bisw // self.num_sweeps
        # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
        local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
        local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1).to(fg.device)

        meta['local_bi_in_inst_bi'] = local_bi_in_inst_bi.sum(dim=0)  # (N_local)
        # this is identical to indices_instance_to_local which is used to
        # broadcast inst_global_feat from (N_inst, C_inst) to (N_local, C_inst)

        return meta

    def add_to_forward_return_dict_(self, name: str, value, name_sub_dict: str = None):
        if name_sub_dict is not None:
            self.forward_return_dict[name_sub_dict][name] = value
        else:
            self.forward_return_dict[name] = value

