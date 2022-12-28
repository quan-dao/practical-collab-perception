import torch
from kornia.losses.focal import BinaryFocalLossWithLogits
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from _dev_space.unet_2d import UNet2D
from _dev_space.resnet import PoseResNet
from _dev_space.loss_utils.pcaccum_ce_lovasz_loss import CELovaszLoss
from _dev_space.external_drop import DropBlock2d
from _dev_space.tail_cutter_utils import *


class PointAligner(nn.Module):
    def __init__(self, full_model_cfg):
        super().__init__()
        self.cfg = full_model_cfg

        pillar_cfg = full_model_cfg.PILLAR_ENCODER
        self.pillar_encoder = PillarEncoder(
            pillar_cfg.NUM_RAW_FEATURES, pillar_cfg.NUM_BEV_FEATURES, pillar_cfg.POINT_CLOUD_RANGE,
            pillar_cfg.VOXEL_SIZE)

        map_net_cfg = full_model_cfg.MAP_NET
        if map_net_cfg.ENABLE:
            if map_net_cfg.NAME == 'MapFusion':
                map_net_channels = [map_net_cfg.NUM_MAP_LAYERS] + map_net_cfg.MAP_FUSION_CHANNELS
                map_net_layers = []
                for ch_idx in range(len(map_net_channels) - 1):
                    is_last_layer = ch_idx == len(map_net_channels) - 2
                    map_net_layers.append(
                        nn.Conv2d(map_net_channels[ch_idx], map_net_channels[ch_idx + 1], 3, padding=1, bias=is_last_layer)
                    )
                    if not is_last_layer:
                        map_net_layers.append(nn.BatchNorm2d(map_net_channels[ch_idx + 1], eps=1e-3, momentum=0.01))
                        map_net_layers.append(nn.ReLU(True))
                self.map_net = nn.Sequential(*map_net_layers)
                num_map_features = map_net_cfg.MAP_FUSION_CHANNELS[-1]
            else:
                self.map_net = PoseResNet(map_net_cfg, map_net_cfg.NUM_MAP_LAYERS)
                num_map_features = self.map_net.num_out_features
            self.drop_map = DropBlock2d(map_net_cfg.DROP_PROB, map_net_cfg.DROP_BLOCK_SIZE)
        else:
            self.map_net, self.drop_map = None, None
            num_map_features = 0

        self.backbone2d = UNet2D(pillar_cfg.NUM_BEV_FEATURES + num_map_features, full_model_cfg.BEV_BACKBONE)
        num_point_features = self.backbone2d.n_output_feat

        # ------------------------------------------------------------------------------------------
        # Aligner stage 1
        # ------------------------------------------------------------------------------------------
        cfg = full_model_cfg.ALIGNER_STAGE_1
        self.max_num_sweeps = cfg.MAX_NUM_SWEEPS
        self.return_corrected_pc = cfg.RETURN_CORRECTED_POINT_CLOUD
        self.thresh_motion_prob = cfg.THRESHOLD_MOTION_RPOB
        self.thresh_foreground_prob = cfg.THRESHOLD_FOREGROUND_RPOB
        # point heads
        self.point_fg_seg = self._make_mlp(num_point_features, 1, cfg.get('HEAD_MID_CHANNELS', None))
        self.point_mos_seg = self._make_mlp(num_point_features, 1, cfg.get('HEAD_MID_CHANNELS', None))
        self.point_inst_assoc = self._make_mlp(num_point_features, 2, cfg.get('HEAD_MID_CHANNELS', None))

        # ---
        self.inst_global_mlp = self._make_mlp(num_point_features, cfg.INSTANCE_OUT_CHANNELS,
                                              cfg.get('INSTANCE_MID_CHANNELS', None))
        self.inst_local_mlp = self._make_mlp(3, cfg.INSTANCE_OUT_CHANNELS, cfg.get('INSTANCE_MID_CHANNELS', None))
        # input channels == 3 because: x - \bar{x}, y - \bar{y}, z - \bar{z}; \bar{} == center

        # ----
        # instance heads
        self.inst_motion_seg = self._make_mlp(cfg.INSTANCE_OUT_CHANNELS, 1,
                                              cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)

        self.inst_proposal_gen = self._make_mlp(3 + 2 * cfg.INSTANCE_OUT_CHANNELS, 8,
                                                cfg.get('INSTANCE_MID_CHANNELS', None), cfg.INSTANCE_HEAD_USE_DROPOUT)
        # in_ch == 3 + 2 * cfg.INSTANCE_OUT_CHANNELS because
        # 3 = |centroid of the local @ target time step|
        # 2 * cfg.INSTANCE_OUT_CHANNELS = |concatenation of feat of local @ target & global|
        # out_ch == 8 because c_x, c_y, c_z, d_x, d_y, d_z, sin_yaw, cos_yaw

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
        self.segmentation_loss_type = cfg.LOSS.SEGMENTATION_LOSS
        if self.segmentation_loss_type == 'focal':
            self.seg_loss = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction='sum')
        elif self.segmentation_loss_type == 'ce_lovasz':
            self.seg_loss = CELovaszLoss(num_classes=2)  # binary classification for both fg seg & motion seg
        else:
            raise NotImplementedError

        self.clusterer = DBSCAN(eps=cfg.CLUSTER.EPS, min_samples=cfg.CLUSTER.MIN_POINTS, metric='precomputed')
        # self.neighbors_finder = NearestNeighbors(radius=cfg.CLUSTER.EPS)

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
        for bidx in range(batch_dict['batch_size']):
            assert batch_dict['metadata'][bidx]['max_num_sweeps'] == self.max_num_sweeps, \
                f"{batch_dict['metadata'][bidx]['max_num_sweeps']} != {self.max_num_sweeps}"
        self.forward_return_dict = dict()  # clear previous output

        # exclude sampled points
        mask_sampled_points = batch_dict['points'][:, -2].long() == -2
        if torch.any(mask_sampled_points):
            sampled_points = batch_dict['points'][mask_sampled_points]  # (N_sp, 9)
            batch_dict['points'] = batch_dict['points'][torch.logical_not(mask_sampled_points)]
        else:
            sampled_points = None

        bev_img = self.pillar_encoder(batch_dict)  # (B, C_bev, H, W), (N, 2)-bev_x, bev_y

        # concatenate hd_map with bev_img before passing to backbone_2d
        if self.map_net is not None:
            map_img = self.map_net(batch_dict['img_map'])
            map_img = self.drop_map(map_img)
            bev_img = torch.cat([bev_img, map_img], dim=1)  # (B, 64 + C_map, H, W)

        bev_img = self.backbone2d(bev_img)  # (B, 64, H, W)

        # interpolate points feature from bev_img
        points_bev_coord = (batch_dict['points'][:, [1, 2]] - self.pillar_encoder.pc_range[:2]) / \
                           self.pillar_encoder.voxel_size[:2]
        points_batch_idx = batch_dict['points'][:, 0].long()
        points_feat = bev_img.new_zeros(points_batch_idx.shape[0], bev_img.shape[1])

        for b_idx in range(batch_dict['batch_size']):
            _img = rearrange(bev_img[b_idx], 'C H W -> H W C')
            batch_mask = points_batch_idx == b_idx
            cur_points_feat = bilinear_interpolate_torch(_img,
                                                         points_bev_coord[batch_mask, 0],
                                                         points_bev_coord[batch_mask, 1])
            points_feat[batch_mask] = cur_points_feat

        # invoke point heads
        pred_points_fg = self.point_fg_seg(points_feat)  # (N, 1)
        pred_points_mos = self.point_mos_seg(points_feat)  # (N, 1)
        pred_points_inst_assoc = self.point_inst_assoc(points_feat)  # (N, 2) - x-,y-offset to mean of points in instance
        pred_dict = {
            'fg': pred_points_fg,  # (N, 1)
            'points_mos': pred_points_mos,  # (N, 1)
            'inst_assoc': pred_points_inst_assoc  # (N, 2)
        }

        # to debug points_mos
        # batch_dict.update({
        #     'original_points': torch.clone(batch_dict['points']),
        #     'pred_fg_prob': sigmoid(pred_points_fg.squeeze(-1)),
        #     'pred_points_mos_prob': sigmoid(pred_points_mos.squeeze(-1))
        # })

        # ---------------------------------------------------------------
        # INSTANCE stuff
        # ---------------------------------------------------------------

        if self.training:
            raise ValueError('This branch is not designed for training Alinger. Switch to main-car-as-meta')
            # use AUGMENTED instance index
            if kwargs.get('use_augmented_instance_idx', True):
                mask_fg = batch_dict['points'][:, -1].long() > -1
            else:
                # use instance_index (just for debugging purpose)
                mask_fg = batch_dict['points'][:, -2].long() > -1

            fg = batch_dict['points'][mask_fg]
            # (N_fg, 9) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_inst_idx
            max_num_inst = batch_dict['instances_tf'].shape[1]
            fg_feat = points_feat[mask_fg]  # (N_fg, C_bev)
        else:
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
            print('no instance found, early return')
            batch_dict['points'] = batch_dict['points'][:, :-3]
            return batch_dict

        meta = self.build_meta_dict(fg, max_num_inst, kwargs.get('use_augmented_instance_idx', True))
        self.forward_return_dict['meta'] = meta

        # ------------
        # compute instance global feature
        fg_feat4glob = self.inst_global_mlp(fg_feat)  # (N_fg, C_inst)
        inst_global_feat = torch_scatter.scatter_max(fg_feat4glob, meta['inst_bi_inv_indices'], dim=0)[0]  # (N_inst, C_inst)

        # use inst_global_feat to predict motion stat
        pred_inst_motion_stat = self.inst_motion_seg(inst_global_feat)  # (N_inst, 1)
        pred_dict['inst_motion_stat'] = pred_inst_motion_stat

        # ------------
        # compute instance local shape encoding
        local_center = torch_scatter.scatter_mean(fg[:, 1: 4], meta['local_bisw_inv_indices'], dim=0)  # (N_local, 3)
        fg_centered_xyz = fg[:, 1: 4] - local_center[meta['local_bisw_inv_indices']]  # (N_fg, 3)
        fg_shape_enc = self.inst_local_mlp(fg_centered_xyz)  # (N_fg, C_inst)

        local_shape_enc = torch_scatter.scatter_max(fg_shape_enc, meta['local_bisw_inv_indices'], dim=0)[0]  # (N_local, C_inst)

        # ------------
        # compute instance shape encoding of the local @ target time step (i.e. local having the largest sweep idx)
        inst_target_center = local_center[meta['indices_local_to_inst_target']]  # (N_inst, 3)
        self.forward_return_dict['meta']['inst_target_center'] = inst_target_center  # (N_inst, 3)

        inst_target_center_shape = torch.cat(
            (inst_target_center, local_shape_enc[meta['indices_local_to_inst_target']]), dim=1)  # (N_inst, 3+C_inst)
        inst_global_feat = torch.cat((inst_global_feat, inst_target_center_shape), dim=1)  # (N_inst, 3+2*C_inst)

        # ------------------------------------
        # generate 3D proposals
        # ------------------------------------
        proposals = self.inst_proposal_gen(inst_global_feat)  # (N_inst, 8)
        pred_dict['proposals'] = proposals  # (N_inst, 8)

        # ------------------------------------
        # generate local_tf
        # ------------------------------------
        local_global_feat = inst_global_feat[meta['local_bi_in_inst_bi']]  # (N_local, 3+2*C_inst)

        # concatenate global feat with local shape enc to make local feat
        local_feat = torch.cat((local_global_feat, local_center, local_shape_enc), dim=1)  # (N_local, 6+3*C_inst)

        # use local_feat to predict local_tf of size (N_local, 3, 4)
        pred_local_transl = self.inst_local_transl(local_feat)  # (N_local, 3)
        pred_dict['local_transl'] = pred_local_transl

        pred_local_rot = self.inst_local_rot(local_feat)  # (N_local, 4)
        pred_dict['local_rot'] = quat2mat(pred_local_rot)  # (N_local, 3, 3)

        # update forward_return_dict with prediction & target for computing loss
        self.forward_return_dict['pred_dict'] = pred_dict
        if self.training:
            self.forward_return_dict['target_dict'] = self.assign_target(batch_dict)
        else:
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
        # -------------------------------------------------------
        # Point-wise target
        # use ORIGINAL instance index
        # -------------------------------------------------------
        # --------------
        # target foreground
        mask_fg = points[:, -2].long() > -1
        target_fg = mask_fg.long()  # (N,) - use original instance index for foreground seg

        # --------------
        # target of inst_assoc as offset toward mean of points inside each instance
        _fg = points[mask_fg]
        max_num_inst = batch_dict['instances_tf'].shape[1]
        _fg_bi_idx = _fg[:, 0].long() * max_num_inst + _fg[:, -2].long()
        _, _inst_bi_inv_indices = torch.unique(_fg_bi_idx, sorted=True, return_inverse=True)

        inst_mean_xy = torch_scatter.scatter_mean(points[mask_fg, 1: 3], _inst_bi_inv_indices, dim=0)  # (N_inst, 2)
        target_inst_assoc = inst_mean_xy[_inst_bi_inv_indices] - points[mask_fg, 1: 3]

        # -------------------------------------------------------
        # Instance-wise target
        # use AUGMENTED instance index
        # -------------------------------------------------------
        instances_tf = batch_dict['instances_tf']  # (B, N_inst_max, N_sweep, 3, 4)

        # --------------
        inst_bi = self.forward_return_dict['meta']['inst_bi']
        # target motion
        inst_motion_stat = torch.linalg.norm(instances_tf[:, :, 0, :, -1], dim=-1) > self.thresh_motion_prob
        inst_motion_stat = rearrange(inst_motion_stat.long(), 'B N_inst_max -> (B N_inst_max)')
        inst_motion_stat = inst_motion_stat[inst_bi]  # (N_inst)

        # target proposal
        gt_boxes = batch_dict['gt_boxes']  # (B, N_inst_max, 11) -center (3), size (3), yaw, dummy_v (2), instance_index, class
        assert gt_boxes.shape[1] == instances_tf.shape[1], f"{gt_boxes.shape[1]} != {instances_tf.shape[1]}"
        batch_boxes = rearrange(gt_boxes, 'B N_inst_max C -> (B N_inst_max) C')
        batch_boxes = batch_boxes[inst_bi]  # (N_inst, 11)

        target_proposals = {
            'offset': batch_boxes[:, :3] - self.forward_return_dict['meta']['inst_target_center'],
            'size': batch_boxes[:, 3: 6],
            'ori': torch.stack([torch.sin(batch_boxes[:, 6]), torch.cos(batch_boxes[:, 6])], dim=1)
        }

        # --------------
        # locals' transformation
        local_bisw = self.forward_return_dict['meta']['local_bisw']
        # local_bisw: (N_local,)

        local_tf = rearrange(instances_tf, 'B N_inst_max N_sweep C1 C2 -> (B N_inst_max N_sweep) C1 C2', C1=3, C2=4)
        local_tf = local_tf[local_bisw]  # (N_local, 3, 4)

        # format output
        target_dict = {
            'fg': target_fg,
            'inst_assoc': target_inst_assoc,
            'inst_motion_stat': inst_motion_stat,
            'local_tf': local_tf,
            'proposals': target_proposals
        }
        return target_dict

    def get_training_loss(self, batch_dict, tb_dict=None):
        if tb_dict is None:
            tb_dict = dict()

        pred_dict = self.forward_return_dict['pred_dict']
        target_dict = self.forward_return_dict['target_dict']

        # -----------------------
        # Point-wise loss
        # -----------------------
        # ---
        # foreground seg
        fg_logit = pred_dict['fg']  # (N, 1)
        fg_target = target_dict['fg']  # (N,)
        assert torch.all(torch.isfinite(fg_logit))

        num_gt_fg = fg_target.sum().item()
        if self.segmentation_loss_type == 'focal':
            loss_fg = self.seg_loss(fg_logit, fg_target[:, None].float()) / max(1., num_gt_fg)
        else:
            loss_fg = self.seg_loss(fg_logit, fg_target, tb_dict, loss_name='fg')
        loss_fg = loss_fg * self.loss_weights.FOREGROUND
        tb_dict['loss_fg'] = loss_fg.item()

        # ---
        # instance assoc
        inst_assoc = pred_dict['inst_assoc']  # (N, 2)
        if num_gt_fg > 0:
            inst_assoc_target = target_dict['inst_assoc']  # (N_fg, 2) ! target was already filtered by foreground mask
            loss_inst_assoc = l2_loss(inst_assoc[fg_target == 1], inst_assoc_target, dim=1, reduction='mean')
        else:
            loss_inst_assoc = l2_loss(inst_assoc, torch.clone(inst_assoc).detach(), dim=1, reduction='mean')
        loss_inst_assoc = loss_inst_assoc * self.loss_weights.INSTANCE_ASSOC
        tb_dict['loss_inst_assoc'] = loss_inst_assoc.item()

        # -----------------------
        # Instance-wise loss
        # -----------------------
        # ---
        # motion seg
        inst_mos_logit = pred_dict['inst_motion_stat']  # (N_inst, 1)
        inst_mos_target = target_dict['inst_motion_stat']  # (N_inst,)

        num_gt_dyn_inst = float(inst_mos_target.sum().item())
        if self.segmentation_loss_type == 'focal':
            loss_inst_mos = self.seg_loss(inst_mos_logit, inst_mos_target[:, None].float()) / max(1., num_gt_dyn_inst)
        else:
            loss_inst_mos = self.seg_loss(inst_mos_logit, inst_mos_target, tb_dict, loss_name='mos')
        loss_inst_mos = loss_inst_mos * self.loss_weights.INSTANCE_MOTION_SEG
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
        with torch.no_grad():
            local_bi_in_inst_bi = self.forward_return_dict['meta']['local_bi_in_inst_bi']  # (N_local,)
            local_mos_target = inst_mos_target[local_bi_in_inst_bi]  # (N_local,)
            local_mos_mask = local_mos_target == 1  # (N_local,)

        # translation
        # logger = logging.getLogger()
        if torch.any(local_mos_mask):
            # logger.info('loss_local_transl has ground truth')
            loss_local_transl = l2_loss(local_transl[local_mos_mask], local_tf_target[local_mos_mask, :, -1],
                                        dim=-1, reduction='mean') * self.loss_weights.LOCAL_TRANSLATION
        else:
            # logger.info('loss_local_transl does not have ground truth')
            loss_local_transl = l2_loss(local_transl, torch.clone(local_transl).detach(), dim=-1, reduction='mean')
        tb_dict['loss_local_transl'] = loss_local_transl.item()

        # rotation
        if torch.any(local_mos_mask):
            # logger.info('loss_local_rot has ground truth')
            loss_local_rot = torch.linalg.norm(
                local_rot_mat[local_mos_mask] - local_tf_target[local_mos_mask, :, :3], dim=(1, 2), ord='fro'
            ).mean() * self.loss_weights.LOCAL_ROTATION
        else:
            # logger.info('loss_local_rot does not have ground truth')
            loss_local_rot = l2_loss(local_rot_mat.reshape(-1, 1), torch.clone(local_rot_mat).detach().reshape(-1, 1),
                                     dim=-1, reduction='mean')
        tb_dict['loss_local_rot'] = loss_local_rot.item()

        # ---
        # Local tf - reconstruction loss

        # get motion mask of foreground
        inst_bi_inv_indices = self.forward_return_dict['meta']['inst_bi_inv_indices']  # (N_fg)
        fg_motion = inst_mos_target[inst_bi_inv_indices] == 1  # (N_fg)

        aug_fg_mask = batch_dict['points'][:, -1].long() > -1  # (N,) - use aug inst index
        fg = batch_dict['points'][aug_fg_mask]  # (N_fg)

        if torch.any(fg_motion):
            # extract dyn fg points
            dyn_fg = fg[fg_motion, 1: 4]  # (N_dyn, 3)

            # reconstruct with ground truth
            local_bisw_inv_indices = self.forward_return_dict['meta']['local_bisw_inv_indices']
            gt_fg_tf = local_tf_target[local_bisw_inv_indices]  # (N_fg, 3, 4)
            gt_dyn_fg_tf = gt_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            gt_recon_dyn_fg = torch.matmul(gt_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1) + \
                              gt_dyn_fg_tf[:, :, -1]  # (N_dyn, 3)

            # reconstruct with prediction
            local_tf_pred = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = local_tf_pred[local_bisw_inv_indices]  # (N_fg, 3, 4)
            pred_dyn_fg_tf = pred_fg_tf[fg_motion]  # (N_dyn, 3, 4)

            pred_recon_dyn_fg = torch.matmul(pred_dyn_fg_tf[:, :, :3], dyn_fg[:, :, None]).squeeze(-1) + \
                                pred_dyn_fg_tf[:, :, -1]  # (N_dyn, 3)

            loss_recon = l2_loss(pred_recon_dyn_fg, gt_recon_dyn_fg, dim=-1,
                                 reduction='mean') * self.loss_weights.RECONSTRUCTION
        else:
            # there is no moving fg -> dummy loss
            local_bisw_inv_indices = self.forward_return_dict['meta']['local_bisw_inv_indices']
            local_tf_pred = torch.cat([local_rot_mat, local_transl.unsqueeze(-1)], dim=-1)  # (N_local, 3, 4)
            pred_fg_tf = local_tf_pred[local_bisw_inv_indices]  # (N_fg, 3, 4)
            pred_recon_dyn_fg = torch.matmul(pred_fg_tf[:, :, :3], fg[:, 1: 4, None]).squeeze(-1) + \
                                pred_fg_tf[:, :, -1]  # (N_dyn, 3)
            loss_recon = l2_loss(pred_recon_dyn_fg, torch.clone(pred_recon_dyn_fg).detach(), dim=-1, reduction='mean')
        tb_dict['loss_recon'] = loss_recon.item()

        # add proposals loss
        pred_proposals = pred_dict['proposals']  # (N_inst, 8)
        target_proposals = target_dict['proposals']

        loss_prop_offset = l2_loss(pred_proposals[:, :3], target_proposals['offset'], reduction='mean')
        tb_dict['loss_prop_offset'] = loss_prop_offset.item()

        loss_prop_size = l2_loss(pred_proposals[:, 3: 6], target_proposals['size'], reduction='mean')
        tb_dict['loss_prop_size'] = loss_prop_size.item()

        loss_prop_ori = l2_loss(pred_proposals[:, 6:], target_proposals['ori'], reduction='mean')
        tb_dict['loss_prop_ori'] = loss_prop_ori.item()

        loss_prop = (loss_prop_offset + loss_prop_size + loss_prop_ori) * self.loss_weights.PROPOSALS
        tb_dict['loss_prop'] = loss_prop.item()

        # --------------
        # total loss
        # --------------
        loss = loss_fg + loss_inst_assoc + loss_inst_mos + loss_local_transl + loss_local_rot + loss_recon + loss_prop
        tb_dict['loss'] = loss.item()

        # eval foregound seg, motion seg during training
        if self.eval_segmentation:
            fg_precision, fg_recall = eval_binary_segmentation(fg_logit.detach().squeeze(-1), fg_target, False)
            tb_dict['fg_P'] = fg_precision
            tb_dict['fg_R'] = fg_recall

            mos_precision, mos_recall = eval_binary_segmentation(inst_mos_logit.detach().squeeze(-1),
                                                                 inst_mos_target, False)
            tb_dict['mos_P'] = mos_precision
            tb_dict['mos_R'] = mos_recall
        out = [loss, tb_dict]
        return out

    @torch.no_grad()
    def build_meta_dict(self, fg: torch.Tensor, max_num_instances: int, use_augmented_instance_idx=True) -> dict:
        """
        Args:
            fg: (N_fg, 9) - batch_idx, x, y, z, intensity, time, sweep_idx, inst_idx, aug_idx
            max_num_instances:
        """
        meta = {}

        fg_sweep_idx = fg[:, -3].long()
        fg_batch_idx = fg[:, 0].long()
        if use_augmented_instance_idx:
            fg_inst_idx = fg[:, -1].long()
        else:
            fg_inst_idx = fg[:, -2].long()

        # merge batch_idx & instance_idx
        fg_bi_idx = fg_batch_idx * max_num_instances + fg_inst_idx  # (N,)

        # merge batch_idx, instance_idx & sweep_idx
        fg_bisw_idx = fg_bi_idx * self.max_num_sweeps + fg_sweep_idx

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
        inst_target_bisw_idx = inst_bi * self.max_num_sweeps + inst_max_sweep_idx  # (N_inst,)
        # for each value in inst_target_bisw_idx find WHERE (i.e., index) it appear in local_bisw
        corr = local_bisw[:, None] == inst_target_bisw_idx[None, :]  # (N_local, N_inst)
        corr = corr.long() * torch.arange(local_bisw.shape[0]).unsqueeze(1).to(fg.device)
        meta['indices_local_to_inst_target'] = corr.sum(dim=0)  # (N_inst)

        # -----------------------------------------------------------------------------
        # the following is to establish correspondence between instances & locals
        # -----------------------------------------------------------------------------
        local_bi = local_bisw // self.max_num_sweeps
        # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
        local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
        local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1).to(fg.device)

        meta['local_bi_in_inst_bi'] = local_bi_in_inst_bi.sum(dim=0)  # (N_local)
        # this is identical to indices_instance_to_local which is used to
        # broadcast inst_global_feat from (N_inst, C_inst) to (N_local, C_inst)

        return meta

