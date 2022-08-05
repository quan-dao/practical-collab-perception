import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.detectors.detector3d_template import Detector3DTemplate


class KnowledgeDistillationLoss(Detector3DTemplate):
    def __init__(self, teacher_model_cfg, num_class, dataset):
        super().__init__(teacher_model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.load_teacher_weights()
        # set teacher to inference mode & freeze teacher's weights
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        # loss
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.aff_cfg = self.model_cfg.KD_AFFINITY_LOSS

    def load_teacher_weights(self):
        filename = self.model_cfg.TEACHER_CKPT
        checkpoint = torch.load(filename, map_location=None)
        model_state_disk = checkpoint['model_state']
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

    def forward(self, batch_dict, tb_dict):
        self.eval()
        # forward prop in teacher to get teacher's encoder_feat_map
        teacher_dict = {'points': batch_dict['points']}
        for cur_module in self.module_list:
            teacher_dict = cur_module(teacher_dict)

        # Distill encoder
        tea_enc = teacher_dict['spatial_features_2d']  # (B, C, H, W)
        stu_enc = batch_dict['spatial_features_2d']  # (B, C, H, W)
        assert tea_enc.shape == stu_enc.shape
        enc_kl_div = self._kd_loss_KLdiv(stu_enc, tea_enc)
        enc_aff_loss = self._kd_loss_affinity(
            stu_enc, tea_enc, self.aff_cfg.NORMALIZE_USING_NORM, self.aff_cfg.DOWNSAMPLE_FACTOR
        )
        enc_l1_loss = self._kd_loss_l1(stu_enc, tea_enc)

        # Distill decoder
        tea_dec = teacher_dict['decoded_spatial_features_2d']
        stu_dec = batch_dict['decoded_spatial_features_2d']  # (B, C, H, W)
        assert tea_dec.shape == stu_dec.shape
        dec_kl_div = self._kd_loss_KLdiv(stu_dec, tea_dec)
        dec_aff_loss = self._kd_loss_affinity(
            stu_dec, tea_dec, self.aff_cfg.NORMALIZE_USING_NORM, self.aff_cfg.DOWNSAMPLE_FACTOR
        )
        dec_l1_loss = self._kd_loss_l1(stu_dec, tea_dec)

        # Distill head's heatmaps
        tea_hms = teacher_dict['pred_heatmaps']
        stu_hms = batch_dict['pred_heatmaps']
        hm_loss = 0.0
        for _stu_hm, _tea_hm in zip(stu_hms, tea_hms):
            hm_loss = hm_loss + self._kd_loss_l1(_stu_hm, _tea_hm)

        # Total loss
        kl_div = (enc_kl_div + dec_kl_div) * self.model_cfg.LOSS_WEIGHTS[0]
        aff_loss = (enc_aff_loss + dec_aff_loss) * self.model_cfg.LOSS_WEIGHTS[1]
        l1_loss = (enc_l1_loss + dec_l1_loss) * self.model_cfg.LOSS_WEIGHTS[2]
        hm_loss = hm_loss * self.model_cfg.LOSS_WEIGHTS[3]

        kd_loss = kl_div + aff_loss + l1_loss + hm_loss
        tb_dict['kd_loss'] = kd_loss.item()
        tb_dict['kd_kl_div'] = kl_div.item()
        tb_dict['kd_aff_loss'] = aff_loss.item()
        tb_dict['kd_l1_loss'] = l1_loss.item()
        tb_dict['hm_loss'] = hm_loss.item()
        return kd_loss, tb_dict

    def _kd_loss_KLdiv(self, student_fmap, teacher_fmap):
        n_bev_channels = student_fmap.shape[1]
        student_fmap = student_fmap.permute(0, 2, 3, 1).reshape(-1, n_bev_channels).contiguous()
        teacher_fmap = teacher_fmap.permute(0, 2, 3, 1).reshape(-1, n_bev_channels).contiguous()
        kl_div = self.kl_loss(F.log_softmax(student_fmap, dim=1), F.softmax(teacher_fmap, dim=1))
        return kl_div

    @staticmethod
    def _kd_loss_affinity(student_fmap, teacher_fmap, normalize_using_norm=True, downsample_factor=1):
        def _compute_affinity(x, _normalize_using_norm):
            assert len(x.shape) == 4, f"input must have shape of (B, C, H, W), current shape: {x.shape}"
            batch_size, n_channels = x.shape[:2]
            x = x.view(batch_size, n_channels, -1)  # (B, C, N) | N = H * W
            aff = torch.bmm(x.transpose(1, 2).contiguous(), x)  # (B, N, N)
            if _normalize_using_norm:
                norm = torch.linalg.norm(x, dim=1, keepdim=True)  # (B, 1, N)
                normalizer = torch.bmm(norm.transpose(1, 2).contiguous(), norm)  # (B, N, N)
                mask_valid_norm = normalizer > 1e-4  # (B, N, N)
                aff[mask_valid_norm] = aff[mask_valid_norm] / normalizer[mask_valid_norm]
            else:
                mask_valid_norm = None
            return aff, mask_valid_norm

        if downsample_factor > 1:
            scale_factor = 1.0 / float(downsample_factor)
            student_fmap = F.interpolate(student_fmap, scale_factor=scale_factor, mode='bilinear')
            teacher_fmap = F.interpolate(teacher_fmap, scale_factor=scale_factor, mode='bilinear')

        student_aff, stud_valid = _compute_affinity(student_fmap, normalize_using_norm)  # (B, N, N) | N = H * W
        teacher_aff, tea_valid = _compute_affinity(teacher_fmap, normalize_using_norm)  # (B, N, N) | N = H * W
        if normalize_using_norm:
            _valid = torch.logical_and(stud_valid, tea_valid)  # (B, N, N)
            aff_loss = F.l1_loss(student_aff[_valid], teacher_aff[_valid], reduction='mean')
        else:
            aff_loss = F.l1_loss(student_aff, teacher_aff, reduction='mean')
        return aff_loss

    def _kd_loss_l1(self, stu_fmap, tea_fmap, gt_mask=None):
        if gt_mask is not None:
            raise NotImplementedError

        l1_loss = F.l1_loss(stu_fmap, tea_fmap, reduction='mean')
        return l1_loss

