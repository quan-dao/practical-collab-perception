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

    def load_teacher_weights(self):
        filename = self.model_cfg.TEACHER_CKPT
        checkpoint = torch.load(filename, map_location=None)
        model_state_disk = checkpoint['model_state']
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

    def forward(self, batch_dict):
        # forward prop in teacher to get teacher's encoder_feat_map
        teacher_dict = {'points': batch_dict['points']}
        for cur_module in self.module_list:
            teacher_dict = cur_module(teacher_dict)

        teacher_encoder_fmap = teacher_dict['spatial_features_2d']  # (B, C, H, W)
        student_encoder_fmap = batch_dict['spatial_features_2d']  # (B, C, H, W)
        assert teacher_encoder_fmap.shape == student_encoder_fmap.shape
        kd_loss = self._kd_loss_KLdiv(student_encoder_fmap, teacher_encoder_fmap) * self.model_cfg.LOSS_WEIGHTS[0]
        batch_dict['kd_loss'] = kd_loss
        return batch_dict

    def _kd_loss_KLdiv(self, student_fmap, teacher_fmap):
        n_bev_channels = student_fmap.shape[1]
        student_fmap = student_fmap.permute(0, 2, 3, 1).reshape(-1, n_bev_channels).contiguous()
        teacher_fmap = teacher_fmap.permute(0, 2, 3, 1).reshape(-1, n_bev_channels).contiguous()
        kl_div = self.kl_loss(F.log_softmax(student_fmap, dim=1), F.softmax(teacher_fmap, dim=1))
        return kl_div

