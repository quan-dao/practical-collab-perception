'''
Src: Dynamic 3D Scene Analysis by Point Cloud Accumulation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pcdet.models.loss_fnc.lovasz_softmax import Lovasz_softmax


_EPS = 1e-5  # To prevent division by zero


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class CELovaszLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.n_classes = num_classes
        assert self.n_classes >= 2
        self.lovasz_loss = Lovasz_softmax()

    def get_cross_entropy_weights(self, gt: torch.Tensor, max_weights=50):
        """
        Compute classes' inverse frequency to use as weights for cross entropy
        Args:
            gt: (N,) - ground truth of segmentation task
        """
        cls_counts = gt.new_zeros(self.n_classes).float() + _EPS
        for cls_idx in range(self.n_classes):
            cls_counts[cls_idx] = torch.sum((gt == cls_idx).float())

        # sanity check
        assert cls_counts.sum() == gt.shape[0]

        inv_freq=  cls_counts.sum() / cls_counts  # (N_cls,)
        seq_weights = torch.clamp(torch.sqrt(inv_freq), 0., max_weights)
        return seq_weights

    def forward(self, pred_logits: torch.Tensor, label: torch.Tensor, tb_dict: dict = None, loss_name=''):
        """
        Args:
            pred_logits: (N, n_cls) - raw, unnormalized scores for each class
            label: (N,) - ground truth class
            tb_dict:
        """

        ce_weights = self.get_cross_entropy_weights(label)

        if self.n_classes == 2:
            loss_ce = F.binary_cross_entropy_with_logits(pred_logits, rearrange(label, 'N -> N 1').float(),
                                                         pos_weight=ce_weights[1])

            positive_prob = sigmoid(pred_logits)  # (N, 1)
            prob = torch.cat([1.0 - positive_prob, positive_prob], dim=1)  # (N, 2)
        else:
            loss_ce = F.cross_entropy(pred_logits, label, ce_weights)

            prob = torch.softmax(pred_logits, dim=1)  # (N, n_cls)

        loss_lovasz = self.lovasz_loss(prob, label)

        loss_seg = loss_ce + loss_lovasz

        if tb_dict is not None:
            tb_dict[f'loss_{loss_name}_ce'] = loss_ce.item()
            tb_dict[f'loss_{loss_name}_lovasz'] = loss_lovasz.item()
        return loss_seg


