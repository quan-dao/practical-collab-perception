import torch
import torch.nn as nn
import torch_scatter
import numpy as np
from einops import rearrange


def single_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices_q2k: torch.Tensor) -> torch.Tensor:
    """
    Args:
        q: (N_q, C)
        k: (N_k, C)
        v: (N_k, C)
        indices_q2k: (N_k,)
    Returns:
        (N_q, C)
    """
    n_channels = k.shape[1]

    Q_for_K = q[indices_q2k]  # (N_k, C)
    attn_logits = torch.sum(Q_for_K * k, dim=1) / np.sqrt(n_channels)  # (N_k,)

    # convert attn_logits to attention weights
    exp = torch.exp(attn_logits)  # (N_k)
    sum_exp = torch_scatter.scatter_sum(exp, indices_q2k)  # (N_q)
    attn = exp / sum_exp[indices_q2k]  # (N_k,)
    out = torch_scatter.scatter_sum(rearrange(attn, 'N_k -> N_k 1') * v, indices_q2k, dim=0)  # (N_q, C)
    return out


def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices_q2k: torch.Tensor,
                         return_attn_weights=False):
    """
    Args:
        q: (head, N_q, k)
        k: (head, N_k, k)
        v: (head, N_k, v)
        indices_q2k: (N_k,)
    Returns:
        (head, N_q, v)
    """
    n_ch = q.shape[-1]
    q_for_k = q[:, indices_q2k]  # (head, N_k, k)
    attn_logits = torch.sum(q_for_k * k, dim=2) / np.sqrt(n_ch)  # (head, N_k)

    # trick for numerical stability
    attn_logits_max = torch_scatter.scatter_max(attn_logits,
                                                rearrange(indices_q2k, 'N_k -> 1 N_k'), dim=1)[0]  # (head, N_q)
    attn_logits = attn_logits - attn_logits_max[:, indices_q2k]  # (head, N_k)

    # convert attention logits to attention weight
    exp = torch.exp(attn_logits)  # (head, N_k)
    sum_exp = torch_scatter.scatter_sum(exp, rearrange(indices_q2k, 'N_k -> 1 N_k'), dim=1)  # (head, N_q)
    attn = exp / sum_exp[:, indices_q2k]  # (head, N_k)

    out = torch_scatter.scatter_sum(rearrange(attn, 'head N_k -> head N_k 1') * v,  # (head, N_k, v)
                                    rearrange(indices_q2k, 'N_k -> 1 N_k'),
                                    dim=1)  # (head, N_q, v)

    if return_attn_weights:
        # attn weights are average over heads
        attn = torch.mean(attn, dim=0)  # (N_k)
        return out, attn
    else:
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k=None, d_v=None, dropout=0.1, return_attn_weight=False):
        super().__init__()
        self.n_head = n_head
        if d_k is None:
            d_k = d_model // self.n_head
            d_v = d_model // self.n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.01)

        self.return_attn_weight = return_attn_weight

    def forward(self, q, k, v, indices_q2k):
        """
        Args:
            q: (N_q, d_model)
            k: (N_k, d_model)
            v: (N_k, d_model)
            indices_q2k: (N_k,)
        Returns:
            - out: (N_q, d_model)
            - attn_weights: (N_k,)
        """
        residual = q

        q = rearrange(self.w_qs(q), 'N_q (head k) -> head N_q k', head=self.n_head)
        k = rearrange(self.w_ks(k), 'N_k (head k) -> head N_k k', head=self.n_head)
        v = rearrange(self.w_vs(v), 'N_k (head v) -> head N_k v', head=self.n_head)

        if not self.return_attn_weight:
            out = rearrange(multi_head_attention(q, k, v, indices_q2k), 'head N_q v -> N_q (head v)')
        else:
            out, attn_weights = multi_head_attention(q, k, v, indices_q2k, return_attn_weights=True)
            out = rearrange(out, 'head N_q v -> N_q (head v)')

        out = self.dropout(self.fc(out))
        out = self.batch_norm(out + residual)

        if not self.return_attn_weight:
            return out
        else:
            return out, attn_weights
