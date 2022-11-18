import torch
import torch_scatter
import numpy as np


def single_head_attention(K: torch.Tensor, V: torch.Tensor, Q: torch.Tensor, indices_Q_to_K: torch.Tensor) -> torch.Tensor:
    """
    Args:
        K: (N_k, C)
        V: (N_k, C)
        Q: (N_q, C)
        indices_Q_to_K: (N_k,)
    """
    assert K.shape[1] == V.shape[1] == Q.shape[1], f"{K.shape[1]} != {V.shape[1]} != {Q.shape[1]}"
    n_channels = K.shape[1]

    Q_for_K = Q[indices_Q_to_K]  # (N_k, C)
    attn_logits = torch.sum(Q_for_K * K, dim=1) / np.sqrt(n_channels)  # (N_k,)

    # convert attn_logits to attention weights
    exp = torch.exp(attn_logits)  # (N_k)
    sum_exp = torch_scatter.scatter_sum(exp, indices_Q_to_K)  # (N_q)
    attn = exp / sum_exp[indices_Q_to_K]  # (N_k,)

    out = torch_scatter.scatter_sum(attn * V, indices_Q_to_K, dim=0)  # (N_q, C)
    return out  # TODO: test this
