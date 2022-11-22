import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from einops import rearrange

from _dev_space.non_batching_attention import single_head_attention, multi_head_attention, MultiHeadAttention
from _dev_space._test.tools_4testing import BackwardHook


def _test_single_head():
    n_ch = 7
    K, V = torch.rand(5, n_ch), torch.rand(5, n_ch)
    Q = torch.rand(2, n_ch)
    indices_Q2K = torch.tensor([0, 1, 0, 1, 1]).long()

    out = []
    for i in range(2):
        mask_cur = indices_Q2K == i
        cur_K, cur_V = K[mask_cur], V[mask_cur]  # (N, C)
        cur_Q = Q[[i]]  # (1, C)
        cur_attn = F.softmax(torch.mm(cur_Q, cur_K.t()) / np.sqrt(n_ch), dim=1)  # (1, N)
        out.append(torch.mm(cur_attn, cur_V))

    out = torch.cat(out)

    comp = single_head_attention(Q, K, V, indices_Q2K)

    diff = torch.abs(comp - out)
    assert torch.all(diff < 1e-5), f"{diff}"
    print('pass test single head\n---')


def _test_multi_head():
    head = 3
    N_q, N_k = 2, 5
    k = v = 7

    q = torch.rand(head, N_q, k)
    k = torch.rand(head, N_k, k)
    v = torch.rand(head, N_k, v)
    indices_q2k = torch.tensor([0, 1, 0, 1, 1]).long()

    out = []
    for h_idx in range(head):
        out.append(single_head_attention(q[h_idx], k[h_idx], v[h_idx], indices_q2k))
    out = rearrange(out, 'head N_q v -> head N_q v')

    other = multi_head_attention(q, k, v, indices_q2k)
    diff = torch.abs(other - out)
    assert torch.all(diff < 1e-5), f"{diff}"
    print('pass test multi-head\n---')


def _test_multi_head_attn_layer():
    n_head = 4
    d_model = 16
    attn_layer = MultiHeadAttention(n_head, d_model)
    bw_hooks = [BackwardHook(name, param) for name, param in attn_layer.named_parameters()]

    N_q, N_k = 2, 5
    q = torch.rand(N_q, d_model)
    k = torch.rand(N_k, d_model)
    v = torch.rand(N_k, d_model)
    indices_q2k = torch.tensor([0, 1, 0, 1, 1]).long()

    out = attn_layer(q, k, v, indices_q2k)
    label = torch.rand(N_q, d_model)
    loss = torch.mean(torch.square(label - out))
    attn_layer.zero_grad()
    loss.backward()

    for hook in bw_hooks:
        if hook.grad_mag < 1e-5:
            print(f'zero grad at {hook.name}')


def _test_batch_scatter_sum():
    b, d = 2, 7
    n_k, n_q = 5, 2
    k = torch.rand(b, n_k, d)
    q = torch.rand(b, n_q, d)
    ids_q2k = torch.tensor([0, 1, 0, 1, 1]).long()

    out = []
    for b_idx in range(b):
        out.append(torch_scatter.scatter_sum(k[b_idx], ids_q2k, dim=0))
    out = rearrange(out, 'b n_q d -> b n_q d')

    other = torch_scatter.scatter_sum(k, ids_q2k.unsqueeze(0), dim=1)

    diff = torch.abs(other - out)
    assert torch.all(diff < 1e-5), f"{diff}"
    print('pass test batch_scatter_sum\n---')


def main():
    _test_single_head()
    _test_batch_scatter_sum()
    _test_multi_head()
    _test_multi_head_attn_layer()


if __name__ == '__main__':
    main()
