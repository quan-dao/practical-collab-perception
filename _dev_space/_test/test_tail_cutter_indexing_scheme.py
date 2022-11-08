import torch


def main():
    batch_size = 1
    n_sweeps = 4
    inst_bi = torch.tensor([0, 1]).long()
    inst_global_feat = torch.tensor([[10, 0, 0], [0, 0, 10]]).float()
    inst_max_sweep_idx = torch.tensor([2, 3]).long()

    local_sweep_idx = torch.tensor([1, 2, 1, 2, 3]).long()
    local_bi = torch.tensor([0, 0, 1, 1, 1]).long()
    local_bisw = local_bi * n_sweeps + local_sweep_idx
    local_center = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])
    local_shape_enc = -torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # -----
    # concatenate instances global feat with center & shape_enc of their target local
    # -----
    inst_target_bisw_idx = inst_bi * n_sweeps + inst_max_sweep_idx  # (N_inst,)

    # for each value in inst_target_bisw_idx find WHERE (i.e., index) it appear in local_bisw
    corr = local_bisw[:, None] == inst_target_bisw_idx[None, :]  # (N_local, N_inst)
    corr = corr.long() * torch.arange(local_bisw.shape[0]).unsqueeze(1)
    corr = corr.sum(dim=0)  # (N_inst)

    inst_target_center_shape = torch.cat((local_center[corr], local_shape_enc[corr]), dim=1)  # (N_inst, 3+C_inst)
    inst_global_feat = torch.cat((inst_global_feat, inst_target_center_shape), dim=1)  # (N_inst, 3+2*C_inst)

    # ==
    # test inst_global_feat
    # ==
    gt_inst_global_feat = torch.tensor([
        [10, 0, 0, 0, 1, 0, 0, -1, 0],
        [0, 0, 10, 0, 1, 1, 0, -1, -1]
    ]).float()
    assert torch.all(inst_global_feat == gt_inst_global_feat), f"inst_global_feat:\n{inst_global_feat}"
    print('test global_feat: passed')

    # ------------
    # broadcast inst_global_feat from (N_inst, C_inst) to (N_local, C_inst)
    local_bi = local_bisw // n_sweeps
    # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
    local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
    local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1)
    local_bi_in_inst_bi = local_bi_in_inst_bi.sum(dim=0)  # (N_local)

    local_global_feat = inst_global_feat[local_bi_in_inst_bi]  # (N_local, 3+2*C_inst)

    # ==
    # test local_global_feat
    # ==
    gt_local_global_feat = torch.tensor([
        [10, 0, 0, 0, 1, 0, 0, -1, 0],
        [10, 0, 0, 0, 1, 0, 0, -1, 0],
        #
        [0, 0, 10, 0, 1, 1, 0, -1, -1],
        [0, 0, 10, 0, 1, 1, 0, -1, -1],
        [0, 0, 10, 0, 1, 1, 0, -1, -1]
    ])
    assert torch.all(local_global_feat == gt_local_global_feat), f"local_global_feat:\n{local_global_feat}"
    print('test local_global_feat: passed')


if __name__ == '__main__':
    main()
