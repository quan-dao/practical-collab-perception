import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.tail_cutter import PointAligner
from _dev_space.viz_tools import print_dict
from tools_4testing import load_data_to_tensor, show_points_in_batch_dict, load_dict_to_gpu, BackwardHook, \
    load_dict_to_cpu
import argparse
from time import time


def main(show_raw_data=False, test_pred_is_gt=False, show_correction_result=False, test_full=False):
    cfg_file = './tail_cutter_cfg.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')

    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    t_tic = time()
    for _ in range(25):
        batch_dict = next(iter_dataloader)
    time_data = (time() - t_tic) / 25.
    print('time_data: ', time_data)

    print_dict(batch_dict)
    load_data_to_tensor(batch_dict)

    if show_raw_data:
        show_points_in_batch_dict(batch_dict, batch_idx=1)

    model = PointAligner(cfg.MODEL)
    print('---\n', model, '\n---\n')
    if test_pred_is_gt:
        assert not test_full
        # ---
        # populate forward return dict
        # ---
        # meta
        mask_fg = batch_dict['points'][:, -1] > -1
        fg = batch_dict['points'][mask_fg]  # (N_fg, 8) - batch_idx, x, y, z, instensity, time, sweep_idx, instacne_idx
        points_batch_idx = batch_dict['points'][:, 0].long()
        fg_batch_idx = points_batch_idx[mask_fg]
        fg_inst_idx = fg[:, -1].long()
        fg_sweep_idx = fg[:, -3].long()

        max_num_inst = batch_dict['instances_tf'].shape[
            1]  # batch_dict['instances_tf']: (batch_size, max_n_inst, n_sweeps, 3, 4)
        fg_bi_idx = fg_batch_idx * max_num_inst + fg_inst_idx  # (N,)
        fg_bisw_idx = fg_bi_idx * model.cfg.get('NUM_SWEEPS', 10) + fg_sweep_idx

        inst_bi, inst_bi_inv_indices = torch.unique(fg_bi_idx, sorted=True, return_inverse=True)
        model.forward_return_dict['meta'] = {'inst_bi': inst_bi, 'inst_bi_inv_indices': inst_bi_inv_indices}

        local_bisw, local_bisw_inv_indices = torch.unique(fg_bisw_idx, sorted=True, return_inverse=True)
        # local_bisw: (N_local,)
        model.forward_return_dict['meta'].update({'local_bisw': local_bisw,
                                                  'local_bisw_inv_indices': local_bisw_inv_indices})

        local_bi = local_bisw // model.cfg.get('NUM_SWEEPS', 10)
        # for each value in local_bi find WHERE (i.e., index) it appear in inst_bi
        local_bi_in_inst_bi = inst_bi[:, None] == local_bi[None, :]  # (N_inst, N_local)
        local_bi_in_inst_bi = local_bi_in_inst_bi.long() * torch.arange(inst_bi.shape[0]).unsqueeze(1).to(fg.device)
        local_bi_in_inst_bi = local_bi_in_inst_bi.sum(dim=0)  # (N_local)
        model.forward_return_dict['meta']['local_bi_in_inst_bi'] = local_bi_in_inst_bi

        # target
        target_dict = model.assign_target(batch_dict)
        print_dict(target_dict)

        # prediction
        pred_dict = dict()
        pred_dict['fg'] = torch.clamp(target_dict['fg'].float().unsqueeze(1), min=1e-4, max=1.0 - 1e-4)  # prob
        pred_dict['fg'] = torch.log(pred_dict['fg'] / (1.0 - pred_dict['fg']))  # logit

        pred_dict['inst_assoc'] = target_dict['inst_assoc'].new_zeros(batch_dict['points'].shape[0], 2)
        pred_dict['inst_assoc'][target_dict['fg'] == 1] = target_dict['inst_assoc']

        pred_dict['inst_motion_stat'] = torch.clamp(target_dict['inst_motion_stat'].float().unsqueeze(1),
                                                    min=1e-4, max=1.0 - 1e-4)  # prob
        pred_dict['inst_motion_stat'] = torch.log(pred_dict['inst_motion_stat'] / (1.0 - pred_dict['inst_motion_stat']))  # logit

        pred_dict['local_transl'] = target_dict['local_tf'][:, :, -1]  # (N_local, 3)
        pred_dict['local_rot'] = target_dict['local_tf'][:, :, :3]  # (N_local, 3, 3)

        model.forward_return_dict['pred_dict'] = pred_dict
        model.forward_return_dict['target_dict'] = target_dict

        # ---
        # ---
        with torch.no_grad():
            loss, tb_dict, debug_dict = model.get_training_loss(batch_dict, debug=True)
        print('loss = ', loss)
        print_dict(tb_dict)
        print_dict(debug_dict)

        if show_correction_result:
            # find static fg
            inst_mos_target = target_dict['inst_motion_stat']  # (N_inst,)
            fg_motion = inst_mos_target[inst_bi_inv_indices] == 1  # (N_fg)
            fg = batch_dict['points'][target_dict['fg'] == 1]  # (N_fg)

            static_fg = fg[torch.logical_not(fg_motion), :4]  # batch_idx, x, y, z
            dyn_fg = torch.cat((fg[fg_motion, 0].unsqueeze(1), debug_dict['gt_recon_dyn_fg']), dim=1)
            bg = batch_dict['points'][target_dict['fg'] == 0, :4]  # (N_bg)
            _points = torch.cat([static_fg, dyn_fg, bg], dim=0)

            color_static_fg = torch.tensor((0, 0, 1)).repeat(static_fg.shape[0], 1).float()
            color_dyn_fg = torch.tensor((1, 0, 0)).repeat(dyn_fg.shape[0], 1).float()
            color_bg = torch.tensor((0, 0, 0)).repeat(bg.shape[0], 1).float()
            _colors = torch.cat([color_static_fg, color_dyn_fg, color_bg], dim=0)
            show_points_in_batch_dict(batch_dict, 1, _points, _colors)

    if test_full:
        assert not test_pred_is_gt, "test_pred_is_gt can't be True if test_full"
        model.cuda()
        bw_hooks = [BackwardHook(name, param) for name, param in model.named_parameters()]

        load_dict_to_gpu(batch_dict)
        batch_dict = model(batch_dict)
        loss, tb_dict = model.get_training_loss(batch_dict, debug=False)

        print('loss = ', loss)
        print_dict(tb_dict)

        model.zero_grad()
        loss.backward()
        for hook in bw_hooks:
            if hook.grad_mag < 1e-4:
                print(f'zero grad at {hook.name}')

        if show_correction_result:
            # find static fg
            target_dict = model.forward_return_dict['target_dict']
            load_dict_to_cpu(target_dict)
            load_dict_to_cpu(batch_dict)
            load_dict_to_cpu(model.forward_return_dict['meta'])
            load_dict_to_cpu(debug_dict)

            inst_mos_target = target_dict['inst_motion_stat']  # (N_inst,)
            inst_bi_inv_indices = model.forward_return_dict['meta']['inst_bi_inv_indices']
            fg_motion = inst_mos_target[inst_bi_inv_indices] == 1  # (N_fg)
            fg = batch_dict['points'][target_dict['fg'] == 1]  # (N_fg)

            static_fg = fg[torch.logical_not(fg_motion), :4]  # batch_idx, x, y, z
            dyn_fg = torch.cat((fg[fg_motion, 0].unsqueeze(1), debug_dict['gt_recon_dyn_fg']), dim=1)
            bg = batch_dict['points'][target_dict['fg'] == 0, :4]  # (N_bg)
            _points = torch.cat([static_fg, dyn_fg, bg], dim=0)

            color_static_fg = torch.tensor((0, 0, 1)).repeat(static_fg.shape[0], 1).float()
            color_dyn_fg = torch.tensor((1, 0, 0)).repeat(dyn_fg.shape[0], 1).float()
            color_bg = torch.tensor((0, 0, 0)).repeat(bg.shape[0], 1).float()
            _colors = torch.cat([color_static_fg, color_dyn_fg, color_bg], dim=0)
            show_points_in_batch_dict(batch_dict, 1, _points, _colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--show_raw_data', action='store_true', default=False)
    parser.add_argument('--test_pred_is_gt', action='store_true', default=False)
    parser.add_argument('--show_correction_result', action='store_true', default=False)
    parser.add_argument('--test_full', action='store_true', default=False)
    args = parser.parse_args()
    main(args.show_raw_data, args.test_pred_is_gt, args.show_correction_result, args.test_full)
