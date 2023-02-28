import torch
from tools import BackwardHook, build_test_infra, to_tensor
from workspace.utils import print_dict
import time
import lovely_tensors as lt


def main(target_dataloader_idx=3, training=False, batch_size=2):
    cfg_file = './cfgs/pointpillar_jr_av2.yaml'
    model, dataloader = build_test_infra(cfg_file, training=training, batch_size=batch_size, ckpt_file='ckpts/pointpillar_jr_av2_ep15.pth')

    # print('---------------------\n', 
    #       model, 
    #       '\n---------------------')

    iter_dataloader = iter(dataloader)
    data_time, counter = 0, 0
    for _ in range(target_dataloader_idx):
        start = time.time()
        batch_dict = next(iter_dataloader)
        data_time += (time.time() - start)
        counter += 1

    print_dict(batch_dict, 'batch_dict')
    print('avg data time: ', data_time / float(counter))

    to_tensor(batch_dict, move_to_gpu=True)
    model.cuda()

    if training:
        bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in model.named_parameters()]
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        print('loss: ', ret_dict['loss'].item())
        print_dict(tb_dict, 'tb_dict')

        model.zero_grad()
        ret_dict['loss'].backward()
        for hook in bw_hooks:
            if hook.grad_mag < 1e-3 and 'bias' not in hook.name:
                print(f'zero grad at {hook.name}')
    else:
        model.eval()
        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)
        
        for k in ('voxel_features', 'pillar_features', 'voxel_coords', 'spatial_features', 'spatial_features_2d'):
            batch_dict.pop(k)

        print_dict(batch_dict, 'batch_dict')
        print_dict(recall_dicts, 'recall_dicts')
        for ii in range(batch_size):
            print('------------')
            print(f'ii: {ii}')
            print_dict(pred_dicts[ii], 'pred_dicts')
            print('------------')
        
        filename = f'artifact/av2_pred_train{training}_bs{batch_size}_dataloaderIdx{target_dataloader_idx}.pth'
        torch.save(batch_dict, filename)
        print(filename)


if __name__ == '__main__':
    lt.monkey_patch()
    main()
