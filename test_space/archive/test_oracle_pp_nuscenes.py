import torch
from tools import BackwardHook, build_test_infra, to_tensor
from workspace.utils import print_dict
import time
import lovely_tensors as lt
import argparse


def main(target_dataloader_idx=3, training=False, batch_size=2, print_model=False):
    cfg_file = './cfgs/oracle_pp_nusc.yaml'
    model, dataloader = build_test_infra(cfg_file, training=training, batch_size=batch_size)

    if print_model:
        print('---------------------\n', 
              model, 
              '\n---------------------')

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
            if k in batch_dict:
                batch_dict.pop(k)

        print_dict(batch_dict, 'batch_dict')
        print_dict(recall_dicts, 'recall_dicts')
        for ii in range(batch_size):
            print('------------')
            print(f'ii: {ii}')
            print_dict(pred_dicts[ii], 'pred_dicts')
            print('------------')
        
        filename = f'artifact/nusc_oracle_pp_train{training}_bs{batch_size}_dataloaderIdx{target_dataloader_idx}.pth'
        torch.save(batch_dict, filename)
        print(filename)


if __name__ == '__main__':
    lt.monkey_patch()
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataloader_idx', type=int, default=3, required=False)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--no-training', dest='training', action='store_false')
    parser.set_defaults(training=True)
    parser.add_argument('--batch_size', type=int, default=2, required=False)
    parser.add_argument('--print_model', action='store_true')
    parser.add_argument('--no-print_model', dest='print_model', action='store_false')
    parser.set_defaults(print_model=False)
    args = parser.parse_args()

    main(target_dataloader_idx=args.dataloader_idx, training=args.training, batch_size=args.batch_size, print_model=args.print_model)
