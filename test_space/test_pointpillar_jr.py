import torch
from tools import BackwardHook, build_test_infra, to_tensor
from workspace.utils import print_dict
import time
import lovely_tensors as lt


def main(target_dataloader_idx=3, training=False, batch_size=2):
    cfg_file = './cfgs/pointpillar_jr.yaml'
    model, dataloader = build_test_infra(cfg_file, training=training, batch_size=batch_size)

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
    bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in model.named_parameters()]

    ret_dict, tb_dict, disp_dict = model(batch_dict)
    print('loss: ', ret_dict['loss'].item())
    print_dict(tb_dict, 'tb_dict')

    model.zero_grad()
    ret_dict['loss'].backward()
    for hook in bw_hooks:
        if hook.grad_mag < 1e-3 and 'bias' not in hook.name:
            print(f'zero grad at {hook.name}')


if __name__ == '__main__':
    lt.monkey_patch()
    main()
