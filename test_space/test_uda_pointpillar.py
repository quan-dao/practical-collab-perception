import torch
from tools import build_test_infra, to_tensor
from workspace.utils import print_dict
import time
from pathlib import Path
import lovely_tensors as lt


def main(target_dataloader_idx=3, training=False, batch_size=2):
    cfg_file = './cfgs/uda_pp_basic.yaml'
    ckpt_dir = Path('/home/jupyter-dao-mq/workspace/learn-to-align-bev/output/cfgs/nuscenes_models/uda_pp_basic/def/ckpt')
    model, dataloader = build_test_infra(cfg_file, 
                                         training=training, 
                                         batch_size=batch_size, 
                                         ckpt_file=ckpt_dir / 'checkpoint_epoch_20.pth')

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

    model.eval()
    with torch.no_grad():
        pred_dicts, recall_dicts = model(batch_dict)

    batch_dict['pred_dicts'] = pred_dicts
    torch.save(batch_dict, 'artifact/uda_pp_batch_dict.pth')



if __name__ == '__main__':
    lt.monkey_patch()
    main(target_dataloader_idx=3,
         training=True,
         batch_size=10)
