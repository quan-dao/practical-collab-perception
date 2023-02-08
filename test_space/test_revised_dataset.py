import torch
from tools import build_test_infra, to_tensor
from workspace.utils import print_dict
import time


def main(target_dataloader_idx=3, training=False, batch_size=2):
    cfg_file = './cfgs/pointpillar_jr.yaml'
    _, dataloader = build_test_infra(cfg_file, training=training, batch_size=batch_size)

    iter_dataloader = iter(dataloader)
    data_time, counter = 0, 0
    for _ in range(target_dataloader_idx):
        start = time.time()
        batch_dict = next(iter_dataloader)
        data_time += (time.time() - start)
        counter += 1

    print_dict(batch_dict, 'batch_dict')
    print('avg data time: ', data_time / float(counter))

    to_tensor(batch_dict, move_to_gpu=False)
    filename = f'./artifact/dataset_train{training}_bs{batch_size}_dataloaderIdx{target_dataloader_idx}.pth'
    torch.save(batch_dict, filename)
    print(filename)


if __name__ == '__main__':
    main()
