import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector
import time
import argparse


def build_test_infra(cfg_file: str, **kwargs):
    np.random.seed(666)
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./artifact/dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=kwargs.get('batch_size', 2),
                                              dist=False, logger=logger, training=kwargs.get('training', False), total_epochs=1, seed=666,
                                              workers=0)
    model = build_detector(cfg.MODEL, num_class=kwargs.get('num_class', 10), dataset=dataset)
    if 'ckpt_file' in kwargs:
        model.load_params_from_file(kwargs['ckpt_file'], logger, to_cpu=True)
        
    return model, dataloader


def to_gpu(batch_dict):
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.cuda()


def to_tensor(batch_dict, move_to_gpu=False) -> None:
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        # elif key in ['images']:
        #     batch_dict[key] = kornia.image_to_tensor(val).float().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int()
        else:
            batch_dict[key] = torch.from_numpy(val).float()
    
    if move_to_gpu:
        to_gpu(batch_dict)


def main(target_dataloader_idx, training, batch_size=10):
    cfg_file = './cfgs/nuscenes_models/pointpillar_jr_withmap.yaml'
    _, dataloader = build_test_infra(cfg_file, training=training, batch_size=batch_size)

    iter_dataloader = iter(dataloader)
    data_time, counter = 0, 0
    for _ in range(target_dataloader_idx):
        start = time.time()
        batch_dict = next(iter_dataloader)
        data_time += (time.time() - start)
        counter += 1

    print('avg data time: ', data_time / float(counter))

    to_tensor(batch_dict, move_to_gpu=False)
    filename = f'./artifact/dataset_train{training}_bs{batch_size}_dataloaderIdx{target_dataloader_idx}.pth'
    torch.save(batch_dict, filename)
    print(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataloader_idx', type=int, default=3, required=False)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--no-training', dest='training', action='store_false')
    parser.set_defaults(training=True)
    parser.add_argument('--batch_size', type=int, default=10, required=False)
    args = parser.parse_args()
    main(args.dataloader_idx, args.training, args.batch_size)
