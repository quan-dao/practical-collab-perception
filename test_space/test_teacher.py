import torch
from tools import to_tensor
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from workspace.utils import print_dict
from workspace.teacher import Teacher


def main(target_dataloader_idx=3, training=False, batch_size=2, print_model=False):
    cfg_file = './cfgs/oracle_pp_nusc.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.MODEL.CKPT = '../tools/pretrained_models/oracle_ep20_nusc4th.pth'
    cfg.MODEL.DEBUG = True

    logger = common_utils.create_logger('./artifact/dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=batch_size,
                                              dist=False, logger=logger, training=training, total_epochs=1, seed=666,
                                              workers=0)
    model = Teacher(cfg.MODEL, len(cfg.CLASS_NAMES), dataset, logger)
    if print_model:
        print('---------------------\n', 
              model, 
              '\n---------------------')
    model.cuda()

    iter_dataloader = iter(dataloader)
    for _ in range(target_dataloader_idx):    
        batch_dict = next(iter_dataloader)
    print_dict(batch_dict, 'batch_dict')
    to_tensor(batch_dict, move_to_gpu=True)

    batch_dict = model(batch_dict)

    filename = Path('./artifact/nusc_teacher_batch_dict.pth')
    torch.save(batch_dict, filename)
    print(filename.resolve())


if __name__ == '__main__':
    main(training=True, print_model=False)
