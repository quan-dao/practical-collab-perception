import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector

from test_space.tools import to_tensor


def gen_exchange_database(model_type: str, 
                          ckpt_path: str,
                          training: bool,
                          batch_size: int = 4):
    np.random.seed(666)
    assert model_type in ['car', 'rsu'], f"{model_type} is invalid"
    cfg_file = f'../tools/cfgs/nuscenes_models/v2x_pointpillar_basic_{model_type}.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    # shutdown data augmentation
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling', 'random_world_flip', 
                                                       'random_world_rotation', 'random_world_scaling']
    cfg.DATA_CONFIG.MINI_TRAINVAL_STRIDE = 1
    logger = common_utils.create_logger(f'log_v2x_gen_exchange_database_{model_type}.txt')

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, 
        batch_size=batch_size,
        dist=False, logger=logger, 
        training=training,  # but dataset_cfg.INFO_PATH's val == train, set training to False to shutdown database_sampling & data_transformation
        total_epochs=1, seed=666,
        workers=0)
    
    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(ckpt_path, logger, to_cpu=True)
    model.cuda()
    model.eval()

    exchange_root = dataset.root_path / f"exchange_database_{model_type}"
    if not exchange_root.exists():
        exchange_root.mkdir(parents=True, exist_ok=True)

    for batch_dict in tqdm(dataloader, total=int(len(dataset) / batch_size) + 1):
        to_tensor(batch_dict, move_to_gpu=True)

        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--training', type=int)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    gen_exchange_database(
        args.model_type,
        args.ckpt_path,
        args.training==1,
        args.batch_size
    )
    print('Remember to generate exchange database for both train & val split')
    # python v2x_gen_exchange_database.py --model_type rsu --ckpt_path ../tools/pretrained_models/v2x_pointpillar_basic_rsu_ep20.pth --training 1
