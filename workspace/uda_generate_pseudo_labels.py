import numpy as np
import torch
from pathlib import Path
import argparse
from copy import deepcopy
from tqdm import tqdm
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector

from test_space.tools import to_tensor


def gen_pseudo_labels(round_idx: int, 
                      class_name: str,
                      batch_size: int, 
                      ckpt_path: str):
    assert class_name in ('car', 'ped'), f"{class_name} is unknown"
    cfg_file = '../tools/cfgs/nuscenes_models/uda_pointpillar_basic.yaml'
    np.random.seed(666)
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('log_gen_pseudo_labels.txt')

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=batch_size,
        dist=False, logger=logger, training=True, total_epochs=1, seed=666,
        workers=0)
    
    keep_class = 1 if class_name == 'car' else 2

    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(ckpt_path, logger, to_cpu=True)
    model.cuda()
    model.eval()

    pseudo_labels_root = dataset.root_path / Path(f"pointpillar_round{round_idx}_pseudo_labels_{class_name}")
    if not pseudo_labels_root.exists():
        pseudo_labels_root.mkdir(parents=True, exist_ok=True)

    for batch_dict in tqdm(dataloader, total=len(dataset) / batch_size):
        metadata = deepcopy(batch_dict['metadata'])
        
        to_tensor(batch_dict, move_to_gpu=True)
        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)

        for b_idx in range(batch_size):
            pred = pred_dicts[b_idx]
            pred_boxes = pred['pred_boxes'].cpu().numpy()
            pred_scores = pred['pred_scores'].cpu().numpy()
            pred_labels = pred['pred_labels'].cpu().numpy()

            kept_pred = pred_labels == keep_class
            pred_boxes = pred_boxes[kept_pred]
            pred_scores = pred_scores[kept_pred]
            pred_labels = pred_labels[kept_pred]

            pseudo_labels = np.concatenate([pred_boxes, 
                                            pred_scores.reshape(-1, 1), 
                                            pred_labels.reshape(-1, 1)], 
                                            axis=1)  # (N, 9) - 7-box, score, class_idx

            sample_token = metadata[b_idx]['token']
            with open(pseudo_labels_root / Path(f"{sample_token}_round{round_idx}_pseudo_labels_{class_name}.pkl"), 'wb') as f:
                pickle.dump(pseudo_labels, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--round_idx', type=int)
    parser.add_argument('--class_name', type=str, )
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    gen_pseudo_labels(args.round_idx, 
                      args.class_name,
                      args.batch_size, 
                      args.ckpt_path)
    # python uda_generate_pseudo_labels.py --round_idx 0 --class_name car --ckpt_path ../tools/ckpt_to_eval/checkpoint_epoch_23.pth
