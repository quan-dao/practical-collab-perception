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
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

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
        dist=False, logger=logger, 
        training=False,  # but dataset_cfg.INFO_PATH's val == train, set training to False to shutdown database_sampling & data_transformation
        total_epochs=1, seed=666,
        workers=0)
    
    keep_class = 1 if class_name == 'car' else 2

    model = build_detector(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(ckpt_path, logger, to_cpu=True)
    model.cuda()
    model.eval()

    pseudo_labels_root = dataset.root_path / Path(f"database_round{round_idx}_pseudo_labels")
    if not pseudo_labels_root.exists():
        pseudo_labels_root.mkdir(parents=True, exist_ok=True)

    for batch_dict in tqdm(dataloader, total=int(len(dataset) / batch_size) + 1):
        metadata = deepcopy(batch_dict['metadata'])
        
        to_tensor(batch_dict, move_to_gpu=True)

        batch_points = torch.copy(batch_dict['points'])

        with torch.no_grad():
            pred_dicts, recall_dicts = model(batch_dict)

        for batch_idx in range(batch_size):
            points = batch_points[batch_points[:, 0].long() == batch_idx]

            pred = pred_dicts[batch_idx]
            pred_boxes = pred['pred_boxes']
            pred_scores = pred['pred_scores']
            pred_labels = pred['pred_labels']

            kept_pred = pred_labels == keep_class
            pred_boxes, pred_scores, pred_labels = pred_boxes[kept_pred], pred_scores[kept_pred], pred_labels[kept_pred]

            # points-to-boxes correspondant here
            box_idx_of_points = roiaware_pool3d_utils.points_in_boxes_gpu(
                points[:, 1: 4].unsqueeze(0), pred_boxes[:, :7].unsqueeze(0),
            ).long().squeeze(0)  # (N_pts,) to index into (N_b)


            # move to cpu
            pred_boxes, pred_scores, pred_labels = pred_boxes.cpu().numpy(), pred_scores.cpu().numpy(), pred_labels.cpu().numpy()

            # assemble  pseudo labels
            pseudo_labels = np.concatenate([pred_boxes,  # 7-box
                                            np.zeros((pred_boxes.shape[0], 1)),  # sweep_idx 
                                            np.arange(pred_boxes.shape[0]).reshape(-1, 1),  # inst_idx
                                            pred_labels.reshape(-1, 1) - 1,  # class_idx| starts @ 0
                                            pred_scores.reshape(-1, 1)],  # score  
                                            axis=1)  # (N, 7 + 3 + 1) - 7-box, sweep_idx, inst, class, score
            # 7 + 3 to be compatible with format of boxes produced by TrajectoryManager

            # save
            sample_token = metadata[batch_idx]['token'] 

            for box_idx in range(pseudo_labels.shape[0]):
                box = pseudo_labels[box_idx]
                
                mask_points_in_box = box_idx_of_points == box_idx
                points_of_box = points[mask_points_in_box].cpu().numpy()
                
                # overwirte box's sweep_idx with the max sweep_idx of its points
                box[-3] = points_of_box[:, -2].max()

                info = {
                    'points_in_lidar': points_of_box,
                    'box_in_lidar': box,  # (7 + 3 + 1,) - 7-box, sweep_idx, inst_idx, class_idx, score
                }

                with open(pseudo_labels_root / Path(f"{sample_token}_round{round_idx}_label{box_idx}_{class_name}.pkl"), 'wb') as f:
                    pickle.dump(info, f)
    

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
