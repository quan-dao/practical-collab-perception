import numpy as np
import torch
from torch_scatter import scatter
from nuscenes import NuScenes
from pathlib import Path
from easydict import EasyDict
from typing import Dict, Tuple
from datetime import datetime
import pickle

from pcdet.utils import common_utils
from pcdet.models.model_utils import model_nms_utils
from pcdet.config import cfg_from_yaml_file
from pcdet.models.detectors import build_detector
from pcdet.datasets import build_dataloader
from pcdet.datasets.v2x_sim.v2x_sim_utils import get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_, roiaware_pool3d_utils


CFG_DIR = Path('../tools/cfgs/v2x_sim_models')
CKPT_DIR = Path('../tools/pretrained_models')

VIZ_OUTPUT_DIR = Path('./viz_output')
if not VIZ_OUTPUT_DIR.exists():
    VIZ_OUTPUT_DIR.mkdir(parents=True)

time_now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
logger = common_utils.create_logger(VIZ_OUTPUT_DIR / f'vis_log_{time_now}.txt')
nusc = NuScenes(dataroot='../data/v2x-sim/v2.0-trainval', version='v2.0-trainval', verbose=True)
ego_lidar_name = 'LIDAR_TOP_id_1'


def _init_config():
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0
    return cfg


def build_model(agent_type: str):
    assert agent_type in ('rsu', 'car', 'collab')
    if agent_type == 'rsu':
        cfg_name = 'v2x_second_rsu.yaml'
        ckpt_name = 'v2x_second_rsu_ep20.pth'
    elif agent_type == 'car':
        cfg_name = 'v2x_pointpillar_basic_car.yaml'
        ckpt_name = 'v2x_pointpillar_basic_car_ep19.pth'
    else:
        cfg_name = 'v2x_pointpillar_basic_ego.yaml'
        ckpt_name = 'v2x_pointpillar_basic_mix_pil4sec1_ep20.pth'

    cfg = _init_config()
    cfg_from_yaml_file(CFG_DIR / cfg_name, cfg)

    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
                                     dist=False, logger=logger, training=False, total_epochs=1, seed=666, workers=0, nusc=nusc)
    model = build_detector(cfg.MODEL, num_class=1, dataset=dataset)
    model.load_params_from_file(CKPT_DIR / ckpt_name, logger, to_cpu=True)
    model.eval()
    model.cuda()

    return model


def get_point_clouds(sample_token: str, target_lidar_token: str, point_cloud_range: np.ndarray) -> Tuple[Dict]:
    target_lidar_se3_glob = np.linalg.inv(get_nuscenes_sensor_pose_in_global(nusc, target_lidar_token))

    sample = nusc.get('sample', sample_token)

    dict_point_clouds: Dict[str, np.ndarray] = dict()
    dict_ego_se3_lidars: Dict[str, np.ndarray] = dict()
    dict_gt_boxes: Dict[str, np.ndarray] = dict()
    
    for idx_lidar in range(6):
        lidar_name = f"LIDAR_TOP_id_{idx_lidar}"
        if lidar_name not in sample['data']:
            continue
        lidar_token = sample['data'][lidar_name]
        stuff = get_pseudo_sweeps_of_1lidar(nusc, lidar_token, num_historical_sweeps=10, points_in_boxes_by_gpu=True)
        mask_in_range = np.logical_and(stuff['points'][:, :3] > point_cloud_range[:3], stuff['points'][:, :3] < point_cloud_range[3:]).all(axis=1) 
        dict_point_clouds[lidar_name] = stuff['points'][mask_in_range]  # in frame of each agent

        glob_se3_lidar = get_nuscenes_sensor_pose_in_global(nusc, lidar_token)
        ego_se3_lidar = target_lidar_se3_glob @ glob_se3_lidar
        dict_ego_se3_lidars[lidar_name] = ego_se3_lidar

        # ---
        gt_boxes = stuff['gt_boxes']  # (N_box, 7) | boxes_in_lidar
        dict_gt_boxes[lidar_name] = gt_boxes  # in frame of each agent

    return dict_point_clouds, dict_ego_se3_lidars, dict_gt_boxes


def filter_overlap_gt_boxes(gt_boxes: np.ndarray, detection_range: float = 60.) -> np.ndarray:
    _nms_config = EasyDict({
        'NMS_TYPE': 'nms_gpu',
        'NMS_THRESH': 0.2,
        'NMS_PRE_MAXSIZE': 10000,
        'NMS_POST_MAXSIZE': 10000,
    })
    gt_boxes = torch.from_numpy(gt_boxes).float().cuda()
    selected, _ = model_nms_utils.class_agnostic_nms(
        box_scores=gt_boxes.new_ones(gt_boxes.shape[0]), 
        box_preds=gt_boxes,
        nms_config=_nms_config
    )
    selected = selected.long().cpu().numpy()
    gt_boxes = gt_boxes.cpu().numpy()[selected]  # (N_tot, 7)
    _in_range = np.linalg.norm(gt_boxes[:, :2], axis=1) < detection_range
    gt_boxes = gt_boxes[_in_range]
    return gt_boxes


@torch.no_grad()
def main(scene_idx: int = 0, start_idx: int = 10, end_idx: int = 90, debug: bool = False):
    point_cloud_range = np.array([-51.2, -51.2, -8.0, 51.2, 51.2, 0.0])

    scene = nusc.scene[scene_idx]
    
    sample_token = scene['first_sample_token']
    for _ in range(start_idx):
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

    dict_viz = dict()

    # MAIN LOOP
    for sample_idx in range(start_idx, end_idx):
        sample = nusc.get('sample', sample_token)
        ego_lidar_token = sample['data'][ego_lidar_name]

        dict_point_clouds_now, dict_ego_se3_lidars_now, dict_gt_boxes_now = get_point_clouds(sample_token, ego_lidar_token, point_cloud_range)

        # for visualization: get point clouds NOW of ego vehicle & other agents
        ego_pc = dict_point_clouds_now[ego_lidar_name]
        other_pc = list()
        for lidar_name, pc in dict_point_clouds_now.items():
            if lidar_name == ego_lidar_name or pc.shape[0] == 0:
                continue
            # map pc to ego_lidar frame
            ego_se3_this_lidar = dict_ego_se3_lidars_now[lidar_name]
            apply_se3_(ego_se3_this_lidar, points_=pc)
            other_pc.append(pc)
        if len(other_pc) > 0:
            other_pc = np.concatenate(other_pc)

        # for visualization: get gt boxes
        gt_boxes = list()
        for lidar_name, boxes in dict_gt_boxes_now.items():
            if boxes.shape[0] == 0:
                continue
            ego_se3_this_lidar = dict_ego_se3_lidars_now[lidar_name]
            apply_se3_(ego_se3_this_lidar, boxes_=boxes)
            gt_boxes.append(boxes)
        gt_boxes = np.concatenate(gt_boxes)
        gt_boxes = filter_overlap_gt_boxes(gt_boxes)

        if debug:
            dict_viz['ego_pc_now'] = ego_pc
            dict_viz['other_pc_now'] = other_pc
            dict_viz['gt_boxes'] = gt_boxes
            with open(VIZ_OUTPUT_DIR / f'debug_dict_viz.pkl', 'wb') as f:
                logger.info(f"debug dict is saved to {VIZ_OUTPUT_DIR}")
                pickle.dump(dict_viz, f)
            return 

        # for computation: get point cloud PREV of other agents

        # move to next sample
        sample_token = sample['next']

    
if __name__ == '__main__':
    main()
    