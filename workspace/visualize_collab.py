import numpy as np
import torch
from torch_scatter import scatter
from nuscenes import NuScenes
from pathlib import Path
from easydict import EasyDict
from typing import Dict, Tuple
from datetime import datetime
import argparse

from pcdet.utils import common_utils
from pcdet.models.model_utils import model_nms_utils
from pcdet.config import cfg_from_yaml_file
from pcdet.models.detectors import build_detector
from pcdet.datasets import build_dataloader
from pcdet.datasets.v2x_sim.v2x_sim_utils import get_pseudo_sweeps_of_1lidar, get_nuscenes_sensor_pose_in_global, apply_se3_, roiaware_pool3d_utils
from workspace.o3d_visualization import PointsPainter


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
    cfg.MODEL.RETURN_BATCH_DICT = True
    if agent_type != 'collab':
        cfg.MODEL.CORRECTOR.RETURN_SCENE_FLOW = True
        cfg.MODEL.DENSE_HEAD.RETURN_MODAR_POINTS =True

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


def propagate_modar(modar_: torch.Tensor, foregr: torch.Tensor) -> None:
    """
    Args:
        modar: (N_modar, 7 + 2) - box-7, score, label
        foregr: (N_fore, 5 + 2 + 3 + 3) - point-5, sweep_idx, inst_idx, cls_prob-3, flow-3
    """
    assert modar_.shape[1] == 9, f"{modar_.shape[1]} != 9 | expect: box-7, score, label"
    assert foregr.shape[1] == 13, f"{foregr.shape[1]} != 13 | expect: point-5, sweep_idx, inst_idx, cls_prob-3, flow-3"
    # pool
    box_idx_of_foregr = roiaware_pool3d_utils.points_in_boxes_gpu(
        foregr[:, :3].unsqueeze(0), modar_[:, :7].unsqueeze(0)
    ).squeeze(0).long()  # (N_foregr,) | == -1 mean not belong to any boxes

    mask_valid_foregr = box_idx_of_foregr > -1
    foregr = foregr[mask_valid_foregr]
    box_idx_of_foregr = box_idx_of_foregr[mask_valid_foregr]
    
    unq_box_idx, inv_unq_box_idx = torch.unique(box_idx_of_foregr, return_inverse=True)

    # weighted sum of foregrounds' offset; weights = foreground's prob dynamic
    boxes_offset = scatter(foregr[:, -3:], inv_unq_box_idx, dim=0, reduce='mean') * 2.  # (N_modar, 3)
    # offset modar; here, assume objects maintain the same speed
    modar_[unq_box_idx, :3] += boxes_offset

    return


def draw_scene(dict_data: dict, sample_idx: int, scene_output_dir: Path, view_point: dict = None):
    ego_pc = dict_data['ego_pc_now']
    other_pc = dict_data['other_pc_now']
    gt_boxes = dict_data['gt_boxes']
    
    pred_dict = dict_data['pred_dict'][0]
    pred_boxes = pred_dict['pred_boxes'].cpu().numpy()  # (N, 7)
    pred_scores = pred_dict['pred_scores'].cpu().numpy()  # (N,)
    pred_boxes = pred_boxes[pred_scores > 0.3]
    
    # ------------
    pc = np.concatenate([ego_pc, other_pc])
    pc_color = np.zeros((pc.shape[0], 3))
    pc_color[:ego_pc.shape[0], 2] = 1.0
    pc_color[ego_pc.shape[0]:] = np.array([0.5, 0.5, 0.5])

    boxes = np.concatenate([gt_boxes, pred_boxes])
    boxes_color = np.zeros((boxes.shape[0], 3))
    boxes_color[:gt_boxes.shape[0]] = np.array([0, 1, 0])
    boxes_color[gt_boxes.shape[0]:] = np.array([0, 0, 1])

    painter = PointsPainter(pc[:, :3], boxes)
    painter.show(pc_color, boxes_color, view_point=view_point, save_to_path=scene_output_dir / f"sample_{sample_idx}.png")


@torch.no_grad()
def main(scene_idx: int, start_idx: int = 10, end_idx: int = 90, debug: bool = False, view_point: dict = None):
    point_cloud_range = np.array([-51.2, -51.2, -8.0, 51.2, 51.2, 0.0])
    scene_output_dir = VIZ_OUTPUT_DIR / f'scene_{scene_idx}'
    if not scene_output_dir.exists():
        scene_output_dir.mkdir(parents=True)

    scene = nusc.scene[scene_idx]
    
    sample_token = scene['first_sample_token']
    for _ in range(start_idx):
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

    model_car = build_model('car')
    model_rsu = build_model('rsu')
    model_ego_collab = build_model('collab')

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

        # x, y, z, instensity, time-lag | dx, dy, dz, heading, box-score, box-label | sweep_idx, inst_idx
        points_ = np.zeros((ego_pc.shape[0], 5 + 6 + 2))
        points_[:, :5] = ego_pc[:, :5]
        points_[:, -2:] = ego_pc[:, -2:]
        max_sweep_idx = ego_pc[:, -2].max()

        # for computation: get point cloud PREV of other agents
        dict_point_clouds_prev, dict_ego_se3_lidars, _ = get_point_clouds(sample['prev'], ego_lidar_token, point_cloud_range)
        dict_modar = dict()
        for lidar_name in dict_point_clouds_prev.keys():
            if lidar_name == 'LIDAR_TOP_id_1':
                continue
            lidar_id = int(lidar_name.split('_')[-1])
            
            points = np.pad(dict_point_clouds_prev[lidar_name], pad_width=[(0, 0), (1, 0)], constant_values=0.0)  # pad batch_idx
            batch_dict = {
                'points': torch.from_numpy(points).float().cuda(),
                'batch_size': 1,
                'metadata': [{
                    'lidar_id': lidar_id,
                    'sample_token': sample['prev'],
                    'batch_size': 1,
                }] 
            }
            if lidar_id == 0:
                pred_dicts, batch_dict = model_rsu(batch_dict)
            else:
                pred_dicts, batch_dict = model_car(batch_dict)

            modar = batch_dict['mo_pts']
            # propagate MoDAR
            propagate_modar(modar, batch_dict['scene_flow'])

            # map modar (center & heading) to target frame
            modar = modar.cpu().numpy()
            modar[:, :7] = apply_se3_(dict_ego_se3_lidars[lidar_name], boxes_=modar[:, :7], return_transformed=True)

            # format modar
            modar_ = np.zeros((modar.shape[0], points_.shape[1]))
            modar_[:, :3] = modar[:, :3]
            modar_[:, 4] = 0.  # after offset, trying set it to zero
            modar_[:, 5: 11] = modar[:, 3:]
            modar_[:, -2] = max_sweep_idx
            modar_[:, -1] = -1  # dummy instance_idx

            # store
            dict_modar[lidar_name] = modar_

        # merge modar & ego_points
        points_ = np.concatenate([points_,] + [modar for _, modar in dict_modar.items()], axis=0)
        points_ = np.pad(points_, pad_width=[(0, 0), (1, 0)], constant_values=0.0)
        _batch_dict = {
            'points': torch.from_numpy(points_).float().cuda(),
            'batch_size': 1,
            'metadata': [{
                'lidar_id': 1,
                'sample_token': sample_token,
                'batch_size': 1,
            }] 
        }
        pred_dicts, _ = model_ego_collab(_batch_dict)

        dict_viz = dict()
        dict_viz['ego_pc_now'] = ego_pc
        dict_viz['other_pc_now'] = other_pc
        dict_viz['gt_boxes'] = gt_boxes
        dict_viz['pred_dict'] = pred_dicts
        if debug:
            logger.info(f"debug dict is saved to {VIZ_OUTPUT_DIR}")
            torch.save(dict_viz, VIZ_OUTPUT_DIR / 'dict_debug_visualize_collab.pth')
            return
        else:
            logger.info(f'saving output of sample {sample_idx}')
            draw_scene(dict_viz, sample_idx, scene_output_dir, view_point)

        # move to next sample
        sample_token = sample['next']

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--scene_idx', type=int, default=0, help='specify scene for visualization')
    args = parser.parse_args()

    scene_idx = args.scene_idx
    if scene_idx == 0:
        view_point = {
            "front" : [ -0.69790464097988081, -0.039023220234865395, 0.71512677224478527 ],
            "lookat" : [ 26.769546783064907, 0.072182490889707351, -15.202964500520691 ],
            "up" : [ 0.71547012226586004, 0.00679146569914983, 0.69861032066419726 ],
            "zoom" : 0.31999999999999962
        }
    else:
        logger.info(f'view_point for scene {scene_idx} is not pre-fixed -> please set it yourself')
        view_point = None
    main(scene_idx, debug=False, view_point=view_point)
    