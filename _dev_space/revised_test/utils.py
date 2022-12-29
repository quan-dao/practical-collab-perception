from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space._test.tools_4testing import load_data_to_tensor
from _dev_space.viz_tools import print_dict, viz_boxes
from _dev_space.tools_box import show_pointcloud
import matplotlib.pyplot as plt
import numpy as np


def make_batch_dict(is_training: bool, batch_size=1, target_batch_idx=5) -> dict:
    cfg_file = './second_aligner_mini.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./a_dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                              batch_size=batch_size, dist=False, logger=logger, training=is_training,
                                              total_epochs=1, seed=666, workers=1)
    iter_dataloader = iter(dataloader)

    batch_dict = None
    for _ in range(target_batch_idx):
        batch_dict = next(iter_dataloader)

    return batch_dict


def color_points_by_detection_cls(points) -> np.ndarray:
    cls_colors = plt.cm.rainbow(np.linspace(0, 1, 11))[:, :3]
    idx_cls = 9
    points_cls_idx = points[:, idx_cls].astype(int)
    points_colors = cls_colors[points_cls_idx]
    # zero-out color of background
    mask_detection_fg = points_cls_idx >= 1
    points_colors *= mask_detection_fg.reshape(-1, 1)
    return points_colors
