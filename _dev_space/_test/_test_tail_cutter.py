import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import load_data_to_gpu
import matplotlib.pyplot as plt
from _dev_space.tools_box import show_pointcloud
from _dev_space.viz_tools import viz_boxes, print_dict
from _dev_space.tail_cutter import PillarEncoder
from _dev_space.tools_4testing import BackwardHook


np.random.seed(666)
cfg_file = '../../tools/cfgs/dataset_configs/nuscenes_dataset.yaml'
cfg_from_yaml_file(cfg_file, cfg)

logger = common_utils.create_logger('./dummy_log.txt')

cfg.CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
    'placeholder',
    'random_world_flip', 'random_world_scaling',
    'gt_sampling',
    'random_world_rotation',
]

cfg.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.POINT_FEATURE_ENCODING.src_feature_list = ['x', 'y', 'z', 'intensity', 'timestamp', 'offset_x', 'offset_y', 'indicator']
cfg.VERSION = 'v1.0-mini'
cfg.DATA_PATH = '../../data/nuscenes'

dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                          logger=logger, training=False, total_epochs=1, seed=666)
iter_dataloader = iter(dataloader)
for _ in range(5):
    data_dict = next(iter_dataloader)

print_dict(data_dict)
load_data_to_gpu(data_dict)

pillar_encoder = PillarEncoder(n_raw_feat=5, n_bev_feat=32, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                               voxel_size=[0.2, 0.2, 8.0])
pillar_encoder.cuda()

bev_img = pillar_encoder(data_dict)
print(f"bev_img: {bev_img.shape}")

bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in pillar_encoder.named_parameters()]

label = torch.rand(bev_img.shape).cuda()
loss = torch.sum(label - bev_img)
pillar_encoder.zero_grad()
loss.backward()

for hook in bw_hooks:
    if hook.grad_mag < 1e-5:
        print(f"Zero grad ({hook.grad_mag}) at {hook.name}")


