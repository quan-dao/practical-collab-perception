import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu

from _dev_space.tail_cutter import PointAligner
from _dev_space.viz_tools import print_dict


cfg_file = './tail_cutter_cfg.yaml'
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger('./dummy_log.txt')

dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                          logger=logger, training=False, total_epochs=1, seed=666)
iter_dataloader = iter(dataloader)
for _ in range(5):
    data_dict = next(iter_dataloader)

print_dict(data_dict)
print('---\n---')

net = PointAligner(cfg.MODEL)
print(net)

net.cuda()
load_data_to_gpu(data_dict)

with torch.no_grad():
    data_dict = net(data_dict)
