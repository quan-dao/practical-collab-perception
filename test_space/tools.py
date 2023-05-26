import torch
import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import build_detector
import kornia


class BackwardHook:
    """Backward hook to check gradient magnitude of parameters (i.e. weights & biases)"""
    def __init__(self, name, param, is_cuda=False):
        """Constructor of BackwardHook

        Args:
            name (str): name of parameter
            param (torch.nn.Parameter): the parameter hook is registered to
            is_cuda (bool): whether parameter is on cuda or not
        """
        self.name = name
        self.hook_handle = param.register_hook(self.hook)
        self.grad_mag = -1.0
        self.is_cuda = is_cuda

    def hook(self, grad):
        """Function to be registered as backward hook

        Args:
            grad (torch.Tensor): gradient of a parameter W (i.e. dLoss/dW)
        """
        if not self.is_cuda:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach())
        else:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach().cpu())

    def remove(self):
        self.hook_handle.remove()


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


def build_dataset_for_testing(dataset_cfg_file: str, class_names: list, **kwargs):
    np.random.seed(666)
    cfg_from_yaml_file(dataset_cfg_file, cfg)
    cfg.CLASS_NAMES = class_names
    if kwargs.get('debug_dataset', False):
        cfg.DEBUG = True
        cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling', 'random_world_flip', 
                                               'random_world_rotation', 'random_world_scaling']

    if kwargs.get('version', None) is not None:
        cfg.VERSION = kwargs['version']
    logger = common_utils.create_logger('./artifact/dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, batch_size=kwargs.get('batch_size', 2),
                                              dist=False, logger=logger, training=kwargs.get('training', False), total_epochs=1, seed=666,
                                              workers=0)
    return dataset, dataloader


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
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int()
        else:
            batch_dict[key] = torch.from_numpy(val).float()
    
    if move_to_gpu:
        to_gpu(batch_dict)
