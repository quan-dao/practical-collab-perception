import numpy as np
import torch
import kornia
from _dev_space.tools_box import show_pointcloud
from _dev_space.viz_tools import viz_boxes


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


def load_data_to_tensor(batch_dict):
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


def show_points_in_batch_dict(batch_dict, batch_idx, points=None, points_color=None):
    if points is None:
        points = batch_dict['points']

    assert isinstance(points, torch.Tensor)
    _points = points[points[:, 0].long() == batch_idx]

    gt_boxes = batch_dict['gt_boxes']
    _boxes = gt_boxes[batch_idx]  # (N_max_gt, 10)
    valid_gt_boxes = torch.linalg.norm(_boxes, dim=1) > 0
    _boxes = viz_boxes(_boxes[valid_gt_boxes].numpy())

    if points_color is None:
        show_pointcloud(_points[:, 1: 4], _boxes, fgr_mask=_points[:, -1] > -1)
    else:
        show_pointcloud(_points[:, 1: 4], _boxes, pc_colors=points_color[points[:, 0].long() == batch_idx])
