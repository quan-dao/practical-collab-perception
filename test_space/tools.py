import torch


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