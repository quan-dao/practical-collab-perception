import torch

from workspace.sc_conv import SCConvBackbone2d
from easydict import EasyDict as edict 
from tools import BackwardHook
import lovely_tensors as lt


lt.monkey_patch()


def main():
    input_channels = 256
    H, W = 128, 128
    B = 2

    model_cfg = edict({'STEM_CHANNELS': 128, 'NUM_BEV_FEATURES': 64})
    backbone2d = SCConvBackbone2d(model_cfg, input_channels)
    print('-----\n', 
            backbone2d,
        '\n-----',)
    backbone2d.cuda()

    bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in backbone2d.named_parameters()]

    inp = {'spatial_features': torch.rand(B, input_channels, H, W).float().cuda()}
    label = torch.rand(B, model_cfg.NUM_BEV_FEATURES, H, W).float().cuda()
    out = backbone2d(inp)
    loss = torch.abs(out['spatial_features_2d'] - label).sum()
    print('loss: ', loss)
    backbone2d.zero_grad()
    loss.backward()
    for hook in bw_hooks:
        if hook.grad_mag < 1e-3:
            print(f'zero grad at {hook.name}')


if __name__ == '__main__':
    main()
