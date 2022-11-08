import torch
from _dev_space.tools_4testing import BackwardHook
from _dev_space.unet_2d import UNet2D


img = torch.rand(2, 32, 256, 256).cuda()
net = UNet2D(32).cuda()
out = net(img)
bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in net.named_parameters()]

label = torch.rand(out.shape).cuda()
loss = torch.sum(label - out)
net.zero_grad()
loss.backward()

for hook in bw_hooks:
    if hook.grad_mag < 1e-5:
        print(f"Zero grad ({hook.grad_mag}) at {hook.name}")