import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from functools import partial


class DeformConv2dPack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding,
                                     dilation, groups, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.conv_offset(x)
        out = self.deform_conv(x, offset)
        return out


class ResBlock(nn.Module):
    kernel_size = 3
    padding = 1

    def __init__(self, n_channels, use_deform_conv=False):
        super().__init__()
        bn2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        conv2d = nn.Conv2d if not use_deform_conv else DeformConv2dPack
        self.conv1 = conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm1 = bn2d(n_channels)
        self.conv2 = conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm2 = bn2d(n_channels)

    def forward(self, x):
        x_conv = F.relu(self.norm1(self.conv1(x)))
        x_conv = self.norm2(self.conv2(x_conv))
        x_conv = F.relu(x + x_conv)
        return x_conv


class UNet2D(nn.Module):
    def __init__(self, n_input_feat: int, cfg):
        super().__init__()
        norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        ch_down = cfg.DOWN_CHANNELS
        ch_up = cfg.UP_CHANNELS
        self.n_output_feat = ch_up[-1]

        # ---
        # Downsampling
        # ---
        self.conv0 = nn.Sequential(nn.Conv2d(n_input_feat, ch_down[0], 7, padding=3, bias=False), norm2d(ch_down[0]))
        self.res0 = ResBlock(ch_down[0])

        self.conv1 = nn.Sequential(nn.Conv2d(ch_down[0], ch_down[1], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[1]))  # tot_stride = 2
        self.res1 = ResBlock(ch_down[1])

        self.conv2 = nn.Sequential(nn.Conv2d(ch_down[1], ch_down[2], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[2]))  # tot_stride = 4
        self.res2 = ResBlock(ch_down[2])

        self.conv3 = nn.Sequential(nn.Conv2d(ch_down[2], ch_down[3], 3, stride=2, padding=1, bias=False),
                                   norm2d(ch_down[3]))  # tot_stride = 8
        self.res3 = ResBlock(ch_down[3])

        # ---
        # Up
        # ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch_down[3], ch_up[1], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[1])
        )  # tot_stride = 4
        self.up_res1 = ResBlock(ch_up[1], cfg.UP_DEFORM_CONV[1])

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[1] + ch_down[2], ch_up[2], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[2])
        )  # tot_stride = 2
        self.up_res2 = ResBlock(ch_up[2], cfg.UP_DEFORM_CONV[2])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ch_up[2] + ch_down[1], ch_up[3], 3, stride=2, padding=1, output_padding=1, bias=False),
            norm2d(ch_up[3])
        )  # tot_stride = 1
        self.up_res3 = ResBlock(ch_up[3], cfg.UP_DEFORM_CONV[3])

        self.up4 = nn.Sequential(
            nn.Conv2d(ch_up[3] + ch_down[0], ch_up[4], 1, bias=False) if not cfg.UP_DEFORM_CONV[4] else
            DeformConv2dPack(ch_up[3] + ch_down[0], ch_up[4], 1, bias=False),
            norm2d(ch_up[4])
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(ch_up[4], ch_up[5], 1, bias=False) if not cfg.UP_DEFORM_CONV[5] else
            DeformConv2dPack(ch_up[4], ch_up[5], 1, bias=False),
            norm2d(ch_up[5])
        )

    def forward(self, x_in):
        # ---
        # Down
        # ---
        x_out = self.conv0(x_in)
        x_skip0 = torch.clone(x_out)
        x_out = self.res0(x_out)

        x_out = self.conv1(x_out)
        x_skip1 = torch.clone(x_out)  # stride = 2
        x_out = self.res1(x_out)

        x_out = self.conv2(x_out)
        x_skip2 = torch.clone(x_out)  # stride = 4
        x_out = self.res2(x_out)

        x_out = self.conv3(x_out)  # stride = 8
        x_out = self.res3(x_out)

        # ---
        # Up
        # ---
        x_out = self.up1(x_out)  # stride = 4
        x_out = self.up_res1(x_out)

        x_out = torch.cat([x_out, x_skip2], dim=1)
        x_out = self.up2(x_out)  # stride = 2
        x_out = self.up_res2(x_out)

        x_out = torch.cat([x_out, x_skip1], dim=1)
        x_out = self.up3(x_out)  # stride = 1
        x_out = self.up_res3(x_out)

        x_out = torch.cat([x_out, x_skip0], dim=1)
        x_out = self.up4(x_out)
        x_out = self.up5(x_out)
        return x_out

