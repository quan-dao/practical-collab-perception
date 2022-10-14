import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ResBlock(nn.Module):
    kernel_size = (3, 3)
    padding = (1, 1)

    def __init__(self, n_channels):
        super().__init__()
        bn2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv1 = nn.Conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm1 = bn2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, self.kernel_size, padding=self.padding, bias=False)
        self.norm2 = bn2d(n_channels)

    def forward(self, x):
        x_conv = F.relu(self.norm1(self.conv1(x)))
        x_conv = self.norm2(self.conv2(x_conv))
        x_conv = F.relu(x + x_conv)
        return x_conv


class UNet2D(nn.Module):
    def __init__(self, n_input_feat: int):
        super().__init__()
        norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        # ---
        # Downsampling
        # ---
        self.conv0 = nn.Sequential(nn.Conv2d(n_input_feat, 64, 7, padding=3, bias=False), norm2d(64))
        self.res0 = ResBlock(64)

        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), norm2d(64))  # tot_stride = 2
        self.res1 = ResBlock(64)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), norm2d(128))  # tot_stride = 4
        self.res2 = ResBlock(128)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False), norm2d(128))  # tot_stride = 8
        self.res3 = ResBlock(128)

        # ---
        # Up
        # ---
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
                                 norm2d(128))  # tot_stride = 4
        self.up_res1 = ResBlock(128)

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
                                 norm2d(128))  # tot_stride = 2
        self.up_res2 = ResBlock(128)

        self.up3 = nn.Sequential(nn.ConvTranspose2d(128 + 64, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
                                 norm2d(128))  # tot_stride = 1
        self.up_res3 = ResBlock(128)

        self.up4 = nn.Sequential(nn.Conv2d(128 + 64, 64, 1, bias=False), norm2d(64))
        self.up5 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), norm2d(64))

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

