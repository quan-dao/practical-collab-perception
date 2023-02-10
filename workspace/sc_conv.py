'''
Src: https://github.com/MCG-NKU/SCNet/blob/master/scnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import lovely_tensors as lt


lt.monkey_patch()


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out


class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        norm_layer(out_channels),
        nn.ReLU(inplace=True)
    )


class SCConvBackbone2dStride1(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.cfg = model_cfg
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        
        stem_ch = model_cfg.STEM_CHANNELS
        self.stem = nn.Sequential(
            conv_bn_relu(input_channels, stem_ch, kernel_size=3, padding=1, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer)
        )

        self.conv_skip = conv_bn_relu(stem_ch, input_channels, kernel_size=1, norm_layer=norm_layer)

        self.main_pass = nn.Sequential(
            conv_bn_relu(stem_ch, input_channels, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer),
            SCBottleneck(input_channels, input_channels, norm_layer=norm_layer),
            SCBottleneck(input_channels, input_channels, norm_layer=norm_layer),
            SCBottleneck(input_channels, input_channels, norm_layer=norm_layer),

            nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2, stride=2, bias=False),
            norm_layer(input_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_out = conv_bn_relu(2 * input_channels, model_cfg.NUM_BEV_FEATURES, kernel_size=3, padding=1)
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']  # (B, C, H, W) - BEV image
        out = self.stem(spatial_features)
        residual = self.conv_skip(out)
        out = self.main_pass(out)
        out = self.conv_out(torch.cat([out, residual], dim=1))
        data_dict['spatial_features_2d'] = out
        return data_dict


class SCConvBackbone2dStride4(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.cfg = model_cfg
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        
        stem_ch = input_channels * 2
        self.stem = nn.Sequential(
            conv_bn_relu(input_channels, stem_ch, kernel_size=3, padding=1, stride=2, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer),
            SCBottleneck(stem_ch, stem_ch, norm_layer=norm_layer)
        )  # block's stride = 2
        
        main_ch = stem_ch * 2
        self.main_pass = nn.Sequential(
            conv_bn_relu(stem_ch, main_ch, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer),
            SCBottleneck(main_ch, main_ch, norm_layer=norm_layer),
            SCBottleneck(main_ch, main_ch, norm_layer=norm_layer),
            SCBottleneck(main_ch, main_ch, norm_layer=norm_layer),

            nn.ConvTranspose2d(main_ch, main_ch, kernel_size=2, stride=2, bias=False),
            norm_layer(main_ch),
            nn.ReLU(inplace=True),
        )  # block's stride = 1
        
        self.conv_skip = conv_bn_relu(stem_ch, main_ch, kernel_size=1, norm_layer=norm_layer)  # block's stride = 1

        self.conv_out = conv_bn_relu(2 * main_ch, model_cfg.NUM_BEV_FEATURES, kernel_size=3, padding=1, stride=2)  # block's stride = 2
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']  # (B, C, H, W) - BEV image
        out = self.stem(spatial_features)
        residual = self.conv_skip(out)
        out = self.main_pass(out)
        out = self.conv_out(torch.cat([out, residual], dim=1))  # (B, num_bev_feat, H/4, W/4)
        data_dict['spatial_features_2d'] = out
        return data_dict
