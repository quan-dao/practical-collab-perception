import torch
import torch.nn as nn
import math


BN_MOMENTUM = 0.01
BN_EPS = 1e-3


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class PoseResNet(nn.Module):
    '''src: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py'''
    def __init__(self, model_cfg, input_channels):
        super(PoseResNet, self).__init__()
        block_type = BasicBlock
        self.model_cfg = model_cfg
        n_downsample_filters = model_cfg.NUM_FILTERS  # [32, 64, 128]
        layers_size = model_cfg.LAYERS_SIZE
        self.inplanes = n_downsample_filters[0]

        # ---
        # ResNet-like downsampling
        # NOTE: after downsampling, total stride=16
        # ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, n_downsample_filters[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(n_downsample_filters[0], momentum=BN_MOMENTUM),
            nn.ReLU(True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, n_downsample_filters[0], layers_size[0])
        self.layer2 = self._make_layer(block_type, n_downsample_filters[1], layers_size[1], stride=2)
        self.layer3 = self._make_layer(block_type, n_downsample_filters[2], layers_size[2], stride=2)
        # self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2)

        # ---
        # Upsampling
        # NOTE: after updsampling, total stride=2
        # ---
        self.deconv_layers = self._make_deconv_layer(model_cfg.NUM_UP_FILTERS)
        self.num_out_features = model_cfg.NUM_UP_FILTERS[-1]

    def _make_layer(self, block_type, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_type.expansion, eps=BN_EPS, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block_type(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_type.expansion
        for i in range(1, num_blocks):
            layers.append(block_type(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise NotImplementedError

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters_per_layer):
        layers = nn.ModuleList()
        for planes in num_filters_per_layer:
            fc = nn.Conv2d(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1)
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            layers.append(nn.Sequential(
                fc, nn.BatchNorm2d(planes, eps=BN_EPS, momentum=BN_MOMENTUM), nn.ReLU(True),
                up
            ))
            # update iterating var
            self.inplanes = planes

        return layers

    def forward(self, input_tensor):
        # ----
        # ResNet
        # ----
        x_stride_2 = self.conv1(input_tensor)  # 32

        x_stride_4 = self.maxpool(x_stride_2)  # 32

        x_stride_4 = self.layer1(x_stride_4)  # 32
        x_stride_8 = self.layer2(x_stride_4)  # 64
        x_stride_16 = self.layer3(x_stride_8)  # 128

        # ----
        # FPN
        # ----
        x_up_stride_8 = self.deconv_layers[0](x_stride_16) + x_stride_8  # 64
        x_up_stride_4 = self.deconv_layers[1](x_up_stride_8) + x_stride_4  # 32
        x_up_stride_2 = self.deconv_layers[2](x_up_stride_4) + x_stride_2  # 32

        out = self.deconv_layers[3](x_up_stride_2)  # 32 - same size as input
        return out
