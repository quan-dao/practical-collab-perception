import torch
import torch.nn as nn
import math

from pcdet.utils.loss_utils import FocalLossCenterNet
from kornia.losses.focal import FocalLoss  # for multiclass focal loss
from _dev_space.bev_segmentation_utils import assign_target_foreground_seg, compute_cls_stats, sigmoid


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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

        out += residual
        out = self.relu(out)

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


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    '''src: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py'''
    def __init__(self, model_cfg, input_channels):
        super(PoseResNet, self).__init__()
        block_type = BasicBlock
        self.model_cfg = model_cfg
        n_downsample_filters = model_cfg.NUM_FILTERS  # [32, 64, 128]
        layers_size = model_cfg.LAYERS_SIZE
        head_conv = model_cfg.HEAD_CONV
        self.inplanes = n_downsample_filters[0]
        self.heads = {
            'cls': 1,  # binary classification: foreground prob
            'to_mean': 2,  # to_cluster_center_x, to_cluster_center_y
            'crt_dir_cls': model_cfg.CRT_DIR_NUM_BINS + 1,  # 1 extra class for outlier
            'crt_dir_res': 1,
            'crt_cls': model_cfg.CRT_NUM_BINS + 1,  # 1 extra class for outlier
        }
        self.deconv_with_bias = False

        self.num_bev_features = n_downsample_filters[0]
        self.forward_ret_dict = {}

        # ---
        # ResNet-like downsampling
        # NOTE: after downsampling, total stride=16
        # ---
        self.conv1 = nn.Conv2d(input_channels, n_downsample_filters[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_downsample_filters[0], momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, n_downsample_filters[0], layers_size[0])
        self.layer2 = self._make_layer(block_type, n_downsample_filters[1], layers_size[1], stride=2)
        self.layer3 = self._make_layer(block_type, n_downsample_filters[2], layers_size[2], stride=2)
        # self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2)

        # ---
        # Upsampling
        # NOTE: after updsampling, total stride=2
        # ---
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,  # up samples 3 times
            num_filters=model_cfg.NUM_UP_FILTERS,
            kernels_size=[4, 4, 4],
        )

        # ---
        # output Head
        # ---
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(n_downsample_filters[0], head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
                if 'cls' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'cls' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(f"head_{head}", fc)

        self.loss_cls = FocalLossCenterNet()
        self.loss_crt_cls = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

    def _make_layer(self, block_type, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_type.expansion, momentum=BN_MOMENTUM),
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

    def _make_deconv_layer(self, num_layers, num_filters, kernels_size):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(kernels_size), 'ERROR: num_deconv_layers is different len(kernels_size)'

        layers = nn.ModuleList()
        for i in range(num_layers):
            up_layer = []
            kernel, padding, output_padding = \
                self._get_deconv_cfg(kernels_size[i])

            planes = num_filters[i]
            fc = nn.Conv2d(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            up_layer.append(fc)
            up_layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            up_layer.append(nn.ReLU(inplace=True))
            up_layer.append(up)
            up_layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            up_layer.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            layers.append(nn.Sequential(*up_layer))

        return layers

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        x = self.conv1(x)  # stride 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # stride 4

        x_stride_4 = self.layer1(x)
        x_stride_8 = self.layer2(x_stride_4)
        x_stride_16 = self.layer3(x_stride_8)

        x_up_stride_8 = self.deconv_layers[0](x_stride_16) + x_stride_8
        x_up_stride_4 = self.deconv_layers[1](x_up_stride_8) + x_stride_4
        x_up_stride_2 = self.deconv_layers[2](x_up_stride_4)

        data_dict['spatial_features_2d'] = x_up_stride_2

        pred_dict = {}
        for head in self.heads:
            pred_dict[f"pred_{head}"] = self.__getattr__(f"head_{head}")(x_up_stride_2)
        self.forward_ret_dict['pred_dict'] = pred_dict

        if self.training or self.model_cfg.get('GENERATE_TARGET_WHILE_TESTING', False):
            target_dict = assign_target_foreground_seg(data_dict, input_stride=self.model_cfg.BEV_IMG_STRIDE,
                                                       crt_mag_max=self.model_cfg.CRT_MAX_MAGNITUDE,
                                                       crt_num_bins=self.model_cfg.CRT_NUM_BINS,
                                                       crt_dir_num_bins=self.model_cfg.CRT_DIR_NUM_BINS)
            self.forward_ret_dict['target_dict'] = target_dict
            data_dict['bev_target_dict'] = target_dict

        data_dict['bev_pred_dict'] = self.forward_ret_dict['pred_dict']

        return data_dict

    def get_loss(self, tb_dict=None):
        if tb_dict is None:
            tb_dict = dict()
        target_dict = self.forward_ret_dict['target_dict']
        pred_dict = self.forward_ret_dict['pred_dict']

        # ---
        # loss: classification - for both bgr & fgr
        # ---
        target_cls = target_dict['target_cls'].unsqueeze(1).contiguous()  # (B, 1, H, W)
        pred_cls = sigmoid(pred_dict['pred_cls'])  # (B, 1, H, W)
        loss_fgr_seg = self.loss_cls(pred_cls, target_cls) * self.model_cfg.LOSS_WEIGHTS[0]
        tb_dict['bev_loss_cls'] = loss_fgr_seg.item()

        # compute segmentation statistic
        for threshold in [0.3, 0.5, 0.7]:
            stats = compute_cls_stats(pred_cls.detach().reshape(-1), target_cls.reshape(-1), threshold)
            for k, v in stats.items():
                tb_dict[f'seg_{k}_{threshold}'] = v

        # ---
        # loss: regression to mean - for fgr only
        # ---
        mask_fgr = target_cls > 0  # (B, 1, H, W)
        has_fgr = torch.any(mask_fgr)
        if has_fgr:
            target_to_mean = target_dict['target_to_mean']  # (2, B, H, W) - offset_to_mean (x & y)
            target_to_mean = target_to_mean.permute(1, 0, 2, 3).contiguous()  # (B, 2, H, W)
            pred_to_mean = pred_dict['pred_to_mean']  # (B, 2, H, W)
            __mask_fgr = mask_fgr.repeat(1, 2, 1, 1)
            loss_to_mean = self.model_cfg.LOSS_WEIGHTS[1] * \
                            nn.functional.l1_loss(pred_to_mean[__mask_fgr], target_to_mean[__mask_fgr], reduction='mean')
            tb_dict['bev_loss_to_mean'] = loss_to_mean.item()
        else:
            loss_to_mean = 0
            tb_dict['bev_loss_to_mean'] = 0.

        # ---
        # loss: correction - for fgr only
        # ---
        if has_fgr:
            target_crt_cls = target_dict['target_crt_cls']  # (B, H, W)
            pred_crt_cls = pred_dict['pred_crt_cls']  # (B, D+1, H, W)
            loss_crt_cls = self.loss_crt_cls(pred_crt_cls, target_crt_cls)  # (B, H, W)
            # apply foreground mask
            loss_crt_cls = loss_crt_cls * mask_fgr.squeeze(1)
            # normalize
            loss_crt_cls = self.model_cfg.LOSS_WEIGHTS[2] * torch.sum(loss_crt_cls) / torch.sum(mask_fgr.int())
            tb_dict['bev_loss_crt_cls'] = loss_crt_cls.item()
        else:
            loss_crt_cls = 0
            tb_dict['bev_loss_crt_cls'] = 0.

        if has_fgr:
            target_crt_dir_cls = target_dict['target_crt_dir_cls']  # (B, H, W)
            pred_crt_dir_cls = pred_dict['pred_crt_dir_cls']  # (B, D+1, H, W)

            loss_crt_dir_cls = self.loss_crt_cls(pred_crt_dir_cls, target_crt_dir_cls)  # (B, H, W)
            # apply foreground mask
            loss_crt_dir_cls = loss_crt_dir_cls * mask_fgr.squeeze(1)
            # normalize
            loss_crt_dir_cls = self.model_cfg.LOSS_WEIGHTS[3] * torch.sum(loss_crt_dir_cls) / torch.sum(mask_fgr.int())
            tb_dict['bev_loss_crt_dir_cls'] = loss_crt_dir_cls.item()
        else:
            loss_crt_dir_cls = 0
            tb_dict['bev_loss_crt_dir_cls'] = 0.

        if has_fgr:
            target_crt_dir_res = target_dict['target_crt_dir_res'].unsqueeze(1)  # (B, 1, H, W)
            pred_crt_dir_res = pred_dict['pred_crt_dir_res']  # (B, 1, H, W)
            loss_crt_dir_res = self.model_cfg.LOSS_WEIGHTS[4] * \
                               nn.functional.l1_loss(pred_crt_dir_res[mask_fgr], target_crt_dir_res[mask_fgr],
                                                     reduction='mean')
            tb_dict['bev_loss_crt_dir_res'] = loss_crt_dir_res.item()
        else:
            loss_crt_dir_res = 0
            tb_dict['bev_loss_crt_dir_res'] = 0.

        bev_loss = loss_fgr_seg + loss_to_mean + loss_crt_cls + loss_crt_dir_cls + loss_crt_dir_res
        tb_dict['bev_loss'] = bev_loss.item()
        return bev_loss, tb_dict
