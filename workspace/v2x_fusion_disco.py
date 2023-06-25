import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 64, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv1_4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1


def transform_bev_img(dst_se3_src: torch.Tensor, bev_in_src:torch.Tensor, pc_range_min: float, pix_size: float) -> torch.Tensor:
    assert len(bev_in_src.shape) == 3, "expect shape of bev_in_src == (C, H, W) "
    rot = dst_se3_src[:2, :2]
    t = dst_se3_src[:2, [-1]]
    t_pix_norm = 2.0 * ((t - pc_range_min) / pix_size) / bev_in_src.shape[1] - 1.0
    theta = torch.cat([rot.T, -torch.matmul(rot.T, t_pix_norm)], dim=1)  # (2, 3)

    # pad theta and bev_in_src with batch dimension
    theta = rearrange(theta, 'C1 C2 -> 1 C1 C2', C1=2, C2=3)
    bev_in_src = rearrange(bev_in_src, 'C H W -> 1 C H W')
    
    # warp
    grid = F.affine_grid(theta, bev_in_src.size())
    bev_in_target = F.grid_sample(bev_in_src, grid, mode='nearest')
    bev_in_target = rearrange(bev_in_target, '1 C H W -> C H W')
    return bev_in_target


class V2XMidFusionDisco(nn.Module):
    def __init__(self, model_cfg, in_channel: int = 384):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Conv2d(in_channel, model_cfg.COMPRESSED_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(model_cfg.COMPRESSED_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_cfg.COMPRESSED_CHANNELS, model_cfg.COMPRESSED_CHANNELS, kernel_size=3, stride=1, padding=1),
        )
        self.pixel_weightor = PixelWeightedFusionSoftmax(model_cfg.COMPRESSED_CHANNELS)
        self.decompressor = nn.Sequential(
            nn.Conv2d(model_cfg.COMPRESSED_CHANNELS, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
        )

        self.pc_min = model_cfg.get("PC_RANGE_MIN", -51.2)
        self.pix_size = model_cfg.get("FINAL_BEV_PIXEL_SIZE", 0.2 * 4)

    def forward(self, batch_dict: dict):
        ego_bev = self.compressor(batch_dict['spatial_features_2d'])
        all_bev, all_weights = list(), list()

        all_bev.append(ego_bev)
        all_weights.append(self.pixel_weightor(torch.cat([ego_bev, ego_bev], dim=1)))
        
        for agent_idx, bev_img in batch_dict['bev_img'].items():
            # bev_img;: (B, C_in, H, W)
            bev_img = self.compressor(bev_img)

            # warp sample in bev_img
            for b_idx, meta in enumerate(batch_dict['metadata']):
                ego_se3_agent = torch.from_numpy(np.linalg.inv(meta['se3_from_ego'][agent_idx])).float().cuda()
                bev_img[b_idx] = transform_bev_img(ego_se3_agent, bev_img[b_idx], self.pc_min, self.pix_size)

            weight = self.pixel_weightor(torch.cat([ego_bev, bev_img], dim=1))  # (B, 1, H, W)

            # store
            all_bev.append(bev_img)
            all_weights.append(weight)
        
        # normalize weights
        all_weights = F.softmax(torch.cat(all_weights, dim=1), dim=1)  # (B, num_agents, H, W)
        # weighted sum
        all_bev = torch.stack(all_bev, dim=0)  # (n_agents, B, C, H, W)
        all_weights = rearrange(all_weights, 'B num_agents H W -> num_agents B 1 H W')
        fused_bev = torch.sum(all_bev * all_weights, dim=1)

        fused_bev = self.decompressor(fused_bev)
        batch_dict['spatial_features_2d'] = fused_bev
        return batch_dict
