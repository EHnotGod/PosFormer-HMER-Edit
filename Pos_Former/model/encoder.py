import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc

# ------------------------------
# SE Layer
# ------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight

# ------------------------------
# DenseNet Components
# ------------------------------
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)

        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        self.se1 = SELayer(n_channels)
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        self.se2 = SELayer(n_channels)
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        self.se3 = SELayer(n_channels)

        self.out_channels = n_channels
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]

        out = self.dense1(out)
        out = self.se1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]

        out = self.dense2(out)
        out = self.se2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]

        out = self.dense3(out)
        out = self.se3(out)
        out = self.post_norm(out)
        return out, out_mask


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)
        self.feature_proj = nn.Conv2d(self.model.out_channels, d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dropout = nn.Dropout(0.1)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)
        feature = self.depthwise_conv(feature)
        feature = self.dropout(feature)
        feature = rearrange(feature, "b d h w -> b h w d")
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)
        return feature, mask
