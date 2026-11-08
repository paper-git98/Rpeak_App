"""ECG U-Net with Omni-Dimensional Dynamic Convolution (architecture copied from your training code)."""
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention1d(nn.Module):
    """Shared attention encoder producing channel / filter / spatial / kernel scalings."""

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4,
                 min_channel=32):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(in_planes, attention_channel, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(num_groups=1, num_channels=attention_channel)
        self.relu = nn.ReLU(inplace=True)

        if in_planes == 1:
            self.func_channel: Callable = self._skip
        else:
            self.channel_fc = nn.Conv1d(attention_channel, in_planes, kernel_size=1, bias=True)
            self.func_channel = self._get_channel_attention
        if in_planes == groups and in_planes == out_planes:
            self.func_filter: Callable = self._skip
        else:
            self.filter_fc = nn.Conv1d(attention_channel, out_planes, kernel_size=1, bias=True)
            self.func_filter = self._get_filter_attention
        if kernel_size == 1:
            self.func_spatial: Callable = self._skip
        else:
            self.spatial_fc = nn.Conv1d(attention_channel, kernel_size, kernel_size=1, bias=True)
            self.func_spatial = self._get_spatial_attention
        if kernel_num == 1:
            self.func_kernel: Callable = self._skip
        else:
            self.kernel_fc = nn.Conv1d(attention_channel, kernel_num, kernel_size=1, bias=True)
            self.func_kernel = self._get_kernel_attention
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def _skip(_):
        return 1.0

    def _get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x) / self.temperature)

    def _get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x) / self.temperature)

    def _get_spatial_attention(self, x):
        s = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size)
        return torch.sigmoid(s / self.temperature)

    def _get_kernel_attention(self, x):
        k = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1)
        return F.softmax(k / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv1d(nn.Module):
    """1-D Omni-Dimensional Dynamic Convolution, drop-in replacement for nn.Conv1d. Input [B, C_in, L]."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super().__init__()
        if in_planes % groups != 0:
            raise ValueError(f"in_planes ({in_planes}) must be divisible by groups ({groups}).")
        if out_planes % groups != 0:
            raise ValueError(f"out_planes ({out_planes}) must be divisible by groups ({groups}).")
        if kernel_num < 1:
            raise ValueError(f"kernel_num must be ≥ 1, got {kernel_num}.")
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        self.attention = Attention1d(in_planes, out_planes, kernel_size, groups=groups,
                                     reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.empty(kernel_num, out_planes, in_planes // groups, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()
        if kernel_size == 1 and kernel_num == 1:
            self._forward_impl = self._forward_pw1x
        else:
            self._forward_impl = self._forward_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out", nonlinearity="relu")

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, _, length = x.size()
        x = x * channel_attention
        x = x.reshape(1, batch_size * self.in_planes, length)
        aggregate_weight = self.weight.unsqueeze(0) * spatial_attention * kernel_attention
        aggregate_weight = aggregate_weight.sum(dim=1)
        aggregate_weight = aggregate_weight.view(batch_size * self.out_planes,
                                                 self.in_planes // self.groups, self.kernel_size)
        output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, -1)
        return output * filter_attention

    def _forward_pw1x(self, x):
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        output = F.conv1d(x, weight=self.weight.squeeze(0), bias=None, stride=self.stride,
                          padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output * filter_attention

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"ODConv1d expects [B, C, L], got {tuple(x.shape)}.")
        if x.size(1) != self.in_planes:
            raise ValueError(f"Channel mismatch: input has C={x.size(1)}, "
                             f"but ODConv1d was built with in_planes={self.in_planes}.")
        return self._forward_impl(x)

    def extra_repr(self):
        return (f"in={self.in_planes}, out={self.out_planes}, k={self.kernel_size}, stride={self.stride}, "
                f"pad={self.padding}, groups={self.groups}, kernel_num={self.kernel_num}")


class ODConvBlock(nn.Module):
    """Two ODConv1d layers, each followed by BN + ReLU (same padding)."""

    def __init__(self, in_ch, out_ch, kernel_size=9, kernel_num=4, reduction=0.0625, dropout=0.05):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            ODConv1d(in_ch, out_ch, kernel_size, padding=pad, kernel_num=kernel_num, reduction=reduction),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ODConv1d(out_ch, out_ch, kernel_size, padding=pad, kernel_num=kernel_num, reduction=reduction),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """Ordinary Conv1d block, used only in the decoder."""

    def __init__(self, in_ch, out_ch, kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: interpolate, reduce channels, concat the skip connection, ConvBlock, dropout."""

    def __init__(self, in_ch, out_ch, kernel_size=9, kernel_num=4, reduction=0.0625, dropout=0.05):
        super().__init__()
        self.reduce = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.conv = self.conv = ConvBlock(in_ch=2 * out_ch, out_ch=out_ch, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        x = self.reduce(x)
        if x.size(-1) != skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class ECGUNet(nn.Module):
    """1-D U-Net for single-lead ECG. Input [B, 1, L] -> raw logits [B, out_channels, L]."""

    def __init__(self, in_channels=1, out_channels=1, base_filters=32, kernel_size=9, kernel_num=4,
                 reduction=0.0625):
        super().__init__()
        f = base_filters
        self.enc1 = ODConvBlock(in_channels, f, kernel_size, kernel_num, reduction)
        self.enc2 = nn.Sequential(ODConvBlock(f, f * 2, kernel_size, kernel_num, reduction), nn.Dropout(0.05))
        self.enc3 = nn.Sequential(ODConvBlock(f * 2, f * 4, kernel_size, kernel_num, reduction), nn.Dropout(0.05))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(ODConvBlock(f * 4, f * 8, kernel_size, kernel_num, reduction),
                                        nn.Dropout(p=0.25))
        self.up1 = UpBlock(f * 8, f * 4, kernel_size, kernel_num, reduction, dropout=0.05)
        self.up2 = UpBlock(f * 4, f * 2, kernel_size, kernel_num, reduction, dropout=0.0)
        self.up3 = UpBlock(f * 2, f, kernel_size, kernel_num, reduction, dropout=0.0)
        self.head = nn.Conv1d(f, out_channels, kernel_size=1)

    def update_temperature(self, temperature):
        for m in self.modules():
            if isinstance(m, ODConv1d):
                m.update_temperature(temperature)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, 1, L], got {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"ECGUNet requires C_in=1, got C_in={x.size(1)}.")
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        neck = self.bottleneck(self.pool(s3))
        d1 = self.up1(neck, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        return self.head(d3)
