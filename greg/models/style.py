"""Style encoder + StyleGAN2 modulated convolution (used by the Regional Experts)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    """Source image I_s -> style code X_style in R^{C_style} (global average pooled CNN)."""

    def __init__(self, in_channels: int = 3, style_dim: int = 256, base: int = 32, num_down: int = 5):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base, 7, padding=3), nn.LeakyReLU(0.2)]
        ch = base
        for i in range(num_down):
            nxt = min(ch * 2, style_dim)
            layers += [nn.Conv2d(ch, nxt, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
            ch = nxt
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(ch, style_dim)
        self.style_dim = style_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).mean(dim=(2, 3))
        return self.fc(h)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated + demodulated convolution with a per-sample style.

    weight' = w * (A(style) + 1) per input channel; demodulated by the rms over
    (in, kh, kw).  Implemented with grouped convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, style_dim: int,
                 demodulate: bool = True):
        super().__init__()
        self.in_channels, self.out_channels, self.k = in_channels, out_channels, kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.affine = nn.Linear(style_dim, in_channels)
        nn.init.ones_(self.affine.bias)
        nn.init.normal_(self.affine.weight, std=0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.demodulate = demodulate

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.affine(style).view(B, 1, C, 1, 1)
        w = self.scale * self.weight.unsqueeze(0) * s                         # (B, O, I, k, k)
        if self.demodulate:
            d = torch.rsqrt((w ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            w = w * d
        w = w.view(B * self.out_channels, C, self.k, self.k)
        out = F.conv2d(x.reshape(1, B * C, H, W), w, padding=self.k // 2, groups=B)
        return out.view(B, self.out_channels, H, W) + self.bias.view(1, -1, 1, 1)


class StyledBlock(nn.Module):
    """Two modulated convs with LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)

    def forward(self, x, style):
        x = F.leaky_relu(self.conv1(x, style), 0.2)
        return F.leaky_relu(self.conv2(x, style), 0.2)
