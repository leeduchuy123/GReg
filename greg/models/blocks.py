"""Common building blocks (conventions follow TPSMM / FOMM)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coordinate_grid(spatial_size, dtype=torch.float32, device=None) -> torch.Tensor:
    """(H, W, 2) grid of (x, y) in [-1, 1] (align_corners=True convention)."""
    h, w = spatial_size
    x = torch.linspace(-1, 1, w, dtype=dtype, device=device)
    y = torch.linspace(-1, 1, h, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def kp2gaussian(kp: torch.Tensor, spatial_size, kp_variance: float) -> torch.Tensor:
    """(B, K, 2) keypoints in [-1, 1] -> (B, K, H, W) Gaussian heatmaps."""
    grid = make_coordinate_grid(spatial_size, kp.dtype, kp.device)      # (H, W, 2)
    grid = grid.view(1, 1, *spatial_size, 2)
    mean = kp.view(kp.shape[0], kp.shape[1], 1, 1, 2)
    return torch.exp(-0.5 * ((grid - mean) ** 2).sum(-1) / kp_variance)


class AntiAliasInterpolation2d(nn.Module):
    """Gaussian blur + strided subsampling (TPSMM's downscale)."""

    def __init__(self, channels: int, scale: float):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        kernel = 1
        mesh = torch.meshgrid([torch.arange(kernel_size, dtype=torch.float32)] * 2, indexing="ij")
        for size, std, m in zip([kernel_size] * 2, [sigma] * 2, mesh):
            mean = (size - 1) / 2
            kernel = kernel * torch.exp(-((m - mean) ** 2) / (2 * std ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale
        self.int_inv_scale = int(1 / scale)

    def forward(self, x):
        if self.scale == 1.0:
            return x
        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        return out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]


class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, lrelu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.act = nn.LeakyReLU(0.2) if lrelu else nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return F.relu(self.norm(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, in_features, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return out + x


class Encoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                      min(max_features, block_expansion * (2 ** (i + 1)))))
        self.down_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x]
        for b in self.down_blocks:
            outs.append(b(outs[-1]))
        return outs


class Decoder(nn.Module):
    """Hourglass decoder returning every level (used by the dense motion net)."""

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        up_blocks = []
        self.level_channels = []  # channels of each returned level, coarse -> fine
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters))
            skip = in_features if i == 0 else min(max_features, block_expansion * (2 ** i))
            self.level_channels.append(out_filters + skip)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
        self.conv = nn.Conv2d(self.out_filters, self.out_filters, 3, padding=1)
        self.norm = nn.BatchNorm2d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        outs = []
        for up in self.up_blocks:
            out = up(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        outs[-1] = F.relu(self.norm(self.conv(outs[-1])))
        return outs  # coarse -> fine; last has out_filters channels


class Hourglass(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.level_channels = self.decoder.level_channels
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))
