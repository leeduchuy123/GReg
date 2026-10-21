"""Geometry-aware generator (Sec. 3.1, "UNet with additive skip connections").

Input: concat(I_st, N_t, R) = 3 + 3 + 3 channels at 256x256.
Decoder levels at the resolutions in ``grm_levels`` are refined by a GRM.
Output: I_gen in [0, 1]; the caller blends with the warped source using the
occlusion mask: I_hat = O * I_st + (1 - O) * I_gen.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from greg.models.gradient_refine import GRM


def _norm(ch: int, kind: str):
    if kind == "batch":
        return nn.BatchNorm2d(ch, affine=True)
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, norm="batch"):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _norm(cout, norm), nn.LeakyReLU(0.2),
                                 nn.Conv2d(cout, cout, 3, padding=1), _norm(cout, norm), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


class GeometryAwareGenerator(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int = 3, base: int = 64, max_channels: int = 512,
                 num_down: int = 4, image_size: int = 256, grm_levels: Sequence[int] = (32, 64, 128),
                 grad_channels: Dict[int, int] | None = None, grm_heads: int = 1, norm: str = "batch",
                 zero_init_grm: bool = True, grm_attn_bf16: bool = True):
        super().__init__()
        self.image_size = image_size
        chs = [min(base * (2 ** i), max_channels) for i in range(num_down + 1)]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, chs[0], 7, padding=3), _norm(chs[0], norm), nn.LeakyReLU(0.2))
        self.enc = nn.ModuleList([ConvBlock(chs[i], chs[i + 1], norm) for i in range(num_down)])
        self.dec = nn.ModuleList([ConvBlock(chs[i + 1], chs[i], norm) for i in range(num_down)][::-1])
        self.grms = nn.ModuleDict()
        self.dec_res = []  # output resolution of each decoder block
        res = image_size // (2 ** num_down)
        for i in range(num_down):
            res *= 2
            self.dec_res.append(res)
        grad_channels = grad_channels or {}
        for res, ch in zip(self.dec_res, [chs[i] for i in range(num_down)][::-1]):
            if res in grm_levels:
                gc = grad_channels.get(res, ch)
                self.grms[str(res)] = GRM(ch, gc, grm_heads, zero_init_grm, grm_attn_bf16)
        self.head = nn.Sequential(nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(chs[0], out_channels, 7, padding=3))

    def forward(self, x: torch.Tensor, grad_feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        h = self.stem(x)
        skips = [h]
        for blk in self.enc:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        skips.pop()
        for blk, res in zip(self.dec, self.dec_res):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = blk(h) + skips.pop()          # additive skip connection
            if str(res) in self.grms:
                h = self.grms[str(res)](h, grad_feats[res])
        return torch.sigmoid(self.head(h))
