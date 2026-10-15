"""VGG-19 perceptual features (relu1_1 ... relu5_1, ImageNet normalisation) and
the multi-scale perceptual loss used by TPSMM / GReg (Eq. 6)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels

from greg.models.blocks import AntiAliasInterpolation2d


class VGG19Features(nn.Module):
    SLICES = ((0, 2), (2, 7), (7, 12), (12, 21), (21, 30))

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvmodels.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        feats = tvmodels.vgg19(weights=weights).features
        self.slices = nn.ModuleList([nn.Sequential(*[feats[i] for i in range(a, b)]) for a, b in self.SLICES])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        h = (x - self.mean) / self.std
        out = []
        for s in self.slices:
            h = s(h)
            out.append(h)
        return out

    def train(self, mode: bool = True):  # always frozen
        return super().train(False)


class PerceptualLoss(nn.Module):
    """sum_scales sum_l w_l |phi_l(x) - phi_l(y)|_1 (TPSMM pyramid perceptual loss)."""

    def __init__(self, vgg: VGG19Features, scales: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
                 layer_weights: Sequence[float] = (10, 10, 10, 10, 10), num_channels: int = 3):
        super().__init__()
        self.vgg = vgg
        self.scales = list(scales)
        self.layer_weights = list(layer_weights)
        self.pyramid = nn.ModuleDict({str(s).replace(".", "_"): AntiAliasInterpolation2d(num_channels, s)
                                      for s in self.scales})

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = x.new_zeros(())
        for s in self.scales:
            down = self.pyramid[str(s).replace(".", "_")]
            xs, ys = down(x), down(y)
            if min(xs.shape[-2:]) < 8:
                continue
            fx, fy = self.vgg(xs), self.vgg(ys.detach())
            for w, a, b in zip(self.layer_weights, fx, fy):
                total = total + w * (a - b).abs().mean()
        return total
