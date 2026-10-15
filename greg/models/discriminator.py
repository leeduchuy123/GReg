"""Regional PatchGAN discriminators (Sec. 3.4, Eq. 9-10).

One PatchGAN per key region; each sees the RGB region crop concatenated with
its gradient map (4 channels).  Decision (paper ambiguity #4): the fake pair
is (I_hat_{t,r}, G_hat_{t,r}) with the gradient computed from the *generated*
region (Eq. 10's form), for both the G and D losses.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 64, num_layers: int = 3, spectral_norm: bool = True):
        super().__init__()
        sn = nn.utils.spectral_norm if spectral_norm else (lambda m: m)
        layers = [sn(nn.Conv2d(in_channels, base, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)]
        ch = base
        for i in range(1, num_layers):
            nxt = min(ch * 2, 512)
            layers += [sn(nn.Conv2d(ch, nxt, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)]
            ch = nxt
        layers += [sn(nn.Conv2d(ch, 1, 4, stride=1, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RegionalDiscriminators(nn.Module):
    def __init__(self, num_regions: int = 3, in_channels: int = 4, base: int = 64, num_layers: int = 3,
                 spectral_norm: bool = True):
        super().__init__()
        self.discs = nn.ModuleList([PatchDiscriminator(in_channels, base, num_layers, spectral_norm)
                                    for _ in range(num_regions)])

    def forward(self, regions: torch.Tensor, grads: torch.Tensor) -> List[torch.Tensor]:
        """regions (B, R, 3, s, s), grads (B, R, 1, s, s) -> list of R patch logits."""
        return [d(torch.cat([regions[:, r], grads[:, r]], dim=1)) for r, d in enumerate(self.discs)]
