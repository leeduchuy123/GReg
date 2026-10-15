"""Regional Prior Generation (Sec. 3.2).

A Regional Expert (RE) is a UNet with StyleGAN2 modulated convolutions.  It
maps a cropped target normal region N_r (3, Hr, Wr) plus the style code of the
source image to a 1-channel structural prior R_r (Hr, Wr).

Decision (paper ambiguity #3): R_r is trained to match the *gradient map of
the target RGB region* (Eq. 2 applied to the grayscale target patch), i.e. the
RE predicts the edge structure of the real region, conditioned on the source
appearance.  Table 2 ("Grayscale" / "RGB" ablations) is reproduced by the
``target`` option of :class:`RegionalPriorGenerator`.

Two experts: one for the face and one shared by both hands (as in the paper).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from greg.data.regions import NUM_REGIONS, crop_regions, paste_regions
from greg.models.style import ModulatedConv2d, StyledBlock


class RegionalExpert(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, style_dim: int = 256,
                 channels=(64, 128, 256, 256)):
        super().__init__()
        self.inc = ModulatedConv2d(in_channels, channels[0], 3, style_dim)
        self.down = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down.append(StyledBlock(channels[i], channels[i + 1], style_dim))
        self.mid = StyledBlock(channels[-1], channels[-1], style_dim)
        self.up = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up.append(StyledBlock(channels[i] + channels[i - 1], channels[i - 1], style_dim))
        self.outc = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.inc(x, style), 0.2)
        skips: List[torch.Tensor] = [h]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2), style)
            skips.append(h)
        h = self.mid(h, style)
        for blk in self.up:
            skip = skips[-2]
            skips.pop()
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], dim=1), style)
        return torch.sigmoid(self.outc(h))


class RegionalPriorGenerator(nn.Module):
    """Crops the K=3 key regions from N_t, runs the experts, pastes the priors
    back into blank canvases -> R (B, 3, H, W)."""

    def __init__(self, style_dim: int = 256, region_size: int = 64, out_channels: int = 1,
                 channels=(64, 128, 256, 256), share_hand_expert: bool = True):
        super().__init__()
        self.region_size = region_size
        self.out_channels = out_channels
        self.face_expert = RegionalExpert(3, out_channels, style_dim, channels)
        self.hand_expert = RegionalExpert(3, out_channels, style_dim, channels)
        self.rhand_expert = None if share_hand_expert else RegionalExpert(3, out_channels, style_dim, channels)

    def forward(self, normal_target: torch.Tensor, boxes: torch.Tensor, style: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        B, _, H, W = normal_target.shape
        crops = crop_regions(normal_target, boxes, self.region_size)        # (B, 3, 3, s, s)
        priors = [self.face_expert(crops[:, 0], style),
                  self.hand_expert(crops[:, 1], style),
                  (self.rhand_expert or self.hand_expert)(crops[:, 2], style)]
        prior_patches = torch.stack(priors, dim=1)                          # (B, 3, C, s, s)
        canvas = paste_regions(prior_patches, boxes, H, W)                  # (B, 3*C, H, W)
        return {"prior_patches": prior_patches, "prior_map": canvas, "normal_crops": crops}
