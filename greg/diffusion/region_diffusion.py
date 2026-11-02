"""D2: Regional Expert as a conditional diffusion (flow-matching) model.

GReg's Regional Expert regresses a deterministic 1-channel gradient prior of a 64x64 region from the target normal
crop and the source style code, trained with L1 (L_struct).  When a hand is self-occluded, turned, or crossing the
other hand, the true structure given only (normal crop, style) is multi-modal and an L1 regressor returns the blurry
median.  Here the same network (``RegionalExpert``: StyleGAN2-modulated UNet, identical channels -> identical
capacity) predicts the flow-matching velocity instead, with the timestep embedded and concatenated to the style:

    x0 = 2 * struct - 1 (struct = gradient map of the real target crop, clipped to [0, 1]),  x1 = eps ~ N(0, I)
    x_t = (1 - t) x0 + t eps,       target v = eps - x0,        L = || v_theta(x_t, N_crop, style, t) - v ||^2
    sampling: Euler from t = 1 to 0 in ``steps`` steps;  prior = (x0_hat + 1) / 2 clipped to [0, 1]

(the rectified-flow parameterisation of Wan / DWM, docs/bai_hoc_diffusion_mesh_cho_greg.md Sec. 2.2.)

The sampled prior goes to GReg exactly where the regression prior went (``RegionalPriorGenerator`` interface:
``prior_patches`` / ``prior_map`` / ``normal_crops``), so the generator, GRM and losses are untouched.

Noise: ``noise_mode=fixed`` uses one noise tensor (seed ``noise_seed``) for every sample and frame -> deterministic
output, and consecutive frames of a clip only differ by their conditions (no flicker from re-sampling);
``random`` re-draws per call (ablation 3 of D2).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from greg.data.regions import crop_regions, paste_regions
from greg.models.regional_expert import RegionalExpert


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) in [0, 1] -> (B, dim) sinusoidal embedding (t scaled to 0..1000 like DDPM timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=1)


class FlowExpert(nn.Module):
    """One diffusion expert: (x_t, normal crop, style, t) -> velocity."""

    def __init__(self, style_dim: int = 256, channels=(64, 128, 256, 256), t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.net = RegionalExpert(in_channels=4, out_channels=1, style_dim=style_dim + t_dim, channels=channels,
                                  out_activation="none")

    def forward(self, x_t: torch.Tensor, normal: torch.Tensor, style: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([style, self.t_mlp(timestep_embedding(t, self.t_dim))], dim=1)
        return self.net(torch.cat([x_t, normal * 2 - 1], dim=1), cond)


class DiffusionRegionalPriorGenerator(nn.Module):
    """Drop-in replacement of ``RegionalPriorGenerator`` whose priors are sampled from flow-matching experts."""

    def __init__(self, style_dim: int = 256, region_size: int = 64, channels=(64, 128, 256, 256),
                 share_hand_expert: bool = True, steps: int = 8, noise_mode: str = "fixed", noise_seed: int = 0,
                 style_dropout: float = 0.0):
        super().__init__()
        self.region_size, self.out_channels = region_size, 1
        self.face_expert = FlowExpert(style_dim, channels)
        self.hand_expert = FlowExpert(style_dim, channels)
        self.rhand_expert = None if share_hand_expert else FlowExpert(style_dim, channels)
        self.steps, self.noise_mode, self.noise_seed = int(steps), str(noise_mode), int(noise_seed)
        if self.noise_mode not in ("fixed", "random"):
            raise ValueError(f"noise_mode must be fixed or random, got {noise_mode!r}")
        self.style_dropout = float(style_dropout)

    def experts(self):
        return [self.face_expert, self.hand_expert, self.rhand_expert or self.hand_expert]

    # --------------------------------------------------------------- noise
    def initial_noise(self, B: int, device, dtype=torch.float32) -> torch.Tensor:
        s = self.region_size
        if self.noise_mode == "random":
            return torch.randn(B, 3, 1, s, s, device=device, dtype=dtype)
        g = torch.Generator(device="cpu").manual_seed(self.noise_seed)
        n = torch.randn(1, 3, 1, s, s, generator=g)                     # same for every sample / frame
        return n.to(device=device, dtype=dtype).expand(B, -1, -1, -1, -1).contiguous()

    # --------------------------------------------------------------- training
    def training_loss(self, normal_crops: torch.Tensor, style: torch.Tensor, struct: torch.Tensor,
                      region_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """normal_crops (B, 3, 3, s, s), style (B, D), struct (B, 3, 1, s, s) in [0, ~1.4] -> flow-matching loss."""
        B = struct.shape[0]
        x0 = struct.clamp(0, 1) * 2 - 1
        eps = torch.randn_like(x0)
        t = torch.rand(B, 3, device=x0.device)
        tt = t[..., None, None, None]
        x_t = (1 - tt) * x0 + tt * eps
        if self.training and self.style_dropout > 0:
            keep = (torch.rand(B, 1, device=style.device) >= self.style_dropout).to(style.dtype)
            style = style * keep
        per = []
        for r, ex in enumerate(self.experts()):
            v = ex(x_t[:, r], normal_crops[:, r], style, t[:, r])
            per.append(((v - (eps[:, r] - x0[:, r])) ** 2).mean(dim=(1, 2, 3)))
        per = torch.stack(per, 1)                                          # (B, 3)
        w = region_weight if region_weight is not None else torch.ones_like(per)
        return {"fm": (per * w).sum() / w.sum().clamp(min=1), "fm_face": per[:, 0].mean(), "fm_hands": per[:, 1:].mean()}

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, normal_crops: torch.Tensor, style: torch.Tensor, noise: Optional[torch.Tensor] = None,
               steps: Optional[int] = None) -> torch.Tensor:
        B = normal_crops.shape[0]
        x = self.initial_noise(B, normal_crops.device) if noise is None else noise
        n = int(steps or self.steps)
        ts = torch.linspace(1.0, 0.0, n + 1, device=x.device)
        out = []
        for r, ex in enumerate(self.experts()):
            xr = x[:, r]
            for i in range(n):
                t = ts[i].expand(B)
                v = ex(xr, normal_crops[:, r], style, t)
                xr = xr + (ts[i + 1] - ts[i]) * v                         # Euler step towards t = 0
            out.append(xr)
        return ((torch.stack(out, 1) + 1) / 2).clamp(0, 1)                 # (B, 3, 1, s, s)

    def forward(self, normal_target: torch.Tensor, boxes: torch.Tensor, style: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = normal_target.shape
        crops = crop_regions(normal_target, boxes, self.region_size)
        prior_patches = self.sample(crops, style)
        canvas = paste_regions(prior_patches, boxes, H, W)
        return {"prior_patches": prior_patches, "prior_map": canvas, "normal_crops": crops}
