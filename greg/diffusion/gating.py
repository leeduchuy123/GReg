"""Where a refiner may change GReg's output, and how the two images are blended (D1).

Gate ``G`` (B, 1, H, W) = dilated hands + face + body pixels not visible in the source (V = 0): where run5's error
budget lives (docs/diffusion_6_huong_ke_hoach.md Sec. 0).  Everything else keeps GReg's pixels, which protects
PSNR / colours and keeps the comparison with the baseline honest.

Blends (ablation 4 of D1): ``none`` (refined image as is), ``hard``, ``soft`` (gate feathered), ``freq`` (low
frequencies from GReg everywhere, high frequencies blended; docs/bai_hoc_diffusion_mesh_cho_greg.md K4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from greg.data.regions import part_mask
from greg.eval.metrics import gaussian_blur

HAND_LABELS, FACE_LABELS = (3, 4), (2,)


def refine_gate(parts: torch.Tensor, visibility: torch.Tensor | None, dilate_px: int = 6,
                include_face: bool = True, include_disocc: bool = True) -> torch.Tensor:
    """(B, H, W) part map, (B, 1, H, W) V -> binary gate (B, 1, H, W)."""
    g = part_mask(parts, HAND_LABELS, dilate_px, soften=False)
    if include_face:
        g = torch.maximum(g, part_mask(parts, FACE_LABELS, dilate_px, soften=False))
    if include_disocc and visibility is not None:
        body = (parts > 0).float()[:, None]
        hidden = body * (1 - visibility).clamp(0, 1)
        if dilate_px > 0:
            hidden = F.max_pool2d(hidden, 2 * (dilate_px // 2) + 1, 1, dilate_px // 2)
        g = torch.maximum(g, hidden)
    return (g > 0.5).float()


def feather(gate: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return gate
    return gaussian_blur(gate, float(px) / 2).clamp(0, 1)


def blend(greg: torch.Tensor, refined: torch.Tensor, gate: torch.Tensor, mode: str = "soft", feather_px: int = 4,
          sigma: float = 8.0) -> torch.Tensor:
    if mode == "none":
        return refined
    if mode == "hard":
        g = (gate > 0.5).float()
        return (1 - g) * greg + g * refined
    if mode == "soft":
        g = feather(gate, feather_px)
        return (1 - g) * greg + g * refined
    if mode == "freq":
        g = feather(gate, feather_px)
        low = gaussian_blur(greg, sigma)
        return (low + (1 - g) * (greg - low) + g * (refined - gaussian_blur(refined, sigma))).clamp(0, 1)
    raise ValueError(f"unknown blend mode {mode!r}")


def refiner_masks(parts: torch.Tensor, vc: torch.Tensor, dilate_px: int = 3) -> torch.Tensor:
    """The 4 latent conditioning masks of the refiner: (B, 4, H, W) = [V*C, face, lhand, rhand]."""
    return torch.cat([vc, part_mask(parts, FACE_LABELS, dilate_px, soften=False),
                      part_mask(parts, (3,), dilate_px, soften=False),
                      part_mask(parts, (4,), dilate_px, soften=False)], 1)


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Bicubic up / area down to ``size`` (area avoids aliasing that would flatter LPIPS)."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    if size < x.shape[-1]:
        return F.interpolate(x, size=(size, size), mode="area")
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
