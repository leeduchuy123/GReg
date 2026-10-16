"""Key regions (face, left hand, right hand): boxes, Gaussians (Eq. 1), crop & paste.

Conventions (docs/decisions.md Sec. 2):
* ``centers`` (B, 3, 2) are pixel coordinates ``(x, y)`` of the region centres,
  order ``face, lhand, rhand`` (= ``greg.render.body.REGION_NAMES``);
* boxes are fixed ``size x size`` squares, clamped inside the frame, stored as
  integer top-left corners ``(x0, y0)`` so that patches paste back at their
  original positions without resampling;
* a region whose centre leaves the frame has visibility 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_REGIONS = 3
REGION_NAMES = ("face", "lhand", "rhand")


def region_boxes(centers: torch.Tensor, size: int, height: int, width: int) -> torch.Tensor:
    """(B, R, 2) centres -> (B, R, 2) integer top-left corners of ``size``² boxes inside the frame."""
    half = size / 2.0
    x0 = torch.round(centers[..., 0] - half).clamp(0, max(width - size, 0))
    y0 = torch.round(centers[..., 1] - half).clamp(0, max(height - size, 0))
    return torch.stack([x0, y0], dim=-1).long()


def _box_indices(boxes: torch.Tensor, size: int, height: int, width: int):
    """-> ys (B, R, size, 1), xs (B, R, 1, size) pixel indices covered by each box."""
    boxes = boxes.long()
    ar = torch.arange(size, device=boxes.device)
    ys = (boxes[..., 1, None] + ar).clamp(0, height - 1)[..., :, None]
    xs = (boxes[..., 0, None] + ar).clamp(0, width - 1)[..., None, :]
    return ys, xs


def crop_regions(x: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, H, W), boxes (B, R, 2) -> (B, R, C, size, size).  Differentiable w.r.t. ``x``."""
    B, C, H, W = x.shape
    R = boxes.shape[1]
    ys, xs = _box_indices(boxes.to(x.device), size, H, W)
    b = torch.arange(B, device=x.device).view(B, 1, 1, 1)
    out = x[b, :, ys, xs]                      # advanced indices first: (B, R, size, size, C)
    return out.permute(0, 1, 4, 2, 3).contiguous()


def paste_regions(patches: torch.Tensor, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, R, C, s, s) patches -> (B, R*C, H, W) canvases, zero outside the boxes.

    Each region gets its own ``C`` channels (region ``r`` -> channels ``r*C:(r+1)*C``).
    """
    B, R, C, s, _ = patches.shape
    ys, xs = _box_indices(boxes.to(patches.device), s, height, width)
    b = torch.arange(B, device=patches.device).view(B, 1, 1, 1)
    r = torch.arange(R, device=patches.device).view(1, R, 1, 1)
    canvas = patches.new_zeros(B, R, height, width, C)
    canvas[b, r, ys, xs] = patches.permute(0, 1, 3, 4, 2)
    return canvas.permute(0, 1, 4, 2, 3).reshape(B, R * C, height, width)


def resize_patches(patches: torch.Tensor, size: int) -> torch.Tensor:
    """(B, R, C, s, s) -> (B, R, C, size, size), bilinear."""
    B, R, C, s, _ = patches.shape
    if s == size:
        return patches
    out = F.interpolate(patches.reshape(B * R, C, s, s), size=(size, size), mode="bilinear", align_corners=False)
    return out.view(B, R, C, size, size)


def region_gaussians(centers: torch.Tensor, height: int, width: int, sigma: float) -> torch.Tensor:
    """Eq. 1: (B, R, 2) centres -> (B, R, H, W) Gaussian weight maps with peak 1 at the centre."""
    ys = torch.arange(height, device=centers.device, dtype=torch.float32).view(1, 1, height, 1)
    xs = torch.arange(width, device=centers.device, dtype=torch.float32).view(1, 1, 1, width)
    cx = centers[..., 0].float()[..., None, None]
    cy = centers[..., 1].float()[..., None, None]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2.0 * float(sigma) ** 2))


def region_visibility(centers: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, R, 2) -> (B, R) float: 1 where the centre lies inside the frame."""
    x, y = centers[..., 0], centers[..., 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height) & torch.isfinite(x) & torch.isfinite(y)
    return inside.float()


def weighted_region_gradients(gradient: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Eq. 3: gradient (B, 1, H, W) x Gaussians (B, R, H, W) -> G_R (B, R, H, W)."""
    return gradient * weights
