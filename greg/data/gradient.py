"""Gradient map of Eq. 2.

    G(I) = sqrt( (dI/dx)^2 + (dI/dy)^2 + eps )

computed on the grayscale image with backward differences (zero padded on the
first row / column, so image borders show up as edges).  Differentiable, so it
can be applied to generated regions inside the losses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ITU-R BT.601 luma weights
_LUMA = (0.299, 0.587, 0.114)


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) with C in {1, 3, 4} -> (B, 1, H, W)."""
    c = x.shape[1]
    if c == 1:
        return x
    if c in (3, 4):
        w = x.new_tensor(_LUMA).view(1, 3, 1, 1)
        return (x[:, :3] * w).sum(dim=1, keepdim=True)
    return x.mean(dim=1, keepdim=True)


def gradient_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(B, C, H, W) image in [0, 1] -> (B, 1, H, W) gradient magnitude (Eq. 2)."""
    g = to_grayscale(x)
    dx = g - F.pad(g, (1, 0, 0, 0))[:, :, :, :-1]   # g[x] - g[x-1]
    dy = g - F.pad(g, (0, 0, 1, 0))[:, :, :-1, :]   # g[y] - g[y-1]
    return torch.sqrt(dx * dx + dy * dy + eps)
