"""Gradient-enhanced Refinement (Sec. 3.3): gradient encoder + GRM.

GRM (Eq. 4)::

    X_out = X_in + softmax(Q(X_g) K(X_in)^T / sqrt(C_in)) V(X_in)

Q comes from the gradient feature X_g, K/V from the generator feature.  The
attention is computed with ``F.scaled_dot_product_attention`` so the N x N
matrix is never materialised explicitly on backends with fused kernels; GRMs
are placed at low-resolution decoder levels (default 32, 64, 128).

Decision: an output 1x1 projection initialised to zero is added after the
attention so each GRM starts as the identity (standard practice for residual
attention; set ``zero_init_out=False`` to disable).
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientEncoder(nn.Module):
    """G_R (B, 3, H, W) -> pyramid of gradient features {res: (B, C_res, res, res)}."""

    def __init__(self, in_channels: int = 3, base: int = 32, max_channels: int = 256,
                 image_size: int = 256, levels: Sequence[int] = (128, 64, 32)):
        super().__init__()
        self.levels = sorted(set(levels), reverse=True)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, base, 7, padding=3), nn.LeakyReLU(0.2))
        self.downs = nn.ModuleDict()
        self.channels: Dict[int, int] = {}
        ch, res = base, image_size
        while res > min(self.levels):
            nxt = min(ch * 2, max_channels)
            self.downs[str(res // 2)] = nn.Sequential(nn.Conv2d(ch, nxt, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
                                                      nn.Conv2d(nxt, nxt, 3, padding=1), nn.LeakyReLU(0.2))
            ch, res = nxt, res // 2
            self.channels[res] = ch
        self.image_size = image_size

    def forward(self, g: torch.Tensor) -> Dict[int, torch.Tensor]:
        h = self.stem(g)
        feats = {}
        res = self.image_size
        while res > min(self.levels):
            h = self.downs[str(res // 2)](h)
            res //= 2
            if res in self.levels:
                feats[res] = h
        return feats


class GRM(nn.Module):
    """Gradient-enhanced Refine Module: cross-attention with gradient-feature queries."""

    def __init__(self, in_channels: int, grad_channels: int, heads: int = 1, zero_init_out: bool = True,
                 attn_bf16: bool = True, qk_norm: bool = False):
        super().__init__()
        self.attn_bf16 = attn_bf16   # attention only (q/k/v cast to bf16 on CUDA): ~2x faster step, the rest stays fp32
        assert in_channels % heads == 0
        self.heads = heads
        self.q = nn.Conv2d(grad_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        # Decision (not in Eq. 4): optional QK-LayerNorm.  The gradient encoder has no normalisation, so the
        # query scale follows its activations; at 32x32 they grew until |q.k|/sqrt(C) reached ~1e4-1e6, the
        # softmax went one-hot and the backward into the gradient encoder exploded (sweep run rper3).  A
        # per-token LayerNorm on q and k bounds the logits to ~sqrt(d) times the learned gains.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(in_channels // heads)
            self.k_norm = nn.LayerNorm(in_channels // heads)

    def forward(self, x: torch.Tensor, grad_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if grad_feat.shape[-2:] != (H, W):
            grad_feat = F.interpolate(grad_feat, size=(H, W), mode="bilinear", align_corners=False)
        # .contiguous(): a transposed view makes SDPA fall back to the O(N^2)-memory "math" kernel (OOM at 128x128)
        q = self.q(grad_feat).view(B, self.heads, C // self.heads, H * W).transpose(-1, -2).contiguous()  # (B, h, N, d)
        k = self.k(x).view(B, self.heads, C // self.heads, H * W).transpose(-1, -2).contiguous()
        v = self.v(x).view(B, self.heads, C // self.heads, H * W).transpose(-1, -2).contiguous()
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.attn_bf16 and q.is_cuda:
            a = F.scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16()).to(x.dtype)
        else:
            a = F.scaled_dot_product_attention(q, k, v)                                        # (B, h, N, d)
        a = a.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.out(a)
