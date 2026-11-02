"""Losses of the D1 refiner (docs/diffusion_6_huong_ke_hoach.md Sec. 3.3).

    L = l2 * L2_w + lpips * LPIPS_w + hand * (L2 + LPIPS on the hand crops) + temp * L_temp

* ``_w``: pixel weights ``w = 1 + w_hand M_hand + w_face M_face + w_disocc (1 - V) body`` (SignRefine Eq. 4 uses
  10x for hands / face).  LPIPS is the ``spatial=True`` map so it is weighted the same way.
* hand crops: boxes fitted to the hand joints (like the ``hand_lpips`` metric), upsampled to 128.
* ``L_temp`` = ||V C (I_t - warp(I_{t-1}, F_{t-1 -> t}))||_1 / sum(V C) with the mesh correspondence between two
  consecutive *targets*: hallucinated detail must move with the surface.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from greg.data.regions import adaptive_region_boxes, crop_regions, part_mask, region_visibility, resize_patches
from greg.models.tpsmm import warp
from greg.render.body import region_extents_from_joints

HAND_LABELS, FACE_LABELS = (3, 4), (2,)


class RefinerLoss:
    def __init__(self, lpips_spatial, weights: Optional[Dict[str, float]] = None, w_hand: float = 9.0,
                 w_face: float = 9.0, w_disocc: float = 1.0, hand_crop: int = 64, hand_up: int = 128,
                 mask_dilate_px: int = 3):
        self.lpips = lpips_spatial                       # callable([-1,1] x2) -> (B, 1, h, w) map
        self.w = {"l2": 1.0, "lpips": 5.0, "hand": 1.0, "temp": 0.0}
        self.w.update({k: float(v) for k, v in (weights or {}).items()})
        self.w_hand, self.w_face, self.w_disocc = float(w_hand), float(w_face), float(w_disocc)
        self.hand_crop, self.hand_up, self.dilate = int(hand_crop), int(hand_up), int(mask_dilate_px)

    def pixel_weights(self, parts: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        body = (parts > 0).float()[:, None]
        return 1 + self.w_hand * part_mask(parts, HAND_LABELS, self.dilate, soften=False) \
            + self.w_face * part_mask(parts, FACE_LABELS, self.dilate, soften=False) \
            + self.w_disocc * body * (1 - vis).clamp(0, 1)

    @staticmethod
    def _wmean(err: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if err.shape[-2:] != w.shape[-2:]:
            w = F.interpolate(w, size=err.shape[-2:], mode="area")
        return (err * w).sum() / w.sum().clamp(min=1.0)

    def hand_terms(self, pred, target, centers, joints):
        size = pred.shape[-1]
        scale = size / float(self.base_size) if hasattr(self, "base_size") else 1.0
        extents = region_extents_from_joints(joints) * scale
        c = centers * scale
        boxes = adaptive_region_boxes(c, extents, self.hand_crop, size, size, min_side=16.0 * scale, max_side=float(size))
        vis = region_visibility(c, size, size)
        ok = ((vis[:, 1:] > 0) & (extents[:, 1:] >= 6.0 * scale)).float()             # (B, 2)
        pc = resize_patches(crop_regions(pred, boxes, self.hand_crop), self.hand_up)[:, 1:]
        tc = resize_patches(crop_regions(target, boxes, self.hand_crop), self.hand_up)[:, 1:]
        B = pc.shape[0]
        pc = pc.reshape(B * 2, 3, self.hand_up, self.hand_up)
        tc = tc.reshape(B * 2, 3, self.hand_up, self.hand_up)
        okf = ok.reshape(B * 2)
        l2 = ((pc - tc) ** 2).mean(dim=(1, 2, 3))
        lp = self.lpips(pc * 2 - 1, tc * 2 - 1).mean(dim=(1, 2, 3))
        n = okf.sum().clamp(min=1.0)
        return (l2 * okf).sum() / n, (lp * okf).sum() / n

    def __call__(self, pred, target, parts, vis, centers, joints) -> Dict[str, torch.Tensor]:
        """``pred``/``target`` (B, 3, S, S) in [0, 1] at the work size; ``parts``/``vis``/``centers``/``joints`` at
        the cache size (256): weights are resized, joints rescaled."""
        S, base = pred.shape[-1], parts.shape[-1]
        self.base_size = base
        w = self.pixel_weights(parts, vis)
        if S != base:
            w = F.interpolate(w, size=(S, S), mode="nearest")
        losses = {"l2": self._wmean(((pred - target) ** 2).mean(1, keepdim=True), w),
                  "lpips": self._wmean(self.lpips(pred * 2 - 1, target * 2 - 1), w)}
        if self.w["hand"] > 0:
            hl2, hlp = self.hand_terms(pred, target, centers, joints)
            losses["hand"] = hl2 + hlp
        losses["total"] = sum(self.w[k] * v for k, v in losses.items() if k in self.w)
        return losses

    @staticmethod
    def temporal(pred_prev: torch.Tensor, pred_t: torch.Tensor, geo: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``geo`` = correspond(prev -> t) at the cache size; predictions are compared at that size."""
        size = geo["grid"].shape[1]
        if pred_t.shape[-1] != size:
            pred_prev = F.interpolate(pred_prev, size=(size, size), mode="area")
            pred_t = F.interpolate(pred_t, size=(size, size), mode="area")
        warped = warp(pred_prev, geo["grid"].to(pred_prev.dtype))
        m = geo["occlusion"].to(pred_t.dtype)
        return ((pred_t - warped).abs().mean(1, keepdim=True) * m).sum() / m.sum().clamp(min=1.0)
