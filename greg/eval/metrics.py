"""Image / region / keypoint metrics used in Table 1.

* L1, PSNR, SSIM (Gaussian-window SSIM as in skimage's default-free variant)
* LPIPS (optional dependency ``lpips``)
* region L1: L1 restricted to the three key-region crops
* AKD: mean Euclidean distance between normalised HRNet COCO-WholeBody
  keypoints of generated vs real frames (face + hands).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from greg.data.regions import crop_regions

# COCO-WholeBody (133 pts): 0-16 body, 17-22 feet, 23-90 face (68), 91-111 left hand, 112-132 right hand
FACE_IDX = list(range(23, 91))
LHAND_IDX = list(range(91, 112))
RHAND_IDX = list(range(112, 133))


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=(1, 2, 3))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-sample SSIM, Gaussian window, valid padding (Wang et al. 2004)."""
    C = pred.shape[1]
    w = _gaussian_window(window, sigma, pred.device, pred.dtype).expand(C, 1, window, window).contiguous()
    mu_x = F.conv2d(pred, w, groups=C)
    mu_y = F.conv2d(target, w, groups=C)
    sxx = F.conv2d(pred * pred, w, groups=C) - mu_x ** 2
    syy = F.conv2d(target * target, w, groups=C) - mu_y ** 2
    sxy = F.conv2d(pred * target, w, groups=C) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def region_l1(pred: torch.Tensor, target: torch.Tensor, boxes: torch.Tensor, size: int,
              weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean L1 over the three region crops (B,)."""
    p = crop_regions(pred, boxes, size)
    t = crop_regions(target, boxes, size)
    per = (p - t).abs().mean(dim=(2, 3, 4))                      # (B, R)
    if weights is None:
        return per.mean(1)
    return (per * weights).sum(1) / weights.sum(1).clamp(min=1)


class LPIPSMetric:
    def __init__(self, device, net: str = "alex"):
        import lpips  # optional dependency
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred * 2 - 1, target * 2 - 1).view(-1)


def akd(kp_pred: np.ndarray, kp_real: np.ndarray, image_size: int, conf_threshold: float = 0.3,
        indices: Sequence[int] = tuple(FACE_IDX + LHAND_IDX + RHAND_IDX)) -> Dict[str, float]:
    """Average Keypoint Distance on normalised coordinates (divide by image size).

    kp_* : (N, 133, 3) arrays (x, y, score).  Only keypoints detected with
    confidence >= threshold in the *real* frame are counted.
    """
    idx = np.asarray(indices)
    out = {}
    for name, sel in (("akd", idx), ("akd_face", np.asarray(FACE_IDX)),
                      ("akd_lhand", np.asarray(LHAND_IDX)), ("akd_rhand", np.asarray(RHAND_IDX))):
        p = kp_pred[:, sel, :2] / image_size
        r = kp_real[:, sel, :2] / image_size
        valid = kp_real[:, sel, 2] >= conf_threshold
        d = np.linalg.norm(p - r, axis=-1)
        out[name] = float(d[valid].mean()) if valid.any() else float("nan")
        out[name + "_n"] = int(valid.sum())
    return out


def tcd(gen: torch.Tensor, real: torch.Tensor, threshold: float = 0.5) -> float:
    """Temporal consistency distance used in the PGMM repo (fraction of pixels whose
    generated value deviates > threshold from the temporal mean of real neighbours)."""
    if gen.shape[0] < 3:
        return float("nan")
    mid = (real[:-2] + real[2:]) / 2
    return float(((gen[1:-1] - mid).abs() > threshold).float().mean())
