"""Frechet Video Distance (Unterthiner et al.) with an I3D (Kinetics-400) feature extractor.

Protocol (same as the PGMM evaluator that ``scripts/10_export_fvd_clips.py`` targets): one clip per test
video = 16 uniformly spaced frames (both end points included), resized to 224, I3D features, Frechet
distance between the real and the generated feature sets.

The I3D weights are NOT shipped.  ``load_i3d`` expects the TorchScript model used by StyleGAN-V /
common FVD implementations (``i3d_torchscript.pt``), called as
``detector(video, rescale=False, resize=False, return_features=True)`` with ``video`` of shape
(N, 3, T, H, W) scaled to [-1, 1].  Put the file anywhere and pass ``--i3d``.  Absolute values depend on
the exact I3D weights / preprocessing, so before comparing with Table 1 check that this implementation
reproduces the numbers of the PGMM evaluator on the same ``.npz`` (``scripts/10_export_fvd_clips.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def uniform_indices(n: int, k: int) -> List[int]:
    """k indices spread over n frames, first and last included (repeat the last one if n < k)."""
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def load_clip(dir_: Path, k: int = 16, size: int = 224) -> np.ndarray:
    """Directory of frames -> (k, size, size, 3) uint8."""
    paths = sorted(Path(dir_).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no frames in {dir_}")
    return np.stack([np.asarray(Image.open(paths[i]).convert("RGB").resize((size, size)))
                     for i in uniform_indices(len(paths), k)]).astype(np.uint8)


def load_i3d(path: str | Path, device) -> torch.nn.Module:
    model = torch.jit.load(str(path), map_location=device).eval()
    return model


def preprocess(videos: np.ndarray, size: int = 224) -> torch.Tensor:
    """(N, T, H, W, 3) uint8 -> (N, 3, T, size, size) float in [-1, 1]."""
    x = torch.from_numpy(videos).permute(0, 4, 1, 2, 3).float() / 255.0
    if x.shape[-1] != size or x.shape[-2] != size:
        n, c, t = x.shape[:3]
        x = F.interpolate(x.reshape(n, c * t, *x.shape[3:]), size=(size, size), mode="bilinear",
                          align_corners=False).reshape(n, c, t, size, size)
    return x * 2 - 1


@torch.no_grad()
def i3d_features(videos: np.ndarray, detector, device, batch_size: int = 16) -> np.ndarray:
    feats = []
    for i in range(0, len(videos), batch_size):
        x = preprocess(videos[i:i + batch_size]).to(device)
        f = detector(x, rescale=False, resize=False, return_features=True)
        feats.append(f.detach().float().cpu().numpy().reshape(x.shape[0], -1))
    return np.concatenate(feats).astype(np.float64)


def statistics(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray, eps: float = 1e-6) -> float:
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2))."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():   # singular product: add a small ridge
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = np.real(covmean)
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fvd_from_features(real: np.ndarray, gen: np.ndarray, num_bootstrap: int = 0, seed: int = 0
                      ) -> Dict[str, float]:
    """FVD of two feature sets; with ``num_bootstrap`` > 0 also mean / std over bootstrap resamples of the clips
    (the paper reports ``mean +- std``; its exact protocol for the std is not stated)."""
    out = {"fvd": frechet_distance(*statistics(real), *statistics(gen)), "num_real": len(real), "num_gen": len(gen)}
    if num_bootstrap > 0:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(num_bootstrap):
            r = rng.integers(0, len(real), len(real))
            g = rng.integers(0, len(gen), len(gen))
            vals.append(frechet_distance(*statistics(real[r]), *statistics(gen[g])))
        out.update(fvd_boot_mean=float(np.mean(vals)), fvd_boot_std=float(np.std(vals)), num_bootstrap=num_bootstrap)
    return out
