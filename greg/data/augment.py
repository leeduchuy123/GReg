"""D4: clip-consistent appearance augmentation with diffusion-generated backgrounds.

Pose-guided generation needs source and target of a pair to show the *same* appearance, so every random choice is
made once per (epoch, clip) and applied identically to both frames:

* ``background_p``: replace the background by one of the Stable-Diffusion-generated images of
  ``scripts/27_make_backgrounds.py``, composited with the person matte of ``scripts/28_matte_cache.py``
  (``matte/<key>/<row>.png``; the SMPL-X mask is thinner than clothes / hair, so it cannot cut the person out);
* ``color_p``: brightness / contrast / saturation / hue jitter of the person (clothes and skin together), so the
  generator sees more appearances than the ~100 signers of the data.

HamNoSys videos have a few studio backgrounds per source corpus; the goal is generalisation to new signers /
backgrounds (the cross-person protocol, scripts/08_eval_cross_person.py), not a better reconstruction score.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


def _load(path: Path, size: int, mode: str = "RGB") -> torch.Tensor:
    img = Image.open(path).convert(mode)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float() / 255.0
    return a.permute(2, 0, 1) if a.dim() == 3 else a[None]


def rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[0], x[1], x[2]
    mx, mn = x.max(0).values, x.min(0).values
    d = mx - mn
    h = torch.zeros_like(mx)
    m = d > 1e-6
    rc, gc, bc = (mx - r) / d.clamp(min=1e-6), (mx - g) / d.clamp(min=1e-6), (mx - b) / d.clamp(min=1e-6)
    h = torch.where(m & (r == mx), bc - gc, h)
    h = torch.where(m & (g == mx) & (r != mx), 2.0 + rc - bc, h)
    h = torch.where(m & (b == mx) & (r != mx) & (g != mx), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    s = torch.where(mx > 1e-6, d / mx.clamp(min=1e-6), torch.zeros_like(mx))
    return torch.stack([h, s, mx])


def hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    h, s, v = x[0], x[1], x[2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    out = torch.zeros(3, *h.shape)
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    for k, (rr, gg, bb) in enumerate(table):
        sel = i == k
        out[0][sel], out[1][sel], out[2][sel] = rr[sel], gg[sel], bb[sel]
    return out


class ClipConsistentAugment:
    def __init__(self, cache_root: str | Path, background_dir: Optional[str | Path] = None, background_p: float = 0.0,
                 color_p: float = 0.0, seed: int = 0, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.3, hue: float = 0.08):
        self.root = Path(cache_root)
        self.bgs: List[Path] = sorted(Path(background_dir).glob("*.png")) if background_dir else []
        if background_p > 0 and not self.bgs:
            raise FileNotFoundError(f"background_p > 0 but no *.png in {background_dir}")
        self.background_p, self.color_p, self.seed = float(background_p), float(color_p), int(seed)
        self.jit = (brightness, contrast, saturation, hue)

    def matte(self, key: str, row: int, size: int) -> torch.Tensor:
        p = self.root / "matte" / key / f"{row:06d}.png"
        if not p.exists():
            raise FileNotFoundError(f"no matte {p}: run scripts/28_matte_cache.py")
        return _load(p, size, "L")

    def draw(self, key: str, epoch: int) -> dict:
        """All random choices of one clip in one epoch (the same for its source and target frame)."""
        rng = random.Random(f"{self.seed}/{epoch}/{key}")
        d = {"bg": None, "jitter": None}
        if self.bgs and rng.random() < self.background_p:
            d["bg"] = self.bgs[rng.randrange(len(self.bgs))]
        if rng.random() < self.color_p:
            b, c, s, h = self.jit
            d["jitter"] = (1 + rng.uniform(-b, b), 1 + rng.uniform(-c, c), 1 + rng.uniform(-s, s), rng.uniform(-h, h))
        return d

    def apply_frame(self, img: torch.Tensor, key: str, row: int, d: dict, bg_img: Optional[torch.Tensor]) -> torch.Tensor:
        if d["bg"] is None and d["jitter"] is None:
            return img
        a = self.matte(key, row, img.shape[-1])
        person = img
        if d["jitter"] is not None:
            bf, cf, sf, hf = d["jitter"]
            x = (person * bf).clamp(0, 1)
            mean = (x * a).sum(dim=(1, 2), keepdim=True) / a.sum().clamp(min=1)
            x = ((x - mean) * cf + mean).clamp(0, 1)
            hsv = rgb_to_hsv(x)
            hsv[0] = (hsv[0] + hf) % 1.0
            hsv[1] = (hsv[1] * sf).clamp(0, 1)
            x = hsv_to_rgb(hsv)
            person = a * x + (1 - a) * img
        if bg_img is not None:
            person = a * person + (1 - a) * bg_img
        return person.clamp(0, 1)

    def __call__(self, item: dict, key: str, epoch: int) -> dict:
        d = self.draw(key, epoch)
        if d["bg"] is None and d["jitter"] is None:
            return item
        size = item["source_image"].shape[-1]
        bg = _load(d["bg"], size) if d["bg"] is not None else None
        item["source_image"] = self.apply_frame(item["source_image"], key, int(item["source_row"]), d, bg)
        item["target_image"] = self.apply_frame(item["target_image"], key, int(item["target_row"]), d, bg)
        return item
