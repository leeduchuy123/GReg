"""Datasets over the preprocessed cache written by ``scripts/02_preprocess_hamnosys.py``.

Cache layout (``root``)::

    meta.json                  {"clips": {key: {"status": "ok", "num_rows", "crop", ...}}, "config": {...}}
    frames/<key>/000012.png    RGB crop (size x size)
    normals/<key>/000012.png   RGBA: normal map (n+1)/2 in RGB, body mask in alpha
    parts/<key>/000012.png     L: 0 bg, 1 body, 2 face, 3 lhand, 4 rhand
    joints/<key>.npy           (N, J, 2) projected SMPL-X joints in crop pixels
    splits/{train,test}.txt    one key per line (``scripts/04_make_splits.py``)

Region centres (face / left hand / right hand) are the means of the projected
SMPL-X joints (``greg.render.body.region_centers_from_joints``), i.e. they come
from the geometry only and are available at inference without a target image.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from greg.data.splits import read_split  # noqa: F401  (re-exported)
from greg.render.body import region_centers_from_joints


def _load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


def _load_rgba(path: Path) -> tuple:
    arr = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    return t[:3], t[3:4]


def _load_l(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return torch.from_numpy(arr.copy())


class HamNoSysCache:
    """Random access to the rows (frames) of the cached clips."""

    def __init__(self, root: str | Path, keys: Optional[Sequence[str]] = None):
        self.root = Path(root)
        with open(self.root / "meta.json", encoding="utf-8") as f:
            full = json.load(f)
        clips = full.get("clips", full)
        self.config = full.get("config", {})
        ok = {k: v for k, v in clips.items() if isinstance(v, dict) and v.get("status") == "ok"}
        if keys is None:
            self.keys: List[str] = sorted(ok)
        else:
            missing = [k for k in keys if k not in ok]
            if missing:
                raise KeyError(f"{len(missing)} keys not in the cache (status ok), e.g. {missing[:5]}")
            self.keys = list(keys)
        self.meta: Dict[str, dict] = {k: ok[k] for k in self.keys}
        self._joints: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self.meta

    def num_rows(self, key: str) -> int:
        return int(self.meta[key]["num_rows"])

    def valid_rows(self, key: str) -> List[int]:
        """Rows usable as source / target: ``valid_range`` from ``scripts/02b_mark_valid_rows.py`` (fade-in /
        fade-out trimmed), or every row for caches written before that step."""
        first, last = self.meta[key].get("valid_range", [0, self.num_rows(key) - 1])
        return list(range(int(first), int(last) + 1))

    def image_size(self, key: Optional[str] = None) -> int:
        k = key or self.keys[0]
        return int(self.meta[k]["crop"]["size"])

    # ------------------------------------------------------------------- paths
    def frame_path(self, key: str, row: int) -> Path:
        return self.root / "frames" / key / f"{row:06d}.png"

    def normal_path(self, key: str, row: int) -> Path:
        return self.root / "normals" / key / f"{row:06d}.png"

    def part_path(self, key: str, row: int) -> Path:
        return self.root / "parts" / key / f"{row:06d}.png"

    def joints_path(self, key: str) -> Path:
        return self.root / "joints" / f"{key}.npy"

    # -------------------------------------------------------------------- data
    def joints(self, key: str) -> np.ndarray:
        """(N, J, 2) projected joints of the whole clip (cached in memory)."""
        j = self._joints.get(key)
        if j is None:
            j = np.load(self.joints_path(key)).astype(np.float32)
            self._joints[key] = j
        return j

    def centers(self, key: str, row: int) -> torch.Tensor:
        """(3, 2) region centres of one row."""
        j = torch.from_numpy(self.joints(key)[row][None])
        return region_centers_from_joints(j)[0]

    def load_row(self, key: str, row: int, with_parts: bool = False) -> Dict[str, torch.Tensor]:
        normal, mask = _load_rgba(self.normal_path(key, row))
        joints = torch.from_numpy(self.joints(key)[row])
        out = {"image": _load_rgb(self.frame_path(key, row)), "normal": normal, "mask": mask,
               "joints_2d": joints, "centers": region_centers_from_joints(joints[None])[0]}
        if with_parts:
            out["parts"] = _load_l(self.part_path(key, row))
        return out


def _pair_item(cache: HamNoSysCache, key: str, s: int, t: int, with_parts: bool) -> dict:
    src = cache.load_row(key, s, with_parts)
    tgt = cache.load_row(key, t, with_parts)
    item = {"source_image": src["image"], "source_normal": src["normal"], "source_mask": src["mask"],
            "source_centers": src["centers"],
            "target_image": tgt["image"], "target_normal": tgt["normal"], "target_mask": tgt["mask"],
            "target_centers": tgt["centers"], "target_joints": tgt["joints_2d"],
            "key": key, "source_row": s, "target_row": t}
    if with_parts:
        item["source_parts"] = src["parts"]
        item["target_parts"] = tgt["parts"]
    return item


class PairDataset(Dataset):
    """``pairs_per_clip`` random (source, target) frame pairs per clip and epoch.

    The pairs are drawn from ``seed + epoch`` so that every epoch sees new pairs
    while a run stays reproducible (``set_epoch``).  ``min_gap`` enforces
    ``|source - target| >= min_gap`` where the clip is long enough.
    """

    def __init__(self, cache: HamNoSysCache, pairs_per_clip: int = 1, seed: int = 0,
                 with_parts: bool = False, min_gap: int = 0):
        self.cache = cache
        self.pairs_per_clip = max(1, int(pairs_per_clip))
        self.seed = int(seed)
        self.with_parts = with_parts
        self.min_gap = max(0, int(min_gap))
        self.keys = [k for k in cache.keys if len(cache.valid_rows(k)) >= 2]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.keys) * self.pairs_per_clip

    def _draw(self, key: str, rng: random.Random) -> tuple:
        rows = self.cache.valid_rows(key)
        n = len(rows)
        gap = min(self.min_gap, n - 1)
        for _ in range(50):
            s, t = rng.randrange(n), rng.randrange(n)
            if s != t and abs(s - t) >= gap:
                return rows[s], rows[t]
        s = rng.randrange(n)
        return rows[s], rows[(s + max(gap, 1)) % n]

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index % len(self.keys)]
        rng = random.Random((self.seed * 1_000_003 + self.epoch) * 7_919 + index)
        s, t = self._draw(key, rng)
        return _pair_item(self.cache, key, s, t, self.with_parts)


class SequenceDataset(Dataset):
    """Reconstruction protocol: source = ``source_row`` of each clip, target = every other row.

    Rows are counted among the clip's valid rows (``HamNoSysCache.valid_rows``): ``source_row=0`` is the first
    frame after a fade-in, and faded rows are never targets.  Without fades this is the plain row index.
    ``max_rows`` limits the targets per clip (first rows, or evenly spread with ``uniform_rows``)."""

    def __init__(self, cache: HamNoSysCache, source_row: int = 0, max_rows: Optional[int] = None,
                 with_parts: bool = False, include_source_row: bool = False, uniform_rows: bool = False):
        self.cache = cache
        self.with_parts = with_parts
        self.items: List[tuple] = []
        for key in cache.keys:
            valid = cache.valid_rows(key)
            if not valid:
                continue
            s = valid[min(max(int(source_row), 0), len(valid) - 1)]
            rows = [t for t in valid if include_source_row or t != s]
            if max_rows and len(rows) > int(max_rows):
                if uniform_rows:   # evenly spread over the clip (quick validation), not just the first rows
                    idx = np.linspace(0, len(rows) - 1, int(max_rows)).round().astype(int)
                    rows = [rows[i] for i in idx]
                else:
                    rows = rows[:int(max_rows)]
            self.items += [(key, s, t) for t in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        key, s, t = self.items[index]
        return _pair_item(self.cache, key, s, t, self.with_parts)


def collate_keep_strings(batch: List[dict]) -> dict:
    """``default_collate`` for tensors / numbers; strings (and other objects) are kept as lists."""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], (torch.Tensor, int, float, bool, np.ndarray, np.generic)):
            out[k] = default_collate(vals)
        else:
            out[k] = vals
    return out


__all__ = ["HamNoSysCache", "PairDataset", "SequenceDataset", "collate_keep_strings", "read_split"]
