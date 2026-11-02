"""Data of the D1 refiner: GReg exports (scripts/20_export_greg.py) joined with the cache.

An item is one target row ``t`` (``pairs=False``) or the consecutive rows ``(t-1, t)`` (``pairs=True``, for the
temporal loss); tensors carry a leading frame dimension F of size 1 or 2::

    greg     (F, 3, H, W)  GReg output of the row            (export)
    target   (F, 3, H, W)  real frame                        (cache frames/)
    normal   (F, 3, H, W)  target normal map                 (cache normals/)
    parts    (F, H, W)     0 bg 1 body 2 face 3 lhand 4 rhand (cache parts/)
    vis, vc  (F, 1, H, W)  mesh visibility V and V*C         (export _vis / _vc; ones if the export has none)
    joints   (F, J, 2), centers (F, 3, 2)
    + with_mesh: faces, smplx, focal, princpt of every frame (for the mesh correspondence t-1 -> t)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from greg.data.dataset import HamNoSysCache, _load_rgb


def _load_gray(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).float()[None] / 255.0


def read_index(export_dir: str | Path) -> dict:
    return json.loads((Path(export_dir) / "index.json").read_text(encoding="utf-8"))["clips"]


class RefinerDataset(Dataset):
    def __init__(self, export_dir: str | Path, cache: HamNoSysCache, keys: Optional[Sequence[str]] = None,
                 pairs: bool = True, with_mesh: bool = False, max_items_per_clip: Optional[int] = None,
                 spread: bool = True):
        self.export = Path(export_dir)
        self.cache = cache
        idx = read_index(self.export)
        keys = [k for k in (keys if keys is not None else cache.keys) if k in idx and k in cache]
        self.pairs, self.with_mesh = bool(pairs), bool(with_mesh)
        self.items: List[tuple] = []
        for key in keys:
            s, rows = int(idx[key]["source"]), sorted(int(r) for r in idx[key]["rows"])
            rs = set(rows)
            its = [(key, s, t) for t in rows if (not pairs) or ((t - 1) in rs)]
            if max_items_per_clip and len(its) > max_items_per_clip:
                if spread:
                    sel = np.linspace(0, len(its) - 1, int(max_items_per_clip)).round().astype(int)
                    its = [its[i] for i in sel]
                else:
                    its = its[:int(max_items_per_clip)]
            self.items += its
        if not self.items:
            raise ValueError(f"no items in {self.export} for {len(keys)} clips (pairs={pairs})")

    def __len__(self) -> int:
        return len(self.items)

    def _frame(self, key: str, t: int) -> dict:
        d = self.export / key
        row = self.cache.load_row(key, t, with_parts=True, with_mesh=self.with_mesh)
        out = {"greg": _load_rgb(d / f"{t:06d}.png"), "target": row["image"], "normal": row["normal"],
               "parts": row["parts"].long(), "joints": row["joints_2d"], "centers": row["centers"]}
        vis, vc = d / f"{t:06d}_vis.png", d / f"{t:06d}_vc.png"
        out["vis"] = _load_gray(vis) if vis.exists() else torch.ones_like(out["normal"][:1])
        out["vc"] = _load_gray(vc) if vc.exists() else out["vis"].clone()
        if self.with_mesh:
            for f in ("faces", "smplx", "focal", "princpt"):
                out[f] = row[f]
        return out

    def __getitem__(self, i: int) -> dict:
        key, s, t = self.items[i]
        frames = [self._frame(key, t - 1), self._frame(key, t)] if self.pairs else [self._frame(key, t)]
        item = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
        item.update(key=key, source_row=s, target_row=t)
        if self.with_mesh:
            item["valid_rect"] = self.cache.valid_rect(key)
        return item


def flatten_frames(batch: dict) -> dict:
    """(B, F, ...) tensors -> (B*F, ...): the refiner sees every frame as a sample (frame order: t-1, t per item)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and k != "valid_rect":
            out[k] = v.flatten(0, 1)
        else:
            out[k] = v
    return out
