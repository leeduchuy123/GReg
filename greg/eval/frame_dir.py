"""A directory of generated frames wrapped as a "model", so that :func:`greg.eval.evaluate.evaluate_reconstruction`
scores any method (diffusion branches, exported GReg runs, baselines) with exactly the same code and protocol as a
GReg checkpoint.

Layouts understood (the first that exists wins)::

    <root>/<key>/<row:06d>.png                 scripts/20_export_greg.py (and most diffusion branches)
    <root>/generated/<key>/<row:06d>.png       scripts/07_eval_reconstruction.py --save-frames

``<row>_warp.png`` next to a frame is reported as ``warped`` (``warped_l1``); without it that metric is NaN.
A missing frame is an error, not a silent skip: evaluate only the rows the directory holds
(``items_filter`` / ``--index`` of scripts/21_eval_frames.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class FrameDirModel(nn.Module):
    uses_mesh = False

    def __init__(self, root: str | Path, image_size: int = 256):
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.register_buffer("_device_probe", torch.zeros(1))
        self.layout = self._detect_layout()

    def _detect_layout(self) -> str:
        if (self.root / "generated").is_dir():
            return "generated"
        return "flat"

    # ------------------------------------------------------------------ paths
    def frame_path(self, key: str, row: int) -> Path:
        base = self.root / "generated" if self.layout == "generated" else self.root
        return base / key / f"{int(row):06d}.png"

    def has(self, key: str, row: int) -> bool:
        return self.frame_path(key, row).exists()

    def available_rows(self, key: str) -> List[int]:
        d = self.frame_path(key, 0).parent
        if not d.is_dir():
            return []
        return sorted(int(p.stem) for p in d.glob("*.png") if p.stem.isdigit())

    # ------------------------------------------------------------------ loading
    def _load(self, path: Path) -> torch.Tensor:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = torch.nn.functional.interpolate(t[None], size=(self.image_size, self.image_size), mode="area")[0]
        return t

    def extra_inputs(self, batch: dict) -> dict:
        return {"keys": list(batch["key"]), "rows": [int(r) for r in batch["target_row"]]}

    def forward(self, image_source: torch.Tensor, normal_source: torch.Tensor, normal_target: torch.Tensor,
                centers_target: torch.Tensor, keys: Optional[List[str]] = None, rows: Optional[List[int]] = None,
                **_) -> Dict[str, torch.Tensor]:
        if keys is None or rows is None:
            raise ValueError("FrameDirModel needs the batch keys / rows (model(..., **model.extra_inputs(batch)))")
        dev = image_source.device
        outs, warps = [], []
        for k, r in zip(keys, rows):
            p = self.frame_path(k, r)
            if not p.exists():
                raise FileNotFoundError(f"no generated frame for clip {k} row {r}: {p}")
            outs.append(self._load(p))
            w = p.with_name(f"{int(r):06d}_warp.png")
            warps.append(self._load(w) if w.exists() else torch.full_like(outs[-1], float("nan")))
        return {"output": torch.stack(outs).to(dev), "warped": torch.stack(warps).to(dev)}
