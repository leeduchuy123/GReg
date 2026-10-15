"""CSV loss logger and image grid helper."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["time", "step", "epoch"] + [f for f in fieldnames if f not in ("time", "step", "epoch")]
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
        if new:
            self._w.writeheader()

    def log(self, step: int, epoch: int, values: Dict[str, float]) -> None:
        row = {"time": f"{time.time():.1f}", "step": step, "epoch": epoch}
        for k, v in values.items():
            row[k] = f"{float(v):.6g}"
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] (C=1 or 3) -> (H, W, 3) uint8."""
    x = x.detach().float().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def save_grid(rows: Sequence[Sequence[torch.Tensor]], path: str | Path) -> None:
    """rows: list of rows, each a list of (C, H, W) tensors with equal H/W within the grid."""
    row_imgs = [np.concatenate([to_uint8(t) for t in row], axis=1) for row in rows]
    w = max(r.shape[1] for r in row_imgs)
    row_imgs = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in row_imgs]
    grid = np.concatenate(row_imgs, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
