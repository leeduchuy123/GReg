"""Reconstruction evaluation (Table 1 protocol), shared by ``scripts/07_eval_reconstruction.py``,
the validation pass of ``scripts/06_train_greg.py`` and the sweep (``scripts/11_sweep.py``).

Source = ``source_row`` of every clip, targets = the other rows.

Region-level L1 is always computed on boxes of the fixed size ``eval_region_size`` (default 64),
*not* on the model's own ``region_size``: otherwise a sweep over ``model.region_size`` would change
the metric together with the model and the runs would not be comparable.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from greg.data.dataset import HamNoSysCache, SequenceDataset, collate_keep_strings
from greg.data.regions import region_boxes, region_visibility
from greg.eval import metrics as M

METRIC_KEYS = ("l1", "psnr", "ssim", "region_l1", "lpips")


@torch.no_grad()
def evaluate_reconstruction(model, cache_root: str | Path, keys: Sequence[str], device, *,
                            batch_size: int = 8, num_workers: int = 0, source_row: int = 0,
                            max_rows: Optional[int] = None, uniform_rows: bool = False,
                            lpips_metric: Optional[M.LPIPSMetric] = None, eval_region_size: int = 64,
                            on_batch: Optional[Callable[[dict, torch.Tensor, torch.Tensor], None]] = None
                            ) -> Tuple[List[dict], Dict[str, float]]:
    """-> (per-frame rows, summary).  ``on_batch(batch, pred, target)`` lets callers save frames."""
    was_training = model.training
    model.eval()
    cache = HamNoSysCache(cache_root, keys)
    ds = SequenceDataset(cache, source_row=source_row, max_rows=max_rows, uniform_rows=uniform_rows)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_keep_strings)
    size = model.image_size
    rows: List[dict] = []
    for batch in loader:
        Is = batch["source_image"].to(device)
        Ns = batch["source_normal"].to(device)
        Nt = batch["target_normal"].to(device)
        It = batch["target_image"].to(device)
        ct = batch["target_centers"].to(device)
        o = model(Is, Ns, Nt, ct)
        pred = o["output"].clamp(0, 1)
        vis = region_visibility(ct, size, size)
        boxes = region_boxes(ct, eval_region_size, size, size)
        res = {"l1": M.l1(pred, It), "psnr": M.psnr(pred, It), "ssim": M.ssim(pred, It),
               "region_l1": M.region_l1(pred, It, boxes, eval_region_size, vis),
               "source_copy_l1": M.l1(Is, It), "warped_l1": M.l1(o["warped"], It)}
        if lpips_metric is not None:
            res["lpips"] = lpips_metric(pred, It)
        for i in range(Is.shape[0]):
            r = {"key": batch["key"][i], "source_row": int(batch["source_row"][i]),
                 "target_row": int(batch["target_row"][i])}
            r.update({k: float(v[i]) for k, v in res.items()})
            rows.append(r)
        if on_batch is not None:
            on_batch(batch, pred, It)
    model.train(was_training)
    return rows, summarize(rows, eval_region_size)


def summarize(rows: List[dict], eval_region_size: int = 64) -> Dict[str, float]:
    """Clip-weighted means: every frame counts equally (as the paper's per-frame image metrics)
    and ``clip_mean_*`` averages the per-clip means (long clips do not dominate)."""
    if not rows:
        return {"num_frames": 0, "num_clips": 0}
    names = [k for k in rows[0] if k not in ("key", "source_row", "target_row")]
    per_clip = defaultdict(list)
    for r in rows:
        per_clip[r["key"]].append(r)
    out: Dict[str, float] = {f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in names}
    for k in names:
        out[f"clip_mean_{k}"] = float(np.mean([np.mean([r[k] for r in v]) for v in per_clip.values()]))
    out.update(num_frames=len(rows), num_clips=len(per_clip), eval_region_size=eval_region_size)
    return out
