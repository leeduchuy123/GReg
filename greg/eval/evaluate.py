"""Reconstruction evaluation (Table 1 protocol), shared by ``scripts/07_eval_reconstruction.py``,
the validation pass of ``scripts/06_train_greg.py`` and the sweep (``scripts/11_sweep.py``).

Source = ``source_row`` of every clip, targets = the other rows.

Region-level L1 is always computed on boxes of the fixed size ``eval_region_size`` (default 64),
*not* on the model's own ``region_size``: otherwise a sweep over ``model.region_size`` would change
the metric together with the model and the runs would not be comparable.

Metrics that see the arm problem (whole-image L1 is dominated by the static background):

* ``hand_l1`` / ``face_l1``  L1 inside the dilated hand / face pixels of ``parts/*.png`` (every cache has them);
* ``arm_l1``                L1 inside the dilated hands + arms of the mesh part map, and ``disocc_frac`` = share of
                            those pixels NOT visible in the source (V = 0).  Need the mesh buffers and SMPL-X:
                            computed for mesh models, and for TPSMM models when ``smplx_dir`` is given;
* ``wrist_disp``            max wrist displacement source -> target (px); every metric is also reported per
                            motion bin ``mean_<metric>@small|medium|large`` (``motion_bins`` thresholds, px @256).

Masked metrics are NaN for frames without such pixels and averaged with ``nanmean``.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from greg.data.dataset import HamNoSysCache, SequenceDataset, collate_keep_strings
from greg.data.regions import part_mask, region_boxes, region_visibility
from greg.eval import metrics as M

METRIC_KEYS = ("l1", "psnr", "ssim", "region_l1", "lpips", "hand_l1", "face_l1", "arm_l1")
MOTION_BINS = (10.0, 40.0)
BIN_NAMES = ("small", "medium", "large")
WRIST_JOINTS = (20, 21)
HAND_LABELS, FACE_LABELS, ARM_HAND_LABELS = (3, 4), (2,), (3, 4, 5, 6)
_ID_KEYS = ("key", "source_row", "target_row")


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, min_pixels: float = 1.0) -> torch.Tensor:
    """(B, 3, H, W) x2, mask (B, 1, H, W) -> (B,) mean absolute error inside the mask, NaN if (almost) empty."""
    err = (pred - target).abs().mean(1, keepdim=True)
    s = mask.sum(dim=(1, 2, 3))
    v = (err * mask).sum(dim=(1, 2, 3)) / s.clamp(min=1e-6)
    return torch.where(s >= min_pixels, v, torch.full_like(v, float("nan")))


def wrist_displacement(joints_source: torch.Tensor, joints_target: torch.Tensor) -> torch.Tensor:
    """(B, J, 2) x2 -> (B,) the larger of the two wrist displacements in pixels."""
    idx = list(WRIST_JOINTS)
    return (joints_target[:, idx] - joints_source[:, idx]).norm(dim=-1).max(dim=1).values


def motion_bin(d: float, bins: Sequence[float] = MOTION_BINS) -> str:
    for name, thr in zip(BIN_NAMES, bins):
        if d < thr:
            return name
    return BIN_NAMES[len(bins)]


def build_metric_mesh(model, cache: HamNoSysCache, smplx_dir: Optional[str]):
    """MeshCorrespondence for the arm metrics of a TPSMM model, or None if impossible (no buffers / smplx)."""
    if getattr(model, "uses_mesh", False) or not smplx_dir:
        return None
    if not cache.has_mesh():
        print("arm metrics disabled: some clips have no mesh buffers (scripts/02c_mesh_buffers.py)")
        return None
    try:
        from greg.geometry.mesh_flow import MeshCorrespondence
        return MeshCorrespondence(smplx_dir, device=next(model.parameters()).device)
    except Exception as e:   # smplx not installed / model files missing: keep the other metrics
        print(f"arm metrics disabled: {type(e).__name__}: {e}")
        return None


@torch.no_grad()
def evaluate_reconstruction(model, cache_root: str | Path, keys: Sequence[str], device, *,
                            batch_size: int = 8, num_workers: int = 0, source_row: int = 0,
                            max_rows: Optional[int] = None, uniform_rows: bool = False,
                            lpips_metric: Optional[M.LPIPSMetric] = None, eval_region_size: int = 64,
                            on_batch: Optional[Callable[[dict, torch.Tensor, torch.Tensor], None]] = None,
                            smplx_dir: Optional[str] = None, motion_bins: Sequence[float] = MOTION_BINS,
                            mask_dilate_px: int = 3) -> Tuple[List[dict], Dict[str, float]]:
    """-> (per-frame rows, summary).  ``on_batch(batch, pred, target)`` lets callers save frames."""
    was_training = model.training
    model.eval()
    cache = HamNoSysCache(cache_root, keys)
    if getattr(model, "uses_mesh", False):
        cache.keep_keys_with_mesh(what="evaluation")
    metric_mesh = build_metric_mesh(model, cache, smplx_dir)
    with_mesh = bool(getattr(model, "uses_mesh", False)) or metric_mesh is not None
    with_parts = (cache.root / "parts").is_dir()        # every preprocessed cache has it; tiny test caches may not
    ds = SequenceDataset(cache, source_row=source_row, max_rows=max_rows, uniform_rows=uniform_rows,
                         with_parts=with_parts, with_mesh=with_mesh)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_keep_strings)
    size = model.image_size
    rows: List[dict] = []
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        Is, Ns, Nt = batch["source_image"], batch["source_normal"], batch["target_normal"]
        It, ct = batch["target_image"], batch["target_centers"]
        extra = model.extra_inputs(batch) if hasattr(model, "extra_inputs") else {}
        o = model(Is, Ns, Nt, ct, **extra)
        pred = o["output"].clamp(0, 1)
        vis = region_visibility(ct, size, size)
        boxes = region_boxes(ct, eval_region_size, size, size)
        res = {"l1": M.l1(pred, It), "psnr": M.psnr(pred, It), "ssim": M.ssim(pred, It),
               "region_l1": M.region_l1(pred, It, boxes, eval_region_size, vis),
               "source_copy_l1": M.l1(Is, It), "warped_l1": M.l1(o["warped"], It),
               "wrist_disp": wrist_displacement(batch["source_joints"], batch["target_joints"])}
        if with_parts:
            parts = batch["target_parts"].long()
            res["hand_l1"] = masked_l1(pred, It, part_mask(parts, HAND_LABELS, mask_dilate_px))
            res["face_l1"] = masked_l1(pred, It, part_mask(parts, FACE_LABELS, mask_dilate_px))
        geo = extra.get("geometry") if extra else None
        if geo is None and metric_mesh is not None:
            geo = metric_mesh(batch["source_smplx"], batch["source_focal"], batch["source_princpt"],
                              batch["source_faces"], batch["target_smplx"], batch["target_focal"],
                              batch["target_princpt"], batch["target_faces"])
        if geo is not None:
            arm = part_mask(geo["parts"], ARM_HAND_LABELS, mask_dilate_px)
            res["arm_l1"] = masked_l1(pred, It, arm)
            core = part_mask(geo["parts"], ARM_HAND_LABELS, 0, soften=False)
            hidden = core * (1 - geo["visibility"])
            n = core.sum(dim=(1, 2, 3))
            res["disocc_frac"] = torch.where(n > 0, hidden.sum(dim=(1, 2, 3)) / n.clamp(min=1),
                                             torch.full_like(n, float("nan")))
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
    return rows, summarize(rows, eval_region_size, motion_bins)


def _nanmean(values) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(values)) if len(values) else float("nan")


def summarize(rows: List[dict], eval_region_size: int = 64,
              motion_bins: Sequence[float] = MOTION_BINS) -> Dict[str, float]:
    """Clip-weighted means: every frame counts equally (as the paper's per-frame image metrics)
    and ``clip_mean_*`` averages the per-clip means (long clips do not dominate).  With ``wrist_disp`` in
    the rows, ``mean_<k>@<bin>`` repeats the frame means per motion bin."""
    if not rows:
        return {"num_frames": 0, "num_clips": 0}
    names = [k for k in rows[0] if k not in _ID_KEYS]
    per_clip = defaultdict(list)
    for r in rows:
        per_clip[r["key"]].append(r)
    out: Dict[str, float] = {f"mean_{k}": _nanmean([r[k] for r in rows]) for k in names}
    for k in names:
        out[f"clip_mean_{k}"] = _nanmean([_nanmean([r[k] for r in v]) for v in per_clip.values()])
    if "wrist_disp" in names:
        by_bin = defaultdict(list)
        for r in rows:
            by_bin[motion_bin(r["wrist_disp"], motion_bins)].append(r)
        for b in BIN_NAMES:
            sel = by_bin.get(b, [])
            out[f"num_frames@{b}"] = len(sel)
            for k in names:
                if k != "wrist_disp":
                    out[f"mean_{k}@{b}"] = _nanmean([r[k] for r in sel])
    out.update(num_frames=len(rows), num_clips=len(per_clip), eval_region_size=eval_region_size)
    return out
