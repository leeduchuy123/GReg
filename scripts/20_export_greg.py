#!/usr/bin/env python
"""Export a GReg checkpoint's outputs plus the geometric signals the diffusion branches condition on.

For every selected (clip, target row) of a split, with the reconstruction protocol's source row::

    <out>/<key>/<row>.png         I_hat   GReg output (what D1 refines; the baseline row of every table)
    <out>/<key>/<row>_warp.png    I_st    warped source of the checkpoint's own flow (TPSMM or mesh)
    <out>/<key>/<row>_occ.png     O       the checkpoint's blending mask (8-bit)
    <out>/<key>/<row>_vis.png     V       mesh visibility of the target pixels in the source (needs --smplx-dir +
                                          mesh buffers; computed for TPSMM checkpoints too, so a D1 gate never
                                          depends on which flow the baseline used)
    <out>/<key>/<row>_vc.png      V*C     visibility times the mesh confidence (A1's O_geo)
    <out>/<key>/<row>_ccm.png     CCM     canonical (T-pose) coordinates of every target pixel, RGB in [0, 1]
    <out>/index.json              {key: {"source": s, "rows": [...]}} + the arguments

Row selection: ``--rows all`` (every valid row; the full test protocol) or ``--rows uniform:K`` (K rows spread over
the clip, as the validation runs); ``--with-prev`` adds row t-1 to every selected t (consecutive pairs for the
temporal losses of D1/D5), ``--consecutive`` keeps only such pairs.

    python scripts/20_export_greg.py --checkpoint ckpt/run5_epoch_100.pth --cache data/hamnosys_cache \
        --split val --rows uniform:16 --with-prev --smplx-dir models_smplx_v1_1/models --out outputs/export/run5_ep100/val
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, _pair_item, collate_keep_strings, read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.eval.evaluate import build_metric_mesh  # noqa: E402
from greg.eval.runner import load_greg  # noqa: E402
from greg.geometry.canonical import canonical_coordinate_map, template_coordinates  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--rows", default="all", help="all | uniform:K")
    p.add_argument("--with-prev", action="store_true", help="also export row t-1 of every selected row t")
    p.add_argument("--consecutive", action="store_true", help="only rows whose predecessor is also exported")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--keys-file", default=None, help="explicit clip list (one key per line) instead of the split")
    p.add_argument("--smplx-dir", default=None, help="enables V / V*C / CCM for TPSMM checkpoints")
    p.add_argument("--no-ccm", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def select_rows(valid: List[int], source: int, spec: str, with_prev: bool, consecutive: bool) -> List[int]:
    rows = [t for t in valid if t != source]
    if spec.startswith("uniform:"):
        k = int(spec.split(":", 1)[1])
        if len(rows) > k:
            idx = np.linspace(0, len(rows) - 1, k).round().astype(int)
            rows = [rows[i] for i in idx]
    elif spec != "all":
        raise ValueError(f"--rows must be all or uniform:K, got {spec!r}")
    vs = set(valid)
    if with_prev:
        rows = sorted({r for t in rows for r in (t - 1, t) if r in vs and r != source})
    if consecutive:
        rs = set(rows)
        rows = [t for t in rows if (t - 1) in rs]
    return rows


class _Items(Dataset):
    def __init__(self, cache: HamNoSysCache, items: List[tuple], with_mesh: bool):
        self.cache, self.items, self.with_mesh = cache, items, with_mesh

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, s, t = self.items[i]
        return _pair_item(self.cache, key, s, t, with_parts=False, with_mesh=self.with_mesh)


def to_png(t: torch.Tensor) -> Image.Image:
    """(C, H, W) float [0, 1] -> PIL (RGB for C=3, L for C=1)."""
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr[..., 0] if arr.shape[-1] == 1 else arr)


def main():
    args = parse_args()
    device = get_device(args.device)
    model, cfg = load_greg(args.checkpoint, device)
    if args.keys_file:
        keys = [l.strip() for l in open(args.keys_file, encoding="utf-8") if l.strip()]
    else:
        keys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.split}.txt"), args.max_clips, args.subset_seed)
    cache = HamNoSysCache(args.cache, keys)
    smplx_dir = args.smplx_dir or cfg["model"].get("flow", {}).get("smplx_dir")
    if getattr(model, "uses_mesh", False):
        cache.keep_keys_with_mesh(what="export")
    metric_mesh = build_metric_mesh(model, cache, smplx_dir)
    uses_mesh = bool(getattr(model, "uses_mesh", False))
    with_mesh = uses_mesh or metric_mesh is not None
    if with_mesh and not uses_mesh:
        cache.keep_keys_with_mesh(what="export")
    body = model.mesh_correspondence().body if uses_mesh else (metric_mesh.body if metric_mesh else None)
    coords = template_coordinates(body).to(device) if (body is not None and not args.no_ccm) else None
    faces = body.faces.to(device) if body is not None else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index: Dict[str, dict] = {}
    items: List[tuple] = []
    for key in cache.keys:
        valid = cache.valid_rows(key)
        if len(valid) < 2:
            continue
        s = valid[min(max(args.source_row, 0), len(valid) - 1)]
        rows = select_rows(valid, s, args.rows, args.with_prev, args.consecutive)
        index[key] = {"source": s, "rows": rows}
        for t in rows:
            if args.overwrite or not (out / key / f"{t:06d}.png").exists():
                items.append((key, s, t))
    print(f"{len(index)} clips, {sum(len(v['rows']) for v in index.values())} rows, {len(items)} to generate "
          f"(mesh signals: {with_mesh}, ccm: {coords is not None})")
    loader = DataLoader(_Items(cache, items, with_mesh), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate_keep_strings)
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            extra = model.extra_inputs(b) if hasattr(model, "extra_inputs") else {}
            o = model(b["source_image"], b["source_normal"], b["target_normal"], b["target_centers"], **extra)
            geo = extra.get("geometry") if extra else None
            if geo is None and metric_mesh is not None:
                geo = metric_mesh(b["source_smplx"], b["source_focal"], b["source_princpt"], b["source_faces"],
                                  b["target_smplx"], b["target_focal"], b["target_princpt"], b["target_faces"],
                                  valid_rect_s=b.get("source_valid_rect"))
            ccm = canonical_coordinate_map(b["target_faces"].long(), geo["bary_target"], faces, coords) \
                if (geo is not None and coords is not None) else None
            for i, key in enumerate(b["key"]):
                t = int(b["target_row"][i])
                d = out / key
                d.mkdir(parents=True, exist_ok=True)
                to_png(o["output"][i]).save(d / f"{t:06d}.png")
                to_png(o["warped"][i]).save(d / f"{t:06d}_warp.png")
                to_png(o["occlusion"][i]).save(d / f"{t:06d}_occ.png")
                if geo is not None:
                    to_png(geo["visibility"][i]).save(d / f"{t:06d}_vis.png")
                    to_png(geo["occlusion"][i]).save(d / f"{t:06d}_vc.png")
                if ccm is not None:
                    to_png(ccm[i]).save(d / f"{t:06d}_ccm.png")
    meta = {"checkpoint": str(args.checkpoint), "split": args.split, "source_row": args.source_row, "rows": args.rows,
            "with_prev": args.with_prev, "consecutive": args.consecutive, "max_clips": args.max_clips,
            "subset_seed": args.subset_seed, "mesh_signals": with_mesh, "ccm": coords is not None,
            "image_size": model.image_size, "seconds": time.time() - t0}
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump({"args": meta, "clips": index}, f)
    print(f"done: {len(items)} frames in {meta['seconds'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
