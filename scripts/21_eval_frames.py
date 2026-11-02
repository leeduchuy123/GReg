#!/usr/bin/env python
"""Evaluate a directory of generated frames (any method) with the GReg reconstruction protocol and metrics.

    python scripts/21_eval_frames.py --frames outputs/export/run5_ep100/test --cache data/hamnosys_cache \
        --split test --out outputs/eval/run5_ep100_test --lpips --smplx-dir models_smplx_v1_1/models

Rows: by default the ``index.json`` written by scripts/20_export_greg.py next to the frames decides which rows are
scored (so a ``uniform:16`` export is scored on exactly those rows); ``--max-rows/--uniform-rows`` reproduce the
sweep protocol instead; ``--all-rows`` scores every valid row (the frames must exist).  ``--keys-file`` fixes a clip
subset shared by every method (the 200-clip test subset of D5/D6).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.eval import metrics as M  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.frame_dir import FrameDirModel  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--keys-file", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--uniform-rows", action="store_true")
    p.add_argument("--all-rows", action="store_true", help="ignore index.json, score every valid row")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--eval-region-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--smplx-dir", default=None, help="enables arm_l1 / disocc_frac / disocc_l1")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    frames = Path(args.frames)
    model = FrameDirModel(frames, args.image_size).to(device)
    if args.keys_file:
        keys = [l.strip() for l in open(args.keys_file, encoding="utf-8") if l.strip()]
    else:
        keys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.split}.txt"), args.max_clips, args.subset_seed)
    items_filter = None
    index_path = frames / "index.json"
    if index_path.exists() and not args.all_rows and args.max_rows is None:
        idx = json.loads(index_path.read_text(encoding="utf-8"))["clips"]
        keys = [k for k in keys if k in idx]
        allowed = {k: set(v["rows"]) for k, v in idx.items()}
        items_filter = lambda key, s, t: t in allowed.get(key, ())   # noqa: E731
        print(f"rows from {index_path}: {sum(len(v) for v in allowed.values())} rows in {len(keys)} clips")
    lp = M.LPIPSMetric(device) if args.lpips else None
    rows, summary = evaluate_reconstruction(
        model, args.cache, keys, device, batch_size=args.batch_size, num_workers=args.num_workers,
        source_row=args.source_row, max_rows=args.max_rows, uniform_rows=args.uniform_rows, lpips_metric=lp,
        eval_region_size=args.eval_region_size, smplx_dir=args.smplx_dir, items_filter=items_filter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary.update(frames=str(frames), split=args.split, max_clips=args.max_clips, subset_seed=args.subset_seed,
                   keys_file=args.keys_file, max_rows=args.max_rows, uniform_rows=args.uniform_rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("mean_") and "@" not in k}, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
