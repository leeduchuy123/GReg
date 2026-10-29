#!/usr/bin/env python
"""Video reconstruction evaluation (Table 1 protocol).

Source = row ``--source-row`` of each test clip, driving = every other row.
Reports image-level L1 / SSIM / LPIPS, region-level L1, and writes the
generated frames (for AKD via ``09_eval_akd.py`` and FVD via ``10_export_fvd_clips.py``).

    python scripts/07_eval_reconstruction.py --checkpoint outputs/greg/checkpoints/epoch_100.pth \
        --cache data/hamnosys_cache --split test --out outputs/greg/eval_recon --lpips
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.eval import metrics as M  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.runner import load_greg, save_frame  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None, help="random subset of the split (see --subset-seed)")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, help="limit driving rows per clip")
    p.add_argument("--uniform-rows", action="store_true", help="with --max-rows: spread rows over the clip")
    p.add_argument("--eval-region-size", type=int, default=64, help="box size of the region-L1 metric")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--save-frames", action="store_true", help="write generated + real frames (needed for AKD/FVD)")
    p.add_argument("--device", default="auto")
    p.add_argument("--smplx-dir", default=None, help="enables arm_l1 / disocc_frac for TPSMM checkpoints "
                                                     "(mesh checkpoints use model.flow.smplx_dir)")
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_greg(args.checkpoint, device)
    keys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.split}.txt"), args.max_clips, args.subset_seed)
    lp = M.LPIPSMetric(device) if args.lpips else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bar = tqdm(total=None)

    def save(batch, pred, target):
        bar.update(1)
        if args.save_frames:
            for i in range(pred.shape[0]):
                trow = int(batch["target_row"][i])
                save_frame(pred[i], out / "generated" / batch["key"][i] / f"{trow:06d}.png")
                save_frame(target[i], out / "real" / batch["key"][i] / f"{trow:06d}.png")

    rows, summary = evaluate_reconstruction(
        model, args.cache, keys, device, batch_size=args.batch_size, num_workers=args.num_workers,
        source_row=args.source_row, max_rows=args.max_rows, uniform_rows=args.uniform_rows, lpips_metric=lp,
        eval_region_size=args.eval_region_size, on_batch=save,
        smplx_dir=args.smplx_dir or cfg["model"].get("flow", {}).get("smplx_dir"))
    bar.close()
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary.update(checkpoint=str(args.checkpoint), split=args.split, max_clips=args.max_clips,
                   subset_seed=args.subset_seed, max_rows=args.max_rows, uniform_rows=args.uniform_rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
