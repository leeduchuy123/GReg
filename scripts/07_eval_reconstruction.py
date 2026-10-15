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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, SequenceDataset, collate_keep_strings, read_split  # noqa: E402
from greg.data.regions import region_visibility  # noqa: E402
from greg.eval import metrics as M  # noqa: E402
from greg.eval.runner import load_greg, save_frame  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="limit driving rows per clip (smoke tests)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--save-frames", action="store_true", help="write generated + real frames (needed for AKD/FVD)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_greg(args.checkpoint, device)
    keys = read_split(Path(args.cache) / "splits" / f"{args.split}.txt")
    if args.max_clips:
        keys = keys[:args.max_clips]
    cache = HamNoSysCache(args.cache, keys)
    ds = SequenceDataset(cache, source_row=args.source_row, max_rows=args.max_rows)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_keep_strings)
    lp = M.LPIPSMetric(device) if args.lpips else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    per_clip = defaultdict(list)
    size = model.image_size
    for batch in tqdm(loader):
        Is = batch["source_image"].to(device)
        Ns = batch["source_normal"].to(device)
        Nt = batch["target_normal"].to(device)
        It = batch["target_image"].to(device)
        ct = batch["target_centers"].to(device)
        with torch.no_grad():
            o = model(Is, Ns, Nt, ct)
        pred = o["output"].clamp(0, 1)
        vis = region_visibility(ct, size, size)
        res = {"l1": M.l1(pred, It), "psnr": M.psnr(pred, It), "ssim": M.ssim(pred, It),
               "region_l1": M.region_l1(pred, It, o["boxes"], model.region_size, vis),
               "source_copy_l1": M.l1(Is, It), "warped_l1": M.l1(o["warped"], It)}
        if lp:
            res["lpips"] = lp(pred, It)
        for i in range(Is.shape[0]):
            key, trow = batch["key"][i], int(batch["target_row"][i])
            r = {"key": key, "source_row": int(batch["source_row"][i]), "target_row": trow}
            r.update({k: float(v[i]) for k, v in res.items()})
            rows.append(r)
            per_clip[key].append(r)
            if args.save_frames:
                save_frame(pred[i], out / "generated" / key / f"{trow:06d}.png")
                save_frame(It[i], out / "real" / key / f"{trow:06d}.png")
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    metric_names = [k for k in rows[0] if k not in ("key", "source_row", "target_row")]
    summary = {f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in metric_names}
    summary["num_frames"] = len(rows)
    summary["num_clips"] = len(per_clip)
    summary["checkpoint"] = str(args.checkpoint)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
