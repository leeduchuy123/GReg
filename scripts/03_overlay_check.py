#!/usr/bin/env python
"""Visual + numeric check of the normal-map alignment (paper Fig. 1).

For sampled clips of the cache: writes a grid [frame | normal | overlay | parts | region Gaussians]
and reports the 2D error between projected SMPL-X joints and the .pkl keypoints
(both transformed into crop coordinates).

    python scripts/03_overlay_check.py --cache data/hamnosys_cache --pkl-dir ... --out outputs/overlay --num 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache  # noqa: E402
from greg.data.regions import region_gaussians  # noqa: E402
from greg.data.signavatars import ClipAnnotation  # noqa: E402
from greg.utils.logging import save_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--pkl-dir", default=None, help="if given, report joint-vs-keypoint distances")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cache = HamNoSysCache(args.cache)
    rng = random.Random(args.seed)
    keys = rng.sample(cache.keys, min(args.num, len(cache.keys)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    for key in keys:
        n = cache.num_rows(key)
        row = rng.randrange(n)
        r = cache.load_row(key, row, with_parts=True)
        g = region_gaussians(r["centers"][None], 256, 256, 32.0)[0]
        overlay = r["image"] * (1 - 0.5 * r["mask"]) + r["normal"] * 0.5 * r["mask"]
        rows.append([r["image"], r["normal"], overlay, r["parts"].float()[None] / 4, g])
        if args.pkl_dir:
            ann = ClipAnnotation.from_pkl(Path(args.pkl_dir) / f"{key}.pkl", key)
            crop = cache.meta[key]["crop"]
            s = crop["size"] / crop["side"]
            kp = ann.keypoints_2d[row]
            kp_c = (kp[:, :2] - np.array([crop["x0"], crop["y0"]])) * s
            valid = kp[:, 2] > 0.3
            j = r["joints_2d"].numpy()
            d = np.linalg.norm(kp_c[valid][:, None] - j[None], axis=-1).min(1)   # nearest joint per keypoint
            stats[key] = {"row": row, "mean_px_at_256": float(d.mean()), "median_px_at_256": float(np.median(d))}
    save_grid(rows, out / "overlay_grid.png")
    if stats:
        with open(out / "joint_keypoint_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("mean nearest-joint distance (px @256):", np.mean([v["mean_px_at_256"] for v in stats.values()]))
    print("wrote", out / "overlay_grid.png")


if __name__ == "__main__":
    main()
