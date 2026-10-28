#!/usr/bin/env python
"""Mark the usable rows of every cached clip (fade-in / fade-out trimming, see ``greg.data.frames.fade_trim``).

Reads the cached RGB crops, computes the mean brightness of every row and stores the inclusive range of
usable rows as ``valid_range`` in ``meta.json``.  ``HamNoSysCache.valid_rows`` then restricts training
pairs, the reconstruction source row and the evaluated targets to that range.

    python scripts/02b_mark_valid_rows.py --cache /data/hamnosys_cache

Sanity checks (exit code 1 on failure, so the pipeline stops): only GSL clips are expected to fade;
trimming more than ``--max-other-trim`` of the rows of the other sources means the threshold is wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.frames import FADE_RATIO, fade_trim  # noqa: E402
from greg.data.splits import source_of_key  # noqa: E402


def clip_brightness(args) -> tuple:
    root, key, n = args
    b = []
    for row in range(n):
        img = Image.open(Path(root) / "frames" / key / f"{row:06d}.png").convert("L")
        b.append(float(np.asarray(img, dtype=np.float32).mean()) / 255.0)
    return key, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ratio", type=float, default=FADE_RATIO)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-other-trim", type=float, default=0.01,
                   help="fail if more than this fraction of the rows of non-GSL clips is trimmed")
    args = p.parse_args()
    root = Path(args.cache)
    with open(root / "meta.json", encoding="utf-8") as f:
        full = json.load(f)
    clips = full.get("clips", full)
    jobs = [(str(root), k, int(v["num_rows"])) for k, v in clips.items() if v.get("status") == "ok"]
    rows_total, rows_trimmed, clips_trimmed = Counter(), Counter(), Counter()
    clips_total = Counter()
    with Pool(args.workers) as pool:
        for key, b in pool.imap_unordered(clip_brightness, jobs, chunksize=8):
            first, last = fade_trim(b, args.ratio)
            clips[key]["valid_range"] = [first, last]
            src = source_of_key(key)
            n = len(b)
            kept = max(last - first + 1, 0)
            clips_total[src] += 1
            rows_total[src] += n
            rows_trimmed[src] += n - kept
            clips_trimmed[src] += int(kept < n)
    full.setdefault("config", {})["fade_ratio"] = args.ratio
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(full, f)
    report = {s: {"clips": clips_total[s], "clips_trimmed": clips_trimmed[s], "rows": rows_total[s],
                  "rows_trimmed": rows_trimmed[s]} for s in sorted(clips_total)}
    print(json.dumps(report, indent=2))
    other = [s for s in report if s != "GSL"]
    other_frac = sum(rows_trimmed[s] for s in other) / max(sum(rows_total[s] for s in other), 1)
    if "GSL" in report and report["GSL"]["clips_trimmed"] == 0:
        sys.exit("sanity check failed: no GSL clip was trimmed (the fades are expected there)")
    if other_frac > args.max_other_trim:
        sys.exit(f"sanity check failed: {other_frac:.1%} of the non-GSL rows were trimmed (> {args.max_other_trim:.0%})")
    print(f"ok: non-GSL rows trimmed {other_frac:.2%}")


if __name__ == "__main__":
    main()
