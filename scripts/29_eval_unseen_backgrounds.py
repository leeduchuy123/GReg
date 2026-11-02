#!/usr/bin/env python
"""D4: generalisation test with ground truth -- reconstruction on val clips composited onto backgrounds that no
model saw in training (a held-out part of the SD background bank, ``--bg-dir``), with the person matte.

Source and target of a clip get the same background (one per clip, fixed seed), so the task is the usual
reconstruction; only the appearance distribution shifts.  Every checkpoint is scored on identical inputs.

    python scripts/29_eval_unseen_backgrounds.py --checkpoint ck.pth --cache CACHE --bg-dir data/backgrounds_heldout \
        --out outputs/d4/unseen/base --lpips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.augment import ClipConsistentAugment  # noqa: E402
from greg.data.dataset import read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.eval import metrics as M  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.runner import load_greg  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--bg-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-clips", type=int, default=100)
    p.add_argument("--max-rows", type=int, default=16)
    p.add_argument("--color-p", type=float, default=0.0, help="also jitter the person's colours (clip-consistent)")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = get_device(a.device)
    model, _ = load_greg(a.checkpoint, dev)
    keys = pick_keys(read_split(Path(a.cache) / "splits" / f"{a.split}.txt"), a.max_clips, seed=0)
    aug = ClipConsistentAugment(a.cache, a.bg_dir, background_p=1.0, color_p=a.color_p, seed=12345)
    _, s = evaluate_reconstruction(model, a.cache, keys, dev, batch_size=a.batch_size, num_workers=a.num_workers,
                                   max_rows=a.max_rows, uniform_rows=True, lpips_metric=M.LPIPSMetric(dev) if a.lpips else None,
                                   item_augment=aug)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s.update(checkpoint=a.checkpoint, bg_dir=a.bg_dir, color_p=a.color_p)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k.startswith("mean_") and "@" not in k}, indent=2))


if __name__ == "__main__":
    main()
