#!/usr/bin/env python
"""Random 8:2 clip split over the preprocessed cache (GReg protocol, seed recorded)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.splits import make_random_split, make_val_split, source_of_key, write_split  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="fraction of the TRAIN clips held out as val.txt (train_fit.txt = the rest); 0 disables")
    p.add_argument("--min-rows", type=int, default=2)
    args = p.parse_args()
    cache = Path(args.cache)
    with open(cache / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)["clips"]
    keys = [k for k, v in meta.items() if v.get("status") == "ok" and v.get("num_rows", 0) >= args.min_rows]
    split = make_random_split(keys, args.train_ratio, args.seed)
    if args.val_ratio > 0:
        split.update(make_val_split(split["train"], args.val_ratio, args.seed))
    info = {"seed": args.seed, "train_ratio": args.train_ratio, "num_train": len(split["train"]),
            "num_test": len(split["test"]),
            "val_ratio": args.val_ratio, "num_train_fit": len(split.get("train_fit", [])),
            "num_val": len(split.get("val", [])),
            "by_source": {n: dict(Counter(source_of_key(k) for k in ks)) for n, ks in split.items()}}
    write_split(split, cache / "splits", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
