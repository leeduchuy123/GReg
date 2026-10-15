#!/usr/bin/env python
"""Export generated / real sequences as an NPZ compatible with the PGMM repo's
FVD evaluator (``PGMM/scripts/10_eval_fvd.py``: I3D Kinetics-400, 16 uniformly
sampled frames per clip, NTHWC uint8).

    python scripts/10_export_fvd_clips.py --eval-dir outputs/greg/eval_recon --out outputs/greg/fvd_clips.npz
    python "D:/Research Thesis/PGMM/scripts/10_eval_fvd.py" --clips outputs/greg/fvd_clips.npz ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def uniform_indices(n: int, k: int):
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def load_clip(dir_: Path, k: int, size: int):
    paths = sorted(dir_.glob("*.png"))
    idx = uniform_indices(len(paths), k)
    frames = [np.asarray(Image.open(paths[i]).convert("RGB").resize((size, size))) for i in idx]
    return np.stack(frames).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip-length", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()
    root = Path(args.eval_dir)
    keys = sorted(d.name for d in (root / "generated").iterdir() if d.is_dir())
    gen = np.stack([load_clip(root / "generated" / k, args.clip_length, args.size) for k in keys])
    real = np.stack([load_clip(root / "real" / k, args.clip_length, args.size) for k in keys])
    meta = {"clip_length": args.clip_length, "size": args.size, "protocol": "uniform-16-incl-endpoints"}
    np.savez_compressed(args.out, real_videos=real, generated_videos=gen, video_ids=np.asarray(keys),
                        source_frame_indices=np.zeros(len(keys), dtype=np.int64), metadata_json=json.dumps(meta))
    print(f"{len(keys)} clips -> {args.out} {gen.shape}")


if __name__ == "__main__":
    main()
