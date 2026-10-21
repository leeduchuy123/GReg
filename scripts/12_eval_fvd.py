#!/usr/bin/env python
"""FVD (I3D, 16 uniform frames per clip) computed directly from the frames of ``07_eval_reconstruction.py --save-frames``
(or from the ``.npz`` written by ``10_export_fvd_clips.py``).

    python scripts/12_eval_fvd.py --eval-dir outputs/greg/eval_recon --i3d weights/i3d_torchscript.pt --bootstrap 10
    python scripts/12_eval_fvd.py --npz outputs/greg/fvd_clips.npz --i3d weights/i3d_torchscript.pt   # cross-check vs PGMM

Real-video features do not depend on the model: ``--real-cache f.npy`` stores/reuses them across runs
(only valid for the same clips; the clip keys are stored next to it and checked).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.eval.fvd import fvd_from_features, i3d_features, load_clip, load_i3d  # noqa: E402
from greg.utils.config import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--npz", default=None)
    p.add_argument("--i3d", required=True, help="TorchScript I3D (see greg/eval/fvd.py)")
    p.add_argument("--clip-length", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for mean/std (0 = off)")
    p.add_argument("--real-cache", default=None)
    p.add_argument("--out", default=None, help="json output (default: <eval-dir>/fvd.json)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if bool(args.eval_dir) == bool(args.npz):
        raise SystemExit("give exactly one of --eval-dir / --npz")
    device = get_device(args.device)
    detector = load_i3d(args.i3d, device)

    if args.npz:
        z = np.load(args.npz, allow_pickle=False)
        keys = [str(k) for k in z["video_ids"]]
        real, gen = z["real_videos"], z["generated_videos"]
    else:
        root = Path(args.eval_dir)
        keys = sorted(d.name for d in (root / "generated").iterdir() if d.is_dir())
        gen = np.stack([load_clip(root / "generated" / k, args.clip_length, args.size) for k in keys])
        real = None
    rf = None
    if args.real_cache and Path(args.real_cache).exists():
        saved = json.loads(Path(args.real_cache).with_suffix(".keys.json").read_text())
        if saved == keys:
            rf = np.load(args.real_cache)
            print(f"reusing real features {rf.shape}")
    if rf is None:
        if real is None:
            real = np.stack([load_clip(root / "real" / k, args.clip_length, args.size) for k in keys])
        rf = i3d_features(real, detector, device, args.batch_size)
        if args.real_cache:
            np.save(args.real_cache, rf)
            Path(args.real_cache).with_suffix(".keys.json").write_text(json.dumps(keys))
    gf = i3d_features(gen, detector, device, args.batch_size)
    res = fvd_from_features(rf, gf, args.bootstrap)
    res.update(clip_length=args.clip_length, size=args.size, feature_dim=int(rf.shape[1]))
    if rf.shape[0] < rf.shape[1]:
        res["warning"] = f"{rf.shape[0]} clips < feature dim {rf.shape[1]}: covariance is rank-deficient, FVD is biased"
    out = Path(args.out) if args.out else (Path(args.eval_dir) / "fvd.json" if args.eval_dir else Path(args.npz).with_suffix(".fvd.json"))
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
