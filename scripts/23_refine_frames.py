#!/usr/bin/env python
"""D1: run a trained refiner over a GReg export and write frame directories for scripts/21_eval_frames.py.

    python scripts/23_refine_frames.py --checkpoint outputs/d1/full/checkpoints/final.pt \
        --export outputs/export/run5_epoch_100/test --cache data/hamnosys_cache \
        --out outputs/d1/full/frames_test --blends soft hard freq none

Writes ``<out>/<blend>/<key>/<row>.png`` (+ ``index.json`` copied from the export, so the evaluation scores exactly
the exported rows) and ``<out>/timing.json`` (seconds per frame of the refiner alone, batch ``--batch-size``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, collate_keep_strings  # noqa: E402
from greg.diffusion.gating import blend, refine_gate  # noqa: E402
from greg.diffusion.refiner_data import RefinerDataset, flatten_frames, read_index  # noqa: E402
from greg.utils.config import apply_overrides, get_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = __import__("22_train_refiner")


def save_png(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--export", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blends", nargs="+", default=["soft"])
    p.add_argument("--override", nargs="*", default=[], help="e.g. gate.dilate_px=8")
    p.add_argument("--keys-file", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = apply_overrides(ck["config"], args.override)
    ref = _train.build_refiner(cfg, device)
    ref.load_trainable_state_dict(ck["trainable"])
    ref.eval()
    amp = torch.bfloat16 if cfg["train"].get("amp_dtype", "bf16") == "bf16" else torch.float16
    idx = read_index(args.export)
    keys = [l.strip() for l in open(args.keys_file, encoding="utf-8") if l.strip()] if args.keys_file else list(idx)
    keys = [k for k in keys if k in idx]
    ds = RefinerDataset(args.export, HamNoSysCache(args.cache, keys), pairs=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_keep_strings)
    out = Path(args.out)
    for bm in args.blends:
        (out / bm).mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.export) / "index.json", out / bm / "index.json")
    g = cfg.get("gate", {})
    t_model, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader):
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in flatten_frames(batch).items()}
            img, nrm, msk = _train.refiner_inputs(b, cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                r = ref(img, nrm, msk)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model += time.perf_counter() - t0
            n += img.shape[0]
            r = _train.resize(r.float(), b["greg"].shape[-1])
            gate = refine_gate(b["parts"], b["vis"], int(g.get("dilate_px", 6)), bool(g.get("include_face", True)),
                               bool(g.get("include_disocc", True)))
            for bm in args.blends:
                o = blend(b["greg"], r, gate, bm, int(g.get("feather_px", 4)))
                for i in range(o.shape[0]):
                    save_png(o[i], out / bm / batch["key"][i] / f"{int(batch['target_row'][i]):06d}.png")
    (out / "timing.json").write_text(json.dumps({"frames": n, "sec_per_frame": t_model / max(n, 1),
                                                 "batch_size": args.batch_size, "work_size": cfg["model"]["work_size"]}))
    print(f"{n} frames, refiner {1000 * t_model / max(n, 1):.1f} ms/frame -> {out}")


if __name__ == "__main__":
    main()
