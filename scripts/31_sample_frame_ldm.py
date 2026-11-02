#!/usr/bin/env python
"""D5: generate frames with a trained FrameLDM on a fixed clip subset (reconstruction protocol), for 21_eval_frames.

    python scripts/31_sample_frame_ldm.py --checkpoint outputs/d5/ng/checkpoints/final.pt --cache CACHE \
        --split test --max-clips 200 --rows 16 --out outputs/d5/ng/frames_test --smplx-dir SMPLX

Writes ``<out>/<key>/<row>.png`` at the cache size (area-downsampled), ``index.json`` (the rows, so every method is
scored on the same ones: score GReg / D1 with ``21_eval_frames.py --index <out>/index.json``) and ``timing.json``.
The initial noise is the same for every frame of a clip (seed = clip index), so frames differ only by condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from greg.data.dataset import HamNoSysCache, SequenceDataset, collate_keep_strings, read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.utils.config import apply_overrides  # noqa: E402

_tr = __import__("30_train_frame_ldm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-clips", type=int, default=200)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--smplx-dir", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    a = p.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = st["config"]
    if a.smplx_dir:
        cfg = apply_overrides(cfg, [f"data.smplx_dir={a.smplx_dir}"])
    model = _tr.build_model(cfg, dev)
    model.load_trainable_state_dict(st["trainable"])
    model.eval()
    work = int(cfg["model"].get("work_size", 512))
    steps = int(a.steps or cfg["model"].get("sample_steps", 25))
    cond_fn = _tr.Conditioner(cfg["model"]["variant"], work, cfg["data"].get("smplx_dir"), dev)
    keys = pick_keys(read_split(Path(a.cache) / "splits" / f"{a.split}.txt"), a.max_clips, seed=0)
    cache = HamNoSysCache(a.cache, keys)
    cache.keep_keys_with_mesh(what="d5 sample")
    ds = SequenceDataset(cache, source_row=0, max_rows=a.rows, uniform_rows=True, with_mesh=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for k, s, t in ds.items:
        index.setdefault(k, {"source": s, "rows": []})["rows"].append(t)
    (out / "index.json").write_text(json.dumps({"args": vars(a), "clips": index}))
    ds.items = [it for it in ds.items if not (out / it[0] / f"{it[2]:06d}.png").exists()]
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=collate_keep_strings)
    clip_id = {k: i for i, k in enumerate(sorted(index))}
    t_model, n = 0.0, 0
    for batch in tqdm(loader):
        b = _tr.to_dev(batch, dev)
        z = torch.stack([torch.randn(4, work // 8, work // 8, generator=torch.Generator().manual_seed(clip_id[k]))
                         for k in batch["key"]]).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = cond_fn(b)
            o = model.sample(_tr.up(b["source_image"], work), c, steps=steps, noise=z)
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        n += o.shape[0]
        o = F.interpolate(o.float(), size=b["target_image"].shape[-2:], mode="area").clamp(0, 1)
        for i, k in enumerate(batch["key"]):
            pth = out / k / f"{int(batch['target_row'][i]):06d}.png"
            pth.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((o[i].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(pth)
    (out / "timing.json").write_text(json.dumps({"frames": n, "sec_per_frame": t_model / max(n, 1), "steps": steps}))
    print(f"{n} frames, {t_model / max(n, 1):.2f} s/frame -> {out}")


if __name__ == "__main__":
    main()
