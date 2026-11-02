#!/usr/bin/env python
"""D6: generate the cached evaluation windows with a trained (or the zero-shot) Wan model and write frames for
scripts/21_eval_frames.py.

    python scripts/34_sample_wan.py --config configs/diffusion/d6_wan.yaml --checkpoint outputs/d6/ng/checkpoints/final.pt \
        --windows outputs/d6/cache/test --out outputs/d6/ng/frames_test
    python scripts/34_sample_wan.py --config ... --variant base --zero-shot ...      # ablation 0 (no LoRA, 36 ch)

Frames: decoded at the work size (fp32 VAE), frame 0 (the given source) and padded frames dropped, area-downsampled to
256, written as ``<out>/<key>/<row>.png`` with ``index.json`` (every generated row).  Noise: the same seed for every
window of a clip.  ``timing.json``: seconds per generated frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from greg.diffusion.wan_video import WanConditionModel, WanLatentCodec, load_fun_inp_transformer, sample  # noqa: E402
from greg.utils.config import apply_overrides, load_config  # noqa: E402

_tr = __import__("33_train_wan")
_cw = __import__("32_cache_wan_windows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--zero-shot", action="store_true", help="the pretrained Fun-InP, no LoRA, 36 input channels")
    p.add_argument("--windows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    dev = torch.device("cuda")
    if a.checkpoint:
        cfg = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["config"]
    else:
        cfg = load_config(a.config)
    if a.variant:
        cfg = apply_overrides(cfg, [f"model.variant={a.variant}"])
    tr = load_fun_inp_transformer(cfg["model"]["checkpoint_dir"], torch.bfloat16)
    model = WanConditionModel(tr, cfg["model"]["variant"], 0 if a.zero_shot else int(cfg["model"].get("lora_rank", 32)))
    if a.checkpoint:
        model.load_trainable_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["trainable"])
    model = model.to(dev).eval()
    from diffusers import AutoencoderKLWan
    codec = WanLatentCodec(AutoencoderKLWan.from_pretrained(_cw.VAE_REPO, subfolder="vae", torch_dtype=torch.float32).to(dev))
    text = torch.load(Path(a.windows).parent / "text.pt", weights_only=False)["text"].to(dev)
    files = sorted(Path(a.windows).glob("*.pt"))[:a.max_windows]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clip_ids = {k: i for i, k in enumerate(sorted({f.stem.rsplit("_", 1)[0] for f in files}))}
    index, t_model, n = {}, 0.0, 0
    for f in tqdm(files):
        r = torch.load(f, map_location="cpu", weights_only=False)
        key = r["key"]
        for row, ok in zip(r["rows"][1:], r["valid_rows"][1:]):
            if ok:
                index.setdefault(key, {"source": r["rows"][0], "rows": []})["rows"].append(int(row))
        w = {k: r[k][None].to(dev) for k in _tr.FIELDS}
        g = torch.Generator().manual_seed(clip_ids[key])
        noise = torch.randn(w["lat_masked"].shape, generator=g).to(dev, torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = sample(model, w, text[None], w["clip"], steps=a.steps, shift=float(cfg["model"].get("shift", 5.0)),
                       noise=noise)
        vid = codec.decode(z.float())[0]                                     # (3, 49, S, S) in [0, 1]
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        frames = F.interpolate(vid.permute(1, 0, 2, 3), size=(a.size, a.size), mode="area").clamp(0, 1)
        for j, (row, ok) in enumerate(zip(r["rows"], r["valid_rows"])):
            if j == 0 or not ok:
                continue
            n += 1
            pth = out / key / f"{int(row):06d}.png"
            pth.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((frames[j].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(pth)
    for v in index.values():
        v["rows"] = sorted(set(v["rows"]))
    (out / "index.json").write_text(json.dumps({"args": vars(a), "clips": index}))
    (out / "timing.json").write_text(json.dumps({"frames": n, "sec_per_frame": t_model / max(n, 1), "steps": a.steps}))
    print(f"{n} frames, {t_model / max(n, 1):.2f} s/frame -> {out}")


if __name__ == "__main__":
    main()
