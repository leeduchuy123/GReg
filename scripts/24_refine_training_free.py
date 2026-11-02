#!/usr/bin/env python
"""D1 run 0: training-free refiner = SD1.5 + ControlNet-Tile img2img on GReg's output, gated like the trained one.

Answers "how far does a pretrained diffusion prior get without any training?" (docs/diffusion_6_huong_ke_hoach.md
Sec. 3.5, run 0).  Every frame: GReg output upsampled to 512 -> img2img with ``--strength`` (fraction of the noise
schedule, 0.3 / 0.5), ControlNet-Tile conditioned on the same image (keeps the layout), fixed prompt, fixed seed per
(clip, row) -> area-downsample to 256 -> gated blend into GReg's output.

    python scripts/24_refine_training_free.py --export outputs/export/run5_epoch_100/val --cache data/hamnosys_cache \
        --out outputs/d1/run0_s03 --strength 0.3
"""
from __future__ import annotations

import argparse
import hashlib
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
from greg.diffusion.gating import blend, refine_gate, resize  # noqa: E402
from greg.diffusion.refiner_data import RefinerDataset, flatten_frames, read_index  # noqa: E402
from greg.diffusion.turbo_refiner import DEFAULT_PROMPT  # noqa: E402

NEG = "blurry, deformed hands, extra fingers, lowres, artifacts"


def seed_of(key: str, row: int) -> int:
    return int(hashlib.md5(f"{key}/{row}".encode()).hexdigest()[:8], 16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=20, help="full schedule length; strength * steps are run")
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--control-scale", type=float, default=1.0)
    p.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--controlnet", default="lllyasviel/control_v11f1e_sd15_tile")
    p.add_argument("--work-size", type=int, default=512)
    p.add_argument("--blends", nargs="+", default=["soft"])
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--max-items-per-clip", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
    dev = torch.device("cuda")
    cn = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(args.base, controlnet=cn, torch_dtype=torch.float16,
                                                                    safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    idx = read_index(args.export)
    keys = sorted(idx)[:args.max_clips] if args.max_clips else sorted(idx)
    ds = RefinerDataset(args.export, HamNoSysCache(args.cache, keys), pairs=False,
                        max_items_per_clip=args.max_items_per_clip)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=collate_keep_strings)
    out = Path(args.out)
    sel = {}
    for k, s, t in ds.items:
        sel.setdefault(k, {"source": s, "rows": []})["rows"].append(t)
    for bm in args.blends:
        (out / bm).mkdir(parents=True, exist_ok=True)
        (out / bm / "index.json").write_text(json.dumps({"args": vars(args), "clips": sel}))
    t_model, n = 0.0, 0
    for batch in tqdm(loader):
        b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in flatten_frames(batch).items()}
        img = resize(b["greg"], args.work_size)
        gens = [torch.Generator(dev).manual_seed(seed_of(k, int(r))) for k, r in zip(batch["key"], batch["target_row"])]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = pipe(prompt=[DEFAULT_PROMPT] * img.shape[0], negative_prompt=[NEG] * img.shape[0], image=img.half(),
                   control_image=img.half(), strength=args.strength, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, controlnet_conditioning_scale=args.control_scale,
                   generator=gens, output_type="pt").images
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        n += img.shape[0]
        r = resize(res.float().to(dev), b["greg"].shape[-1])
        gate = refine_gate(b["parts"], b["vis"])
        for bm in args.blends:
            o = blend(b["greg"], r, gate, bm)
            for i in range(o.shape[0]):
                pth = out / bm / batch["key"][i] / f"{int(batch['target_row'][i]):06d}.png"
                pth.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((o[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(pth)
    (out / "timing.json").write_text(json.dumps({"frames": n, "sec_per_frame": t_model / max(n, 1), **vars(args)}))
    print(f"{n} frames, {1000 * t_model / max(n, 1):.0f} ms/frame -> {out}")


if __name__ == "__main__":
    main()
