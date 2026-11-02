#!/usr/bin/env python
"""D4: generate a bank of empty-background images with Stable Diffusion 1.5 (text-to-image, 512 -> 256).

    python scripts/27_make_backgrounds.py --out data/backgrounds --num 500

Prompts are combinations of plain indoor / studio backgrounds and lighting (sign-language videos are filmed
indoors in front of a wall); a negative prompt keeps people and text out.  Seeds 0..N-1, so the bank is
reproducible.
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch

PLACES = ["a plain painted wall", "an office wall", "a living room wall with a bookshelf", "a classroom wall",
          "a studio backdrop", "a kitchen wall", "a brick wall", "a wooden wall", "a corridor", "a library",
          "a white wall with a window", "a meeting room", "a curtain", "a green screen studio", "a bedroom wall",
          "a concrete wall", "a hallway with doors", "a cafe interior", "a gym wall", "a tiled wall"]
LIGHT = ["soft daylight", "warm indoor light", "cool fluorescent light", "evening lamp light", "bright studio light"]
COLORS = ["", "light blue ", "beige ", "grey ", "dark ", "pastel green ", "warm orange "]
NEG = "person, people, man, woman, hands, face, text, watermark, logo, blurry"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=500)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    combos = list(itertools.product(COLORS, PLACES, LIGHT))
    random.Random(0).shuffle(combos)
    prompts = [f"photo of {c}{pl}, empty, no people, {li}, background, high quality" for c, pl, li in combos]
    todo = [i for i in range(a.num) if not (out / f"{i:04d}.png").exists()]
    for k in range(0, len(todo), a.batch):
        ids = todo[k:k + a.batch]
        gens = [torch.Generator("cuda").manual_seed(i) for i in ids]
        imgs = pipe([prompts[i % len(prompts)] for i in ids], negative_prompt=[NEG] * len(ids),
                    num_inference_steps=a.steps, guidance_scale=6.0, generator=gens).images
        for i, im in zip(ids, imgs):
            im.resize((a.size, a.size)).save(out / f"{i:04d}.png")
        print(f"{k + len(ids)}/{len(todo)}", flush=True)
    (out / "prompts.txt").write_text("\n".join(f"{i:04d}\t{prompts[i % len(prompts)]}" for i in range(a.num)))


if __name__ == "__main__":
    main()
