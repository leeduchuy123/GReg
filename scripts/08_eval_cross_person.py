#!/usr/bin/env python
"""Cross-person video generation: source frame of clip A driven by the SMPL-X
motion of clip B.

The driving normal maps are re-rendered with the *source* person's shape
(``betas`` of A) and the camera of A's crop, so that the body proportions match
the source image -- this is where having both the default- and optimized-shape
SignAvatars releases matters.  Needs the renderer and the SMPL-X model files.

    python scripts/08_eval_cross_person.py --checkpoint ... --cache ... --pkl-dir ... \
        --smplx-dir models_smplx_v1_1/models --out outputs/greg/eval_cross --num-pairs 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, read_split  # noqa: E402
from greg.data.signavatars import ClipAnnotation  # noqa: E402
from greg.eval.runner import load_greg, save_frame  # noqa: E402
from greg.eval.visualize import labelled, save_gif, tile_grid, to_pil  # noqa: E402
from greg.render.body import Camera, SMPLXBody, region_centers_from_joints  # noqa: E402
from greg.render.normals import NormalRenderer  # noqa: E402
from greg.utils.config import get_device  # noqa: E402
from greg.utils.logging import save_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--pkl-dir", required=True)
    p.add_argument("--smplx-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--num-pairs", type=int, default=50)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--shape-from", choices=["source", "driving"], default="source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif-frames", type=int, default=120, help="frames per clip.gif (source | driving normal | warped | output)")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--backend", default="torch")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_greg(args.checkpoint, device)
    keys = read_split(Path(args.cache) / "splits" / f"{args.split}.txt")
    cache = HamNoSysCache(args.cache, keys)
    rng = random.Random(args.seed)
    pairs = []
    while len(pairs) < args.num_pairs:
        a, b = rng.sample(cache.keys, 2)
        pairs.append((a, b))
    body = SMPLXBody(args.smplx_dir, device=device)
    renderer = NormalRenderer(body, backend=args.backend)
    size = model.image_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a, b in tqdm(pairs):
        src = cache.load_row(a, args.source_row, with_mesh=model.uses_mesh)
        ann_a = ClipAnnotation.from_pkl(Path(args.pkl_dir) / f"{a}.pkl", a)
        ann_b = ClipAnnotation.from_pkl(Path(args.pkl_dir) / f"{b}.pkl", b)
        crop = cache.meta[a]["crop"]
        # camera of the source crop (frame `source_row` of A), applied to B's poses
        cam_a = Camera(torch.as_tensor(ann_a.focal[[args.source_row]]), torch.as_tensor(ann_a.princpt[[args.source_row]]),
                       torch.as_tensor(ann_a.cam_trans[[args.source_row]])).cropped(crop["x0"], crop["y0"], crop["side"], size)
        betas_a = torch.as_tensor(ann_a.smplx[args.source_row, 159:169])[None]
        rows = list(range(ann_b.num_rows))[:args.max_rows] if args.max_rows else list(range(ann_b.num_rows))
        Is = src["image"][None].to(device)
        Ns = src["normal"][None].to(device)
        frames = []
        gif = []
        for i in range(0, len(rows), args.batch):
            rr = rows[i:i + args.batch]
            vec = torch.as_tensor(ann_b.smplx[rr]).clone()
            # keep A's camera translation so the body stays where A's body is
            vec[:, 179:182] = torch.as_tensor(ann_a.smplx[args.source_row, 179:182])
            cam = Camera(cam_a.focal.expand(len(rr), -1), cam_a.princpt.expand(len(rr), -1), vec[:, 179:182])
            betas_t = betas_a if args.shape_from == "source" else None
            r = renderer.render(vec, cam, size, size, betas_override=betas_t, keep_fragments=model.uses_mesh)
            Nt = r.normal.to(device)
            ct = region_centers_from_joints(r.joints_2d).to(device)
            extra = {"joints_target": r.joints_2d.to(device)}
            if model.uses_mesh:   # A1: correspondence between A's source mesh and the re-posed target mesh
                n = len(rr)
                rep = lambda t: t[None].expand(n, *t.shape).to(device)   # noqa: E731
                extra["geometry"] = model.mesh_correspondence()(
                    rep(src["smplx"]), rep(src["focal"]), rep(src["princpt"]), rep(src["faces"]),
                    vec.to(device), cam.focal.to(device), cam.princpt.to(device), r.frag.face_idx.to(device),
                    betas_override_t=betas_t)
            with torch.no_grad():
                o = model(Is.expand(len(rr), -1, -1, -1), Ns.expand(len(rr), -1, -1, -1), Nt, ct, **extra)
            for j, row in enumerate(rr):
                save_frame(o["output"][j], out / f"{a}__{b}" / f"{row:06d}.png")
                if len(gif) < args.gif_frames:
                    gif.append(tile_grid([[labelled(to_pil(Is[0]), "source (A)"), labelled(to_pil(Nt[j][:3]), "driving N_t (B)"),
                                           labelled(to_pil(o["warped"][j]), "warped"),
                                           labelled(to_pil(o["output"][j].clamp(0, 1)), "output")]]))
                if len(frames) < 6 and (row % max(1, len(rows) // 6) == 0):
                    frames.append([Is[0], Nt[j], o["warped"][j], o["output"][j]])
        save_grid(frames, out / f"{a}__{b}" / "_grid.png")
        if gif:
            save_gif(gif, out / f"{a}__{b}" / "clip.gif", args.fps)
        manifest.append({"source": a, "driving": b, "num_frames": len(rows)})
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": manifest, "shape_from": args.shape_from, "seed": args.seed}, f, indent=2)
    print(f"wrote {len(manifest)} cross-person sequences to {out}")


if __name__ == "__main__":
    main()
