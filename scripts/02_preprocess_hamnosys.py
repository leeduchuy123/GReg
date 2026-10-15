#!/usr/bin/env python
"""Build the 256x256 training cache for the HamNoSys subset.

For every clip ``<key>`` with ``<pkl_dir>/<key>.pkl`` and ``<video_dir>/<key>.mp4``:

1. probe the video; extract frames with the fps rule (``-r 25`` only for >30 fps);
2. check that ``total_valid_index`` fits inside the extracted frames;
3. run SMPL-X for every row, project the vertices, and compute ONE square crop
   box per clip (union of the projected body over all rows + margin);
4. crop + resize the RGB frames to ``--size`` and render the normal map / mask /
   part map directly in the crop camera at ``--size``;
5. store projected joints in crop coordinates and a meta entry.

Run this where you have a GPU (Colab / Kaggle); the pure-torch rasteriser needs
no compilation.  ``--backend nvdiffrast`` is faster if that package is installed.

Example::

    python scripts/02_preprocess_hamnosys.py --pkl-dir /data/hamnosys_pkls --video-dir /data/videos \
        --out /data/hamnosys_cache --smplx-dir models_smplx_v1_1/models --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.frames import check_alignment, extract_frames, probe_video  # noqa: E402
from greg.data.signavatars import ClipAnnotation  # noqa: E402
from greg.render.body import Camera, SMPLXBody, project  # noqa: E402
from greg.render.normals import NormalRenderer, encode_normal_png  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl-dir", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smplx-dir", required=True, help="folder containing smplx/SMPLX_NEUTRAL.npz")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--margin", type=float, default=0.15, help="crop margin relative to the body box side")
    p.add_argument("--keys", default=None, help="optional txt file with the keys to process")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", default="torch", choices=["torch", "nvdiffrast"])
    p.add_argument("--batch", type=int, default=16, help="rows rendered per batch")
    p.add_argument("--jpeg-quality", type=int, default=2, help="ffmpeg -q:v for the temp frames")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tmp", default=None, help="temp dir for extracted frames (default: system temp)")
    return p.parse_args()


def compute_crop_box(uv_all: torch.Tensor, width: int, height: int, margin: float):
    """uv_all (N, V, 2) projected vertices of all rows -> (x0, y0, side) square crop.

    Only the part of the body that is inside the frame counts (legs are usually
    below the image in HamNoSys), so the projections are clipped to the image
    first.  The square is shifted to stay inside the image when it fits;
    otherwise it is centred and the frame is padded with black.
    """
    pts = uv_all.reshape(-1, 2).clone()
    pts[:, 0] = pts[:, 0].clamp(0, width - 1)
    pts[:, 1] = pts[:, 1].clamp(0, height - 1)
    mn = pts.min(0).values
    mx = pts.max(0).values
    center = (mn + mx) / 2
    side = float((mx - mn).max()) * (1 + margin)
    side = max(side, 32.0)
    x0 = float(center[0]) - side / 2
    y0 = float(center[1]) - side / 2
    if side <= width:
        x0 = min(max(x0, 0.0), width - side)
    else:
        x0 = (width - side) / 2
    if side <= height:
        y0 = min(max(y0, 0.0), height - side)
    else:
        y0 = (height - side) / 2
    return x0, y0, side


def crop_resize(img: Image.Image, x0: float, y0: float, side: float, size: int) -> Image.Image:
    # PIL pads with black when the box exceeds the image, which is what we want
    return img.crop((int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side)))).resize(
        (size, size), Image.BILINEAR)


def process_clip(key: str, args, body: SMPLXBody, renderer: NormalRenderer, out: Path, tmp_root: Path) -> dict:
    pkl = Path(args.pkl_dir) / f"{key}.pkl"
    video = Path(args.video_dir) / f"{key}.mp4"
    if not video.exists():
        return {"status": "missing_video"}
    ann = ClipAnnotation.from_pkl(pkl, key=key)
    info = probe_video(video)
    if (info["width"], info["height"]) != (ann.width, ann.height):
        return {"status": "size_mismatch", "video_size": [info["width"], info["height"]],
                "pkl_size": [ann.width, ann.height]}
    frames_dir = tmp_root / key
    n_frames = extract_frames(video, frames_dir, fps=info["fps"], quality=args.jpeg_quality, overwrite=True)
    align = check_alignment(n_frames, ann.frame_index)
    if align == "MISMATCH":
        shutil.rmtree(frames_dir, ignore_errors=True)
        return {"status": "frame_mismatch", "num_extracted": n_frames,
                "tvi_max": int(ann.frame_index.max()), "fps": info["fps"]}

    dev = body.device
    n = ann.num_rows
    size = args.size
    vec = torch.as_tensor(ann.smplx, device=dev)
    cam_native = Camera(torch.as_tensor(ann.focal, device=dev), torch.as_tensor(ann.princpt, device=dev),
                        torch.as_tensor(ann.cam_trans, device=dev))

    # pass 1: crop box from all projected vertices
    uv_chunks = []
    for i in range(0, n, args.batch):
        o = body(vec[i:i + args.batch])
        c = Camera(cam_native.focal[i:i + args.batch], cam_native.princpt[i:i + args.batch],
                   cam_native.cam_trans[i:i + args.batch])
        uv_chunks.append(project(o["vertices"], c).cpu())
    x0, y0, side = compute_crop_box(torch.cat(uv_chunks), ann.width, ann.height, args.margin)

    (out / "frames" / key).mkdir(parents=True, exist_ok=True)
    (out / "normals" / key).mkdir(parents=True, exist_ok=True)
    (out / "parts" / key).mkdir(parents=True, exist_ok=True)
    joints_all = np.zeros((n, 0, 2), dtype=np.float32)
    jl = []
    # pass 2: render in the crop camera and crop the frames
    for i in range(0, n, args.batch):
        rows = list(range(i, min(i + args.batch, n)))
        c = Camera(cam_native.focal[rows], cam_native.princpt[rows], cam_native.cam_trans[rows]).cropped(
            x0, y0, side, size)
        r = renderer.render(vec[rows], c, size, size)
        jl.append(r.joints_2d.cpu().numpy().astype(np.float32))
        for j, row in enumerate(rows):
            Image.fromarray(encode_normal_png(r.normal[j], r.mask[j]), "RGBA").save(out / "normals" / key / f"{row:06d}.png")
            Image.fromarray(r.part[j].cpu().numpy().astype(np.uint8), "L").save(out / "parts" / key / f"{row:06d}.png")
            fr = Image.open(frames_dir / f"{int(ann.frame_index[row]):06d}.jpg").convert("RGB")
            crop_resize(fr, x0, y0, side, size).save(out / "frames" / key / f"{row:06d}.png")
    joints_all = np.concatenate(jl, 0)
    (out / "joints").mkdir(parents=True, exist_ok=True)
    np.save(out / "joints" / f"{key}.npy", joints_all)
    shutil.rmtree(frames_dir, ignore_errors=True)
    return {"status": "ok", "num_rows": n, "num_extracted": n_frames, "alignment": align,
            "fps": info["fps"], "native_size": [ann.width, ann.height],
            "crop": {"x0": x0, "y0": y0, "side": side, "size": size},
            "frame_index": ann.frame_index.tolist()}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = {"clips": {}, "config": vars(args).copy()}
    if meta_path.exists() and not args.overwrite:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if args.keys:
        keys = [l.strip() for l in open(args.keys, encoding="utf-8") if l.strip()]
    else:
        keys = sorted(p.stem for p in Path(args.pkl_dir).glob("*.pkl"))
    if args.limit:
        keys = keys[:args.limit]
    body = SMPLXBody(args.smplx_dir, device=args.device)
    renderer = NormalRenderer(body, backend=args.backend)
    tmp_root = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="greg_frames_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    for key in tqdm(keys):
        if key in meta["clips"] and meta["clips"][key].get("status") == "ok" and not args.overwrite:
            continue
        try:
            meta["clips"][key] = process_clip(key, args, body, renderer, out, tmp_root)
        except Exception as e:  # keep going; record the failure
            meta["clips"][key] = {"status": f"error: {type(e).__name__}: {e}"[:300]}
        done += 1
        if done % 20 == 0:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    from collections import Counter
    print(Counter(v.get("status", "?").split(":")[0] for v in meta["clips"].values()))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {meta_path}")


if __name__ == "__main__":
    main()
