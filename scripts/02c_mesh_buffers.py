#!/usr/bin/env python
"""Add the mesh buffers needed by the analytic mesh flow (``model.flow.type: mesh``) to an existing cache.

Re-renders every row of every ``ok`` clip with exactly the camera of ``02_preprocess_hamnosys.py`` (crop box
from ``meta.json``) but does NOT decode the videos again.  Writes::

    faces/<key>/000012.png   uint16 PNG, rasterised SMPL-X face index + 1 (0 = background)
    mesh/<key>.npz           smplx (N, 182) float32, focal (N, 2), princpt (N, 2)  -- crop intrinsics

The SMPL-X vertices are recomputed from ``smplx`` at training time (greg.geometry.mesh_flow), which keeps the
cache small (~1 KB per row on top of the face-index PNG).  ``--check`` compares the re-rendered body mask with
the alpha channel of ``normals/*.png`` (the two renders must agree, otherwise the camera is wrong).

    python scripts/02c_mesh_buffers.py --cache data/hamnosys_cache --pkl-dir PKLS \
        --smplx-dir models_smplx_v1_1/models --device cuda [--shard 0/4] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.signavatars import ClipAnnotation  # noqa: E402
from greg.render.body import Camera, SMPLXBody  # noqa: E402
from greg.render.normals import NormalRenderer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--pkl-dir", required=True)
    p.add_argument("--smplx-dir", required=True, help="folder containing smplx/SMPLX_NEUTRAL.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", default="torch", choices=["torch", "nvdiffrast"])
    p.add_argument("--batch", type=int, default=16, help="rows rendered per batch")
    p.add_argument("--keys", default=None, help="optional txt file with the keys to process")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="I/N: process keys[I::N] only (run N processes in parallel)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check", action="store_true", help="compare the body mask with normals/*.png alpha")
    return p.parse_args()


def save_face_png(face_idx: torch.Tensor, path: Path) -> None:
    """(H, W) long, -1 = background -> uint16 PNG of face_idx + 1."""
    arr = (face_idx.cpu().numpy() + 1).astype(np.uint16)
    Image.fromarray(arr).save(path)


def process_clip(key: str, meta: dict, args, body: SMPLXBody, renderer: NormalRenderer, cache: Path) -> dict:
    ann = ClipAnnotation.from_pkl(Path(args.pkl_dir) / f"{key}.pkl", key=key)
    n = int(meta["num_rows"])
    if ann.num_rows != n:
        return {"status": "row_mismatch", "pkl_rows": ann.num_rows, "cache_rows": n}
    crop = meta["crop"]
    size = int(crop["size"])
    dev = body.device
    vec = torch.as_tensor(ann.smplx, device=dev)
    cam_native = Camera(torch.as_tensor(ann.focal, device=dev), torch.as_tensor(ann.princpt, device=dev),
                        torch.as_tensor(ann.cam_trans, device=dev))
    out_faces = cache / "faces" / key
    out_faces.mkdir(parents=True, exist_ok=True)
    focal, princpt, mismatch = [], [], []
    for i in range(0, n, args.batch):
        rows = list(range(i, min(i + args.batch, n)))
        c = Camera(cam_native.focal[rows], cam_native.princpt[rows], cam_native.cam_trans[rows]).cropped(
            crop["x0"], crop["y0"], crop["side"], size)
        r = renderer.render(vec[rows], c, size, size, keep_fragments=True)
        focal.append(c.focal.cpu().numpy())
        princpt.append(c.princpt.cpu().numpy())
        for j, row in enumerate(rows):
            save_face_png(r.frag.face_idx[j], out_faces / f"{row:06d}.png")
            if args.check:
                alpha = np.asarray(Image.open(cache / "normals" / key / f"{row:06d}.png").convert("RGBA"))[..., 3] > 127
                mismatch.append(float((alpha != r.frag.mask[j].cpu().numpy()).mean()))
    np.savez(cache / "mesh" / f"{key}.npz", smplx=ann.smplx.astype(np.float32),
             focal=np.concatenate(focal).astype(np.float32), princpt=np.concatenate(princpt).astype(np.float32))
    res = {"status": "ok", "num_rows": n}
    if mismatch:
        res["mask_mismatch"] = float(np.mean(mismatch))
    return res


def main():
    args = parse_args()
    cache = Path(args.cache)
    with open(cache / "meta.json", encoding="utf-8") as f:
        clips = json.load(f)["clips"]
    if args.keys:
        keys = [l.strip() for l in open(args.keys, encoding="utf-8") if l.strip()]
    else:
        keys = sorted(k for k, v in clips.items() if isinstance(v, dict) and v.get("status") == "ok")
    if args.limit:
        keys = keys[:args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        keys = keys[i::n]
    (cache / "mesh").mkdir(exist_ok=True)
    body = SMPLXBody(args.smplx_dir, device=args.device)
    renderer = NormalRenderer(body, backend=args.backend)
    stats = {}
    for key in tqdm(keys, desc="mesh buffers"):
        if (cache / "mesh" / f"{key}.npz").exists() and not args.overwrite:
            continue
        try:
            stats[key] = process_clip(key, clips[key], args, body, renderer, cache)
        except Exception as e:  # keep going, report at the end
            stats[key] = {"status": f"error: {type(e).__name__}: {e}"}
    bad = {k: v for k, v in stats.items() if v["status"] != "ok"}
    print(f"processed {len(stats)} clips, {len(bad)} failed" + (f", e.g. {list(bad.items())[:3]}" if bad else ""))
    mm = [v["mask_mismatch"] for v in stats.values() if "mask_mismatch" in v]
    if mm:
        print(f"body mask vs normals alpha: mean mismatch {np.mean(mm):.5f}, max {np.max(mm):.5f} (should be ~0)")
    tag = args.shard.replace("/", "of") if args.shard else "all"
    with open(cache / "mesh" / f"_report_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f)
    # fail loudly: a wrong camera silently corrupts every flow; a few failed clips are skipped by the loaders
    if mm and np.mean(mm) > 0.005:
        raise SystemExit(f"re-rendered body masks disagree with normals/*.png (mean {np.mean(mm):.4f}): "
                         "the camera / crop does not match 02_preprocess_hamnosys.py")
    if stats and len(bad) > 0.01 * len(stats):
        raise SystemExit(f"{len(bad)}/{len(stats)} clips failed (see mesh/_report_{tag}.json)")


if __name__ == "__main__":
    main()
