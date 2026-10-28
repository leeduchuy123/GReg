#!/usr/bin/env python
"""Average Keypoint Distance with HRNet-W32 COCO-WholeBody (MMPose), as in the
PGMM repo (``scripts/03_extract_hrnet_pose.py``): full-frame box, no detector.

Input: the ``generated/`` and ``real/`` trees written by ``07_eval_reconstruction.py --save-frames``.

    pip install openmim mmengine==0.10.7 mmcv-lite==2.1.0 mmdet==3.3.0 mmpose==1.1.0 xtcocotools "numpy<2"
    (separate venv: the xtcocotools wheel is built against numpy 1.x and fails to import under numpy 2)
    wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth
    python scripts/09_eval_akd.py --eval-dir outputs/greg/eval_recon --config <mmpose cfg> --checkpoint <pth>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.eval.metrics import akd  # noqa: E402

DEFAULT_CFG = ("td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py")


def extract(model, paths, size):
    from mmpose.apis import inference_topdown
    kps = []
    bbox = np.asarray([[0, 0, size, size]], dtype=np.float32)
    for p in tqdm(paths, leave=False):
        res = inference_topdown(model, str(p), bboxes=bbox, bbox_format="xyxy")
        inst = res[0].pred_instances
        kp = np.concatenate([inst.keypoints[0][:, :2], inst.keypoint_scores[0][:, None]], axis=1)
        kps.append(kp.astype(np.float32))
    return np.stack(kps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--config", required=True, help="mmpose HRNet-W32 COCO-WholeBody config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--real-cache", default=None,
                   help="dir with <key>.npy HRNet keypoints of the REAL frames, shared between runs "
                        "(real frames do not depend on the model); reused only if the frame count matches")
    args = p.parse_args()
    from mmpose.apis import init_model
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.test_cfg["flip_test"] = False
    root = Path(args.eval_dir)
    clips = sorted(d.name for d in (root / "generated").iterdir() if d.is_dir())
    if args.max_clips:
        clips = clips[:args.max_clips]
    gen_all, real_all = [], []
    per_clip = {}
    for key in tqdm(clips):
        gen_paths = sorted((root / "generated" / key).glob("*.png"))
        real_paths = [root / "real" / key / p.name for p in gen_paths]
        g = extract(model, gen_paths, args.image_size)
        cached = Path(args.real_cache) / f"{key}.npy" if args.real_cache else None
        if cached is not None and cached.exists() and np.load(cached).shape[0] == len(real_paths):
            r = np.load(cached)
        else:
            r = extract(model, real_paths, args.image_size)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, r)
        np.save(root / "generated" / key / "_keypoints.npy", g)
        np.save(root / "real" / key / "_keypoints.npy", r)
        per_clip[key] = akd(g, r, args.image_size, args.conf)
        gen_all.append(g)
        real_all.append(r)
    summary = akd(np.concatenate(gen_all), np.concatenate(real_all), args.image_size, args.conf)
    summary["num_clips"] = len(clips)
    with open(root / "akd.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
