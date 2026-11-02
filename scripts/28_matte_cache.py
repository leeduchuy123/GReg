#!/usr/bin/env python
"""D4: soft person mattes for cached frames (torchvision DeepLabV3-ResNet101, COCO/VOC 'person' class).

    python scripts/28_matte_cache.py --cache data/hamnosys_cache --keys-file data/hamnosys_cache/splits/train_fit.txt

Writes ``matte/<key>/<row>.png`` (8-bit soft alpha) = person probability, united with the SMPL-X body mask
(``normals`` alpha, dilated 2 px: segmentation sometimes drops thin fingers) and feathered by a small blur.  The
SMPL-X mask alone cannot be used for compositing (no clothes or hair).  ``--check`` reports how much of the SMPL-X
body the raw segmentation covers (should be ~1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greg.data.dataset import HamNoSysCache, read_split  # noqa: E402

PERSON = 15                       # VOC label index of 'person' in torchvision's segmentation weights
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--keys-file", nargs="+", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--work-size", type=int, default=512, help="segmentation input size (upsampled from the cache)")
    p.add_argument("--check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(dev).eval()
    keys = sorted({k for f in a.keys_file for k in read_split(f)}) if a.keys_file else None
    cache = HamNoSysCache(a.cache, keys)
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    cov = []
    for key in tqdm(cache.keys):
        n = cache.num_rows(key)
        out = Path(a.cache) / "matte" / key
        if out.is_dir() and len(list(out.glob("*.png"))) == n and not a.overwrite:
            continue
        out.mkdir(parents=True, exist_ok=True)
        for s in range(0, n, a.batch):
            rows = list(range(s, min(s + a.batch, n)))
            img = torch.stack([torch.from_numpy(np.asarray(Image.open(cache.frame_path(key, r)).convert("RGB"),
                                                           dtype=np.uint8).copy()) for r in rows])
            img = img.permute(0, 3, 1, 2).float().div(255).to(dev)
            body = torch.stack([torch.from_numpy(np.asarray(Image.open(cache.normal_path(key, r)).convert("RGBA"),
                                                            dtype=np.uint8)[..., 3].copy()) for r in rows]).float().div(255)
            body = body.to(dev)[:, None]
            H = img.shape[-1]
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                x = F.interpolate(img, size=(a.work_size, a.work_size), mode="bilinear", align_corners=False)
                logits = model((x - mean) / std)["out"].float()
            prob = logits.softmax(1)[:, PERSON:PERSON + 1]
            prob = F.interpolate(prob, size=(H, H), mode="bilinear", align_corners=False)
            if a.check:
                bm = body > 0.5
                for i in range(len(rows)):
                    if bm[i].any():
                        cov.append(float((prob[i][bm[i]] > 0.5).float().mean()))
            alpha = torch.maximum(prob, F.max_pool2d(body, 5, 1, 2))
            alpha = F.avg_pool2d(alpha, 3, 1, 1, count_include_pad=False).clamp(0, 1)
            for i, r in enumerate(rows):
                Image.fromarray((alpha[i, 0].cpu().numpy() * 255).round().astype(np.uint8)).save(out / f"{r:06d}.png")
    if cov:
        print(f"person segmentation covers {np.mean(cov):.3f} of the SMPL-X body (5th percentile {np.percentile(cov, 5):.3f})")
        (Path(a.cache) / "matte" / "_check.json").write_text(json.dumps({"mean_cover": float(np.mean(cov))}))


if __name__ == "__main__":
    main()
