#!/usr/bin/env python
"""D6: encode 49-frame windows (source frame + 48 consecutive targets) into everything the Wan DiT trains on.

    python scripts/32_cache_wan_windows.py text --out outputs/d6/cache
    python scripts/32_cache_wan_windows.py windows --cache CACHE --split train_fit --windows-per-clip 1 \
        --smplx-dir SMPLX --out outputs/d6/cache/train
    python scripts/32_cache_wan_windows.py windows --cache CACHE --split test --max-clips 200 --all-windows \
        --smplx-dir SMPLX --out outputs/d6/cache/test

A window = [source row (the clip's first valid row, the reconstruction protocol's source)] + 48 consecutive target
rows (repeat-padded at the clip end; padded frames get ``valid = 0``).  Per window (bf16, ``<key>_<start>.pt``):

    lat_mean, lat_std   (16, 13, h, w)  normalised VAE posterior of the real video (sampled at every training step)
    lat_masked          (16, 13, h, w)  VAE(I_st * M), I_st = source warped by the mesh correspondence, M = 1[V*C>=0.5]
    lat_i2v             (16, 13, h, w)  VAE([source, 0, 0, ...])  (i2v mode of the keypoint / normal variants)
    lat_normal, lat_ccm, lat_skel        VAE of the target normal map / canonical coordinate map / 2D skeleton videos
    mask, mask_i2v      (4, 13, h, w)   packed "known" masks; region (1, 13, h, w) hands + face; valid (1, 13, h, w)
    clip                (257, 1280)     CLIP vision hidden_states[-2] of the source frame
    rows                the cache rows of the 49 frames (frame 0 = source)

The video is upsampled from the 256^2 cache to ``--work`` (default 320, multiple of 16).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, read_split  # noqa: E402
from greg.data.regions import part_mask  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.diffusion.frame_ldm import draw_skeleton  # noqa: E402
from greg.diffusion.wan_video import TEXT_LEN, WanLatentCodec, pack_frames  # noqa: E402
from greg.geometry.canonical import canonical_coordinate_map, template_coordinates  # noqa: E402
from greg.models.tpsmm import warp  # noqa: E402

PROMPT = "a person signing in sign language in front of a static camera, sharp hands and fingers, studio video"
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CLIP_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
FRAMES = 49


def cache_text(a):
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(VAE_REPO, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(VAE_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
    t = tok([PROMPT], padding="max_length", max_length=TEXT_LEN, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids.to("cuda"), t.attention_mask.to("cuda")).last_hidden_state[0, :n]
    out = torch.zeros(TEXT_LEN, e.shape[-1], dtype=torch.bfloat16)
    out[:n] = e.to(torch.bfloat16).cpu()                              # truncated to its length, zero-padded to 512
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({"text": out, "prompt": PROMPT, "tokens": n}, Path(a.out) / "text.pt")
    print(f"text embedding {tuple(out.shape)}, {n} tokens -> {a.out}/text.pt")


def windows_of(valid, frames: int, all_windows: bool, per_clip: int, rng: random.Random):
    s0, targets = valid[0], valid[1:]
    n = frames - 1
    if len(targets) == 0:
        return []
    if all_windows:
        starts = list(range(0, len(targets), n))
    else:
        cand = list(range(0, max(len(targets) - n, 0) + 1))
        starts = sorted(rng.sample(cand, min(per_clip, len(cand))))
    out = []
    for st in starts:
        chunk = targets[st:st + n]
        valid_flags = [1] * len(chunk) + [0] * (n - len(chunk))
        chunk = chunk + [chunk[-1]] * (n - len(chunk))
        out.append((st, [s0] + chunk, [1] + valid_flags))
    return out


def up(x, size, mode="bicubic"):
    if mode == "nearest":
        return F.interpolate(x, size=(size, size), mode="nearest")
    return F.interpolate(x, size=(size, size), mode=mode, align_corners=False).clamp(0, 1)


def cache_windows(a):
    from diffusers import AutoencoderKLWan
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from greg.geometry.mesh_flow import MeshCorrespondence
    dev = torch.device("cuda")
    codec = WanLatentCodec(AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=torch.bfloat16).to(dev))
    clip_proc = CLIPImageProcessor.from_pretrained(CLIP_REPO, subfolder="image_processor")
    clip = CLIPVisionModel.from_pretrained(CLIP_REPO, subfolder="image_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    mesh = MeshCorrespondence(a.smplx_dir, device=dev, edge_px=4)
    coords, faces = template_coordinates(mesh.body).to(dev), mesh.body.faces.to(dev)
    keys = [l.strip() for l in open(a.keys_file, encoding="utf-8") if l.strip()] if a.keys_file else \
        pick_keys(read_split(Path(a.cache) / "splits" / f"{a.split}.txt"), a.max_clips, a.subset_seed)
    cache = HamNoSysCache(a.cache, keys)
    cache.keep_keys_with_mesh(what="wan cache")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keys.txt").write_text("\n".join(cache.keys))
    S = a.work
    index = {}
    for key in tqdm(cache.keys):
        rng = random.Random(f"{a.seed}/{key}")
        wins = windows_of(cache.valid_rows(key), FRAMES, a.all_windows, a.windows_per_clip, rng)
        for st, rows, flags in wins:
            index.setdefault(key, []).append({"start": st, "rows": rows, "valid": flags})
            path = out / f"{key}_{st:05d}.pt"
            if path.exists():
                continue
            fr = [cache.load_row(key, r, with_parts=True, with_mesh=True) for r in rows]
            g = lambda k: torch.stack([f[k] for f in fr]).to(dev)  # noqa: E731
            img, nrm, parts, joints = g("image"), g("normal"), g("parts").long(), g("joints_2d")
            T = len(rows)
            with torch.no_grad():
                geo = mesh(g("smplx")[:1].expand(T, -1), g("focal")[:1].expand(T, -1), g("princpt")[:1].expand(T, -1),
                           g("faces")[:1].expand(T, -1, -1), g("smplx"), g("focal"), g("princpt"), g("faces"),
                           valid_rect_s=cache.valid_rect(key)[None].expand(T, -1).to(dev))
                warped = warp(img[:1].expand(T, -1, -1, -1), geo["grid"])
                M = (geo["occlusion"] >= 0.5).float()
                M[0], warped[0] = 1.0, img[0]                                 # frame 0 = the real source
                ccm = canonical_coordinate_map(g("faces").long(), geo["bary_target"], faces, coords)
                skel = torch.stack([draw_skeleton(j, 256) for j in joints.cpu()]).to(dev)
                region = part_mask(parts, (2, 3, 4), 3, soften=False)
                vid = lambda x: (up(x, S) * 2 - 1).permute(1, 0, 2, 3)[None]  # noqa: E731  (1, 3, T, S, S)
                Mu = up(M, S, "nearest")
                lat_mean, lat_std = codec.encode(vid(img), moments=True)
                masked = ((up(warped, S) * 2 - 1) * Mu).permute(1, 0, 2, 3)[None]
                i2v = torch.zeros(1, 3, T, S, S, device=dev)
                i2v[:, :, 0] = (up(img[:1], S) * 2 - 1)[0]
                rec = {"lat_mean": lat_mean, "lat_std": lat_std, "lat_masked": codec.encode(masked),
                       "lat_i2v": codec.encode(i2v), "lat_normal": codec.encode(vid(nrm)),
                       "lat_ccm": codec.encode(vid(ccm)), "lat_skel": codec.encode(vid(skel))}
                m_i2v = torch.zeros(1, 1, T, S, S, device=dev)
                m_i2v[:, :, 0] = 1
                rec["mask"] = pack_frames(Mu.permute(1, 0, 2, 3)[None])
                rec["mask_i2v"] = pack_frames(m_i2v)
                rec["region"] = pack_frames(up(region, S, "nearest").permute(1, 0, 2, 3)[None]).amax(1, keepdim=True)
                vf = torch.tensor(flags, device=dev, dtype=torch.float32).view(1, 1, T, 1, 1).expand(1, 1, T, S // 8, S // 8)
                rec["valid"] = (pack_frames(vf, pool=1).amax(1, keepdim=True) > 0).float()
                ci = clip_proc(images=[(img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)], return_tensors="pt")
                rec["clip"] = clip(ci.pixel_values.to(dev, torch.bfloat16), output_hidden_states=True).hidden_states[-2][0]
            rec = {k: v.squeeze(0).to(torch.bfloat16).cpu() if v.dim() > 2 else v.to(torch.bfloat16).cpu() for k, v in rec.items()}
            rec.update(key=key, start=st, rows=rows, valid_rows=flags, work=S)
            torch.save(rec, path)
    (out / "index.json").write_text(json.dumps({"args": vars(a), "windows": index}))
    print(f"{sum(len(v) for v in index.values())} windows of {len(index)} clips -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["text", "windows"])
    p.add_argument("--out", required=True)
    p.add_argument("--cache")
    p.add_argument("--split", default="train_fit")
    p.add_argument("--keys-file", default=None)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--windows-per-clip", type=int, default=1)
    p.add_argument("--all-windows", action="store_true")
    p.add_argument("--work", type=int, default=320)
    p.add_argument("--smplx-dir")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.work % 16:
        raise SystemExit("--work must be a multiple of 16")
    cache_text(a) if a.what == "text" else cache_windows(a)


if __name__ == "__main__":
    main()
