#!/usr/bin/env python
"""D1: train the one-step geometry-gated refiner on GReg exports (docs/diffusion_6_huong_ke_hoach.md Sec. 3).

    python scripts/22_train_refiner.py --config configs/diffusion/d1_refiner.yaml \
        --override data.cache=/workspace/data/hamnosys_cache output_dir=/workspace/outputs/d1/full

Train pairs: GReg epoch_040 outputs on the train_fit clips (``data.export_train``), consecutive rows (t-1, t) for the
temporal loss.  Validation: GReg epoch_100 outputs on val (``data.export_val``), gated blend, L1 / LPIPS / hand LPIPS
against the GReg input itself (the refiner must beat the frame it refines).  Losses are computed at the cache size
(256): the work-size (512) prediction is area-downsampled first, exactly as for the evaluation.
Resume-safe: ``<output_dir>/checkpoints/last.pt`` holds the trainable tensors + optimizer + step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, collate_keep_strings, read_split  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.diffusion.gating import blend, refine_gate, refiner_masks, resize  # noqa: E402,F401  (resize re-used by 23)
from greg.diffusion.refiner_data import RefinerDataset, flatten_frames, read_index  # noqa: E402
from greg.diffusion.refiner_losses import RefinerLoss  # noqa: E402
from greg.diffusion.turbo_refiner import TurboRefiner  # noqa: E402
from greg.eval.evaluate import hand_lpips  # noqa: E402
from greg.eval.metrics import LPIPSMetric  # noqa: E402
from greg.utils.config import apply_overrides, get_device, load_config, save_config, set_seed  # noqa: E402
from greg.utils.logging import CSVLogger, save_grid  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    return p.parse_args()


def build_refiner(cfg: dict, device) -> TurboRefiner:
    m = cfg["model"]
    ref = TurboRefiner.from_pretrained(m.get("pretrained", "stabilityai/sd-turbo"),
                                       cond_normal=bool(m.get("cond_normal", True)), num_masks=int(m.get("num_masks", 4)),
                                       lora_rank_unet=int(m.get("lora_rank_unet", 8)),
                                       lora_rank_vae=int(m.get("lora_rank_vae", 4)))
    if bool(m.get("gradient_checkpointing", True)):
        ref.enable_gradient_checkpointing()
    return ref.to(device)


def refiner_inputs(b: dict, cfg: dict):
    """-> image, normal, masks at the work size."""
    S = int(cfg["model"].get("work_size", 512))
    masks = refiner_masks(b["parts"], b["vc"]) if int(cfg["model"].get("num_masks", 4)) > 0 else None
    return (resize(b["greg"], S), resize(b["normal"], S) if cfg["model"].get("cond_normal", True) else None,
            F.interpolate(masks, size=(S, S), mode="nearest") if masks is not None else None)


def gated(b: dict, refined256: torch.Tensor, cfg: dict) -> torch.Tensor:
    g = cfg.get("gate", {})
    gate = refine_gate(b["parts"], b["vis"], int(g.get("dilate_px", 6)), bool(g.get("include_face", True)),
                       bool(g.get("include_disocc", True)))
    return blend(b["greg"], refined256, gate, g.get("blend", "soft"), int(g.get("feather_px", 4)))


@torch.no_grad()
def validate(ref, loader, cfg, device, lp: LPIPSMetric, amp_dtype, out_png=None) -> dict:
    ref.eval()
    acc = {k: [] for k in ("l1", "lpips", "hand_lpips", "greg_l1", "greg_lpips", "greg_hand_lpips")}
    shown = False
    for batch in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in flatten_frames(batch).items()}
        img, nrm, msk = refiner_inputs(b, cfg)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            r = ref(img, nrm, msk)
        out = gated(b, resize(r.float(), b["greg"].shape[-1]), cfg)
        for name, x in (("", out), ("greg_", b["greg"])):
            acc[name + "l1"] += (x - b["target"]).abs().mean(dim=(1, 2, 3)).tolist()
            acc[name + "lpips"] += lp(x, b["target"]).tolist()
            acc[name + "hand_lpips"] += hand_lpips(lp, x, b["target"], b["centers"], b["joints"], x.shape[-1]).tolist()
        if out_png is not None and not shown:
            rows = [[b["greg"][i], b["normal"][i], b["vc"][i].expand(3, -1, -1), out[i], b["target"][i]]
                    for i in range(min(4, out.shape[0]))]
            save_grid(rows, out_png)
            shown = True
    ref.train()
    import numpy as np
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    d, tr, lc = cfg["data"], cfg["train"], cfg.get("loss", {})
    amp_dtype = torch.bfloat16 if tr.get("amp_dtype", "bf16") == "bf16" else torch.float16

    # ---------------------------------------------------------------- data
    use_temp = float(lc.get("weights", {}).get("temp", 0)) > 0 and bool(d.get("pairs", True))
    train_keys = read_split(Path(d["cache"]) / "splits" / f"{d.get('train_split', 'train_fit')}.txt")
    train_keys = [k for k in train_keys if k in read_index(d["export_train"])]
    cache_tr = HamNoSysCache(d["cache"], train_keys)
    if use_temp:
        cache_tr.keep_keys_with_mesh(what="refiner train")
    ds = RefinerDataset(d["export_train"], cache_tr, pairs=bool(d.get("pairs", True)), with_mesh=use_temp)
    loader = DataLoader(ds, batch_size=int(tr["batch_pairs"]), shuffle=True, drop_last=True,
                        num_workers=int(tr.get("num_workers", 4)), collate_fn=collate_keep_strings,
                        pin_memory=True, persistent_workers=int(tr.get("num_workers", 4)) > 0)
    val_keys = pick_keys([k for k in read_split(Path(d["cache"]) / "splits" / f"{d.get('val_split', 'val')}.txt")
                          if k in read_index(d["export_val"])], d.get("val_clips", 40), seed=0)
    vds = RefinerDataset(d["export_val"], HamNoSysCache(d["cache"], val_keys), pairs=False,
                         max_items_per_clip=int(d.get("val_items_per_clip", 4)))
    vloader = DataLoader(vds, batch_size=8, num_workers=2, collate_fn=collate_keep_strings)
    print(f"train items {len(ds)} ({len(cache_tr)} clips, temporal={use_temp}), val items {len(vds)}")

    mesh = None
    if use_temp:
        from greg.geometry.mesh_flow import MeshCorrespondence
        mesh = MeshCorrespondence(d["smplx_dir"], device=device)

    # ---------------------------------------------------------------- model / loss
    ref = build_refiner(cfg, device)
    params = ref.trainable_parameters()
    print(f"refiner trainable {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=float(tr["lr"]), weight_decay=float(tr.get("weight_decay", 0.01)))
    import lpips as lpips_pkg
    lp_spatial = lpips_pkg.LPIPS(net="vgg", spatial=True, verbose=False).to(device).eval()
    for p in lp_spatial.parameters():
        p.requires_grad_(False)
    crit = RefinerLoss(lp_spatial, lc.get("weights"), float(lc.get("w_hand", 9)), float(lc.get("w_face", 9)),
                       float(lc.get("w_disocc", 1)), int(lc.get("hand_crop", 64)), int(lc.get("hand_up", 128)))
    lp_metric = LPIPSMetric(device)

    step, ck_path = 0, out_dir / "checkpoints" / "last.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ref.load_trainable_state_dict(ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        print(f"resumed at step {step}")
    total = int(args.max_steps or tr["steps"])
    accum = int(tr.get("grad_accum", 1))
    logger = CSVLogger(out_dir / "logs" / "loss.csv", ["lr", "total", "l2", "lpips", "hand", "temp", "gradnorm",
                                                       "sec_per_step", "vram_gb"])
    vlog = CSVLogger(out_dir / "logs" / "val.csv", ["l1", "lpips", "hand_lpips", "greg_l1", "greg_lpips",
                                                   "greg_hand_lpips"])

    def save(tag="last"):
        st = {"trainable": ref.trainable_state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg}
        p = out_dir / "checkpoints" / f"{tag}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st, p.with_suffix(".tmp"))
        p.with_suffix(".tmp").replace(p)

    ref.train()
    t0, it, warm = time.time(), iter(loader), int(tr.get("warmup", 200))
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    tl = time.time()
    while step < total:
        lr = float(tr["lr"]) * min(1.0, (step + 1) / max(warm, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in flatten_frames(batch).items()}
            img, nrm, msk = refiner_inputs(b, cfg)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                r = ref(img, nrm, msk)
            pred = resize(r.float(), b["target"].shape[-1])               # compare at the cache size
            losses = crit(pred, b["target"], b["parts"], b["vis"], b["centers"], b["joints"])
            if use_temp:
                with torch.no_grad():
                    geo = mesh(b["smplx"][0::2], b["focal"][0::2], b["princpt"][0::2], b["faces"][0::2],
                               b["smplx"][1::2], b["focal"][1::2], b["princpt"][1::2], b["faces"][1::2])
                losses["temp"] = RefinerLoss.temporal(pred[0::2], pred[1::2], geo)
                losses["total"] = losses["total"] + crit.w["temp"] * losses["temp"]
            if not torch.isfinite(losses["total"]):
                raise SystemExit(f"non-finite loss at step {step}: {({k: float(v) for k, v in losses.items()})}")
            (losses["total"] / accum).backward()
            for k, v in losses.items():
                logs[k] = logs.get(k, 0.0) + float(v) / accum
        gn = torch.nn.utils.clip_grad_norm_(params, float(tr.get("grad_clip", 1.0)))
        opt.step()
        step += 1
        if step % int(tr.get("log_every", 50)) == 0:
            dt = (time.time() - tl) / int(tr.get("log_every", 50))
            tl = time.time()
            vram = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0
            logs.update(lr=lr, gradnorm=float(gn), sec_per_step=dt, vram_gb=vram)
            logger.log(step, 0, logs)
            print(f"step {step}/{total} " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()) +
                  f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if step % int(tr.get("val_every", 1000)) == 0 or step == total:
            vs = validate(ref, vloader, cfg, device, lp_metric, amp_dtype,
                          out_dir / "samples" / f"val_{step:06d}.jpg")
            vlog.log(step, 0, vs)
            print(f"[val {step}] " + " ".join(f"{k}={v:.4f}" for k, v in vs.items()), flush=True)
        if step % int(tr.get("save_every", 2500)) == 0 or step == total:
            save()
    save()
    save("final")
    logger.close()
    vlog.close()
    (out_dir / "train_done.json").write_text(json.dumps({"steps": step, "minutes": (time.time() - t0) / 60}))
    print("done")


if __name__ == "__main__":
    main()
