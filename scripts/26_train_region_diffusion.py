#!/usr/bin/env python
"""D2 stage A: train the flow-matching Regional Experts (+ their style encoder) on region crops.

    python scripts/26_train_region_diffusion.py --cache data/hamnosys_cache --out outputs/d2/re \
        --regression-checkpoint ckpt/run5_greg_final_checkpoints_epoch_100.pth

Data = the GReg training pairs (train_fit clips, random source/target rows): crops of the target normal map at the
three fixed 64x64 region boxes, style of the source frame, target = gradient map of the real target crop (the RE
target of ``loss.struct_target: gradient``).  Weights are kept as an EMA (0.999); the checkpoint holds the EMA
``rpg`` and ``style_encoder`` states that ``GReg.load_pretrained_rpg`` expects.

Validation (reconstruction protocol: source row 0, ``--val-rows`` targets spread over each val clip) compares on the
same crops: the diffusion RE (fixed noise = what GReg will see; mean of K random samples; per-pixel std of the K
samples = how multi-modal it thinks the region is) against the regression RE of a trained GReg checkpoint (run5).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, PairDataset, SequenceDataset, collate_keep_strings, read_split  # noqa: E402
from greg.data.gradient import gradient_map  # noqa: E402
from greg.data.regions import crop_regions, region_boxes, region_visibility  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.diffusion.region_diffusion import DiffusionRegionalPriorGenerator  # noqa: E402
from greg.models.regional_expert import RegionalPriorGenerator  # noqa: E402
from greg.models.style import StyleEncoder  # noqa: E402
from greg.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from greg.utils.logging import CSVLogger, save_grid  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-split", default="train_fit")
    p.add_argument("--val-split", default="val")
    p.add_argument("--regression-checkpoint", default=None, help="GReg checkpoint whose RE is the baseline")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--region-size", type=int, default=64)
    p.add_argument("--style-dim", type=int, default=256)
    p.add_argument("--style-dropout", type=float, default=0.1)
    p.add_argument("--sample-steps", type=int, default=8)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-clips", type=int, default=60)
    p.add_argument("--val-rows", type=int, default=8)
    p.add_argument("--val-samples", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def region_batch(batch, device, size: int, image_size: int):
    It, Nt, Is = batch["target_image"].to(device), batch["target_normal"].to(device), batch["source_image"].to(device)
    ct = batch["target_centers"].to(device)
    boxes = region_boxes(ct, size, image_size, image_size)
    crops_n = crop_regions(Nt, boxes, size)                              # (B, 3, 3, s, s)
    crops_i = crop_regions(It, boxes, size)
    B, R, C, s, _ = crops_i.shape
    struct = gradient_map(crops_i.reshape(B * R, C, s, s)).view(B, R, 1, s, s)
    return Is, crops_n, struct, region_visibility(ct, image_size, image_size)


@torch.no_grad()
def validate(rpg, style_enc, loader, args, device, reg=None, png=None) -> dict:
    rpg.eval()
    acc = {k: [] for k in ("diff_l1", "diff_mean_l1", "diff_std", "reg_l1")}
    per_region = {k: [[], [], []] for k in ("diff_l1", "reg_l1", "diff_std")}
    first = True
    for batch in loader:
        Is, cn, struct, vis = region_batch(batch, device, args.region_size, int(Is_size(batch)))
        style = style_enc(Is)
        tgt = struct.clamp(0, 1)
        fixed = rpg.sample(cn, style)                                     # what GReg sees
        rpg.noise_mode = "random"
        samples = torch.stack([rpg.sample(cn, style) for _ in range(args.val_samples)])   # (K, B, 3, 1, s, s)
        rpg.noise_mode = "fixed"
        l1 = (fixed - tgt).abs().mean(dim=(2, 3, 4))                      # (B, 3)
        l1m = (samples.mean(0) - tgt).abs().mean(dim=(2, 3, 4))
        std = samples.std(0).mean(dim=(2, 3, 4))
        m = vis > 0
        acc["diff_l1"] += l1[m].tolist()
        acc["diff_mean_l1"] += l1m[m].tolist()
        acc["diff_std"] += std[m].tolist()
        for r in range(3):
            per_region["diff_l1"][r] += l1[:, r][m[:, r]].tolist()
            per_region["diff_std"][r] += std[:, r][m[:, r]].tolist()
        rp = None
        if reg is not None:
            rg, rs = reg
            rp = rg.forward_patches(cn, rs(Is))
            rl = (rp - tgt).abs().mean(dim=(2, 3, 4))
            acc["reg_l1"] += rl[m].tolist()
            for r in range(3):
                per_region["reg_l1"][r] += rl[:, r][m[:, r]].tolist()
        if png is not None and first:
            rows = []
            for i in range(min(4, cn.shape[0])):
                for r in (1, 2):
                    row = [cn[i, r], tgt[i, r], fixed[i, r]] + [samples[k, i, r] for k in range(min(3, samples.shape[0]))]
                    if rp is not None:
                        row.append(rp[i, r])
                    rows.append(row)
            save_grid(rows, png)
            first = False
    rpg.train()
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    for k, lists in per_region.items():
        for r, name in enumerate(("face", "lhand", "rhand")):
            out[f"{k}_{name}"] = float(np.mean(lists[r])) if lists[r] else float("nan")
    return out


def Is_size(batch) -> int:
    return batch["target_image"].shape[-1]


class RegressionRE:
    """The RE + style encoder of a trained GReg checkpoint, applied to given crops."""

    def __init__(self, path, device, style_dim, region_size):
        ck = load_checkpoint(path)
        m = ck["config"]["model"]
        st = ck["model"]
        self.rpg = RegionalPriorGenerator(style_dim, region_size, 1, tuple(m.get("rpg", {}).get("channels", (64, 128, 256, 256))),
                                          bool(m.get("rpg", {}).get("share_hand_expert", True))).to(device).eval()
        self.rpg.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("rpg.")})
        self.style = StyleEncoder(3, style_dim, base=int(m.get("style_base", 32))).to(device).eval()
        self.style.load_state_dict({k[len("style_encoder."):]: v for k, v in st.items() if k.startswith("style_encoder.")})

        def fwd(cn, style):
            exs = [self.rpg.face_expert, self.rpg.hand_expert, self.rpg.rhand_expert or self.rpg.hand_expert]
            return torch.stack([ex(cn[:, r], style) for r, ex in enumerate(exs)], 1)
        self.rpg.forward_patches = fwd


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    tr_keys = read_split(Path(args.cache) / "splits" / f"{args.train_split}.txt")
    ds = PairDataset(HamNoSysCache(args.cache, tr_keys), pairs_per_clip=8, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        collate_fn=collate_keep_strings, pin_memory=True, persistent_workers=args.num_workers > 0)
    vkeys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.val_split}.txt"), args.val_clips, seed=0)
    vds = SequenceDataset(HamNoSysCache(args.cache, vkeys), source_row=0, max_rows=args.val_rows, uniform_rows=True)
    vloader = DataLoader(vds, batch_size=32, num_workers=4, collate_fn=collate_keep_strings)
    print(f"train pairs/epoch {len(ds)}, val items {len(vds)}")

    rpg = DiffusionRegionalPriorGenerator(args.style_dim, args.region_size, steps=args.sample_steps,
                                          style_dropout=args.style_dropout).to(device)
    style_enc = StyleEncoder(3, args.style_dim).to(device)
    params = list(rpg.parameters()) + list(style_enc.parameters())
    print(f"diffusion RE {sum(p.numel() for p in rpg.parameters()) / 1e6:.2f}M + style "
          f"{sum(p.numel() for p in style_enc.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    ema_rpg, ema_style = copy.deepcopy(rpg).eval(), copy.deepcopy(style_enc).eval()
    for p in list(ema_rpg.parameters()) + list(ema_style.parameters()):
        p.requires_grad_(False)
    reg = None
    if args.regression_checkpoint:
        r = RegressionRE(args.regression_checkpoint, device, args.style_dim, args.region_size)
        reg = (r.rpg, r.style)

    step, ck_path = 0, out / "last.pth"
    if ck_path.exists():
        ck = load_checkpoint(ck_path)
        rpg.load_state_dict(ck["rpg_raw"]); style_enc.load_state_dict(ck["style_raw"])
        ema_rpg.load_state_dict(ck["rpg"]); ema_style.load_state_dict(ck["style_encoder"])
        opt.load_state_dict(ck["optimizer"]); step = int(ck["step"])
        print(f"resumed at step {step}")
    log = CSVLogger(out / "loss.csv", ["fm", "fm_face", "fm_hands", "lr", "sec_per_step"])
    vlog = CSVLogger(out / "val.csv", ["diff_l1", "diff_mean_l1", "diff_std", "reg_l1"])

    def save():
        save_checkpoint({"rpg": ema_rpg.state_dict(), "style_encoder": ema_style.state_dict(),
                         "rpg_raw": rpg.state_dict(), "style_raw": style_enc.state_dict(),
                         "optimizer": opt.state_dict(), "step": step, "args": vars(args)}, ck_path)

    t0, tl, it = time.time(), time.time(), iter(loader)
    rpg.train()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            ds.set_epoch(ds.epoch + 1)
            it = iter(loader)
            batch = next(it)
        lr = args.lr * min(1.0, (step + 1) / 500)
        for g in opt.param_groups:
            g["lr"] = lr
        Is, cn, struct, vis = region_batch(batch, device, args.region_size, Is_size(batch))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = rpg.training_loss(cn, style_enc(Is), struct, vis)
        if not torch.isfinite(losses["fm"]):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        losses["fm"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            for e, s in ((ema_rpg, rpg), (ema_style, style_enc)):
                for pe, ps in zip(e.parameters(), s.parameters()):
                    pe.lerp_(ps, 1 - args.ema)
        if step % 100 == 0:
            dt = (time.time() - tl) / 100
            tl = time.time()
            log.log(step, 0, {**{k: float(v) for k, v in losses.items()}, "lr": lr, "sec_per_step": dt})
            print(f"step {step} fm={float(losses['fm']):.4f} face={float(losses['fm_face']):.4f} "
                  f"hands={float(losses['fm_hands']):.4f} {dt:.3f}s/step ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vs = validate(ema_rpg, ema_style, vloader, args, device, reg, out / f"val_{step:06d}.jpg")
            vlog.log(step, 0, vs)
            print("[val] " + json.dumps({k: round(v, 4) for k, v in vs.items()}), flush=True)
            (out / "val_last.json").write_text(json.dumps({"step": step, **vs}, indent=2))
            save()
    save()
    print("done")


if __name__ == "__main__":
    main()
