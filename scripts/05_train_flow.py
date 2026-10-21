#!/usr/bin/env python
"""Stage 1: pre-train the TPSMM flow estimator on SMPL-X normal maps.

Keypoints and dense motion are computed from (N_s, N_t); the inpainting
network reconstructs the RGB target from the warped RGB source so that the
flow/occlusion are useful for image synthesis.  Losses follow TPSMM:
pyramid perceptual + keypoint equivariance (on normal maps) + warp loss.

    python scripts/05_train_flow.py --config configs/hamnosys_flow.yaml [--override key=value ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.models.tpsmm import FlowEstimator, InpaintingNetwork, resize_occlusion  # noqa: E402
from greg.models.vgg import PerceptualLoss, VGG19Features  # noqa: E402
from greg.train_utils import (SpeedMeter, StepLR, batch_to_device, build_pair_loader, count_params, dropout_schedule,  # noqa: E402
                              equivariance_loss, warp_loss)
from greg.utils.checkpoint import load_checkpoint, load_module_state, save_checkpoint  # noqa: E402
from greg.utils.config import apply_overrides, get_device, load_config, save_config, set_seed  # noqa: E402
from greg.utils.logging import CSVLogger, save_grid  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    return p.parse_args()


def build_models(cfg, device):
    f = cfg["model"]["flow"]
    flow = FlowEstimator(num_tps=int(f.get("num_tps", 10)), num_channels=3, dense_motion=f.get("dense_motion", {}),
                         multi_mask=bool(f.get("multi_mask", False)),
                         pretrained_backbone=bool(f.get("pretrained_backbone", False))).to(device)
    g = cfg["model"].get("inpainting", {})
    inpaint = InpaintingNetwork(3, int(g.get("block_expansion", 64)), int(g.get("max_features", 512)),
                                int(g.get("num_down_blocks", 3)), multi_mask=bool(f.get("multi_mask", False))).to(device)
    return flow, inpaint


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    tr = cfg["train"]

    flow, inpaint = build_models(cfg, device)
    print(f"flow {count_params(flow):.1f}M  inpainting {count_params(inpaint):.1f}M  device {device}")
    params = list(flow.parameters()) + list(inpaint.parameters())
    opt = torch.optim.Adam(params, lr=float(tr["lr"]), betas=(0.5, 0.999), weight_decay=float(tr.get("weight_decay", 1e-4)))
    sched = StepLR(opt, float(tr["lr"]), tr.get("milestones", [70, 90]))
    vgg = VGG19Features(pretrained=bool(tr.get("vgg_pretrained", True))).to(device)
    perceptual = PerceptualLoss(vgg, scales=tr.get("scales", (1, 0.5, 0.25, 0.125)),
                                layer_weights=tr.get("perceptual_weights", (10,) * 5)).to(device)
    lw = tr.get("loss_weights", {})
    w_equi, w_warp = float(lw.get("equivariance", 10)), float(lw.get("warp", 10))
    tparams = tr.get("transform_params", {"sigma_affine": 0.05, "sigma_tps": 0.005, "points_tps": 5})
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tr.get("amp", False)) and device.type == "cuda")

    loader = build_pair_loader(cfg, "train", seed=int(cfg.get("seed", 0)))
    step, start_epoch = 0, 0
    if args.resume:
        ck = load_checkpoint(args.resume)
        load_module_state(flow, ck["flow"])
        load_module_state(inpaint, ck["inpainting"])
        opt.load_state_dict(ck["optimizer"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")
    logger = CSVLogger(out_dir / "logs" / "loss.csv", ["lr", "total", "perceptual", "equivariance", "warp", "sec_per_step", "vram_gb"])
    epochs = int(tr["epochs"])
    t0 = time.time()
    meter = SpeedMeter(device)
    for epoch in range(start_epoch, epochs):
        loader.dataset.set_epoch(epoch)
        lr = sched.set_epoch(epoch)
        drop_flag, drop_p = dropout_schedule(epoch, tr)
        flow.train(); inpaint.train()
        for batch in loader:
            batch = batch_to_device(batch, device)
            Is, It = batch["source_image"], batch["target_image"]
            Ns, Nt = batch["source_normal"], batch["target_normal"]
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                fo = flow(Ns, Nt, dropout_flag=drop_flag, dropout_p=drop_p)
                gen = inpaint(Is, fo["dense_motion"])
                losses = {"perceptual": perceptual(gen["prediction"], It)}
                if w_equi > 0:
                    losses["equivariance"] = w_equi * equivariance_loss(flow.kp_detector, Nt, fo["kp_driving"]["fg_kp"], tparams)
                if w_warp > 0:
                    losses["warp"] = w_warp * warp_loss(gen["warped_encoder_maps"], inpaint.encode(It))
                total = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0, norm_type=float("inf"))
            scaler.step(opt)
            scaler.update()
            step += 1
            meter.step()
            if step % int(tr.get("log_every", 50)) == 0:
                vals = {k: float(v) for k, v in losses.items()}
                vals.update({"total": float(total), "lr": lr, **meter.read()})
                logger.log(step, epoch, vals)
                print(f"ep {epoch} step {step} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()) +
                      f" ({(time.time() - t0) / 60:.1f} min)")
            if step % int(tr.get("sample_every", 500)) == 0:
                # occlusion lives at the dense-motion resolution (scale_factor * H); upsample for the grid
                occ = resize_occlusion(fo["occlusion"], Is.shape[-2:])
                rows = [[Is[i], Ns[i], Nt[i], gen["deformed"][i], occ[i].expand(3, -1, -1),
                         gen["prediction"][i], It[i]] for i in range(min(4, Is.shape[0]))]
                save_grid(rows, out_dir / "samples" / f"step_{step:07d}.png")
            if args.max_steps and step >= args.max_steps:
                break
        state = {"flow": flow.state_dict(), "inpainting": inpaint.state_dict(), "optimizer": opt.state_dict(),
                 "step": step, "epoch": epoch, "config": cfg}
        save_checkpoint(state, out_dir / "checkpoints" / "last.pth")
        if (epoch + 1) in set(tr.get("keep_epochs", [20, 40, 60, 80, 100])) or epoch + 1 == epochs:
            save_checkpoint(state, out_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth")
        if args.max_steps and step >= args.max_steps:
            break
    logger.close()
    print("done")


if __name__ == "__main__":
    main()
