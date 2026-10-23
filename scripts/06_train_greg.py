#!/usr/bin/env python
"""Stage 2: train GReg with the flow estimator frozen.

    python scripts/06_train_greg.py --config configs/hamnosys_greg.yaml \
        --flow-checkpoint outputs/flow/checkpoints/epoch_100.pth

Ablations (Table 2) are config switches: ``model.ablation.no_rpg``,
``model.ablation.no_gr``, ``loss.weights.adv: 0`` (w/o adv),
``loss.struct_target: grayscale|rgb``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.splits import pick_keys  # noqa: E402
from greg.data.dataset import read_split  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.visualize import fixed_panels  # noqa: E402
from greg.losses import GRegLoss  # noqa: E402
from greg.models.discriminator import RegionalDiscriminators  # noqa: E402
from greg.models.greg import GReg  # noqa: E402
from greg.models.vgg import PerceptualLoss, VGG19Features  # noqa: E402
from greg.train_utils import (NonFiniteGuard, SpeedMeter, StepLR, batch_to_device, build_pair_loader,  # noqa: E402
                              count_params, region_weights)
from greg.utils.checkpoint import load_checkpoint, load_module_state, save_checkpoint  # noqa: E402
from greg.utils.config import apply_overrides, get_device, load_config, save_config, set_seed  # noqa: E402
from greg.utils.logging import CSVLogger, save_grid  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--flow-checkpoint", default=None, help="stage-1 checkpoint (required unless resuming)")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def apply_ablation(model: GReg, cfg: dict):
    ab = cfg["model"].get("ablation", {}) or {}
    model.ablation_no_rpg = bool(ab.get("no_rpg", False))
    model.ablation_no_gr = bool(ab.get("no_gr", False))
    if model.ablation_no_gr:
        for g in model.generator.grms.values():
            for p in g.parameters():
                p.requires_grad_(False)
        model.generator.grms.clear()
    return model


def _nonfinite(t) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point() and not bool(torch.isfinite(t).all())


def nonfinite_report(kind: str, step: int, epoch: int, batch: dict, out: dict, losses: dict, model, disc) -> str:
    """Everything needed to find where a NaN/inf came from, printed once before training stops."""
    lines = [f"NON-FINITE {kind} at epoch {epoch} step {step} (the update was NOT applied)"]
    lines.append("  losses: " + " ".join(f"{k}={float(v):.4g}" for k, v in losses.items()))
    for i, key in enumerate(batch["key"]):
        per = {k: bool(torch.isfinite(batch[k][i]).all()) for k in
               ("source_image", "source_normal", "target_image", "target_normal", "source_centers", "target_centers",
                "target_joints") if k in batch}
        outs = {k: bool(torch.isfinite(v[i]).all()) for k, v in out.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.shape[0] == len(batch["key"])}
        bad_in = [k for k, ok in per.items() if not ok]
        bad_out = [k for k, ok in outs.items() if not ok]
        lines.append(f"  sample {i}: clip {key} rows {int(batch['source_row'][i])}->{int(batch['target_row'][i])} "
                     f"target_centers={batch['target_centers'][i].tolist()} non-finite inputs={bad_in} outputs={bad_out}")
    for name, mod in (("model", model), ("discriminator", disc)):
        if mod is None:
            continue
        grads = [n for n, p in mod.named_parameters() if p.grad is not None and _nonfinite(p.grad)]
        state = [n for n, t in mod.state_dict().items() if _nonfinite(t)]
        lines.append(f"  {name}: {len(grads)} params with non-finite grads {grads[:8]}; "
                     f"{len(state)} non-finite params/buffers {state[:8]}")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    tr, lc = cfg["train"], cfg.get("loss", {})

    model = GReg(cfg["model"]).to(device)
    apply_ablation(model, cfg)
    if args.flow_checkpoint:
        ck = load_checkpoint(args.flow_checkpoint)
        load_module_state(model.flow, ck["flow"])
        print(f"loaded flow estimator from {args.flow_checkpoint} (epoch {ck.get('epoch')})")
    elif not args.resume:
        raise SystemExit("--flow-checkpoint is required (or --resume)")
    if bool(cfg["model"].get("freeze_flow", True)):
        model.freeze_flow()
    use_adv = float(lc.get("weights", {}).get("adv", 1.0)) > 0
    disc = RegionalDiscriminators(3, 4, int(cfg["model"].get("disc_base", 64)),
                                  int(cfg["model"].get("disc_layers", 3))).to(device) if use_adv else None
    print(f"GReg trainable {sum(p.numel() for p in model.trainable_parameters()) / 1e6:.1f}M  "
          f"flow {count_params(model.flow):.1f}M  D {count_params(disc) if disc else 0:.1f}M  device {device}")

    g_params = [p for p in model.trainable_parameters() if p.requires_grad]
    opt_g = torch.optim.Adam(g_params, lr=float(tr["lr"]), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(tr.get("lr_d", tr["lr"])), betas=(0.5, 0.999)) if disc else None
    sched_g = StepLR(opt_g, float(tr["lr"]), tr.get("milestones", [70, 90]))
    sched_d = StepLR(opt_d, float(tr.get("lr_d", tr["lr"])), tr.get("milestones", [70, 90])) if opt_d else None

    vgg = VGG19Features(pretrained=bool(tr.get("vgg_pretrained", True))).to(device)
    perceptual = PerceptualLoss(vgg, scales=tr.get("scales", (1, 0.5, 0.25, 0.125)),
                                layer_weights=tr.get("perceptual_weights", (10,) * 5)).to(device)
    criterion = GRegLoss(perceptual, lc.get("weights", {}), gan_type=lc.get("gan_type", "bce"),
                         region_size=model.region_size, region_perceptual_size=int(lc.get("region_perceptual_size", 128)),
                         struct_target=lc.get("struct_target", "gradient"))
    if getattr(model, "ablation_no_rpg", False):
        criterion.w["struct"] = 0.0   # no prior -> no prior structure loss
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tr.get("amp", False)) and device.type == "cuda")

    loader = build_pair_loader(cfg, "train", seed=int(cfg.get("seed", 0)))
    step, start_epoch = 0, 0
    if args.resume:
        ck = load_checkpoint(args.resume)
        load_module_state(model, ck["model"])
        if disc and ck.get("discriminator"):
            load_module_state(disc, ck["discriminator"])
        opt_g.load_state_dict(ck["optimizer_g"])
        if opt_d and ck.get("optimizer_d"):
            opt_d.load_state_dict(ck["optimizer_d"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")
    logger = CSVLogger(out_dir / "logs" / "loss.csv",
                       ["lr", "total", "perceptual", "regional_perceptual", "struct", "adv", "d_total", "g_gradnorm",
                        "d_gradnorm", "amp_scale", "skipped", "sec_per_step", "vram_gb"])
    guard = NonFiniteGuard([model, disc])

    def drop(kind, report_losses, report_model, report_disc):
        """Discard this iteration's non-finite update (reads step/epoch/batch/out of the training loop)."""
        # report first (it lists the parameters holding non-finite gradients), then discard them
        msg = nonfinite_report(kind, step, epoch, batch, out, report_losses, report_model, report_disc) \
            if guard.total < 3 else f"NON-FINITE {kind} at epoch {epoch} step {step}: update skipped"
        opt_g.zero_grad(set_to_none=True)
        if opt_d:
            opt_d.zero_grad(set_to_none=True)
        diverging = guard.skip(step)
        print(f"{msg}\n  skipped updates so far: {guard.total}", flush=True)
        if diverging:
            raise SystemExit(f"more than {guard.max_events} non-finite updates within {guard.window} steps "
                             f"(epoch {epoch} step {step}): the run is diverging, stopping")
    epochs = int(tr["epochs"])
    t0 = time.time()
    meter = SpeedMeter(device)

    # periodic validation (train.val_every epochs; 0 = off).  Uses splits/<val.split>.txt, a held-out part of
    # the TRAIN clips (scripts/04_make_splits.py); with data.train_split=train these clips are also trained on.
    va = tr.get("val", {}) or {}
    val_every = int(tr.get("val_every", 0))
    val_keys = None
    if val_every > 0:
        vpath = Path(cfg["data"]["cache"]) / "splits" / f"{va.get('split', 'val')}.txt"
        if vpath.exists():
            val_keys = pick_keys(read_split(vpath), va.get("max_clips", 60), seed=0)
        else:
            print(f"WARNING: {vpath} not found -> validation disabled (run scripts/04_make_splits.py)")
    val_logger = CSVLogger(out_dir / "logs" / "val.csv",
                           ["l1", "psnr", "ssim", "region_l1", "warped_l1", "source_copy_l1"]) if val_keys else None
    for epoch in range(start_epoch, epochs):
        loader.dataset.set_epoch(epoch)
        lr = sched_g.set_epoch(epoch)
        if sched_d:
            sched_d.set_epoch(epoch)
        model.train()
        for batch in loader:
            batch = batch_to_device(batch, device)
            Is, It = batch["source_image"], batch["target_image"]
            Ns, Nt = batch["source_normal"], batch["target_normal"]
            rw = region_weights(batch, model.image_size)
            guard.snapshot()
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                out = model(Is, Ns, Nt, batch["target_centers"])
                losses = criterion.generator_loss(out, It, disc, rw)
            # Without AMP nothing skips a non-finite update: one NaN gradient goes straight into Adam and
            # every weight and moment turns NaN (that is what killed the sigma16/struct1 sweep runs).  The
            # guard drops such updates the way GradScaler does under AMP; with AMP, GradScaler handles
            # overflowing gradients itself, but its unscale multiplies by inf unflagged once the scale
            # collapses below ~3e-39, so stop on scale < 1.
            if not torch.isfinite(losses["total"]):
                drop("loss", losses, model, disc)
                continue
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt_g)
            g_norm = torch.nn.utils.clip_grad_norm_(g_params, max_norm=10.0, norm_type=float("inf"))
            if not scaler.is_enabled() and not torch.isfinite(g_norm):
                drop("generator gradient", losses, model, None)
                continue
            scaler.step(opt_g)
            if disc is not None:
                with torch.autocast(device.type, enabled=scaler.is_enabled()):
                    d_losses = criterion.discriminator_loss(out, It, disc)
                d_ok = bool(torch.isfinite(d_losses["d_total"]))
                if d_ok:
                    opt_d.zero_grad(set_to_none=True)
                    scaler.scale(d_losses["d_total"]).backward()
                    scaler.unscale_(opt_d)
                    d_norm = torch.stack([p.grad.detach().abs().max() for p in disc.parameters()
                                          if p.grad is not None]).max()
                    d_ok = scaler.is_enabled() or bool(torch.isfinite(d_norm))
                    losses["d_gradnorm"] = d_norm
                if d_ok:
                    scaler.step(opt_d)
                    losses.update(d_losses)
                else:   # the generator step of this iteration is kept, only the discriminator update is dropped
                    drop("discriminator loss/gradient", {**losses, **d_losses}, None, disc)
                    losses.pop("d_gradnorm", None)
            losses["g_gradnorm"] = g_norm
            losses["skipped"] = torch.tensor(float(guard.total))
            scaler.update()
            if scaler.is_enabled():
                losses["amp_scale"] = torch.tensor(scaler.get_scale())
                if scaler.get_scale() < 1.0:
                    raise SystemExit(f"GradScaler scale collapsed to {scaler.get_scale():.3g} at epoch {epoch} "
                                     f"step {step}: gradients keep overflowing; stopping before weights go NaN")
            step += 1
            meter.step()
            if step % int(tr.get("log_every", 50)) == 0:
                vals = {k: float(v) for k, v in losses.items()}
                vals["lr"] = lr
                vals.update(meter.read())
                logger.log(step, epoch, vals)
                print(f"ep {epoch} step {step} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()) +
                      f" ({(time.time() - t0) / 60:.1f} min)")
            if step % int(tr.get("sample_every", 500)) == 0:
                rows = []
                for i in range(min(4, Is.shape[0])):
                    rows.append([Is[i], Nt[i], out["warped"][i], out["occlusion"][i].expand(3, -1, -1),
                                 out["prior_map"][i], out["region_gradients"][i], out["generated"][i],
                                 out["output"][i], It[i]])
                save_grid(rows, out_dir / "samples" / f"step_{step:07d}.png")
            if args.max_steps and step >= args.max_steps:
                break
        state = {"model": model.state_dict(), "discriminator": disc.state_dict() if disc else None,
                 "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict() if opt_d else None,
                 "step": step, "epoch": epoch, "config": cfg}
        bad = [f"{m}.{k}" for m in ("model", "discriminator") if state[m] for k, v in state[m].items() if _nonfinite(v)]
        if bad:   # never overwrite a good last.pth with a corrupted state
            raise SystemExit(f"non-finite weights after epoch {epoch} ({len(bad)} tensors, e.g. {bad[:3]}); not saved")
        save_checkpoint(state, out_dir / "checkpoints" / "last.pth")
        if val_keys and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            _, vs = evaluate_reconstruction(model, cfg["data"]["cache"], val_keys, device,
                                            batch_size=int(va.get("batch_size", 16)), num_workers=int(tr.get("num_workers", 2)),
                                            max_rows=va.get("max_rows", 8), uniform_rows=True,
                                            eval_region_size=int(va.get("eval_region_size", 64)))
            val_logger.log(step, epoch, {k[5:]: v for k, v in vs.items() if k.startswith("mean_")})
            print(f"[val ep {epoch + 1}] " + " ".join(f"{k[5:]}={v:.4f}" for k, v in vs.items() if k.startswith("mean_")))
            # the same clips at every validation epoch -> compare the images across epochs
            fixed_panels(model, cfg["data"]["cache"], val_keys[:int(va.get("fixed_clips", 3))], device,
                         out_dir / "samples" / f"fixed_ep{epoch + 1:03d}.jpg")
            meter.reset()
        if (epoch + 1) in set(tr.get("keep_epochs", [20, 40, 60, 80, 100])) or epoch + 1 == epochs:
            save_checkpoint(state, out_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pth")
        if args.max_steps and step >= args.max_steps:
            break
    logger.close()
    if val_logger:
        val_logger.close()
    print("done")


if __name__ == "__main__":
    main()
