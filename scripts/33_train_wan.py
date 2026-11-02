#!/usr/bin/env python
"""D6: LoRA + widened patch embedding training of Wan2.1-Fun-1.3B-InP on cached windows (32_cache_wan_windows.py).

    python scripts/33_train_wan.py --config configs/diffusion/d6_wan.yaml --override model.variant=ng \
        data.cache_dir=outputs/d6/cache output_dir=outputs/d6/ng

Variants (greg.diffusion.wan_video.VARIANT_EXTRA): ``ng`` mesh warp + V*C mask + normal + CCM; ``ng_ones`` the same
with an all-ones mask (DWM's mode); ``n`` / ``k`` i2v mode (source frame only) + normal / 2D-skeleton video.
Resume-safe (``checkpoints/last.pt``: trainable tensors + optimizer + step).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.diffusion.wan_video import (WanConditionModel, flow_schedule, load_fun_inp_transformer,  # noqa: E402
                                      training_loss)
from greg.utils.config import apply_overrides, load_config, save_config, set_seed  # noqa: E402
from greg.utils.logging import CSVLogger  # noqa: E402

FIELDS = ("lat_mean", "lat_std", "lat_masked", "lat_i2v", "lat_normal", "lat_ccm", "lat_skel", "mask", "mask_i2v",
          "region", "valid", "clip")


class WindowSet(Dataset):
    def __init__(self, root: str | Path):
        self.files = sorted(Path(root).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no cached windows in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        r = torch.load(self.files[i], map_location="cpu", weights_only=False)
        out = {k: r[k] for k in FIELDS}
        out["name"] = self.files[i].stem
        return out


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in FIELDS}
    out["name"] = [x["name"] for x in b]
    return out


def build(cfg, device) -> WanConditionModel:
    m = cfg["model"]
    tr = load_fun_inp_transformer(m["checkpoint_dir"], torch.bfloat16)
    model = WanConditionModel(tr, m["variant"], int(m.get("lora_rank", 32)))
    if bool(m.get("gradient_checkpointing", True)):
        tr.enable_gradient_checkpointing()
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.override)
    set_seed(int(cfg.get("seed", 0)))
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    d, tr = cfg["data"], cfg["train"]
    ds = WindowSet(Path(d["cache_dir"]) / d.get("train_subdir", "train"))
    loader = DataLoader(ds, batch_size=int(tr["batch_size"]), shuffle=True, drop_last=True, collate_fn=collate,
                        num_workers=int(tr.get("num_workers", 4)), persistent_workers=True, pin_memory=True)
    text = torch.load(Path(d["cache_dir"]) / "text.pt", weights_only=False)["text"].to(dev)
    model = build(cfg, dev)
    params = model.trainable_parameters()
    print(f"D6 {cfg['model']['variant']}: {len(ds)} windows, trainable {sum(p.numel() for p in params) / 1e6:.1f}M, "
          f"patch_embedding in={model.transformer.config.in_channels}")
    opt = torch.optim.AdamW(params, lr=float(tr["lr"]), betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    total, warm = int(a.max_steps or tr["steps"]), int(tr.get("warmup", 100))
    lr_at = lambda s: float(tr["lr"]) * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))  # noqa: E731
    step, ck = 0, out / "checkpoints" / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(st["trainable"])
        opt.load_state_dict(st["optimizer"])
        step = int(st["step"])
        print(f"resumed at step {step}")
    sched = flow_schedule(float(cfg["model"].get("shift", 5.0)))
    log = CSVLogger(out / "logs" / "loss.csv", ["loss", "lr", "gradnorm", "sec_per_step", "vram_gb"])

    def save(tag="last"):
        pth = out / "checkpoints" / f"{tag}.pt"
        pth.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable": model.trainable_state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "config": cfg}, pth.with_suffix(".tmp"))
        pth.with_suffix(".tmp").replace(pth)

    model.train()
    it, t0, tl, acc = iter(loader), time.time(), time.time(), 0.0
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        w = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k != "name"}
        w = {k: (v.float() if k in ("lat_mean", "lat_std") else v) for k, v in w.items()}
        B = w["lat_mean"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = training_loss(model, sched, w, text[None].expand(B, -1, -1), w["clip"],
                                 float(cfg.get("loss", {}).get("w_region", 9.0)))
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step} ({b['name']})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        acc += float(loss)
        if step % int(tr.get("log_every", 20)) == 0:
            n = int(tr.get("log_every", 20))
            dt = (time.time() - tl) / n
            tl = time.time()
            vram = torch.cuda.max_memory_allocated() / 2 ** 30
            log.log(step, 0, {"loss": acc / n, "lr": lr_at(step), "gradnorm": float(gn), "sec_per_step": dt,
                              "vram_gb": vram})
            print(f"step {step}/{total} loss={acc / n:.4f} {dt:.2f}s/step vram={vram:.1f}GB "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            acc = 0.0
        if step % int(tr.get("save_every", 500)) == 0:
            save()
    save()
    save("final")
    (out / "train_done.json").write_text(json.dumps({"steps": step, "minutes": (time.time() - t0) / 60}))
    print("done")


if __name__ == "__main__":
    main()
