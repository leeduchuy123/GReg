#!/usr/bin/env python
"""D5: train SD-1.5 + ControlNet (+ reference attention, LoRA) with one pose-condition variant (k | n | ng).

    python scripts/30_train_frame_ldm.py --config configs/diffusion/d5_frame_ldm.yaml \
        --override model.variant=ng data.cache=CACHE data.smplx_dir=SMPLX output_dir=outputs/d5/ng

Pairs: the GReg training pairs (train_fit clips) with half of the sources at the clip's first valid row (the test
protocol's rest pose).  Every variant sees exactly the same pairs and steps (same seed) -> only the condition differs.
The ``ng`` conditions (mesh warp, V*C, CCM) are computed on the GPU from the mesh buffers for every pair.
Resume-safe (``checkpoints/last.pt``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, PairDataset, SequenceDataset, collate_keep_strings, read_split  # noqa: E402
from greg.data.regions import part_mask  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.diffusion.frame_ldm import CONDITION_CHANNELS, FrameLDM, build_condition  # noqa: E402
from greg.geometry.canonical import canonical_coordinate_map, template_coordinates  # noqa: E402
from greg.models.tpsmm import warp  # noqa: E402
from greg.utils.config import apply_overrides, get_device, load_config, save_config, set_seed  # noqa: E402
from greg.utils.logging import CSVLogger, save_grid  # noqa: E402


def up(x, size):
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


class Conditioner:
    """Builds the ControlNet input of a batch for one variant (mesh correspondence only for ``ng``)."""

    def __init__(self, variant: str, work: int, smplx_dir: str | None, device):
        self.variant, self.work = variant, work
        self.mesh = None
        if variant == "ng":
            from greg.geometry.mesh_flow import MeshCorrespondence
            self.mesh = MeshCorrespondence(smplx_dir, device=device, edge_px=4)
            self.coords = template_coordinates(self.mesh.body).to(device)
            self.faces = self.mesh.body.faces.to(device)

    @torch.no_grad()
    def __call__(self, b: dict) -> torch.Tensor:
        if self.variant == "k":
            return build_condition("k", self.work, joints=b["target_joints"])
        if self.variant == "n":
            return build_condition("n", self.work, normal=b["target_normal"])
        g = self.mesh(b["source_smplx"], b["source_focal"], b["source_princpt"], b["source_faces"],
                      b["target_smplx"], b["target_focal"], b["target_princpt"], b["target_faces"],
                      valid_rect_s=b.get("source_valid_rect"))
        ccm = canonical_coordinate_map(b["target_faces"].long(), g["bary_target"], self.faces, self.coords)
        return build_condition("ng", self.work, normal=b["target_normal"], warped=warp(b["source_image"], g["grid"]),
                               vc=g["occlusion"], ccm=ccm)


def build_model(cfg, device) -> FrameLDM:
    m = cfg["model"]
    model = FrameLDM.from_pretrained(m.get("base", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
                                     CONDITION_CHANNELS[m["variant"]], lora_rank=int(m.get("lora_rank", 16)))
    if bool(m.get("gradient_checkpointing", True)):
        model.enable_gradient_checkpointing()
    return model.to(device)


def to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.get("device", "auto"))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    d, m, tr = cfg["data"], cfg["model"], cfg["train"]
    work = int(m.get("work_size", 512))
    amp = torch.bfloat16

    keys = read_split(Path(d["cache"]) / "splits" / f"{d.get('train_split', 'train_fit')}.txt")
    cache = HamNoSysCache(d["cache"], keys)
    cache.keep_keys_with_mesh(what="d5 train")
    ds = PairDataset(cache, pairs_per_clip=int(d.get("pairs_per_clip", 8)), seed=int(cfg.get("seed", 0)),
                     with_parts=True, with_mesh=True, source_first_prob=float(d.get("source_first_prob", 0.5)))
    loader = DataLoader(ds, batch_size=int(tr["batch_size"]), shuffle=True, drop_last=True,
                        num_workers=int(tr.get("num_workers", 8)), collate_fn=collate_keep_strings, pin_memory=True,
                        persistent_workers=True)
    vkeys = pick_keys(read_split(Path(d["cache"]) / "splits" / "val.txt"), 8, seed=0)
    vcache = HamNoSysCache(d["cache"], vkeys)
    vds = SequenceDataset(vcache, source_row=0, max_rows=1, uniform_rows=True, with_parts=True, with_mesh=True)
    vbatch = collate_keep_strings([vds[i] for i in range(len(vds))])

    cond_fn = Conditioner(m["variant"], work, d.get("smplx_dir"), device)
    model = build_model(cfg, device)
    cn_params = [p for p in model.controlnet.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.unet.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": cn_params, "lr": float(tr.get("lr_controlnet", 1e-5))},
                             {"params": lora_params, "lr": float(tr.get("lr_lora", 1e-4))}], weight_decay=0.01)
    print(f"D5 variant {m['variant']}: ControlNet {sum(p.numel() for p in cn_params) / 1e6:.1f}M + LoRA "
          f"{sum(p.numel() for p in lora_params) / 1e6:.2f}M trainable; {len(ds)} pairs/pass")
    step, ck = 0, out / "checkpoints" / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(st["trainable"])
        opt.load_state_dict(st["optimizer"])
        step = int(st["step"])
        print(f"resumed at step {step}")
    total, accum = int(args.max_steps or tr["steps"]), int(tr.get("grad_accum", 1))
    log = CSVLogger(out / "logs" / "loss.csv", ["loss", "sec_per_step", "vram_gb"])

    def save(tag="last"):
        pth = out / "checkpoints" / f"{tag}.pt"
        pth.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable": model.trainable_state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "config": cfg}, pth.with_suffix(".tmp"))
        pth.with_suffix(".tmp").replace(pth)

    def preview():
        model.eval()
        b = to_dev(vbatch, device)
        with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
            c = cond_fn(b)
            g = torch.Generator(device="cpu").manual_seed(0)
            z = torch.randn(c.shape[0], 4, work // 8, work // 8, generator=g).to(device)
            o = model.sample(up(b["source_image"], work), c, steps=int(m.get("sample_steps", 25)), noise=z)
        o = F.interpolate(o.float(), size=b["target_image"].shape[-2:], mode="area")
        rows = [[b["source_image"][i], F.interpolate(c[i:i + 1, :3].float(), size=o.shape[-2:])[0], o[i],
                 b["target_image"][i]] for i in range(o.shape[0])]
        save_grid(rows, out / "samples" / f"step_{step:06d}.jpg")
        model.train()
        return float((o - b["target_image"]).abs().mean())

    model.train()
    it, t0, tl = iter(loader), time.time(), time.time()
    warm = int(tr.get("warmup", 500))
    while step < total:
        f = min(1.0, (step + 1) / warm)
        opt.param_groups[0]["lr"] = float(tr.get("lr_controlnet", 1e-5)) * f
        opt.param_groups[1]["lr"] = float(tr.get("lr_lora", 1e-4)) * f
        opt.zero_grad(set_to_none=True)
        lv = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            b = to_dev(batch, device)
            cond = cond_fn(b)
            mask = part_mask(b["target_parts"].long(), (2, 3, 4), 3, soften=False)
            # cache_enabled=False: training_loss runs the LoRA UNet twice in this region, first under no_grad (reference
            # write); a cached bf16 copy of the LoRA weights from that pass carries no autograd history, so the grad pass
            # would reuse it (no LoRA gradient) and checkpoint recomputation saw different tensors (CheckpointError)
            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda", cache_enabled=False):
                loss = model.training_loss(up(b["target_image"], work), up(b["source_image"], work), cond, mask,
                                           float(cfg.get("loss", {}).get("w_region", 9.0)))
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            (loss / accum).backward()
            lv += float(loss) / accum
        torch.nn.utils.clip_grad_norm_(cn_params + lora_params, 1.0)
        opt.step()
        step += 1
        if step % int(tr.get("log_every", 50)) == 0:
            dt = (time.time() - tl) / int(tr.get("log_every", 50))
            tl = time.time()
            vram = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0
            log.log(step, 0, {"loss": lv, "sec_per_step": dt, "vram_gb": vram})
            print(f"step {step}/{total} loss={lv:.4f} {dt:.2f}s/step vram={vram:.1f}GB ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
        if step % int(tr.get("preview_every", 2000)) == 0 or step == total:
            print(f"[preview {step}] l1={preview():.4f}", flush=True)
        if step % int(tr.get("save_every", 1000)) == 0 or step == total:
            save()
    save()
    save("final")
    (out / "train_done.json").write_text(json.dumps({"steps": step, "minutes": (time.time() - t0) / 60}))
    print("done")


if __name__ == "__main__":
    main()
