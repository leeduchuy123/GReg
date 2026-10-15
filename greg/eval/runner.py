"""Load a trained GReg checkpoint and run it on batches / sequences."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from greg.models.greg import GReg
from greg.utils.checkpoint import load_checkpoint, load_module_state


def load_greg(checkpoint: str | Path, device, override_cfg: Optional[dict] = None) -> tuple:
    ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    if override_cfg:
        from greg.utils.config import deep_update
        cfg = deep_update(cfg, override_cfg)
    model = GReg(cfg["model"]).to(device)
    ab = cfg["model"].get("ablation", {}) or {}
    model.ablation_no_rpg = bool(ab.get("no_rpg", False))
    if bool(ab.get("no_gr", False)):
        model.generator.grms.clear()
    load_module_state(model, ck["model"], strict=False)
    model.freeze_flow()
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(model: GReg, image_source, normal_source, normal_target, centers_target) -> Dict[str, torch.Tensor]:
    return model(image_source, normal_source, normal_target, centers_target)


def save_frame(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)
