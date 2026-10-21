"""Shared pieces of the two training scripts."""
from __future__ import annotations

import math
from typing import Dict, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from greg.data.dataset import HamNoSysCache, PairDataset, collate_keep_strings, read_split
from greg.data.regions import region_visibility
from greg.models.tpsmm import TPS


def build_pair_loader(cfg: dict, split: str = "train", with_parts: bool = False, seed: int = 0) -> DataLoader:
    data = cfg["data"]
    if split == "train":   # sweeps train on "train_fit" (train minus the validation clips), final runs on "train"
        split = data.get("train_split", "train")
    keys = read_split(f"{data['cache']}/splits/{split}.txt")
    cache = HamNoSysCache(data["cache"], keys)
    ds = PairDataset(cache, pairs_per_clip=int(data.get("pairs_per_clip", 1)), seed=seed,
                     with_parts=with_parts, min_gap=int(data.get("min_gap", 0)))
    train = cfg["train"]
    return DataLoader(ds, batch_size=int(train["batch_size"]), shuffle=(split == "train"),
                      num_workers=int(train.get("num_workers", 2)), drop_last=(split == "train"),
                      collate_fn=collate_keep_strings, pin_memory=torch.cuda.is_available(),
                      persistent_workers=int(train.get("num_workers", 2)) > 0)


def batch_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def region_weights(batch: dict, image_size: int) -> torch.Tensor:
    """(B, 3) 1 where the target region centre is inside the frame."""
    return region_visibility(batch["target_centers"], image_size, image_size)


class StepLR:
    """lr * 0.1 ** (#milestones passed), applied per epoch (paper: 2e-4, x0.1 at 70 and 90)."""

    def __init__(self, optimizer, base_lr: float, milestones, gamma: float = 0.1):
        self.opt, self.base_lr, self.milestones, self.gamma = optimizer, base_lr, sorted(milestones), gamma

    def set_epoch(self, epoch: int) -> float:
        k = sum(epoch >= m for m in self.milestones)
        lr = self.base_lr * (self.gamma ** k)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def dropout_schedule(epoch: int, cfg: dict) -> tuple:
    """TPSMM contribution-map dropout: (flag, p)."""
    if epoch >= int(cfg.get("dropout_epoch", 35)):
        return False, 0.0
    start, maxp = float(cfg.get("dropout_startp", 0.0)), float(cfg.get("dropout_maxp", 0.7))
    inc = int(cfg.get("dropout_inc_epoch", 10))
    p = min(start + (maxp - start) * epoch / max(inc, 1), maxp)
    return True, p


def equivariance_loss(kp_detector, driving: torch.Tensor, kp_driving: torch.Tensor, params: dict) -> torch.Tensor:
    """TPSMM equivariance on keypoints: kp(T(x)) == T(kp(x)) for a random TPS T."""
    bs = driving.shape[0]
    t = TPS("random", bs, **params)
    grid = t.transform_frame(driving)
    transformed = F.grid_sample(driving, grid, padding_mode="reflection", align_corners=True)
    kp_t = kp_detector(transformed)["fg_kp"]
    warped_kp = t.warp_coordinates(kp_t)
    return (kp_driving - warped_kp).abs().mean()


def warp_loss(warped_encoder_maps, driving_encoder_maps) -> torch.Tensor:
    """Multi-scale L1 between warped source features and driving features (TPSMM 'warp_loss')."""
    total = warped_encoder_maps[0].new_zeros(())
    n = len(warped_encoder_maps)
    for w, d in zip(warped_encoder_maps, driving_encoder_maps[::-1][:n]):
        total = total + (w - d.detach()).abs().mean()
    return total


def infinite(loader: DataLoader) -> Iterator:
    while True:
        for b in loader:
            yield b


class SpeedMeter:
    """Seconds / step and peak GPU memory since the last ``read`` (cost measurement for the GPU budget)."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.reset()

    def reset(self):
        import time
        self.t, self.n = time.perf_counter(), 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def step(self):
        self.n += 1

    def read(self) -> dict:
        import time
        if self.cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self.t
        out = {"sec_per_step": dt / max(self.n, 1),
               "vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if self.cuda else 0.0}
        self.reset()
        return out


def count_params(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6
