"""GReg objective (Sec. 3.4).

    L = l_per * L_per + l_rper * L_r-per + l_struct * L_struct + l_adv * L_adv

Decisions for the paper's gaps:
* ``lambda_prior`` (Eq. 5) and ``lambda_struct`` (text) are the same weight.
* No weights are given; defaults are in ``configs/hamnosys_greg.yaml``.
* The fake pair fed to the discriminators uses the gradient of the *generated*
  region (Eq. 10's form) in both G and D losses.
* GAN loss type is configurable (``bce`` = the paper's log form, non-saturating
  for G; ``hinge``; ``lsgan``).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from greg.data.gradient import gradient_map
from greg.data.regions import crop_regions, resize_patches
from greg.models.vgg import PerceptualLoss


def gan_loss_d(real: List[torch.Tensor], fake: List[torch.Tensor], kind: str) -> torch.Tensor:
    total = real[0].new_zeros(())
    for r, f in zip(real, fake):
        if kind == "bce":
            total = total + F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + \
                F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
        elif kind == "hinge":
            total = total + F.relu(1 - r).mean() + F.relu(1 + f).mean()
        elif kind == "lsgan":
            total = total + ((r - 1) ** 2).mean() + (f ** 2).mean()
        else:
            raise ValueError(kind)
    return total / len(real)


def gan_loss_g(fake: List[torch.Tensor], kind: str) -> torch.Tensor:
    total = fake[0].new_zeros(())
    for f in fake:
        if kind == "bce":
            total = total + F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
        elif kind == "hinge":
            total = total - f.mean()
        elif kind == "lsgan":
            total = total + ((f - 1) ** 2).mean()
        else:
            raise ValueError(kind)
    return total / len(fake)


class GRegLoss:
    def __init__(self, perceptual: PerceptualLoss, weights: Dict[str, float], gan_type: str = "bce",
                 region_size: int = 64, region_perceptual_size: int = 128, struct_target: str = "gradient",
                 region_visibility_weighting: bool = True):
        self.perceptual = perceptual
        self.w = {"perceptual": 10.0, "regional_perceptual": 10.0, "struct": 10.0, "adv": 1.0}
        self.w.update({k: float(v) for k, v in (weights or {}).items()})
        self.gan_type = gan_type
        self.region_size = region_size
        self.region_perceptual_size = region_perceptual_size
        self.struct_target = struct_target
        self.region_visibility_weighting = region_visibility_weighting

    # --------------------------------------------------------------- region targets
    def region_targets(self, image_target: torch.Tensor, boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Real crops (B, R, 3, s, s), their gradients (B, R, 1, s, s), and the RE target."""
        crops = crop_regions(image_target, boxes, self.region_size)
        B, R, C, s, _ = crops.shape
        grads = gradient_map(crops.view(B * R, C, s, s)).view(B, R, 1, s, s)
        if self.struct_target == "gradient":
            struct = grads
        elif self.struct_target == "grayscale":
            from greg.data.gradient import to_grayscale
            struct = to_grayscale(crops.view(B * R, C, s, s)).view(B, R, 1, s, s)
        elif self.struct_target == "rgb":
            struct = crops
        else:
            raise ValueError(self.struct_target)
        return {"crops": crops, "grads": grads, "struct": struct}

    def fake_regions(self, output: torch.Tensor, boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        crops = crop_regions(output, boxes, self.region_size)
        B, R, C, s, _ = crops.shape
        grads = gradient_map(crops.view(B * R, C, s, s)).view(B, R, 1, s, s)
        return {"crops": crops, "grads": grads}

    # ------------------------------------------------------------------ generator
    def generator_loss(self, out: Dict[str, torch.Tensor], image_target: torch.Tensor,
                       discriminators=None, region_weight: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        boxes = out["boxes"]
        real = self.region_targets(image_target, boxes)
        fake = self.fake_regions(out["output"], boxes)
        B, R = real["crops"].shape[:2]
        rw = region_weight if (region_weight is not None and self.region_visibility_weighting) else \
            real["crops"].new_ones(B, R)

        losses: Dict[str, torch.Tensor] = {}
        losses["perceptual"] = self.perceptual(out["output"], image_target)

        # regional perceptual loss on upsampled crops (so VGG sees enough pixels)
        fr = resize_patches(fake["crops"], self.region_perceptual_size)
        rr = resize_patches(real["crops"], self.region_perceptual_size)
        rper = out["output"].new_zeros(())
        for r in range(R):
            # weight each region by the fraction of samples where it is visible
            rper = rper + self.perceptual(fr[:, r], rr[:, r]) * rw[:, r].mean()
        losses["regional_perceptual"] = rper / R

        # prior structure loss (RE output vs. target region structure)
        prior = out["prior_patches"]                                   # (B, R, 1, s, s)
        struct = real["struct"]
        if struct.shape[2] != prior.shape[2]:
            struct = struct.mean(2, keepdim=True)
        l1 = (prior - struct).abs().mean(dim=(2, 3, 4))                 # (B, R)
        losses["struct"] = (l1 * rw).sum() / rw.sum().clamp(min=1)

        if discriminators is not None and self.w["adv"] > 0:
            fake_logits = discriminators(fake["crops"], fake["grads"])
            losses["adv"] = gan_loss_g(fake_logits, self.gan_type)
        total = sum(self.w[k] * v for k, v in losses.items())
        losses["total"] = total
        return losses

    # -------------------------------------------------------------- discriminator
    def discriminator_loss(self, out: Dict[str, torch.Tensor], image_target: torch.Tensor, discriminators
                           ) -> Dict[str, torch.Tensor]:
        boxes = out["boxes"]
        real = self.region_targets(image_target, boxes)
        fake = self.fake_regions(out["output"].detach(), boxes)
        real_logits = discriminators(real["crops"], real["grads"])
        fake_logits = discriminators(fake["crops"], fake["grads"])
        d = gan_loss_d(real_logits, fake_logits, self.gan_type)
        return {"d_total": d}
