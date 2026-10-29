"""GReg objective (Sec. 3.4).

    L = l_per * L_per + l_rper * L_r-per + l_struct * L_struct + l_adv * L_adv

Decisions for the paper's gaps:
* ``lambda_prior`` (Eq. 5) and ``lambda_struct`` (text) are the same weight.
* No weights are given; defaults are in ``configs/hamnosys_greg.yaml``.
* The fake pair fed to the discriminators uses the gradient of the *generated*
  region (Eq. 10's form) in both G and D losses.
* GAN loss type is configurable (``bce`` = the paper's log form, non-saturating
  for G; ``hinge``; ``lsgan``).

Improvement B1 (``region_mask_mode: multiply``): the regional crops are multiplied by the mesh mask of their
own part (``out["region_masks"]``: face pixels in the face crop, left-hand pixels in the left-hand crop, ...)
before the regional VGG loss and the regional discriminators, and ``L_struct`` weights the mask by 1 and the
rest of the crop by ``struct_bg_weight``.  The background around a hand then no longer dominates these losses.
Two optional terms: ``occ`` = |O - O_geo| keeps the learned occlusion residual of A1 close to the geometry,
``masked_l1`` = pixel L1 on the dilated hands + arms (labels 3-6 of the mesh part map).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from greg.data.gradient import gradient_map
from greg.data.regions import crop_regions, part_mask, resize_patches
from greg.models.vgg import PerceptualLoss

ARM_HAND_LABELS = (3, 4, 5, 6)      # lhand, rhand, larm, rarm (arms only exist in the mesh part map)


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
                 region_visibility_weighting: bool = True, region_mask_mode: str = "none",
                 struct_bg_weight: float = 1.0, masked_l1_dilate_px: int = 4):
        self.perceptual = perceptual
        self.w = {"perceptual": 10.0, "regional_perceptual": 10.0, "struct": 10.0, "adv": 1.0,
                  "occ": 0.0, "masked_l1": 0.0}
        self.w.update({k: float(v) for k, v in (weights or {}).items()})
        self.gan_type = gan_type
        self.region_size = region_size
        self.region_perceptual_size = region_perceptual_size
        self.struct_target = struct_target
        self.region_visibility_weighting = region_visibility_weighting
        if region_mask_mode not in ("none", "multiply"):
            raise ValueError(f"loss.region_mask_mode must be none or multiply, got {region_mask_mode!r}")
        self.region_mask_mode = region_mask_mode
        self.struct_bg_weight = float(struct_bg_weight)
        self.masked_l1_dilate_px = int(masked_l1_dilate_px)

    def masks(self, out: Dict[str, torch.Tensor]):
        """B1 region masks (B, R, 1, s, s) when enabled and available, else None."""
        if self.region_mask_mode == "multiply":
            if "region_masks" not in out:
                raise ValueError("loss.region_mask_mode=multiply needs the target part map "
                                 "(mesh flow, or a loader built with with_parts=True)")
            return out["region_masks"]
        return None

    # --------------------------------------------------------------- region targets
    def region_targets(self, image_target: torch.Tensor, boxes: torch.Tensor,
                       masks: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Real crops (B, R, 3, s, s), their gradients (B, R, 1, s, s), and the RE target.  With ``masks``
        the crops and gradients fed to VGG / D are masked; the RE target is not (``L_struct`` weights it)."""
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
        if masks is not None:
            crops, grads = crops * masks, grads * masks
        return {"crops": crops, "grads": grads, "struct": struct}

    def fake_regions(self, output: torch.Tensor, boxes: torch.Tensor,
                     masks: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        crops = crop_regions(output, boxes, self.region_size)
        B, R, C, s, _ = crops.shape
        grads = gradient_map(crops.view(B * R, C, s, s)).view(B, R, 1, s, s)
        if masks is not None:
            crops, grads = crops * masks, grads * masks
        return {"crops": crops, "grads": grads}

    # ------------------------------------------------------------------ generator
    def generator_loss(self, out: Dict[str, torch.Tensor], image_target: torch.Tensor,
                       discriminators=None, region_weight: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        boxes = out["boxes"]
        masks = self.masks(out)
        real = self.region_targets(image_target, boxes, masks)
        fake = self.fake_regions(out["output"], boxes, masks)
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
        if masks is None:
            l1 = (prior - struct).abs().mean(dim=(2, 3, 4))                 # (B, R)
        else:   # B1: the region's own pixels count fully, the rest of the crop by struct_bg_weight
            wm = masks + self.struct_bg_weight * (1 - masks)
            l1 = ((prior - struct).abs() * wm).sum(dim=(2, 3, 4)) / wm.sum(dim=(2, 3, 4)).clamp(min=1e-6)
        losses["struct"] = (l1 * rw).sum() / rw.sum().clamp(min=1)

        if discriminators is not None and self.w["adv"] > 0:
            fake_logits = discriminators(fake["crops"], fake["grads"])
            losses["adv"] = gan_loss_g(fake_logits, self.gan_type)

        if self.w["occ"] > 0 and "occlusion_geo" in out and out["occlusion"] is not out["occlusion_geo"]:
            losses["occ"] = (out["occlusion"] - out["occlusion_geo"]).abs().mean()
        if self.w["masked_l1"] > 0 and "parts_target" in out:
            m = part_mask(out["parts_target"], ARM_HAND_LABELS, self.masked_l1_dilate_px)
            err = (out["output"] - image_target).abs().mean(1, keepdim=True)
            losses["masked_l1"] = (err * m).sum() / m.sum().clamp(min=1.0)
        total = sum(self.w[k] * v for k, v in losses.items())
        losses["total"] = total
        return losses

    # -------------------------------------------------------------- discriminator
    def discriminator_loss(self, out: Dict[str, torch.Tensor], image_target: torch.Tensor, discriminators
                           ) -> Dict[str, torch.Tensor]:
        boxes = out["boxes"]
        masks = self.masks(out)
        real = self.region_targets(image_target, boxes, masks)
        fake = self.fake_regions(out["output"].detach(), boxes, masks)
        real_logits = discriminators(real["crops"], real["grads"])
        fake_logits = discriminators(fake["crops"], fake["grads"])
        d = gan_loss_d(real_logits, fake_logits, self.gan_type)
        return {"d_total": d}
