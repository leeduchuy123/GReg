"""D3: a frozen diffusion model's latent space as a training loss for GReg (no diffusion at inference).

E-LatentLPIPS (Kang et al. 2024, "Diffusion2GAN"; pip ``elatentlpips``, CC BY-NC): an LPIPS whose VGG-16-BN trunk
works on Stable-Diffusion VAE latents and is calibrated on perceptual judgements, evaluated on an ensemble of
differentiable augmentations (``augment='bg'``: without it the loss gives wavy / patchy artefacts and does not
converge, paper Sec. 4).  GReg's output goes through the frozen SD-1.5 VAE encoder *with* gradients; the target
latent is computed without.  docs/code_notes_diffusion_as_loss.md explains why DMD (distribution matching) was
dropped (its critic is a full trainable SD UNet: neither frozen nor 24 GB-friendly).

``scale``: GReg works at 256^2 -> latent 32^2 (SD's latent is 64^2 at 512); ``scale=2`` upsamples first (4x cost).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def use_reference_style_ops() -> None:
    """Make the StyleGAN2-ADA ops inside ``elatentlpips`` (augmentation: upfirdn2d, bias_act, filtered_lrelu) use
    their pure-PyTorch reference path instead of JIT-compiling CUDA plugins, which needs nvcc / CUDA_HOME (absent on
    the vast image: the first E-LatentLPIPS forward raised OSError).  On 4x32^2 latents the cost is negligible."""
    import importlib
    for name in ("upfirdn2d", "bias_act", "filtered_lrelu"):
        try:
            mod = importlib.import_module(f"elatentlpips.style_ops.{name}")
        except ImportError:
            continue
        if hasattr(mod, "_init"):
            mod._init = lambda: False


class ELatentLPIPSLoss(nn.Module):
    def __init__(self, vae_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", encoder: str = "sd15",
                 augment: str = "bg", scale: float = 1.0, add_l1: bool = True, vae=None, metric=None):
        super().__init__()
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
            vae.decoder = None                        # only the encoder is used
        if metric is None:
            from elatentlpips import ELatentLPIPS
            use_reference_style_ops()
            metric = ELatentLPIPS(encoder=encoder, augment=augment, verbose=False)
        self.vae, self.metric = vae.eval(), metric.eval()
        for p in list(self.vae.parameters()) + list(self.metric.parameters()):
            p.requires_grad_(False)
        self.scale, self.add_l1 = float(scale), bool(add_l1)

    def train(self, mode: bool = True):          # frozen: always eval (BN statistics of the latent VGG)
        return super().train(False)

    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        if self.scale != 1.0:
            x01 = F.interpolate(x01, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.vae.encode(x01 * 2 - 1).latent_dist.mode()     # unscaled: ELatentLPIPS(normalize=True) scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> scalar loss (mean over the batch)."""
        z0 = self.encode(pred)
        with torch.no_grad():
            z1 = self.encode(target)
        return self.metric(z0, z1, normalize=True, add_l1_loss=self.add_l1).mean()
