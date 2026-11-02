"""D1: one-step diffusion refiner on top of GReg (pix2pix-turbo re-implemented on current diffusers / peft).

What pix2pix-turbo does (Parmar et al. 2024; docs/code_notes_pix2pix_turbo.md, read line by line):

* the conditioning image goes through the SD-Turbo VAE encoder, the latent is fed to the UNet at the single
  timestep t = 999, the epsilon prediction is turned into x0 in closed form (that is what ``DDPMScheduler.step``
  does at the last timestep: ``x0 = (z - sqrt(1 - a) eps) / sqrt(a)``, a = alphas_cumprod[999] ~ 0.0047, i.e. a
  ~14.6x gain on the UNet error) and decoded by the VAE decoder;
* four 1x1 "skip" convolutions carry the encoder activations (captured before each down block) into the decoder
  (added before each up block, in reverse order) so pixel detail of the input survives the 8x latent bottleneck;
  they start at ~0 so the net begins as the plain VAE;
* LoRA (rank 8 UNet, rank 4 VAE) + the UNet ``conv_in`` trained in full; ~9.6 M trainable parameters.

Differences here (GReg refiner):

* ``conv_in`` is widened by zero-initialised columns for extra latent-space conditions (VAE latent of the target
  normal map, visibility / region masks pooled to the latent grid) -- the DWM recipe; at step 0 they change nothing;
* skips are captured / injected with forward pre-hooks instead of monkey-patching ``Encoder.forward`` /
  ``Decoder.forward`` (the original pins diffusers 0.25.1 for that);
* ``latent_dist.mode()`` (no posterior noise -> no frame flicker), inputs and targets both in [-1, 1],
  ``lora_alpha = r`` explicitly, a fixed prompt embedding computed once (the text encoder is not kept).

The pixel-space gate (which pixels of GReg's output are replaced) is applied outside, see :mod:`greg.diffusion.gating`.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

UNET_LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                     "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
VAE_LORA_TARGETS = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
DEFAULT_PROMPT = "a person signing in sign language, sharp hands and fingers, studio video, uniform lighting"


def widen_conv_in(conv: nn.Conv2d, extra_in: int) -> nn.Conv2d:
    """Copy of ``conv`` taking ``extra_in`` more input channels whose weights start at zero (output unchanged)."""
    new = nn.Conv2d(conv.in_channels + extra_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new.to(conv.weight.device, conv.weight.dtype)


class VAESkips(nn.Module):
    """Encoder -> decoder skip connections of pix2pix-turbo, attached with hooks.

    ``capture`` switches the encoder hooks off for conditioning images that are encoded for the UNet only, so they
    do not overwrite the activations of the image being refined."""

    def __init__(self, vae):
        super().__init__()
        enc_ch = [b.resnets[0].in_channels for b in vae.encoder.down_blocks]       # inputs of the down blocks
        dec_ch = [b.resnets[0].in_channels for b in vae.decoder.up_blocks]         # inputs of the up blocks
        if len(enc_ch) != len(dec_ch):
            raise ValueError("encoder / decoder depth mismatch")
        self.convs = nn.ModuleList(nn.Conv2d(ci, co, 1, bias=False) for ci, co in zip(enc_ch[::-1], dec_ch))
        for c in self.convs:
            nn.init.zeros_(c.weight)
        self.gamma = 1.0
        self.capture = True
        self.enabled = True
        self._acts: List[torch.Tensor] = []
        self._handles = []
        for i, blk in enumerate(vae.encoder.down_blocks):
            self._handles.append(blk.register_forward_pre_hook(self._make_capture(i)))
        for i, blk in enumerate(vae.decoder.up_blocks):
            self._handles.append(blk.register_forward_pre_hook(self._make_inject(i), with_kwargs=True))

    def _make_capture(self, i: int):
        def hook(module, args):
            if self.capture:
                if i == 0:
                    self._acts = []
                self._acts.append(args[0])
        return hook

    def _make_inject(self, i: int):
        def hook(module, args, kwargs):
            if not self.enabled or not self._acts:
                return None
            acts = self._acts[::-1]
            if len(acts) != len(self.convs):
                raise RuntimeError(f"{len(acts)} skip activations captured, {len(self.convs)} expected: "
                                   "encode the image right before decoding, with capture on")
            x = args[0]
            skip = self.convs[i](acts[i].to(x.dtype) * self.gamma)
            if skip.shape[-2:] != x.shape[-2:]:
                raise RuntimeError(f"skip {i}: activation {tuple(acts[i].shape)} vs decoder input {tuple(x.shape)}")
            return (x + skip,) + tuple(args[1:]), kwargs
        return hook

    def clear(self):
        self._acts = []


class TurboRefiner(nn.Module):
    """One-step SD-Turbo refiner with extra latent conditions.

    ``forward(image, normal=None, masks=None)``: ``image`` (B, 3, H, W) in [0, 1] (GReg output upsampled to the work
    size), ``normal`` (B, 3, H, W) in [0, 1] (target normal map, encoded by the frozen VAE), ``masks`` (B, M, H, W) in
    [0, 1], pooled to the latent grid.  Returns the refined image in [0, 1]."""

    def __init__(self, vae, unet, prompt_embeds: torch.Tensor, alphas_cumprod: torch.Tensor, *,
                 cond_normal: bool = True, num_masks: int = 4, lora_rank_unet: int = 8, lora_rank_vae: int = 4,
                 timestep: int = 999):
        super().__init__()
        self.vae, self.unet = vae, unet
        self.cond_normal, self.num_masks = bool(cond_normal), int(num_masks)
        self.timestep = int(timestep)
        self.scaling = float(vae.config.scaling_factor)
        a = float(alphas_cumprod[self.timestep])
        self.sqrt_a, self.sqrt_1ma = a ** 0.5, (1 - a) ** 0.5
        self.register_buffer("prompt_embeds", prompt_embeds.detach().clone(), persistent=False)
        for p in list(self.vae.parameters()) + list(self.unet.parameters()):
            p.requires_grad_(False)
        from peft import LoraConfig
        if lora_rank_unet > 0:
            unet.add_adapter(LoraConfig(r=lora_rank_unet, lora_alpha=lora_rank_unet, init_lora_weights="gaussian",
                                        target_modules=UNET_LORA_TARGETS), adapter_name="default")
        if lora_rank_vae > 0:
            vae.add_adapter(LoraConfig(r=lora_rank_vae, lora_alpha=lora_rank_vae, init_lora_weights="gaussian",
                                       target_modules=VAE_LORA_TARGETS), adapter_name="vae_skip")
        latent_ch = int(unet.config.in_channels)
        extra = (latent_ch if self.cond_normal else 0) + self.num_masks
        if extra > 0:
            unet.conv_in = widen_conv_in(unet.conv_in, extra)
        for p in unet.conv_in.parameters():
            p.requires_grad_(True)
        self.latent_ch = latent_ch
        self.skips = VAESkips(vae)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, model_id: str = "stabilityai/sd-turbo", prompt: str = DEFAULT_PROMPT,
                        torch_dtype=torch.float32, **kw) -> "TurboRefiner":
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import AutoTokenizer, CLIPTextModel
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)
        sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        tok = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        with torch.no_grad():
            ids = tok(prompt, max_length=tok.model_max_length, padding="max_length", truncation=True,
                      return_tensors="pt").input_ids
            emb = te(ids)[0].to(torch_dtype)
        del te
        return cls(vae, unet, emb, sched.alphas_cumprod, **kw)

    # ------------------------------------------------------------------ parameters
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in names}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        missing = [n for n in sd if n not in params]
        if missing:
            raise KeyError(f"{len(missing)} saved tensors have no parameter here, e.g. {missing[:3]}")
        with torch.no_grad():
            for n, v in sd.items():
                params[n].copy_(v.to(params[n].dtype))

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ forward
    def encode(self, x01: torch.Tensor, capture: bool) -> torch.Tensor:
        """[0, 1] image -> scaled latent (mode of the posterior)."""
        self.skips.capture = capture
        try:
            z = self.vae.encode(x01 * 2 - 1).latent_dist.mode() * self.scaling
        finally:
            self.skips.capture = True
        return z

    def forward(self, image: torch.Tensor, normal: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = image.shape
        conds = []
        if self.cond_normal:
            if normal is None:
                raise ValueError("this refiner conditions on the normal map (cond_normal=True)")
            with torch.no_grad():
                conds.append(self.encode(normal, capture=False))      # no skips from the normal map
        if self.num_masks > 0:
            if masks is None or masks.shape[1] != self.num_masks:
                raise ValueError(f"masks (B, {self.num_masks}, H, W) required")
            conds.append(F.interpolate(masks.to(image.dtype), size=(H // 8, W // 8), mode="area") * 2 - 1)
        z = self.encode(image, capture=True)                           # skips of the image being refined
        unet_in = torch.cat([z] + [c.to(z.dtype) for c in conds], dim=1)
        t = torch.full((B,), self.timestep, device=image.device, dtype=torch.long)
        emb = self.prompt_embeds.to(z.dtype).expand(B, -1, -1)
        eps = self.unet(unet_in, t, encoder_hidden_states=emb).sample
        x0 = (z - self.sqrt_1ma * eps) / self.sqrt_a
        out = self.vae.decode(x0 / self.scaling).sample
        self.skips.clear()
        return (out.clamp(-1, 1) + 1) / 2
