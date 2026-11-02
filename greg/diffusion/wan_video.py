"""D6: Wan2.1-Fun-V1.1-1.3B-InP (video DiT, inpainting variant) with mesh conditions, the DWM recipe on diffusers.

docs/code_notes_dwm_wan.md is the line-by-line read of DWM's Wan code (commit 19de79b) this follows.  Summary of
the conventions that silently break training when wrong:

* channels into ``patch_embedding``: ``[z_t (16) | mask (4) | masked-video latent (16) | extra (16 each)]``; the
  extra columns start at zero (DWM ``_extend_patch_embedding``); the whole patch embedding is trained, kept in
  **fp32** and gradient-clipped (DWM updates it in bf16 without clipping);
* the mask channel means **1 = known** in latent space.  Our V*C already means "1 = trusted source pixel", so the
  packed mask is ``1[V*C >= 0.5]`` itself (NOT 1 - V*C); the masked video is ``I_st * M`` in [-1, 1] (unknown = 0);
* mask packing: frame 0 repeated 4x, reshaped to 4 channels per latent frame, then reduced 8x spatially by
  **average pooling** (DWM's trilinear resize aliases fingers); frame count must be 4n + 1;
* diffusers' ``AutoencoderKLWan`` does NOT normalise latents (VideoX-Fun's does): ``(z - mean) / std`` with the
  per-channel ``latents_mean / latents_std`` of the VAE config, inverted before decoding;
* text: umT5 embedding of a fixed prompt, truncated to its length and zero-padded to exactly 512 tokens (diffusers
  splits image / text context with a hard-coded 512); CLIP: ``hidden_states[-2]`` of the CLIP vision tower on the
  real source frame, in training AND inference (DWM trains without and infers with zeros -> mismatch);
* flow matching on the shift-5 schedule: ``z_t = (1 - s) z0 + s eps``, target ``eps - z0``, MSE ignoring elements
  with |error| > 50 (DWM / VideoX-Fun); timestep index uniform in [0, 1000).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FUN_INP_13B_CONFIG = dict(patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128, in_channels=36,
                          out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=8960, num_layers=30,
                          cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=1280,
                          added_kv_proj_dim=1536, rope_max_seq_len=1024)
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "ffn.net.0.proj", "ffn.net.2"]
TEXT_LEN = 512
# extra conditioning blocks (16 latent channels each) per variant; the mask / masked-video slots always exist
VARIANT_EXTRA = {"base": [], "k": ["skel"], "n": ["normal"], "ng": ["normal", "ccm"], "ng_ones": ["normal", "ccm"]}


def load_fun_inp_transformer(ckpt_dir: str | Path, dtype=torch.bfloat16):
    """Original-format Wan2.1-Fun-1.3B-InP checkpoint -> ``diffusers.WanTransformer3DModel`` (strict load).
    diffusers' single-file loader would mistake it for the t2v-1.3B (16 input channels, no image branch)."""
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_dir) / "diffusion_pytorch_model.safetensors"))
    conv = convert_wan_transformer_to_diffusers(sd)
    model = WanTransformer3DModel(**FUN_INP_13B_CONFIG)
    missing, unexpected = model.load_state_dict(conv, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Wan Fun-InP conversion: {len(missing)} missing (e.g. {missing[:4]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})")
    return model.to(dtype)


def widen_patch_embedding(transformer, extra_channels: int) -> None:
    """Zero-initialised input columns for ``extra_channels`` more channels; kept in fp32 (trained)."""
    old = transformer.patch_embedding
    new = nn.Conv3d(old.in_channels + extra_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight.float()
        if old.bias is not None:
            new.bias.copy_(old.bias.float())
    transformer.patch_embedding = new.to(old.weight.device, torch.float32)
    transformer.register_to_config(in_channels=old.in_channels + extra_channels)


def pack_frames(m: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """(B, 1, F, H, W), F = 4n+1 -> (B, 4, n+1, H/pool, W/pool): Wan-Fun's 4-channels-per-latent-frame layout
    (frame 0 repeated 4x), average-pooled spatially."""
    B, C, Fr, H, W = m.shape
    if C != 1 or (Fr - 1) % 4 != 0:
        raise ValueError(f"pack_frames needs (B, 1, 4n+1, H, W), got {tuple(m.shape)}")
    x = torch.cat([m[:, :, :1].repeat(1, 1, 4, 1, 1), m[:, :, 1:]], dim=2)          # (B, 1, 4(n+1), H, W)
    x = x.view(B, (Fr + 3) // 4, 4, H, W).transpose(1, 2)                            # (B, 4, n+1, H, W)
    if pool > 1:
        x = F.avg_pool3d(x, (1, pool, pool))
    return x


class WanLatentCodec:
    """``AutoencoderKLWan`` with the per-channel latent normalisation of Wan (done inside encode by VideoX-Fun)."""

    def __init__(self, vae):
        self.vae = vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, video_m11: torch.Tensor, moments: bool = False):
        """(B, 3, F, H, W) in [-1, 1] -> normalised mode (B, 16, F', h, w), or (mean, std) if ``moments``."""
        d = self.vae.encode(video_m11.to(next(self.vae.parameters()).dtype)).latent_dist
        m, s = self.mean.to(d.mean.device, d.mean.dtype), self.std.to(d.mean.device, d.mean.dtype)
        mean = (d.mean - m) / s
        return (mean, d.std / s) if moments else mean

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalised latent -> (B, 3, F, H, W) in [0, 1]."""
        dt = next(self.vae.parameters()).dtype
        m, s = self.mean.to(z.device, dt), self.std.to(z.device, dt)
        x = self.vae.decode(z.to(dt) * s + m).sample
        return (x.float().clamp(-1, 1) + 1) / 2


def flow_schedule(shift: float = 5.0):
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)


def filtered_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None,
                 threshold: float = 50.0) -> torch.Tensor:
    """DWM / VideoX-Fun ``custom_mse_loss``: squared error, elements with |error| > threshold dropped (counted in
    the mean's denominator), optionally weighted per position."""
    diff = pred.float() - target.float()
    se = diff ** 2 * (diff.abs() <= threshold).float()
    if weight is None:
        return se.mean()
    w = weight.float().expand_as(se)
    return (se * w).sum() / w.sum().clamp(min=1e-6)


class WanConditionModel(nn.Module):
    """Transformer + LoRA + widened patch embedding; assembles the input channels for one variant."""

    def __init__(self, transformer, variant: str = "ng", lora_rank: int = 32):
        super().__init__()
        if variant not in VARIANT_EXTRA:
            raise ValueError(f"variant must be one of {list(VARIANT_EXTRA)}")
        self.variant = variant
        self.extra = VARIANT_EXTRA[variant]
        self.transformer = transformer
        for p in transformer.parameters():
            p.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig
            transformer.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights=True,
                                               target_modules=LORA_TARGETS))
        if self.extra:
            widen_patch_embedding(transformer, 16 * len(self.extra))
        else:
            transformer.patch_embedding.float()
        for p in transformer.patch_embedding.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in names}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        bad = [n for n in sd if n not in params]
        if bad:
            raise KeyError(f"{len(bad)} saved tensors have no parameter, e.g. {bad[:3]}")
        with torch.no_grad():
            for n, v in sd.items():
                params[n].copy_(v.to(params[n].dtype))

    def assemble(self, z_t: torch.Tensor, w: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``w``: cached window tensors (batched).  -> (B, C, F', h, w) input of the patch embedding."""
        if self.variant in ("k", "n"):          # i2v mode: only frame 0 (the real source) is known
            mask, masked = w["mask_i2v"], w["lat_i2v"]
        elif self.variant == "ng_ones":         # DWM mode: everything "known", warp as a static condition
            mask, masked = torch.ones_like(w["mask"]), w["lat_masked"]
        else:                                   # base / ng: the real mesh visibility mask
            mask, masked = w["mask"], w["lat_masked"]
        blocks = [z_t, mask.to(z_t.dtype), masked.to(z_t.dtype)]
        blocks += [w[f"lat_{e}"].to(z_t.dtype) for e in self.extra]
        return torch.cat(blocks, dim=1)

    def forward(self, z_t, t, w, text, clip) -> torch.Tensor:
        inp = self.assemble(z_t, w)
        return self.transformer(hidden_states=inp, timestep=t, encoder_hidden_states=text,
                                encoder_hidden_states_image=clip, return_dict=False)[0]


def training_loss(model: WanConditionModel, sched, w: Dict[str, torch.Tensor], text, clip,
                  w_region: float = 9.0) -> torch.Tensor:
    z0 = w["lat_mean"] + w["lat_std"] * torch.randn_like(w["lat_std"])          # sample the cached posterior
    B = z0.shape[0]
    idx = torch.randint(0, sched.config.num_train_timesteps, (B,))
    sig = sched.sigmas[idx].to(z0.device, z0.dtype).view(B, 1, 1, 1, 1)
    t = sched.timesteps[idx].to(z0.device)
    eps = torch.randn_like(z0)
    z_t = (1 - sig) * z0 + sig * eps
    v = model(z_t, t, w, text, clip)
    weight = (1 + w_region * w["region"]) * w["valid"]                         # (B, 1, F', h, w)
    return filtered_mse(v, eps - z0, weight)


@torch.no_grad()
def sample(model: WanConditionModel, w: Dict[str, torch.Tensor], text, clip, steps: int = 25, shift: float = 5.0,
           noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    sched = flow_schedule(shift)
    sched.set_timesteps(steps, device=w["lat_masked"].device)
    z = noise if noise is not None else torch.randn_like(w["lat_masked"])
    for t in sched.timesteps:
        v = model(z, t.expand(z.shape[0]).to(z.device), w, text, clip)
        z = sched.step(v, t, z, return_dict=False)[0]
    return z
