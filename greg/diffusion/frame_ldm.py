"""D5: per-frame latent diffusion (SD-1.5 + ControlNet) with interchangeable pose conditions.

The controlled comparison of the thesis: *the same backbone, the same identity mechanism, the same training*, only
the pose condition changes (docs/diffusion_6_huong_ke_hoach.md Sec. 7):

* ``k``   2D keypoints drawn as a skeleton (body + 2 x 21 hand joints + face landmarks), the SignRefine-style
          condition, taken from the *same* SMPL-X fit (projected joints) so only the representation differs;
* ``n``   the SMPL-X normal map of the target (GReg's condition);
* ``ng``  normal map + source warped by the mesh correspondence (only where V*C) + V*C + the canonical coordinate
          map (CCM): what only a mesh can give (visibility, source <-> target correspondence, canonical position).

Identity: "reference attention".  The frozen SD UNet (with the shared LoRA) runs once on the clean latent of the
source frame at t = 0 and every self-attention layer stores its input; in the denoising pass each self-attention
attends to [its own tokens, the stored reference tokens].  Because t = 0 the reference bank is computed once per
frame and reused at every sampling step.  (Animate Anyone uses a trainable copy of the UNet for this, ReferenceNet;
too heavy for a 24 GB card next to a ControlNet.)  LoRA (rank ``lora_rank``) on the UNet attention lets the frozen
UNet learn to read the bank; the ControlNet is trained in full.

Training: epsilon prediction (SD-1.5 DDPM schedule), MSE weighted ``1 + w_region`` on the latent cells of hands and
face (SignRefine Eq. 4 uses 10x).  Sampling: DDIM, ``steps`` steps, no CFG (guidance 1; halves the cost), the same
initial noise for every frame of a clip.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# same prompt as the D1 refiner (greg/diffusion/turbo_refiner.py lives on the D1 branch only)
DEFAULT_PROMPT = "a person signing in sign language, sharp hands and fingers, studio video, uniform lighting"

# --------------------------------------------------------------------------- skeleton condition ("k")
BODY_BONES = [(12, 15), (12, 16), (12, 17), (16, 18), (18, 20), (17, 19), (19, 21), (9, 12), (9, 13), (9, 14)]
# SMPL-X hand joints: index (+0..2), middle (+3..5), pinky (+6..8), ring (+9..11), thumb (+12..14); tips in the
# extra joints in the order thumb, index, middle, ring, pinky
_FINGERS = [("thumb", 12, 0), ("index", 0, 1), ("middle", 3, 2), ("ring", 9, 3), ("pinky", 6, 4)]


def hand_bones(wrist: int, first: int, tips: int) -> List[tuple]:
    bones = []
    for _, off, tip in _FINGERS:
        chain = [wrist, first + off, first + off + 1, first + off + 2, tips + tip]
        bones += list(zip(chain[:-1], chain[1:]))
    return bones


LHAND_BONES = hand_bones(20, 25, 66)
RHAND_BONES = hand_bones(21, 40, 71)
FACE_POINTS = list(range(76, 144))


def _colour(i: int, n: int) -> tuple:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_skeleton(joints: torch.Tensor, size: int, src_size: int = 256, width: Optional[int] = None) -> torch.Tensor:
    """(J, 2) projected SMPL-X joints in ``src_size`` pixels -> (3, size, size) skeleton image in [0, 1].
    Every bone has its own colour (left / right and fingers distinguishable), face landmarks are white dots."""
    from PIL import Image, ImageDraw
    s = size / float(src_size)
    w = width or max(1, int(round(2 * s)))
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    j = (joints.float() * s).tolist()
    bones = BODY_BONES + LHAND_BONES + RHAND_BONES
    for i, (a, b) in enumerate(bones):
        d.line([tuple(j[a]), tuple(j[b])], fill=_colour(i, len(bones)), width=w)
    for i in (20, 21):
        x, y = j[i]
        d.ellipse([x - w, y - w, x + w, y + w], fill=(255, 255, 255))
    r = max(1, w // 2)
    for i in FACE_POINTS:
        x, y = j[i]
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    import numpy as np
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0


CONDITION_CHANNELS = {"k": 3, "n": 3, "ng": 3 + 3 + 1 + 3}


def build_condition(variant: str, size: int, *, joints=None, normal=None, warped=None, vc=None, ccm=None) -> torch.Tensor:
    """-> (B, C, size, size) in [0, 1] for the ControlNet."""
    up = lambda x, m="bilinear": F.interpolate(x, size=(size, size), mode=m, align_corners=False) if m != "nearest" \
        else F.interpolate(x, size=(size, size), mode="nearest")  # noqa: E731
    if variant == "k":
        return torch.stack([draw_skeleton(j, size) for j in joints.cpu()]).to(joints.device)
    if variant == "n":
        return up(normal)
    if variant == "ng":
        return torch.cat([up(normal), up(warped * vc), up(vc), up(ccm, "nearest")], dim=1)
    raise ValueError(f"unknown condition variant {variant!r}")


# --------------------------------------------------------------------------- reference attention
class ReferenceBank:
    def __init__(self):
        self.mode = "off"            # off | write | read


class RefSelfAttnProcessor:
    """Self-attention that stores its input (``write``) or also attends to the stored tokens (``read``)."""

    def __init__(self, bank: ReferenceBank):
        self.bank = bank
        self.stored: Optional[torch.Tensor] = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kw):
        residual = hidden_states
        nd = hidden_states.ndim
        if nd == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if self.bank.mode == "write":
            self.stored = hidden_states.detach()
        kv = hidden_states
        if self.bank.mode == "read":
            if self.stored is None or self.stored.shape[0] != hidden_states.shape[0]:
                raise RuntimeError("reference bank empty or batch mismatch: run write_reference() first")
            kv = torch.cat([hidden_states, self.stored.to(hidden_states.dtype)], dim=1)
        B = hidden_states.shape[0]
        q, k, v = attn.to_q(hidden_states), attn.to_k(kv), attn.to_v(kv)
        hd = k.shape[-1] // attn.heads
        q, k, v = (t.view(B, -1, attn.heads, hd).transpose(1, 2) for t in (q, k, v))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if nd == 4:
            out = out.transpose(-1, -2).reshape(B, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install_reference_attention(unet) -> ReferenceBank:
    from diffusers.models.attention_processor import AttnProcessor2_0
    bank = ReferenceBank()
    procs = {}
    for name in unet.attn_processors:
        procs[name] = RefSelfAttnProcessor(bank) if name.endswith("attn1.processor") else AttnProcessor2_0()
    unet.set_attn_processor(procs)
    return bank


# --------------------------------------------------------------------------- model
class FrameLDM(nn.Module):
    def __init__(self, vae, unet, controlnet, scheduler, prompt_embeds: torch.Tensor, lora_rank: int = 16):
        super().__init__()
        self.vae, self.unet, self.controlnet, self.scheduler = vae, unet, controlnet, scheduler
        self.scaling = float(vae.config.scaling_factor)
        self.register_buffer("prompt_embeds", prompt_embeds.detach().clone(), persistent=False)
        for p in list(vae.parameters()) + list(unet.parameters()):
            p.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig
            unet.add_adapter(LoraConfig(r=lora_rank, lora_alpha=lora_rank, init_lora_weights="gaussian",
                                        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        self.bank = install_reference_attention(unet)
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod.clone().float(), persistent=False)

    @classmethod
    def from_pretrained(cls, base: str, cond_channels: int, lora_rank: int = 16, prompt: str = None,
                        torch_dtype=torch.float32) -> "FrameLDM":
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import AutoTokenizer, CLIPTextModel
        vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch_dtype)
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch_dtype)
        cn = ControlNetModel.from_unet(unet, conditioning_channels=cond_channels)
        sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
        tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
        with torch.no_grad():
            ids = tok(prompt or DEFAULT_PROMPT, max_length=tok.model_max_length, padding="max_length",
                      truncation=True, return_tensors="pt").input_ids
            emb = te(ids)[0].to(torch_dtype)
        del te
        return cls(vae, unet, cn, sched, emb, lora_rank)

    # ------------------------------------------------------------- parameters
    def trainable_parameters(self) -> List[nn.Parameter]:
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

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        self.controlnet.enable_gradient_checkpointing()

    # ------------------------------------------------------------- pieces
    def emb(self, B: int, dtype) -> torch.Tensor:
        return self.prompt_embeds.to(dtype).expand(B, -1, -1)

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x01 * 2 - 1).latent_dist.mode() * self.scaling

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (self.vae.decode(z / self.scaling).sample.clamp(-1, 1) + 1) / 2

    def write_reference(self, ref_latent: torch.Tensor) -> None:
        self.bank.mode = "write"
        t = torch.zeros(ref_latent.shape[0], device=ref_latent.device, dtype=torch.long)
        with torch.no_grad():
            self.unet(ref_latent, t, encoder_hidden_states=self.emb(ref_latent.shape[0], ref_latent.dtype))
        self.bank.mode = "read"

    def eps(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.emb(z_t.shape[0], z_t.dtype)
        down, mid = self.controlnet(z_t, t, encoder_hidden_states=emb, controlnet_cond=cond.to(z_t.dtype),
                                    return_dict=False)
        return self.unet(z_t, t, encoder_hidden_states=emb, down_block_additional_residuals=down,
                         mid_block_additional_residual=mid).sample

    # ------------------------------------------------------------- training / sampling
    def training_loss(self, target01: torch.Tensor, ref01: torch.Tensor, cond: torch.Tensor,
                      region_mask: Optional[torch.Tensor] = None, w_region: float = 9.0) -> torch.Tensor:
        z0 = self.encode(target01)
        self.write_reference(self.encode(ref01))
        B = z0.shape[0]
        t = torch.randint(0, self.alphas_cumprod.numel(), (B,), device=z0.device)
        noise = torch.randn_like(z0)
        a = self.alphas_cumprod.to(z0.device)[t].view(B, 1, 1, 1)
        z_t = a.sqrt() * z0 + (1 - a).sqrt() * noise
        pred = self.eps(z_t, t, cond)
        self.bank.mode = "off"
        err = ((pred.float() - noise.float()) ** 2).mean(1, keepdim=True)
        if region_mask is None:
            return err.mean()
        w = 1 + w_region * F.adaptive_max_pool2d(region_mask.float(), err.shape[-2:])
        return (err * w).sum() / w.sum()

    @torch.no_grad()
    def sample(self, ref01: torch.Tensor, cond: torch.Tensor, steps: int = 25, noise: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        from diffusers import DDIMScheduler
        sch = DDIMScheduler.from_config(self.scheduler.config)
        sch.set_timesteps(steps, device=ref01.device)
        ref = self.encode(ref01)
        self.write_reference(ref)
        z = noise if noise is not None else torch.randn_like(ref)
        z = z * sch.init_noise_sigma
        for t in sch.timesteps:
            e = self.eps(z, t.expand(z.shape[0]).to(z.device), cond)
            z = sch.step(e, t, z).prev_sample
        self.bank.mode = "off"
        return self.decode(z)
