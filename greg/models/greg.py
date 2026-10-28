"""The full GReg model (Fig. 3).

forward(I_s, N_s, N_t, centers_t) ->
    flow (frozen TPSMM on normals)  -> F, O
    I_st = warp(I_s, F)
    R    = RPG(crop(N_t), style(I_s))              (regional priors, Sec. 3.2)
    G_R  = Gaussians(centers) * grad(N_t)          (Eq. 1-3)
    X_g  = GradientEncoder(G_R)
    I_gen = Generator(cat[I_st, N_t, R]; GRM(X_g))
    I_hat = O * I_st + (1 - O) * I_gen
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from greg.data.gradient import gradient_map
from greg.data.regions import (NUM_REGIONS, crop_regions, region_boxes, region_gaussians,
                               weighted_region_gradients)
from greg.models.generator import GeometryAwareGenerator
from greg.models.gradient_refine import GradientEncoder
from greg.models.regional_expert import RegionalPriorGenerator
from greg.models.style import StyleEncoder
from greg.models.tpsmm import FlowEstimator, resize_occlusion, warp


class GReg(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg
        self.image_size = int(m.get("image_size", 256))
        self.region_size = int(m.get("region_size", 64))
        self.sigma = float(m.get("gaussian_sigma", self.region_size / 2))
        self.detach_prior = bool(m.get("detach_prior", True))
        style_dim = int(m.get("style_dim", 256))

        flow_cfg = m.get("flow", {})
        self.flow = FlowEstimator(num_tps=int(flow_cfg.get("num_tps", 10)), num_channels=3,
                                  dense_motion=flow_cfg.get("dense_motion", {}),
                                  multi_mask=bool(flow_cfg.get("multi_mask", False)))
        self.style_encoder = StyleEncoder(3, style_dim, base=int(m.get("style_base", 32)))
        rpg = m.get("rpg", {})
        self.rpg = RegionalPriorGenerator(style_dim, self.region_size, out_channels=1,
                                          channels=tuple(rpg.get("channels", (64, 128, 256, 256))),
                                          share_hand_expert=bool(rpg.get("share_hand_expert", True)))
        gr = m.get("gr", {})
        levels = tuple(int(x) for x in gr.get("grm_levels", (32, 64, 128)))
        self.gradient_encoder = GradientEncoder(NUM_REGIONS, base=int(gr.get("base", 32)),
                                                max_channels=int(gr.get("max_channels", 256)),
                                                image_size=self.image_size, levels=levels)
        gen = m.get("generator", {})
        self.generator = GeometryAwareGenerator(
            in_channels=9, out_channels=3, base=int(gen.get("base", 64)),
            max_channels=int(gen.get("max_channels", 512)), num_down=int(gen.get("num_down", 4)),
            image_size=self.image_size, grm_levels=levels, grad_channels=self.gradient_encoder.channels,
            grm_heads=int(gr.get("heads", 1)), norm=gen.get("norm", "batch"),
            zero_init_grm=bool(gr.get("zero_init_out", True)), grm_attn_bf16=bool(gr.get("attn_bf16", True)),
            grm_qk_norm=bool(gr.get("qk_norm", False)))
        self.flow_frozen = False

    # ----------------------------------------------------------------- flow control
    def freeze_flow(self):
        self.flow.freeze()
        self.flow_frozen = True
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.flow_frozen:
            self.flow.eval()
        return self

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if self.flow_frozen and name.startswith("flow."):
                continue
            yield p

    # --------------------------------------------------------------------- helpers
    def boxes_from_centers(self, centers: torch.Tensor) -> torch.Tensor:
        return region_boxes(centers, self.region_size, self.image_size, self.image_size)

    def region_gradient_input(self, normal_target: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """G_R (B, 3, H, W): Gaussian-weighted gradient of the target normal map (Eq. 1-3)."""
        g = gradient_map(normal_target)
        w = region_gaussians(centers, self.image_size, self.image_size, self.sigma)
        return weighted_region_gradients(g, w)

    # --------------------------------------------------------------------- forward
    def forward(self, image_source: torch.Tensor, normal_source: torch.Tensor, normal_target: torch.Tensor,
                centers_target: torch.Tensor, flow_out: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        H = W = self.image_size
        if flow_out is None:
            if self.flow_frozen:
                with torch.no_grad():
                    flow_out = self.flow(normal_source, normal_target)
            else:
                flow_out = self.flow(normal_source, normal_target)
        deformation = flow_out["deformation"]
        occlusion = resize_occlusion(flow_out["occlusion"], (H, W))
        warped = warp(image_source, deformation)

        boxes = self.boxes_from_centers(centers_target)
        style = self.style_encoder(image_source)
        rpg = self.rpg(normal_target, boxes, style)
        prior = rpg["prior_map"]
        prior_in = prior.detach() if self.detach_prior else prior
        if getattr(self, "ablation_no_rpg", False):   # Table 2 "w/o RPG": generator gets no prior
            prior_in = torch.zeros_like(prior_in)

        g_r = self.region_gradient_input(normal_target, centers_target)
        grad_feats = self.gradient_encoder(g_r)
        gen_in = torch.cat([warped, normal_target, prior_in], dim=1)
        generated = self.generator(gen_in, grad_feats)
        output = occlusion * warped + (1 - occlusion) * generated

        return {"output": output, "generated": generated, "warped": warped, "occlusion": occlusion,
                "deformation": deformation, "prior_map": prior, "prior_patches": rpg["prior_patches"],
                "normal_crops": rpg["normal_crops"], "boxes": boxes, "region_gradients": g_r,
                "style": style}

    def crop(self, x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return crop_regions(x, boxes, self.region_size)
