"""The full GReg model (Fig. 3) with the two improvements A1 / B1 behind config switches.

forward(I_s, N_s, N_t, centers_t, [geometry], [joints_t], [parts_t]) ->
    flow:   model.flow.type = tpsmm  -> frozen TPSMM on normals -> F, O              (the paper)
            model.flow.type = mesh   -> F, O_geo = V * C from the SMPL-X meshes        (A1, greg.geometry.mesh_flow)
                                        O = sigmoid(logit O_geo + OcclusionRefiner)   (optional, residual)
    I_st = warp(I_s, F)
    boxes:  regions.mode = fixed     -> region_size squares around the centres         (the paper)
            regions.mode = adaptive  -> hands: squares fitted to the hand joints        (B1)
    R    = RPG(crop(N_t), style(I_s))              (regional priors, Sec. 3.2)
    G_R  = Gaussians(centers) * grad(N_t)          (Eq. 1-3)
    X_g  = GradientEncoder(G_R)
    I_gen = Generator(cat[I_st, N_t, R]; GRM(X_g))
    I_hat = O * I_st + (1 - O) * I_gen
    region_masks = face / hand pixels of the target inside each box (B1, used by the regional losses)

Call sites pass the optional inputs with ``model(..., **model.extra_inputs(batch))`` so that the same code
runs every configuration.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from greg.data.gradient import gradient_map
from greg.data.regions import (NUM_REGIONS, adaptive_region_boxes, crop_regions, region_boxes, region_gaussians,
                               region_masks, weighted_region_gradients)
from greg.models.generator import GeometryAwareGenerator
from greg.models.gradient_refine import GradientEncoder
from greg.models.regional_expert import RegionalPriorGenerator
from greg.models.style import StyleEncoder
from greg.models.tpsmm import FlowEstimator, resize_occlusion, warp
from greg.render.body import region_extents_from_joints

MESH_BATCH_FIELDS = ("smplx", "focal", "princpt", "faces")


class OcclusionRefiner(nn.Module):
    """A1: small full-resolution residual on the geometric occlusion O_geo = V * C.

    The mesh has no clothes or hair: right next to the silhouette V is sometimes wrong (a loose sleeve).  This
    net sees the warped source, the target normals and O_geo and predicts a logit residual; its last conv is
    zero-initialised, so training starts from O = O_geo exactly (up to ``clamp``)."""

    def __init__(self, in_channels: int = 7, hidden: int = 32, clamp: float = 0.02):
        super().__init__()
        self.clamp = float(clamp)
        act = nn.LeakyReLU(0.2)
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), act,
                                 nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), act,
                                 nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4), act,
                                 nn.Conv2d(hidden, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, warped: torch.Tensor, normal_target: torch.Tensor, occ_geo: torch.Tensor) -> torch.Tensor:
        logit = torch.logit(occ_geo.clamp(self.clamp, 1 - self.clamp))
        return torch.sigmoid(logit + self.net(torch.cat([warped, normal_target, occ_geo], dim=1)))


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
        self.flow_cfg = dict(flow_cfg)
        self.flow_type = str(flow_cfg.get("type", "tpsmm"))
        if self.flow_type not in ("tpsmm", "mesh"):
            raise ValueError(f"model.flow.type must be tpsmm or mesh, got {self.flow_type!r}")
        if self.flow_type == "tpsmm":
            self.flow = FlowEstimator(num_tps=int(flow_cfg.get("num_tps", 10)), num_channels=3,
                                      dense_motion=flow_cfg.get("dense_motion", {}),
                                      multi_mask=bool(flow_cfg.get("multi_mask", False)))
        else:
            self.flow = None          # A1: the flow is computed from the meshes, nothing to learn or load
        self.occlusion_refiner = OcclusionRefiner(clamp=float(m.get("occlusion_clamp", 0.02))) \
            if self.flow_type == "mesh" and bool(m.get("occlusion_refine", False)) else None
        self._mesh = None             # MeshCorrespondence, built lazily (needs smplx + the SMPL-X files)

        rc = m.get("regions", {}) or {}
        self.region_mode = str(rc.get("mode", "fixed"))
        if self.region_mode not in ("fixed", "adaptive"):
            raise ValueError(f"model.regions.mode must be fixed or adaptive, got {self.region_mode!r}")
        self.hand_margin = float(rc.get("hand_margin", 0.25))
        self.min_side = float(rc.get("min_side", 24))
        self.max_side = float(rc.get("max_side", 96))
        self.adaptive_face = bool(rc.get("adaptive_face", False))
        self.mask_dilate_px = int(rc.get("mask_dilate_px", 3))

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
        if self.flow is not None:
            self.flow.freeze()
        self.flow_frozen = True
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.flow_frozen and self.flow is not None:
            self.flow.eval()
        return self

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if self.flow_frozen and name.startswith("flow."):
                continue
            yield p

    # ----------------------------------------------------------------- mesh geometry (A1)
    @property
    def uses_mesh(self) -> bool:
        return self.flow_type == "mesh"

    def mesh_correspondence(self):
        """The (lazily built) :class:`greg.geometry.mesh_flow.MeshCorrespondence` on the model's device."""
        dev = next(self.parameters()).device
        if self._mesh is None or self._mesh.device != dev:
            from pathlib import Path
            from greg.geometry.mesh_flow import MeshCorrespondence
            c = self.flow_cfg
            if not c.get("smplx_dir"):
                raise ValueError("model.flow.smplx_dir is required with model.flow.type=mesh")
            if not (Path(c["smplx_dir"]) / "smplx" / "SMPLX_NEUTRAL.npz").exists():
                raise FileNotFoundError(f"model.flow.smplx_dir={c['smplx_dir']!r} has no smplx/SMPLX_NEUTRAL.npz "
                                        "(on vast.ai: --override model.flow.smplx_dir=/workspace/data/raw/smplx_dl/models)")
            self._mesh = MeshCorrespondence(c["smplx_dir"], device=dev, eps_depth=float(c.get("eps_depth", 0.01)),
                                            bg_dilate_px=int(c.get("bg_dilate_px", 4)),
                                            stretch_tau=float(c.get("stretch_tau", 0.5)))
        return self._mesh

    @torch.no_grad()
    def geometry(self, batch: dict) -> Optional[Dict[str, torch.Tensor]]:
        """Mesh correspondence of a batch carrying the ``source_*`` / ``target_*`` mesh fields
        (``HamNoSysCache.load_row(with_mesh=True)``), or None for the TPSMM flow."""
        if not self.uses_mesh:
            return None
        g = {f"{side}_{f}": batch[f"{side}_{f}"] for side in ("source", "target") for f in MESH_BATCH_FIELDS}
        return self.mesh_correspondence()(g["source_smplx"], g["source_focal"], g["source_princpt"], g["source_faces"],
                                          g["target_smplx"], g["target_focal"], g["target_princpt"], g["target_faces"])

    def extra_inputs(self, batch: dict) -> dict:
        """Keyword arguments for :meth:`forward` from a dataset batch: mesh geometry (A1), target joints for the
        adaptive boxes and the target part map for the region masks (B1)."""
        dev = next(self.parameters()).device
        out = {"geometry": self.geometry(batch)}
        if "target_joints" in batch:
            out["joints_target"] = batch["target_joints"].to(dev)
        if "target_parts" in batch:
            out["parts_target"] = batch["target_parts"].to(dev)
        return out

    # --------------------------------------------------------------------- helpers
    def boxes_from_centers(self, centers: torch.Tensor, joints: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.region_mode == "adaptive":
            if joints is None:
                raise ValueError("regions.mode=adaptive needs the target joints (model(..., joints_target=...))")
            return adaptive_region_boxes(centers, region_extents_from_joints(joints), self.region_size,
                                         self.image_size, self.image_size, self.hand_margin, self.min_side,
                                         self.max_side, self.adaptive_face)
        return region_boxes(centers, self.region_size, self.image_size, self.image_size)

    def region_gradient_input(self, normal_target: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """G_R (B, 3, H, W): Gaussian-weighted gradient of the target normal map (Eq. 1-3)."""
        g = gradient_map(normal_target)
        w = region_gaussians(centers, self.image_size, self.image_size, self.sigma)
        return weighted_region_gradients(g, w)

    # --------------------------------------------------------------------- forward
    def forward(self, image_source: torch.Tensor, normal_source: torch.Tensor, normal_target: torch.Tensor,
                centers_target: torch.Tensor, flow_out: Optional[Dict[str, torch.Tensor]] = None,
                geometry: Optional[Dict[str, torch.Tensor]] = None, joints_target: Optional[torch.Tensor] = None,
                parts_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        H = W = self.image_size
        extra: Dict[str, torch.Tensor] = {}
        if self.uses_mesh:
            if geometry is None:
                raise ValueError("model.flow.type=mesh needs geometry=model.geometry(batch)")
            deformation = geometry["grid"]
            occ_geo = geometry["occlusion"]
            warped = warp(image_source, deformation)
            occlusion = occ_geo if self.occlusion_refiner is None else \
                self.occlusion_refiner(warped, normal_target, occ_geo)
            extra = {"occlusion_geo": occ_geo, "visibility": geometry["visibility"],
                     "confidence": geometry["confidence"]}
            if parts_target is None:
                parts_target = geometry.get("parts")
        else:
            if flow_out is None:
                if self.flow_frozen:
                    with torch.no_grad():
                        flow_out = self.flow(normal_source, normal_target)
                else:
                    flow_out = self.flow(normal_source, normal_target)
            deformation = flow_out["deformation"]
            occlusion = resize_occlusion(flow_out["occlusion"], (H, W))
            warped = warp(image_source, deformation)

        boxes = self.boxes_from_centers(centers_target, joints_target)
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

        out = {"output": output, "generated": generated, "warped": warped, "occlusion": occlusion,
               "deformation": deformation, "prior_map": prior, "prior_patches": rpg["prior_patches"],
               "normal_crops": rpg["normal_crops"], "boxes": boxes, "region_gradients": g_r,
               "style": style, **extra}
        if parts_target is not None:
            out["parts_target"] = parts_target
            out["region_masks"] = region_masks(parts_target, boxes, self.region_size, self.mask_dilate_px)
        return out

    def crop(self, x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return crop_regions(x, boxes, self.region_size)
