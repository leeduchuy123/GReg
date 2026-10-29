"""Normal-map rendering of SMPL-X meshes with the SignAvatars camera.

Outputs, per frame:

* ``normal``  (3, H, W) float in [0, 1]: camera-space unit normal encoded as
  ``(n + 1) / 2`` (background = 0.5 gray would hide edges, so background is 0).
* ``mask``    (1, H, W) float: 1 where the body covers the pixel.
* ``part``    (H, W) uint8: 0 bg, 1 body, 2 face, 3 left hand, 4 right hand.
* ``depth``   (H, W) float.

Normals are *vertex* normals (area weighted) interpolated with barycentrics,
which gives the smooth shading seen in the paper's Fig. 1.  Camera axes are
OpenCV (z forward); visible surfaces therefore have ``n_z < 0`` and the
encoded normal map of a front-facing flat region is ``(0.5, 0.5, 0)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from greg.render.body import Camera, SMPLXBody, project, to_camera
from greg.render.rasterize import Fragments, face_attribute_map, interpolate_vertex_attribute, rasterize


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals. verts (B, V, 3), faces (F, 3) -> (B, V, 3) unit."""
    v0, v1, v2 = verts[:, faces[:, 0]], verts[:, faces[:, 1]], verts[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)              # (B, F, 3), |fn| = 2*area
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(1, faces[:, k], fn)
    return torch.nn.functional.normalize(vn, dim=-1, eps=1e-8)


def face_parts(faces: torch.Tensor, vertex_part: torch.Tensor) -> torch.Tensor:
    """(F, 3) faces + (V,) vertex labels -> (F,) uint8 per-face label (majority of the 3 vertices)."""
    return torch.mode(vertex_part[faces], dim=1).values.to(torch.uint8)


@dataclass
class RenderOutput:
    normal: torch.Tensor   # (B, 3, H, W) in [0, 1]
    mask: torch.Tensor     # (B, 1, H, W)
    part: torch.Tensor     # (B, H, W) uint8
    depth: torch.Tensor    # (B, H, W)
    joints_2d: torch.Tensor  # (B, J, 2) projected joints in the output image
    frag: Optional[Fragments] = None


class NormalRenderer:
    def __init__(self, body: SMPLXBody, backend: str = "torch", tile: int = 32):
        self.body = body
        self.backend = backend
        self.tile = tile
        self.faces = body.faces
        # per-face part label = majority of the 3 vertices (hands/face are contiguous)
        self.face_part = face_parts(self.faces, body.vertex_part)

    @torch.no_grad()
    def render(self, smplx_vec: torch.Tensor, cam: Camera, height: int, width: int,
               betas_override: Optional[torch.Tensor] = None, keep_fragments: bool = False
               ) -> RenderOutput:
        """Render (B, 182) parameter vectors with per-frame cameras at ``height x width``."""
        dev = self.body.device
        cam = cam.to(dev)
        out = self.body(smplx_vec, betas_override=betas_override)
        verts_cam = to_camera(out["vertices"], cam)          # (B, V, 3)
        joints_2d = project(out["joints"], cam)
        uv = project(out["vertices"], cam)
        z = verts_cam[..., 2]
        frag = rasterize(uv, z, self.faces, height, width, backend=self.backend, tile=self.tile)
        vn = vertex_normals(verts_cam, self.faces)
        n = interpolate_vertex_attribute(frag, vn, self.faces)             # (B, 3, H, W)
        n = torch.nn.functional.normalize(n, dim=1, eps=1e-8)
        mask = frag.mask[:, None].to(n.dtype)
        normal = ((n + 1) * 0.5) * mask
        part = face_attribute_map(frag, self.face_part, fill=0)
        return RenderOutput(normal=normal, mask=mask, part=part, depth=frag.depth,
                            joints_2d=joints_2d, frag=frag if keep_fragments else None)


def encode_normal_png(normal: torch.Tensor, mask: torch.Tensor):
    """(3, H, W) [0,1] + (1, H, W) -> (H, W, 4) uint8 RGBA numpy for lossless PNG storage."""
    import numpy as np
    rgb = (normal.clamp(0, 1) * 255).round().to(torch.uint8)
    a = (mask.clamp(0, 1) * 255).round().to(torch.uint8)
    return torch.cat([rgb, a], 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def decode_normal_png(rgba) -> tuple:
    """(H, W, 4) uint8 -> (normal (3, H, W) float [0,1], mask (1, H, W) float)."""
    t = torch.as_tensor(rgba).permute(2, 0, 1).float() / 255.0
    return t[:3], t[3:4]
