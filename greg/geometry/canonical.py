"""Canonical coordinate map (CCM): every target pixel carries the T-pose position of the surface point it shows.

Two independent papers use the same idea as a conditioning signal for diffusion models (Mesh4D: positional
embedding from the canonical 3D position shared across frames; RealisDance / Uni3C: SMPL-X rendered with the
canonical coordinates as colours, ``cs_map``), see docs/bai_hoc_diffusion_mesh_cho_greg.md K2.  Compared with
the source (u, v) map of GenWarp it (1) exists for points that are *not* visible in the source (the palm still
has a T-pose position) and (2) gives the same value to the same body point in every frame, which is what a
temporal attention needs to know which tokens are the same physical point.

The coordinates come from ``smplx``'s ``v_template`` (the neutral T-pose), which is the same for every signer, so
a CCM leaks neither identity nor shape.  Values are normalised to [0, 1] by the template's bounding box so the map
can be stored as an 8-bit RGB PNG (1/255 of the box ~ 2-8 mm, enough for a conditioning image).
"""
from __future__ import annotations

import torch


def template_coordinates(body) -> torch.Tensor:
    """(V, 3) T-pose vertex positions of a :class:`greg.render.body.SMPLXBody`, normalised to [0, 1]."""
    vt = body.model.v_template.detach().float()
    lo, hi = vt.min(0).values, vt.max(0).values
    return (vt - lo) / (hi - lo).clamp(min=1e-6)


def canonical_coordinate_map(faces_t: torch.Tensor, bary_t: torch.Tensor, faces: torch.Tensor,
                             coords: torch.Tensor) -> torch.Tensor:
    """(B, H, W) long face map (-1 = background), (B, H, W, 3) barycentrics, (F, 3) topology, (V, 3) coordinates
    -> (B, 3, H, W) CCM, 0 on the background."""
    B, H, W = faces_t.shape
    fa = coords.to(bary_t.dtype)[faces.to(coords.device)]                  # (F, 3, 3) per-face vertex coordinates
    picked = fa.to(faces_t.device)[faces_t.clamp(min=0)]                   # (B, H, W, 3, 3)
    out = (picked * bary_t.to(picked.dtype)[..., None]).sum(-2)             # (B, H, W, 3)
    out = torch.where((faces_t >= 0)[..., None], out, torch.zeros_like(out))
    return out.permute(0, 3, 1, 2).contiguous()


def ccm_from_geometry(geometry: dict, faces_t: torch.Tensor, faces: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Convenience: CCM from the output of :func:`greg.geometry.mesh_flow.correspond` (``bary_target``)."""
    return canonical_coordinate_map(faces_t.long(), geometry["bary_target"], faces, coords)
