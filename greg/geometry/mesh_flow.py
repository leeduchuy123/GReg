"""Analytic source <-> target correspondence from the SMPL-X meshes (improvement A1).

GReg's TPSMM estimator *learns* a 2D warp from the two normal maps.  A smooth 2D warp cannot fold a forearm
over the torso (it stretches the sleeve instead) and its 64x64 occlusion mask is soft.  Here the warp is
*computed* instead: both frames are the same mesh topology, so a target pixel that shows face ``f`` at
barycentrics ``b`` shows the same surface point as ``sum_k b_k * uv_s[f_k]`` in the source.

For every target pixel ``p`` (all at the cache resolution, pixel centres at integer coordinates, the
rasteriser's convention):

* body pixel (``face_t(p) = f >= 0``)::

      b      = barycentrics of p in the target triangle uv_t[f]          (2D, analytic)
      q      = sum_k b_k * uv_s[f_k]                                       (where it is in the source)
      z_q    = sum_k b_k *  z_s[f_k]                                       (its source camera depth)
      V(p)   = q inside the frame
               and face f is front-facing in the source too (same 2D winding as in the target: a triangle the
                   target sees but the source sees from behind -- the palm of a hand that was turned towards the
                   thigh -- can never be visible; the depth test alone misses this for 1-3 cm thick fingers)
               and the source pixel that will be sampled, round(q), shows the point's own surface patch: face f or
                   a face sharing a vertex with f (a finger hidden behind another finger 5 mm closer, or an arm
                   point that lands on the torso right next to the arm's silhouette, is rejected here; a pure
                   depth threshold cannot separate them)
               and depth_s(round(q)) >= z_q - eps_depth (nothing clearly in front of it)
           Errs on the side of V = 0: a wrongly hidden pixel is generated, a wrongly visible one copies the wrong
           surface (ghosts), which is the failure being fixed.  On a real SMPL-X pair (arm folded up, see
           docs/A1_B1_kien_truc.md Sec. 9) this keeps 94% of the body visible with a warp error of 0.0016.

* background pixel (``face_t(p) = -1``): the camera is static, so ``q = p``; visible iff the source mesh does
  not cover p either AND, when p lies within ``bg_dilate_px`` of the source body, it also lies within that
  distance of the target body.  The mesh is thinner than the clothed body: next to a limb that moved away the
  source still shows the sleeve, so copying it would leave the old sleeve outline in the target; next to a
  part that did not move (both silhouettes close) the source pixel is the right one and stays visible, so
  static silhouettes are not handed to the generator.

* ``C(p) = clamp(|det dq/dp| / stretch_tau, 0, 1)`` on interior body pixels: a target pixel that samples less
  than ``stretch_tau`` source pixels is magnified (e.g. a hand seen edge-on in the source, palm-on in the
  target) and will be blurry, so it gets less trust.

Outputs ``grid`` (B, H, W, 2) in [-1, 1] for ``grid_sample(align_corners=True)`` (``greg.models.tpsmm.warp``),
``visibility`` V, ``confidence`` C and ``occlusion = V * C`` (B, 1, H, W), plus the extended part map of the
target (0 bg, 1 body, 2 face, 3 lhand, 4 rhand, 5 larm, 6 rarm).

:func:`correspond` is pure torch (no SMPL-X needed, unit-tested on synthetic meshes);
:class:`MeshCorrespondence` adds the SMPL-X forward pass that turns the cached parameter vectors into
projected vertices with exactly the camera used to render the normal maps.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from greg.render.body import Camera, SMPLXBody, project, to_camera


def barycentric_2d(p: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """p (..., 2), tri (..., 3, 2) -> (..., 3) barycentric coordinates (unclamped)."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return torch.stack([1 - v - w, v, w], dim=-1)


def signed_area_2d(tri: torch.Tensor) -> torch.Tensor:
    """tri (..., 3, 2) -> (...) twice the signed 2D area (sign = winding = which side faces the camera)."""
    e1, e2 = tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]
    return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]


def _clamped_barycentric(p: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    b = barycentric_2d(p, tri).clamp(0, 1)
    return b / b.sum(-1, keepdim=True).clamp(min=1e-8)


def gather_triangles(uvz: torch.Tensor, faces: torch.Tensor, face_map: torch.Tensor) -> torch.Tensor:
    """uvz (B, V, 3), faces (F, 3), face_map (B, H, W) long (-1 = none) -> (B, H, W, 3, 3) vertex (u, v, z)."""
    B = uvz.shape[0]
    vids = faces[face_map.clamp(min=0)]                                    # (B, H, W, 3)
    b = torch.arange(B, device=uvz.device).view(B, 1, 1, 1)
    return uvz[b, vids]


def pixel_grid(height: int, width: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """(H, W, 2) pixel-centre coordinates (x, y) in the rasteriser convention."""
    ys = torch.arange(height, dtype=dtype, device=device)
    xs = torch.arange(width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def pixels_to_grid(q: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pixel coordinates (..., 2) -> normalised [-1, 1] coordinates for ``grid_sample(align_corners=True)``."""
    return torch.stack([2 * q[..., 0] / (width - 1) - 1, 2 * q[..., 1] / (height - 1) - 1], dim=-1)


def _erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    """(B, H, W) bool -> bool, True where the whole (2r+1)^2 neighbourhood is True."""
    m = mask.float()[:, None]
    return (-F.max_pool2d(-m, 2 * r + 1, 1, r))[:, 0] > 0.5


def stretch_confidence(q: torch.Tensor, body: torch.Tensor, tau: float) -> torch.Tensor:
    """Area ratio |det dq/dp| (source pixels per target pixel) -> confidence in [0, 1].

    q (B, H, W, 2) source pixel coordinates, body (B, H, W) bool.  Only interior body pixels are rated
    (a 3x3 neighbourhood fully on the body); everything else -- background, silhouettes, occlusion edges
    where q jumps -- keeps confidence 1 (V already handles those)."""
    if tau <= 0:
        return torch.ones_like(q[..., 0])
    qp = F.pad(q.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")    # (B, 2, H+2, W+2)
    dx = (qp[:, :, 1:-1, 2:] - qp[:, :, 1:-1, :-2]) / 2
    dy = (qp[:, :, 2:, 1:-1] - qp[:, :, :-2, 1:-1]) / 2
    det = (dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]).abs()
    conf = (det / tau).clamp(0, 1)
    return torch.where(_erode(body, 1), conf, torch.ones_like(conf))


def segment_depth(seg: torch.Tensor, max_px: int) -> torch.Tensor:
    """(B, H, W) integer label map -> (B, H, W) float: how many pixels (capped at ``max_px``) the pixel lies inside
    its own segment, i.e. the largest r <= max_px such that the whole (2r+1)^2 window has its label.  Background
    (label 0) gets 0; a boundary with the background or with another part both count as an edge."""
    s = seg.float()[:, None]
    depth = torch.zeros_like(s)
    for r in range(1, max_px + 1):
        k = 2 * r + 1
        same = F.max_pool2d(s, k, 1, r) == -F.max_pool2d(-s, k, 1, r)                # window has one label only
        depth += (same & (s > 0)).float()
    return depth[:, 0]


def edge_confidence(q: torch.Tensor, p: torch.Tensor, seg_s: torch.Tensor, labels, edge_px: int,
                    move_px: float = 1.0, move_ramp_px: float = 2.0) -> torch.Tensor:
    """Confidence of sampling the source at q, given how close q is to the edge of its limb segment.

    E0 on real data: a hanging forearm is only 10-15 px wide next to the (often dark) shirt, and the SMPL-X fit is
    off by a few px, so mesh-warped forearm pixels sampled near the limb edge take the shirt / background colour
    (L1 0.139 -> 0.095 when only samples >= 5 px inside the limb are counted).  For samples whose source segment
    is in ``labels`` (hands, arms): conf = depth(q) / edge_px, applied only where the point moved (|q - p| beyond
    ``move_px``, fully after ``move_px + move_ramp_px``): where nothing moved, q = p and copying is exact whatever
    the fit error.  Returns (B, H, W) in [0, 1]; 1 everywhere when ``edge_px <= 0``."""
    if edge_px <= 0:
        return torch.ones_like(q[..., 0])
    B, H, W = seg_s.shape
    depth = segment_depth(seg_s, edge_px)
    qx = q[..., 0].round().clamp(0, W - 1).long()
    qy = q[..., 1].round().clamp(0, H - 1).long()
    b = torch.arange(B, device=q.device).view(B, 1, 1)
    d_q, lab_q = depth[b, qy, qx], seg_s[b, qy, qx]
    limb = torch.zeros_like(lab_q, dtype=torch.bool)
    for lab in labels:
        limb |= lab_q == int(lab)
    near_edge = 1 - (d_q / edge_px).clamp(0, 1)
    moved = ((q - p).norm(dim=-1) - move_px).clamp(min=0) / max(move_ramp_px, 1e-6)
    return torch.where(limb, 1 - moved.clamp(0, 1) * near_edge, torch.ones_like(near_edge))


def correspond(uvz_s: torch.Tensor, faces_s: torch.Tensor, uvz_t: torch.Tensor, faces_t: torch.Tensor,
               faces: torch.Tensor, face_part: Optional[torch.Tensor] = None, eps_depth: float = 0.01,
               bg_dilate_px: int = 4, stretch_tau: float = 0.5, edge_px: int = 0,
               edge_labels=(3, 4, 5, 6)) -> Dict[str, torch.Tensor]:
    """Target -> source correspondence of two renders of the same mesh topology.

    Args:
        uvz_s, uvz_t: (B, V, 3) projected vertices (u, v in pixels, z = camera depth) of source / target.
        faces_s, faces_t: (B, H, W) long rasterised face index (-1 = background) of source / target.
        faces: (F, 3) long mesh topology.  face_part: optional (F,) per-face labels -> ``parts``.
        edge_px, edge_labels: confidence ramp near the edges of the moved source limbs (``edge_confidence``);
            needs ``face_part``.  0 = off.
    """
    B, H, W = faces_t.shape
    dev, dt = uvz_t.device, uvz_t.dtype
    p = pixel_grid(H, W, dt, dev).expand(B, H, W, 2)
    body_t = faces_t >= 0

    # where every target body pixel is in the source (same face, same barycentrics)
    tri_t = gather_triangles(uvz_t, faces, faces_t)
    bary = _clamped_barycentric(p, tri_t[..., :2])
    tri_s = gather_triangles(uvz_s, faces, faces_t)
    q_body = (bary[..., None] * tri_s[..., :2]).sum(-2)
    z_q = (bary * tri_s[..., 2]).sum(-1)
    q = torch.where(body_t[..., None], q_body, p)

    # the target sees face f from the front; in the source f must face the camera as well
    front_s = signed_area_2d(tri_s[..., :2]) * torch.sign(signed_area_2d(tri_t[..., :2])) > 0

    # is that point the visible surface of the source at q? (z-buffer test against the source render)
    inside = (q[..., 0] >= -0.5) & (q[..., 0] <= W - 0.5) & (q[..., 1] >= -0.5) & (q[..., 1] <= H - 0.5)
    qx = q[..., 0].round().clamp(0, W - 1).long()
    qy = q[..., 1].round().clamp(0, H - 1).long()
    b = torch.arange(B, device=dev).view(B, 1, 1)
    fs = faces_s[b, qy, qx]
    tri_sq = gather_triangles(uvz_s, faces, fs)
    bq = _clamped_barycentric(q, tri_sq[..., :2])
    depth_s = (bq * tri_sq[..., 2]).sum(-1)
    # does the source pixel we will sample show the point's own surface patch (f or a face sharing a vertex)?
    vf_t = faces[faces_t.clamp(min=0)]                                                # (B, H, W, 3)
    vf_s = faces[fs.clamp(min=0)]
    patch_seen = (fs >= 0) & (vf_s[..., :, None] == vf_t[..., None, :]).flatten(-2).any(-1)
    # silhouette rounding: a target silhouette pixel whose round(q) fell just outside the source coverage is
    # accepted if its patch is right next to q (static outline).  An interior target pixel landing outside the
    # source coverage is a surface seen edge-on in the source (a turned forearm): sampling there reads the
    # background, so it stays hidden.
    edge = (fs < 0) & body_t & ~_erode(body_t, 1)
    if bool(edge.any()):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                fn = faces_s[b, (qy + dy).clamp(0, H - 1), (qx + dx).clamp(0, W - 1)]
                vf_n = faces[fn.clamp(min=0)]
                patch_seen |= edge & (fn >= 0) & (vf_n[..., :, None] == vf_t[..., None, :]).flatten(-2).any(-1)
    vis_body = inside & front_s & patch_seen & (edge | (depth_s >= z_q - eps_depth))

    # background: visible where the source mesh does not cover p and no limb moved away from next to p
    src_body = faces_s >= 0
    vis_bg = ~src_body
    if bg_dilate_px > 0:
        k = 2 * bg_dilate_px + 1
        near_src = F.max_pool2d(src_body.float()[:, None], k, 1, bg_dilate_px)[:, 0] > 0.5
        near_tgt = F.max_pool2d(body_t.float()[:, None], k, 1, bg_dilate_px)[:, 0] > 0.5
        vis_bg = vis_bg & (~near_src | near_tgt)

    vis = torch.where(body_t, vis_body, vis_bg)
    # single-pixel holes are not disocclusions: SMPL-X faces are smaller than a pixel on the fingers / face
    # (round(q) then hits a face two rings away) and the outline shifts by ~1 px when a limb moves (skinning);
    # a hidden pixel with >= 7 visible neighbours is made visible (body pixels only if q is a valid sample)
    v = vis.float()[:, None]
    neigh = (F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, 1) * 9 - v)[:, 0]
    fillable = torch.where(body_t, inside & front_s, torch.ones_like(body_t))
    vis = (vis | ((neigh >= 7) & fillable)).to(dt)
    conf = stretch_confidence(q, body_t, stretch_tau)
    if edge_px > 0:
        if face_part is None:
            raise ValueError("edge_px > 0 needs face_part (the source limb segments)")
        fp_s = face_part.to(dev)[faces_s.clamp(min=0)].long()
        seg_s = torch.where(faces_s >= 0, fp_s, torch.zeros_like(fp_s))
        conf_edge = torch.where(body_t, edge_confidence(q, p, seg_s, edge_labels, edge_px), torch.ones_like(conf))
        conf = conf * conf_edge
    conf = conf.to(dt)
    out = {"grid": pixels_to_grid(q, H, W), "source_coords": q, "visibility": vis[:, None],
           "confidence": conf[:, None], "occlusion": (vis * conf)[:, None], "target_body": body_t[:, None].to(dt)}
    if face_part is not None:
        fp = face_part.to(dev)[faces_t.clamp(min=0)].long()
        out["parts"] = torch.where(body_t, fp, torch.zeros_like(fp))
    return out


class MeshCorrespondence:
    """SMPL-X forward + :func:`correspond`.  Not an ``nn.Module``: it has no learnable parameters and is
    never saved in a checkpoint (built from ``model.flow`` config, see ``GReg.mesh_correspondence``)."""

    def __init__(self, smplx_dir: str, device="cpu", eps_depth: float = 0.01, bg_dilate_px: int = 4,
                 stretch_tau: float = 0.5, edge_px: int = 0):
        self.body = SMPLXBody(smplx_dir, device=device)
        self.device = self.body.device
        self.faces = self.body.faces
        from greg.render.normals import face_parts
        self.face_part = face_parts(self.faces, self.body.vertex_part_arms)
        self.eps_depth, self.bg_dilate_px, self.stretch_tau = float(eps_depth), int(bg_dilate_px), float(stretch_tau)
        self.edge_px = int(edge_px)

    @torch.no_grad()
    def vertices(self, smplx_vec: torch.Tensor, focal: torch.Tensor, princpt: torch.Tensor,
                 betas_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, 182) SignAvatars vectors + crop intrinsics -> (B, V, 3) projected (u, v, z), same camera as
        ``NormalRenderer.render`` (cam_trans comes from the vector)."""
        out = self.body(smplx_vec.to(self.device), betas_override=betas_override)
        cam = Camera(focal.to(self.device).float(), princpt.to(self.device).float(), out["cam_trans"].float())
        uv = project(out["vertices"], cam)
        z = to_camera(out["vertices"], cam)[..., 2:3]
        return torch.cat([uv, z], dim=-1)

    @torch.no_grad()
    def __call__(self, smplx_s, focal_s, princpt_s, faces_s, smplx_t, focal_t, princpt_t, faces_t,
                 betas_override_t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if betas_override_t is None:      # one SMPL-X pass for both frames
            uvz = self.vertices(torch.cat([smplx_s, smplx_t]), torch.cat([focal_s, focal_t]),
                                torch.cat([princpt_s, princpt_t]))
            uvz_s, uvz_t = uvz.chunk(2)
        else:
            uvz_s = self.vertices(smplx_s, focal_s, princpt_s)
            uvz_t = self.vertices(smplx_t, focal_t, princpt_t, betas_override=betas_override_t)
        return correspond(uvz_s, faces_s.to(self.device).long(), uvz_t, faces_t.to(self.device).long(),
                          self.faces, self.face_part, self.eps_depth, self.bg_dilate_px, self.stretch_tau,
                          edge_px=self.edge_px)
