"""Mesh rasterisation to per-pixel barycentrics / face ids.

Two backends with the same interface:

* ``"torch"``  -- pure PyTorch tiled rasteriser (no compilation; CPU or GPU).
  Good enough for 10k-face SMPL-X meshes at 256^2 (tens of ms on a GPU,
  a few seconds on CPU).  Used by the unit tests.
* ``"nvdiffrast"`` -- optional, much faster on GPU if the package is installed.

Pixel convention: pixel (i, j) has its centre at (x=j, y=i) in the same
coordinate system as the projected vertices, i.e. ``u = X/Z*fx + cx`` lands on
the centre of column ``round(u)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Fragments:
    face_idx: torch.Tensor   # (B, H, W) long, -1 for background
    bary: torch.Tensor       # (B, H, W, 3) float, barycentric coords of the visible face
    depth: torch.Tensor      # (B, H, W) float, +inf for background

    @property
    def mask(self) -> torch.Tensor:
        return self.face_idx >= 0


def rasterize(uv: torch.Tensor, z: torch.Tensor, faces: torch.Tensor, height: int, width: int,
              backend: str = "torch", tile: int = 32, cull_backfaces: bool = False) -> Fragments:
    """Rasterise projected triangles.

    Args:
        uv: (B, V, 2) pixel coordinates of the vertices.
        z: (B, V) camera-space depth (larger = farther).
        faces: (F, 3) long.
    """
    if backend == "nvdiffrast":
        return _rasterize_nvdiffrast(uv, z, faces, height, width)
    if backend != "torch":
        raise ValueError(f"unknown rasteriser backend {backend!r}")
    return _rasterize_torch(uv, z, faces, height, width, tile=tile, cull_backfaces=cull_backfaces)


# ----------------------------------------------------------------------------- torch backend
def _rasterize_torch(uv, z, faces, height, width, tile=32, cull_backfaces=False) -> Fragments:
    B = uv.shape[0]
    dev = uv.device
    face_idx = torch.full((B, height, width), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((B, height, width, 3), dtype=uv.dtype, device=dev)
    depth = torch.full((B, height, width), float("inf"), dtype=uv.dtype, device=dev)

    ys = torch.arange(height, device=dev, dtype=uv.dtype)
    xs = torch.arange(width, device=dev, dtype=uv.dtype)

    for b in range(B):
        tri = uv[b][faces]                       # (F, 3, 2)
        tz = z[b][faces]                         # (F, 3)
        # signed double area; drop degenerate (and optionally back-facing) triangles
        e0 = tri[:, 1] - tri[:, 0]
        e1 = tri[:, 2] - tri[:, 0]
        area = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
        valid = area.abs() > 1e-12
        if cull_backfaces:
            valid &= area < 0  # OpenCV (y down) + CCW faces => front faces have negative area
        tmin = tri.min(1).values
        tmax = tri.max(1).values
        valid &= (tmax[:, 0] >= 0) & (tmax[:, 1] >= 0) & (tmin[:, 0] <= width - 1) & (tmin[:, 1] <= height - 1)

        for ty in range(0, height, tile):
            for tx in range(0, width, tile):
                y1, x1 = min(ty + tile, height), min(tx + tile, width)
                cand = valid & (tmax[:, 0] >= tx) & (tmin[:, 0] <= x1 - 1) & (tmax[:, 1] >= ty) & (tmin[:, 1] <= y1 - 1)
                fids = cand.nonzero(as_tuple=False).squeeze(1)
                if fids.numel() == 0:
                    continue
                t = tri[fids]                      # (K, 3, 2)
                a = area[fids]                     # (K,)
                gy, gx = torch.meshgrid(ys[ty:y1], xs[tx:x1], indexing="ij")
                p = torch.stack([gx, gy], -1).reshape(-1, 1, 2)   # (P, 1, 2)
                v0, v1, v2 = t[:, 0][None], t[:, 1][None], t[:, 2][None]   # (1, K, 2)
                # edge functions -> barycentrics
                w0 = (v1[..., 0] - p[..., 0]) * (v2[..., 1] - p[..., 1]) - (v1[..., 1] - p[..., 1]) * (v2[..., 0] - p[..., 0])
                w1 = (v2[..., 0] - p[..., 0]) * (v0[..., 1] - p[..., 1]) - (v2[..., 1] - p[..., 1]) * (v0[..., 0] - p[..., 0])
                w2 = a[None] - w0 - w1
                bc = torch.stack([w0, w1, w2], -1) / a[None, :, None]      # (P, K, 3)
                inside = (bc >= -1e-6).all(-1)
                zz = (bc * tz[fids][None]).sum(-1)                           # (P, K)
                zz = torch.where(inside, zz, torch.full_like(zz, float("inf")))
                zmin, kmin = zz.min(1)                                       # (P,)
                hit = torch.isfinite(zmin)
                if not hit.any():
                    continue
                pidx = hit.nonzero(as_tuple=False).squeeze(1)
                py = pidx // (x1 - tx) + ty
                px = pidx % (x1 - tx) + tx
                face_idx[b, py, px] = fids[kmin[pidx]]
                bary[b, py, px] = bc[pidx, kmin[pidx]]
                depth[b, py, px] = zmin[pidx]
    return Fragments(face_idx=face_idx, bary=bary, depth=depth)


# ------------------------------------------------------------------------ nvdiffrast backend
_NVDR_CTX = None


def _rasterize_nvdiffrast(uv, z, faces, height, width) -> Fragments:
    import nvdiffrast.torch as dr  # type: ignore

    global _NVDR_CTX
    if _NVDR_CTX is None:
        _NVDR_CTX = dr.RasterizeCudaContext()
    B = uv.shape[0]
    # pixel coords -> clip space; nvdiffrast's y axis points up and the image is flipped later
    x = (uv[..., 0] + 0.5) / width * 2 - 1
    y = -((uv[..., 1] + 0.5) / height * 2 - 1)
    zmin, zmax = z.min(), z.max()
    zc = (z - zmin) / (zmax - zmin + 1e-6) * 2 - 1
    pos = torch.stack([x, y, zc, torch.ones_like(x)], -1).contiguous()  # (B, V, 4)
    rast, _ = dr.rasterize(_NVDR_CTX, pos, faces.int().contiguous(), resolution=[height, width])
    rast = torch.flip(rast, dims=[1])  # back to y-down image orientation
    face_idx = rast[..., 3].long() - 1
    u, v = rast[..., 0], rast[..., 1]
    bary = torch.stack([1 - u - v, u, v], -1)
    bary = torch.where(face_idx[..., None] >= 0, bary, torch.zeros_like(bary))
    # depth: barycentric interpolation of camera z
    zf = z[:, faces]  # (B, F, 3)
    fi = face_idx.clamp(min=0)
    zpix = torch.gather(zf, 1, fi.view(B, -1, 1).expand(-1, -1, 3)).view(B, height, width, 3)
    depth = (zpix * bary).sum(-1)
    depth = torch.where(face_idx >= 0, depth, torch.full_like(depth, float("inf")))
    return Fragments(face_idx=face_idx, bary=bary, depth=depth)


def interpolate_vertex_attribute(frag: Fragments, attr: torch.Tensor, faces: torch.Tensor,
                                 fill: float = 0.0) -> torch.Tensor:
    """Interpolate a per-vertex attribute (B, V, C) to (B, C, H, W) using the fragments."""
    B, H, W = frag.face_idx.shape
    C = attr.shape[-1]
    fa = attr[:, faces]                      # (B, F, 3, C)
    fi = frag.face_idx.clamp(min=0).view(B, -1, 1, 1).expand(-1, -1, 3, C)
    picked = torch.gather(fa, 1, fi).view(B, H, W, 3, C)
    out = (picked * frag.bary[..., None]).sum(-2)           # (B, H, W, C)
    out = torch.where(frag.mask[..., None], out, torch.full_like(out, fill))
    return out.permute(0, 3, 1, 2).contiguous()


def face_attribute_map(frag: Fragments, face_attr: torch.Tensor, fill=0) -> torch.Tensor:
    """Look up a per-face attribute (F,) -> (B, H, W)."""
    B, H, W = frag.face_idx.shape
    fi = frag.face_idx.clamp(min=0)
    out = face_attr[fi.view(-1)].view(B, H, W)
    return torch.where(frag.mask, out, torch.full_like(out, fill))
