"""Key regions (face, left hand, right hand): boxes, Gaussians (Eq. 1), crop & paste.

Conventions (docs/decisions.md Sec. 2):
* ``centers`` (B, 3, 2) are pixel coordinates ``(x, y)`` of the region centres,
  order ``face, lhand, rhand`` (= ``greg.render.body.REGION_NAMES``);
* two box formats, told apart by the last dimension:
    - ``(B, R, 2)`` long  ``(x0, y0)``: fixed ``size x size`` squares (the paper; ``regions.mode: fixed``),
      clamped inside the frame, cropped / pasted by indexing without resampling;
    - ``(B, R, 3)`` float ``(x0, y0, side)``: squares of any side (``regions.mode: adaptive``, improvement B1),
      resampled to ``size x size`` patches with bilinear ``grid_sample`` and pasted back by the inverse map.
  A box covers the pixel centres ``x0 .. x0 + side - 1``, i.e. the continuous interval
  ``[x0 - 0.5, x0 + side - 0.5]``, so an adaptive box with ``side == size`` and integer ``x0`` crops exactly
  what the fixed box crops;
* a region whose centre leaves the frame has visibility 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_REGIONS = 3
REGION_NAMES = ("face", "lhand", "rhand")
REGION_PART_LABELS = (2, 3, 4)          # labels of face / lhand / rhand in the part maps


def adaptive_region_boxes(centers: torch.Tensor, extents: torch.Tensor, size: int, height: int, width: int,
                          hand_margin: float = 0.25, min_side: float = 24.0, max_side: float = 96.0,
                          adaptive_face: bool = False) -> torch.Tensor:
    """B1: (B, R, 2) centres + (B, R) joint extents -> (B, R, 3) float ``(x0, y0, side)`` boxes.

    Hands get ``side = clamp(extent * (1 + 2 * hand_margin), min_side, max_side)`` so the hand fills the patch
    instead of ~15% of a fixed 64x64 box; the face keeps ``size`` (it already fits) unless ``adaptive_face``.
    ``extents`` come from ``greg.render.body.region_extents_from_joints`` (geometry only, available at test)."""
    side = (extents.float() * (1 + 2 * hand_margin)).clamp(min_side, max_side)
    if not adaptive_face:
        side = torch.cat([torch.full_like(side[:, :1], float(size)), side[:, 1:]], dim=1)
    x0 = centers[..., 0].float() - side / 2 + 0.5
    y0 = centers[..., 1].float() - side / 2 + 0.5
    x0 = torch.minimum(torch.maximum(x0, torch.zeros_like(x0)), (width - side).clamp(min=0))
    y0 = torch.minimum(torch.maximum(y0, torch.zeros_like(y0)), (height - side).clamp(min=0))
    return torch.stack([x0, y0, side], dim=-1)


def box_sides(boxes: torch.Tensor, size: int) -> torch.Tensor:
    """(B, R, 2|3) -> (B, R) float side length of every box."""
    if boxes.shape[-1] == 3:
        return boxes[..., 2].float()
    return torch.full(boxes.shape[:-1], float(size), device=boxes.device)


def region_boxes(centers: torch.Tensor, size: int, height: int, width: int) -> torch.Tensor:
    """(B, R, 2) centres -> (B, R, 2) integer top-left corners of ``size``² boxes inside the frame."""
    half = size / 2.0
    x0 = torch.round(centers[..., 0] - half).clamp(0, max(width - size, 0))
    y0 = torch.round(centers[..., 1] - half).clamp(0, max(height - size, 0))
    return torch.stack([x0, y0], dim=-1).long()


def _box_indices(boxes: torch.Tensor, size: int, height: int, width: int):
    """-> ys (B, R, size, 1), xs (B, R, 1, size) pixel indices covered by each box."""
    boxes = boxes.long()
    ar = torch.arange(size, device=boxes.device)
    ys = (boxes[..., 1, None] + ar).clamp(0, height - 1)[..., :, None]
    xs = (boxes[..., 0, None] + ar).clamp(0, width - 1)[..., None, :]
    return ys, xs


def _patch_coords(boxes: torch.Tensor, size: int):
    """(B, R, 3) float boxes -> pixel x (B, R, size), y (B, R, size) of the patch pixel centres."""
    k = (torch.arange(size, device=boxes.device, dtype=torch.float32) + 0.5) / size
    x0, y0, side = boxes[..., 0, None].float(), boxes[..., 1, None].float(), boxes[..., 2, None].float()
    return x0 - 0.5 + k * side, y0 - 0.5 + k * side


def crop_regions_affine(x: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, H, W), boxes (B, R, 3) float -> (B, R, C, size, size), bilinear.  Differentiable w.r.t. ``x``."""
    B, C, H, W = x.shape
    R = boxes.shape[1]
    px, py = _patch_coords(boxes.to(x.device), size)                       # (B, R, s)
    gx = (2 * px / (W - 1) - 1)[:, :, None, :].expand(B, R, size, size)
    gy = (2 * py / (H - 1) - 1)[:, :, :, None].expand(B, R, size, size)
    grid = torch.stack([gx, gy], dim=-1).reshape(B * R, size, size, 2).to(x.dtype)
    src = x[:, None].expand(B, R, C, H, W).reshape(B * R, C, H, W)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(B, R, C, size, size)


def paste_regions_affine(patches: torch.Tensor, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`crop_regions_affine`: (B, R, C, s, s) -> (B, R*C, H, W), zero outside the boxes."""
    B, R, C, s, _ = patches.shape
    dev = patches.device
    boxes = boxes.to(dev).float()
    x0, y0, side = boxes[..., 0, None, None], boxes[..., 1, None, None], boxes[..., 2, None, None]
    xs = torch.arange(width, device=dev, dtype=torch.float32).view(1, 1, 1, width)
    ys = torch.arange(height, device=dev, dtype=torch.float32).view(1, 1, height, 1)
    kx = (xs - x0 + 0.5) * s / side - 0.5                                   # patch coordinates of every pixel
    ky = (ys - y0 + 0.5) * s / side - 0.5
    inside = ((xs >= x0 - 0.5) & (xs < x0 + side - 0.5) & (ys >= y0 - 0.5) & (ys < y0 + side - 0.5))
    grid = torch.stack([(2 * kx / (s - 1) - 1).expand(B, R, height, width),
                        (2 * ky / (s - 1) - 1).expand(B, R, height, width)], dim=-1)
    out = F.grid_sample(patches.reshape(B * R, C, s, s), grid.reshape(B * R, height, width, 2).to(patches.dtype),
                        mode="bilinear", padding_mode="border", align_corners=True)
    out = out.view(B, R, C, height, width) * inside[:, :, None].to(patches.dtype)
    return out.reshape(B, R * C, height, width)


def crop_regions(x: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, H, W), boxes (B, R, 2|3) -> (B, R, C, size, size).  Differentiable w.r.t. ``x``."""
    if boxes.shape[-1] == 3:
        return crop_regions_affine(x, boxes, size)
    B, C, H, W = x.shape
    R = boxes.shape[1]
    ys, xs = _box_indices(boxes.to(x.device), size, H, W)
    b = torch.arange(B, device=x.device).view(B, 1, 1, 1)
    out = x[b, :, ys, xs]                      # advanced indices first: (B, R, size, size, C)
    return out.permute(0, 1, 4, 2, 3).contiguous()


def paste_regions(patches: torch.Tensor, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, R, C, s, s) patches -> (B, R*C, H, W) canvases, zero outside the boxes.

    Each region gets its own ``C`` channels (region ``r`` -> channels ``r*C:(r+1)*C``).
    """
    if boxes.shape[-1] == 3:
        return paste_regions_affine(patches, boxes, height, width)
    B, R, C, s, _ = patches.shape
    ys, xs = _box_indices(boxes.to(patches.device), s, height, width)
    b = torch.arange(B, device=patches.device).view(B, 1, 1, 1)
    r = torch.arange(R, device=patches.device).view(1, R, 1, 1)
    canvas = patches.new_zeros(B, R, height, width, C)
    canvas[b, r, ys, xs] = patches.permute(0, 1, 3, 4, 2)
    return canvas.permute(0, 1, 4, 2, 3).reshape(B, R * C, height, width)


def resize_patches(patches: torch.Tensor, size: int) -> torch.Tensor:
    """(B, R, C, s, s) -> (B, R, C, size, size), bilinear."""
    B, R, C, s, _ = patches.shape
    if s == size:
        return patches
    out = F.interpolate(patches.reshape(B * R, C, s, s), size=(size, size), mode="bilinear", align_corners=False)
    return out.view(B, R, C, size, size)


def region_gaussians(centers: torch.Tensor, height: int, width: int, sigma: float) -> torch.Tensor:
    """Eq. 1: (B, R, 2) centres -> (B, R, H, W) Gaussian weight maps with peak 1 at the centre."""
    ys = torch.arange(height, device=centers.device, dtype=torch.float32).view(1, 1, height, 1)
    xs = torch.arange(width, device=centers.device, dtype=torch.float32).view(1, 1, 1, width)
    cx = centers[..., 0].float()[..., None, None]
    cy = centers[..., 1].float()[..., None, None]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2.0 * float(sigma) ** 2))


def region_visibility(centers: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, R, 2) -> (B, R) float: 1 where the centre lies inside the frame."""
    x, y = centers[..., 0], centers[..., 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height) & torch.isfinite(x) & torch.isfinite(y)
    return inside.float()


def weighted_region_gradients(gradient: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Eq. 3: gradient (B, 1, H, W) x Gaussians (B, R, H, W) -> G_R (B, R, H, W)."""
    return gradient * weights


def part_mask(parts: torch.Tensor, labels, dilate_px: int = 0, soften: bool = True) -> torch.Tensor:
    """(B, H, W) integer part map -> (B, 1, H, W) float mask of ``labels``, dilated by ``dilate_px`` (the mesh is
    ~2.5 px off and has no sleeves) and softened by a 3x3 box filter."""
    m = torch.zeros_like(parts, dtype=torch.bool)
    for lab in labels:
        m |= parts == int(lab)
    m = m.float()[:, None]
    if dilate_px > 0:
        m = F.max_pool2d(m, 2 * dilate_px + 1, 1, dilate_px)
    if soften:
        m = F.avg_pool2d(m, 3, 1, 1, count_include_pad=False)
    return m


def region_masks(parts: torch.Tensor, boxes: torch.Tensor, size: int, dilate_px: int = 3) -> torch.Tensor:
    """B1: (B, H, W) part map -> (B, R, 1, size, size) soft mask of each region's own part inside its box
    (face crop -> face pixels, lhand crop -> left-hand pixels, ...)."""
    full = torch.cat([part_mask(parts, (lab,), dilate_px) for lab in REGION_PART_LABELS], dim=1)   # (B, R, H, W)
    crops = crop_regions(full, boxes, size)                                  # (B, R, R, s, s)
    idx = torch.arange(crops.shape[1], device=crops.device)
    return crops[:, idx, idx][:, :, None]
