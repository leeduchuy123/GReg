"""Visual diagnostics: labelled panels, error maps, region close-ups and GIFs.

Numbers say *how much* a model is off; these say *where and why*:

* ``frame_panels``  two labelled images for one (source, target) pair
    - full: source | N_s | N_t | warped I_s->t | occlusion O | output  /  target | |error| | generator
      output before the blend | G_R (Gaussian-weighted gradient: face=R, lhand=G, rhand=B) | regional
      priors R (same colours) | output with the region boxes
    - regions: per region (face, lhand, rhand)  output | target | |error| | RE prior | target gradient
* ``clip_gif``      animated  N_t | target | output | |error|  over a whole clip
* ``heat``          black -> red -> yellow -> white colour map for scalar maps

Everything works on CPU tensors in [0, 1]; only PIL is needed (no matplotlib).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from greg.data.gradient import gradient_map
from greg.data.regions import crop_regions

REGION_NAMES = ("face", "lhand", "rhand")
REGION_COLORS = ((255, 64, 64), (64, 255, 64), (80, 140, 255))
LABEL_H = 16


def heat(x: torch.Tensor) -> torch.Tensor:
    """(1, H, W) in [0, 1] -> (3, H, W) black-red-yellow-white."""
    x = x.clamp(0, 1)
    return torch.cat([(3 * x).clamp(0, 1), (3 * x - 1).clamp(0, 1), (3 * x - 2).clamp(0, 1)], dim=0)


def error_map(pred: torch.Tensor, target: torch.Tensor, gain: float = 4.0) -> torch.Tensor:
    """(3, H, W) x2 -> (3, H, W) heat map of the mean absolute error (x ``gain``, 1.0 = error 1/gain)."""
    return heat(((pred - target).abs().mean(0, keepdim=True) * gain))


def to_pil(t: torch.Tensor, size: Optional[int] = None) -> Image.Image:
    t = t.detach().float().cpu().clamp(0, 1)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    if size is not None and t.shape[-1] != size:
        t = F.interpolate(t[None], size=(size, size), mode="nearest")[0]
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))


def labelled(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + LABEL_H), (20, 20, 24))
    out.paste(img, (0, LABEL_H))
    ImageDraw.Draw(out).text((3, 2), text, fill=(235, 235, 235))
    return out


def tile_grid(rows: Sequence[Sequence[Image.Image]]) -> Image.Image:
    w = max(sum(i.width for i in r) for r in rows)
    h = sum(max(i.height for i in r) for r in rows)
    canvas = Image.new("RGB", (w, h), (20, 20, 24))
    y = 0
    for r in rows:
        x = 0
        for i in r:
            canvas.paste(i, (x, y))
            x += i.width
        y += max(i.height for i in r)
    return canvas


def draw_boxes(img: Image.Image, boxes: torch.Tensor, size: int, visible: Optional[Sequence[float]] = None) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    for r in range(boxes.shape[0]):
        if visible is not None and visible[r] < 0.5:
            continue
        x0, y0 = int(boxes[r, 0]), int(boxes[r, 1])
        d.rectangle([x0, y0, x0 + size - 1, y0 + size - 1], outline=REGION_COLORS[r], width=2)
    return img


def frame_panels(out: Dict[str, torch.Tensor], i: int, image_source: torch.Tensor, normal_source: torch.Tensor,
                 normal_target: torch.Tensor, image_target: torch.Tensor, region_size: int,
                 error_gain: float = 4.0, region_zoom: int = 128, visible: Optional[Sequence[float]] = None
                 ) -> Dict[str, Image.Image]:
    """Panels for sample ``i`` of a GReg forward ``out`` (see ``GReg.forward``)."""
    pred, tgt = out["output"][i].clamp(0, 1), image_target[i]
    err = error_map(pred, tgt, error_gain)
    size = pred.shape[-1]
    tiles = lambda items: [labelled(to_pil(t), name) for name, t in items]   # noqa: E731
    row1 = tiles([("source I_s", image_source[i]), ("N_s", normal_source[i][:3]), ("N_t", normal_target[i][:3]),
                  ("warped I_s->t", out["warped"][i]), ("occlusion O (1=keep warped)", out["occlusion"][i]),
                  ("output", pred)])
    row2 = tiles([("target I_t", tgt), (f"|error| x{error_gain:g}", err), ("generator (pre-blend)", out["generated"][i]),
                  ("G_R  face/lh/rh = R/G/B", out["region_gradients"][i]),
                  ("regional priors R", out["prior_map"][i])])
    row2.append(labelled(draw_boxes(to_pil(pred), out["boxes"][i], region_size, visible), "output + boxes"))
    full = tile_grid([row1, row2])

    boxes = out["boxes"][i:i + 1]
    pc = crop_regions(pred[None], boxes, region_size)[0]                 # (R, 3, s, s)
    tc = crop_regions(tgt[None], boxes, region_size)[0]
    tg = gradient_map(tc)                                                 # (R, 1, s, s)
    gmax = tg.flatten(1).max(1).values.clamp(min=1e-6).view(-1, 1, 1, 1)
    prior = out["prior_patches"][i]                                       # (R, C, s, s)
    rows = []
    for r in range(boxes.shape[1]):
        name = REGION_NAMES[r] + ("" if visible is None or visible[r] >= 0.5 else "(out)")
        rows.append([labelled(to_pil(t, region_zoom), f"{name} {title}" if k == 0 else title) for k, (title, t) in enumerate((
            ("output", pc[r]), ("target", tc[r]), ("|error|", error_map(pc[r], tc[r], error_gain)),
            ("RE prior", prior[r][:1].expand(3, -1, -1)), ("target gradient", (tg[r] / gmax[r]).expand(3, -1, -1))))])
    return {"full": full, "regions": tile_grid(rows)}


def clip_gif(frames: Sequence[Dict[str, torch.Tensor]], path: str | Path, fps: float = 12.0, scale: float = 1.0,
             error_gain: float = 4.0) -> Path:
    """frames: dicts with ``normal`` (3,H,W), ``target``, ``output`` -> GIF ``N_t | target | output | |error|``."""
    imgs = []
    for f in frames:
        row = [labelled(to_pil(f["normal"][:3]), "N_t"), labelled(to_pil(f["target"]), "target"),
               labelled(to_pil(f["output"].clamp(0, 1)), "output"),
               labelled(to_pil(error_map(f["output"].clamp(0, 1), f["target"], error_gain)), f"|error| x{error_gain:g}")]
        im = tile_grid([row])
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        imgs.append(im)
    return save_gif(imgs, path, fps)


def save_gif(imgs: Sequence[Image.Image], path: str | Path, fps: float = 12.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=list(imgs[1:]), duration=int(1000 / fps), loop=0, optimize=False)
    return path


@torch.no_grad()
def render_clip(model, cache_root, key: str, device, source_row: int = 0, stride: int = 1, batch_size: int = 8,
                max_frames: Optional[int] = None, rows: Optional[Sequence[int]] = None) -> List[dict]:
    """Run the model over the rows of one clip (source = ``source_row``).  Returns per-frame dicts of CPU
    tensors (``normal``, ``target``, ``output``, ``warped``, ...) plus ``row``, ``l1`` and ``region_l1`` (model boxes)."""
    from greg.data.dataset import HamNoSysCache, SequenceDataset
    ds = SequenceDataset(HamNoSysCache(cache_root, [key]), source_row=source_row)
    items = list(range(len(ds)))[::max(stride, 1)]
    if rows is not None:   # only these target rows
        items = [k for k in range(len(ds)) if ds.items[k][2] in set(rows)]
    if max_frames:
        items = items[:max_frames]
    was = model.training
    model.eval()
    frames: List[dict] = []
    for b in range(0, len(items), batch_size):
        batch = [ds[k] for k in items[b:b + batch_size]]
        stack = lambda k: torch.stack([x[k] for x in batch]).to(device)   # noqa: E731
        Is, Ns, Nt, It, ct = (stack("source_image"), stack("source_normal"), stack("target_normal"),
                              stack("target_image"), stack("target_centers"))
        o = model(Is, Ns, Nt, ct)
        pred = o["output"].clamp(0, 1)
        crops_p = crop_regions(pred, o["boxes"], model.region_size)
        crops_t = crop_regions(It, o["boxes"], model.region_size)
        reg = (crops_p - crops_t).abs().mean(dim=(2, 3, 4))                # (B, R)
        for j, x in enumerate(batch):
            frames.append({"row": int(x["target_row"]), "l1": float((pred[j] - It[j]).abs().mean()),
                           "region_l1": float(reg[j].mean()), "normal": Nt[j].cpu(), "target": It[j].cpu(),
                           "output": pred[j].cpu(), "out": {k: (v[j:j + 1].cpu() if torch.is_tensor(v) and v.shape[0] == len(batch) else v)
                                                            for k, v in o.items()},
                           "source": Is[j].cpu(), "normal_source": Ns[j].cpu(), "centers": ct[j].cpu()})
    model.train(was)
    return frames


def fixed_panels(model, cache_root, keys: Sequence[str], device, path: str | Path, source_row: int = 0,
                 error_gain: float = 4.0, quality: int = 90) -> Path:
    """One image with the full panel of the middle frame of each clip in ``keys``.  Called with the SAME clips
    at every validation epoch, so the images can be compared across epochs (training samples are random)."""
    from greg.data.dataset import HamNoSysCache
    from greg.data.regions import region_visibility
    panels = []
    cache = HamNoSysCache(cache_root, list(keys))
    for key in keys:
        mid = min(max(cache.num_rows(key) // 2, 1), cache.num_rows(key) - 1)
        f = render_clip(model, cache_root, key, device, source_row=source_row, rows=[mid])[0]
        o = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in f["out"].items()}
        vis = region_visibility(f["centers"][None], model.image_size, model.image_size)[0].tolist()
        ps = frame_panels(o, 0, f["source"][None].to(device), f["normal_source"][None].to(device),
                          f["normal"][None].to(device), f["target"][None].to(device), model.region_size,
                          error_gain=error_gain, visible=vis)
        panels.append(labelled(ps["full"], f"clip {key}, row {mid}"))
    img = tile_grid([[p] for p in panels])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality) if path.suffix.lower() in (".jpg", ".jpeg") else img.save(path)
    return path
