#!/usr/bin/env python
"""E0 diagnostic: is the warp the bottleneck?  TPSMM flow (stage 1) vs the analytic mesh flow (A1).

No generator is involved: both flows warp the RGB source (reconstruction protocol: source = first valid row),
and the warped images are compared with the target inside the arm + hand pixels that are visible in the
source (V = 1 of the mesh), binned by wrist displacement.  Also reports how much of the arms / hands is NOT
visible in the source (``disocc_frac``: what any warp must leave to the generator) and saves panels

    source | warp TPSMM | O TPSMM | warp mesh | O_geo = V*C | target

for the clips with the largest motion.

    python scripts/14_diag_mesh_flow.py --cache data/hamnosys_cache --flow-checkpoint outputs/flow/checkpoints/epoch_100.pth \
        --smplx-dir models_smplx_v1_1/models --out outputs/diag_mesh_flow [--split val --max-clips 50 --max-rows 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import HamNoSysCache, SequenceDataset, collate_keep_strings, read_split  # noqa: E402
from greg.data.regions import part_mask  # noqa: E402
from greg.data.splits import pick_keys  # noqa: E402
from greg.eval.evaluate import BIN_NAMES, MOTION_BINS, masked_l1, motion_bin, wrist_displacement  # noqa: E402
from greg.eval.visualize import labelled, tile_grid, to_pil  # noqa: E402
from greg.geometry.mesh_flow import MeshCorrespondence  # noqa: E402
from greg.models.tpsmm import FlowEstimator, resize_occlusion, warp  # noqa: E402
from greg.utils.checkpoint import load_checkpoint, load_module_state  # noqa: E402
from greg.utils.config import get_device  # noqa: E402

ARM_HAND = (3, 4, 5, 6)
HAND = (3, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--flow-checkpoint", required=True, help="stage-1 checkpoint (05_train_flow.py)")
    p.add_argument("--smplx-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-clips", type=int, default=50)
    p.add_argument("--max-rows", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--panels", type=int, default=12, help="panels for the largest-motion frames")
    p.add_argument("--edge-px", type=int, default=4, help="edge confidence of the mesh flow (0 = off, as in v1)")
    p.add_argument("--no-valid-rect", dest="valid_rect", action="store_false",
                   help="ignore the black crop padding (v1 / v2 behaviour)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ck = load_checkpoint(args.flow_checkpoint)
    f = ck["config"]["model"]["flow"]
    tpsmm = FlowEstimator(num_tps=int(f.get("num_tps", 10)), num_channels=3, dense_motion=f.get("dense_motion", {}),
                          multi_mask=bool(f.get("multi_mask", False))).to(device)
    load_module_state(tpsmm, ck["flow"])
    tpsmm.eval()
    mesh = MeshCorrespondence(args.smplx_dir, device=device, edge_px=args.edge_px)

    keys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.split}.txt"), args.max_clips, seed=0)
    cache = HamNoSysCache(args.cache, keys)
    cache.keep_keys_with_mesh(what="E0")
    ds = SequenceDataset(cache, source_row=0, max_rows=args.max_rows, uniform_rows=True, with_mesh=True)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2, collate_fn=collate_keep_strings)

    rows, panels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="E0"):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            Is, It = b["source_image"], b["target_image"]
            fo = tpsmm(b["source_normal"], b["target_normal"])
            warp_t = warp(Is, fo["deformation"])
            occ_t = resize_occlusion(fo["occlusion"], Is.shape[-2:])
            g = mesh(b["source_smplx"], b["source_focal"], b["source_princpt"], b["source_faces"],
                     b["target_smplx"], b["target_focal"], b["target_princpt"], b["target_faces"],
                     valid_rect_s=b["source_valid_rect"] if args.valid_rect else None)
            warp_m = warp(Is, g["grid"])
            vis = g["visibility"]
            arm_vis = part_mask(g["parts"], ARM_HAND, 0, soften=False) * vis
            hand_vis = part_mask(g["parts"], HAND, 0, soften=False) * vis
            core = part_mask(g["parts"], ARM_HAND, 0, soften=False)
            n = core.sum(dim=(1, 2, 3))
            disocc = (core * (1 - vis)).sum(dim=(1, 2, 3)) / n.clamp(min=1)
            # *_Ow: weighted by O_geo = V * C (stretch + limb-edge confidence), i.e. on the pixels the model copies
            arm_ow = part_mask(g["parts"], ARM_HAND, 0, soften=False) * g["occlusion"]
            res = {"arm_vis_l1_tpsmm": masked_l1(warp_t, It, arm_vis), "arm_vis_l1_mesh": masked_l1(warp_m, It, arm_vis),
                   "hand_vis_l1_tpsmm": masked_l1(warp_t, It, hand_vis), "hand_vis_l1_mesh": masked_l1(warp_m, It, hand_vis),
                   "copy_arm_vis_l1": masked_l1(Is, It, arm_vis),
                   "arm_Ow_l1_mesh": masked_l1(warp_m, It, arm_ow, min_pixels=1.0),
                   "arm_Ow_l1_tpsmm": masked_l1(warp_t, It, arm_ow, min_pixels=1.0),
                   "arm_Ow_l1_copy": masked_l1(Is, It, arm_ow, min_pixels=1.0),
                   "arm_O_coverage": torch.where(n > 0, arm_ow.sum(dim=(1, 2, 3)) / n.clamp(min=1),
                                                 torch.full_like(disocc, float("nan"))),
                   "disocc_frac": torch.where(n > 0, disocc, torch.full_like(disocc, float("nan"))),
                   "wrist_disp": wrist_displacement(b["source_joints"], b["target_joints"])}
            for i in range(Is.shape[0]):
                r = {"key": b["key"][i], "target_row": int(b["target_row"][i])}
                r.update({k: float(v[i]) for k, v in res.items()})
                rows.append(r)
                panels.append((r["wrist_disp"], [Is[i].cpu(), warp_t[i].cpu(), occ_t[i].cpu(), warp_m[i].cpu(),
                                                 g["occlusion"][i].cpu(), It[i].cpu()], r["key"], r["target_row"]))
                panels = sorted(panels, key=lambda x: -x[0])[:args.panels]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[motion_bin(r["wrist_disp"])].append(r)
    names = [k for k in rows[0] if k not in ("key", "target_row", "wrist_disp")]
    summary = {"num_frames": len(rows), "motion_bins_px": list(MOTION_BINS), "edge_px": args.edge_px,
               "valid_rect": args.valid_rect}
    for bn in ("all",) + BIN_NAMES:
        sel = rows if bn == "all" else by_bin.get(bn, [])
        summary[bn] = {"num_frames": len(sel), **{k: float(np.nanmean([r[k] for r in sel])) if sel else float("nan")
                                                  for k in names}}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    titles = ("source", "warp TPSMM", "O TPSMM", "warp mesh (A1)", "O_geo = V*C", "target")
    grid = tile_grid([[labelled(to_pil(t), f"{titles[0]} {k}:{row} d={d:.0f}px" if j == 0 else titles[j])
                       for j, t in enumerate(ts)] for d, ts, k, row in panels])
    grid.save(out / "panels_largest_motion.jpg", quality=90)
    # the clips with the largest motion, for side-by-side GIFs of several models (13_visualize.py --keys)
    keys = list(dict.fromkeys(k for _, _, k, _ in panels))
    (out / "largest_motion_keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    print(f"panels -> {out / 'panels_largest_motion.jpg'}")


if __name__ == "__main__":
    main()
