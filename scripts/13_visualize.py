#!/usr/bin/env python
"""Visual error analysis of a trained GReg checkpoint: GIFs, labelled panels and an HTML report.

1. rank a random subset of the split by a metric (``--metric``, default region_l1) with a quick evaluation;
2. pick the WORST / median / BEST clips (+ any ``--keys``);
3. for each: ``clip.gif`` (N_t | target | output | |error|), and at its worst and its middle frame the panels
   ``*_full.png`` (warping, occlusion, G_R, priors R, generator output, error map, boxes) and ``*_regions.png``
   (face / hands: output | target | |error| | RE prior | target gradient);
4. ``report.html`` with the per-corpus breakdown and everything above.

    python scripts/13_visualize.py --checkpoint outputs/greg/checkpoints/epoch_100.pth \
        --cache data/hamnosys_cache --split test --out outputs/greg/vis
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greg.data.dataset import read_split  # noqa: E402
from greg.data.regions import region_visibility  # noqa: E402
from greg.data.splits import pick_keys, source_of_key  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.runner import load_greg  # noqa: E402
from greg.eval.visualize import clip_gif, frame_panels, render_clip  # noqa: E402
from greg.utils.config import get_device  # noqa: E402

CSS = """body{font-family:system-ui,sans-serif;background:#14151a;color:#e6e6e6;margin:24px}
h1,h2,h3{font-weight:600}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #333;padding:4px 10px;text-align:right}
th{background:#22242c}td:first-child,th:first-child{text-align:left}.clip{border-top:1px solid #333;margin-top:28px;padding-top:8px}
img{max-width:100%;image-rendering:pixelated}.tag{background:#33415c;border-radius:4px;padding:1px 8px;font-size:.85em}
.worst{background:#7a2e2e}.best{background:#2e6b3f}.median{background:#555}"""


def panels_for(frame, model, device, out_dir: Path, tag: str):
    import torch
    o = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in frame["out"].items()}
    vis = region_visibility(frame["centers"][None], model.image_size, model.image_size)[0].tolist()
    ps = frame_panels(o, 0, frame["source"][None].to(device), frame["normal_source"][None].to(device),
                      frame["normal"][None].to(device), frame["target"][None].to(device), model.region_size, visible=vis)
    names = {}
    for k, im in ps.items():
        names[k] = f"{tag}_{k}.png"
        im.save(out_dir / names[k])
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--metric", default="region_l1", choices=["l1", "region_l1", "ssim", "psnr"])
    p.add_argument("--rank-clips", type=int, default=120, help="random clips used to rank the clips")
    p.add_argument("--rank-rows", type=int, default=8, help="rows per clip used to rank")
    p.add_argument("--worst", type=int, default=6)
    p.add_argument("--median", type=int, default=2)
    p.add_argument("--best", type=int, default=3)
    p.add_argument("--keys", nargs="*", default=[], help="extra clips to render")
    p.add_argument("--stride", type=int, default=1, help="use every n-th frame in the GIF")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--gif-scale", type=float, default=0.75)
    p.add_argument("--error-gain", type=float, default=4.0)
    p.add_argument("--source-row", type=int, default=0)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_greg(args.checkpoint, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = pick_keys(read_split(Path(args.cache) / "splits" / f"{args.split}.txt"), args.rank_clips, args.subset_seed)
    print(f"ranking {len(keys)} clips by {args.metric} ...")
    rows, summary = evaluate_reconstruction(model, args.cache, keys, device, batch_size=16, num_workers=2,
                                            source_row=args.source_row, max_rows=args.rank_rows, uniform_rows=True,
                                            smplx_dir=cfg["model"].get("flow", {}).get("smplx_dir"))
    per_clip = defaultdict(list)
    by_source = defaultdict(list)
    for r in rows:
        per_clip[r["key"]].append(r[args.metric])
        by_source[source_of_key(r["key"])].append(r)
    clip_score = {k: float(np.mean(v)) for k, v in per_clip.items()}
    higher_better = args.metric in ("ssim", "psnr")
    order = sorted(clip_score, key=clip_score.get, reverse=higher_better)       # best first
    worst = order[::-1][:args.worst]
    best = order[:args.best]
    mid = len(order) // 2
    median = order[max(0, mid - args.median // 2): max(0, mid - args.median // 2) + args.median]
    selected = [(k, "worst") for k in worst] + [(k, "median") for k in median] + [(k, "best") for k in best]
    selected += [(k, "chosen") for k in args.keys if k not in {s[0] for s in selected}]

    cards = []
    for key, cat in selected:
        d = out / f"{cat}_{key}"
        d.mkdir(parents=True, exist_ok=True)
        frames = render_clip(model, args.cache, key, device, source_row=args.source_row, stride=args.stride,
                             max_frames=args.max_frames)
        clip_gif(frames, d / "clip.gif", args.fps, args.gif_scale, args.error_gain)
        worst_i = int(np.argmax([f["region_l1"] for f in frames]))
        imgs = {"worst": panels_for(frames[worst_i], model, device, d, f"worst_row{frames[worst_i]['row']}"),
                "middle": panels_for(frames[len(frames) // 2], model, device, d, f"mid_row{frames[len(frames) // 2]['row']}")}
        cards.append({"key": key, "cat": cat, "source": source_of_key(key), "score": clip_score.get(key, float("nan")),
                      "frames": len(frames), "mean_l1": float(np.mean([f["l1"] for f in frames])),
                      "mean_region_l1": float(np.mean([f["region_l1"] for f in frames])),
                      "worst_row": frames[worst_i]["row"], "dir": d.name, "imgs": imgs})
        print(f"[{cat}] {key}: {args.metric}={clip_score.get(key, float('nan')):.4f}  -> {d}")

    h = [f"<html><head><meta charset='utf-8'><style>{CSS}</style><title>GReg visual report</title></head><body>",
         f"<h1>GReg visual report</h1><p>checkpoint <code>{html.escape(str(args.checkpoint))}</code>, split <b>{args.split}</b>, "
         f"ranking by <b>{args.metric}</b> on {len(keys)} clips x {args.rank_rows} rows. Error maps are |output - target| x{args.error_gain:g}.</p>",
         "<h2>By corpus</h2><table><tr><th>corpus</th><th>clips</th><th>L1</th><th>SSIM</th><th>region-L1</th></tr>"]
    for src, rs in sorted(by_source.items()):
        h.append(f"<tr><td>{src}</td><td>{len({r['key'] for r in rs})}</td><td>{np.mean([r['l1'] for r in rs]):.4f}</td>"
                 f"<td>{np.mean([r['ssim'] for r in rs]):.4f}</td><td>{np.mean([r['region_l1'] for r in rs]):.4f}</td></tr>")
    h.append(f"<tr><td><b>all</b></td><td>{summary['num_clips']}</td><td>{summary['mean_l1']:.4f}</td>"
             f"<td>{summary['mean_ssim']:.4f}</td><td>{summary['mean_region_l1']:.4f}</td></tr></table>")
    h.append(f"<p>Baselines: copying the source frame gives L1 {summary['mean_source_copy_l1']:.4f}; the warped source alone "
             f"{summary['mean_warped_l1']:.4f}.</p>")
    for c in cards:
        h.append(f"<div class='clip'><h3>{html.escape(c['key'])} <span class='tag {c['cat']}'>{c['cat']}</span> "
                 f"<small>{c['source']}, {c['frames']} frames, {args.metric}={c['score']:.4f}, L1 {c['mean_l1']:.4f}, "
                 f"region-L1 {c['mean_region_l1']:.4f}</small></h3><img src='{c['dir']}/clip.gif'>")
        for which, names in c["imgs"].items():
            h.append(f"<h4>{which} frame{' (row ' + str(c['worst_row']) + ')' if which == 'worst' else ''}</h4>"
                     f"<img src='{c['dir']}/{names['full']}'><br><img src='{c['dir']}/{names['regions']}'>")
        h.append("</div>")
    h.append("</body></html>")
    (out / "report.html").write_text("\n".join(h), encoding="utf-8")
    (out / "selection.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "imgs"} for c in cards], indent=2))
    print(f"report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
