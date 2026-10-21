#!/usr/bin/env python
"""Hyper-parameter sweep for the values the paper leaves open (Hr x Wr, sigma, lambdas, detach_prior ...).

For every run of a sweep file: train stage 2 with the short schedule (``06_train_greg.py``, flow frozen and
shared), evaluate the final checkpoint on the held-out VALIDATION clips, then collect all runs in one table
ranked by the mean rank over ``rank_metrics``.  The test split is refused unless ``--allow-test``.

    python scripts/04_make_splits.py --cache data/hamnosys_cache          # creates val.txt / train_fit.txt
    python scripts/11_sweep.py --sweep configs/sweeps/hamnosys_sweep.yaml \
        --flow-checkpoint outputs/flow/checkpoints/epoch_100.pth --out-root outputs/sweep [--cache ...]
    python scripts/11_sweep.py --sweep ... --out-root outputs/sweep --collect-only

Resume-safe: finished runs (``val_metrics.json`` present) are skipped, interrupted ones resume from ``last.pth``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HIGHER_IS_BETTER = {"ssim", "psnr"}


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def train_run(name: str, overrides: List[str], sweep: dict, args, out_dir: Path) -> float:
    cmd = [sys.executable, "scripts/06_train_greg.py", "--config", sweep["base_config"], "--override",
           *sweep.get("common", []), *overrides, f"output_dir={out_dir}"]
    if args.cache:
        cmd.append(f"data.cache={args.cache}")
    if args.device:
        cmd.append(f"device={args.device}")
    last = out_dir / "checkpoints" / "last.pth"
    cmd += ["--resume", str(last)] if last.exists() else ["--flow-checkpoint", args.flow_checkpoint]
    t0 = time.time()
    run_cmd(cmd)
    return (time.time() - t0) / 60


def cost_from_log(out_dir: Path) -> Dict[str, float]:
    """Median sec/step and max VRAM from the training log (cost measurement for the GPU budget)."""
    path = out_dir / "logs" / "loss.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sec = sorted(float(r["sec_per_step"]) for r in rows if r.get("sec_per_step"))
    vram = [float(r["vram_gb"]) for r in rows if r.get("vram_gb")]
    return {"sec_per_step_median": sec[len(sec) // 2] if sec else float("nan"), "vram_gb_max": max(vram) if vram else float("nan")}


def eval_run(out_dir: Path, sweep: dict, cache: str, device_name: str) -> Dict[str, float]:
    import torch
    from greg.data.dataset import read_split
    from greg.data.splits import pick_keys
    from greg.eval import metrics as M
    from greg.eval.evaluate import evaluate_reconstruction
    from greg.eval.runner import load_greg
    from greg.utils.config import get_device

    ev = sweep.get("eval", {})
    device = get_device(device_name or "auto")
    model, cfg = load_greg(out_dir / "checkpoints" / "last.pth", device)
    cache = cache or cfg["data"]["cache"]
    keys = pick_keys(read_split(Path(cache) / "splits" / f"{ev.get('split', 'val')}.txt"), ev.get("max_clips", 100), seed=0)
    lp = M.LPIPSMetric(device) if ev.get("lpips", True) else None
    _, summary = evaluate_reconstruction(
        model, cache, keys, device, batch_size=int(ev.get("batch_size", 16)), num_workers=2,
        max_rows=ev.get("max_rows", 16), uniform_rows=True, lpips_metric=lp,
        eval_region_size=int(ev.get("eval_region_size", 64)))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def collect(sweep: dict, out_root: Path) -> None:
    runs = []
    for name in sweep["runs"]:
        f = out_root / name / "val_metrics.json"
        if f.exists():
            runs.append(json.loads(f.read_text(encoding="utf-8")))
    if not runs:
        print("no finished runs")
        return
    metrics = [m for m in sweep.get("rank_metrics", ["l1", "ssim", "lpips", "region_l1"]) if f"mean_{m}" in runs[0]]
    for m in metrics:   # rank 1 = best
        order = sorted(runs, key=lambda r: r[f"mean_{m}"], reverse=m in HIGHER_IS_BETTER)
        for i, r in enumerate(order):
            r.setdefault("ranks", {})[m] = i + 1
    for r in runs:
        r["mean_rank"] = sum(r["ranks"].values()) / max(len(metrics), 1)
    runs.sort(key=lambda r: r["mean_rank"])
    base = next((r for r in runs if r["name"] == "base"), None)
    cols = ["name", "mean_rank"] + [f"mean_{m}" for m in metrics] + ["sec_per_step_median", "vram_gb_max", "train_min"]
    with open(out_root / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["overrides"], extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({**r, "overrides": " ".join(r["overrides"])})
    print("\n| run | rank | " + " | ".join(metrics) + " | s/step | VRAM GB |")
    print("|---|---|" + "---|" * (len(metrics) + 2))
    for r in runs:
        cells = []
        for m in metrics:
            v = r[f"mean_{m}"]
            d = f" ({100 * (v - base[f'mean_{m}']) / abs(base[f'mean_{m}']):+.1f}%)" if base and r is not base else ""
            cells.append(f"{v:.4f}{d}")
        print(f"| {r['name']} | {r['mean_rank']:.2f} | " + " | ".join(cells) +
              f" | {r.get('sec_per_step_median', float('nan')):.2f} | {r.get('vram_gb_max', float('nan')):.1f} |")
    print(f"\nwritten {out_root / 'sweep_results.csv'}  (differences in % are relative to 'base'; "
          "single seed and ~100 val clips: treat small differences as noise, re-run the finalists with `seed=1` in `common`)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--flow-checkpoint", default=None)
    p.add_argument("--cache", default=None, help="overrides data.cache")
    p.add_argument("--device", default=None)
    p.add_argument("--only", nargs="*", default=None, help="run only these names")
    p.add_argument("--force", action="store_true", help="re-train and re-evaluate finished runs")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--allow-test", action="store_true")
    args = p.parse_args()

    sweep = yaml.safe_load(open(args.sweep, encoding="utf-8"))
    out_root = Path(args.out_root)
    if sweep.get("eval", {}).get("split", "val") == "test" and not args.allow_test:
        raise SystemExit("eval.split=test: hyper-parameters must be selected on val (use --allow-test to override)")
    if not args.collect_only:
        if not args.flow_checkpoint and not all((out_root / n / "checkpoints" / "last.pth").exists() for n in sweep["runs"]):
            raise SystemExit("--flow-checkpoint is required")
        for name, overrides in sweep["runs"].items():
            if args.only and name not in args.only:
                continue
            out_dir = out_root / name
            done = out_dir / "val_metrics.json"
            if done.exists() and not args.force:
                print(f"[skip] {name} (finished)")
                continue
            print(f"\n=== {name}: {overrides}")
            minutes = train_run(name, overrides, sweep, args, out_dir)
            summary = eval_run(out_dir, sweep, args.cache, args.device)
            summary.update(name=name, overrides=overrides, train_min=minutes, **cost_from_log(out_dir))
            done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in summary.items() if k.startswith("mean_")}, indent=2))
    collect(sweep, out_root)


if __name__ == "__main__":
    main()
