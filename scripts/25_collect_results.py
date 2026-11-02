#!/usr/bin/env python
"""Collect every ``summary.json`` under a results root into one markdown table, relative to a baseline.

    python scripts/25_collect_results.py --root outputs/d1 --baseline outputs/d1/baseline_greg_val --out outputs/d1/results_val.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ["l1", "psnr", "ssim", "lpips", "hand_l1", "hand_lpips", "disocc_l1", "lowfreq_drift", "arm_l1",
        "hand_l1@large", "lpips@large"]
HIGHER = {"psnr", "ssim"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    rows = []
    for f in sorted(root.rglob("summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        rows.append((str(f.parent.relative_to(root)), s))
    base = json.loads((Path(a.baseline) / "summary.json").read_text()) if a.baseline else None
    lines = ["| run | frames | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for name, s in rows:
        cells = []
        for c in COLS:
            v = s.get(f"mean_{c}")
            if v is None:
                cells.append("–")
                continue
            d = ""
            if base and base.get(f"mean_{c}") not in (None, 0) and s is not base:
                rel = 100 * (v - base[f"mean_{c}"]) / abs(base[f"mean_{c}"])
                better = rel > 0 if c in HIGHER else rel < 0
                d = f" ({rel:+.1f}%{'✓' if better and abs(rel) > 0.5 else ''})"
            cells.append(f"{v:.4f}{d}")
        lines.append(f"| {name} | {s.get('num_frames')} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
