"""Post-D6 (C): paired clip-level bootstrap of the differences between runs (per_frame.csv of the eval scripts).

Each comparison is paired on the (key, target_row) rows both runs scored; clips are resampled with replacement
(frames of a clip are correlated, so resampling frames would make the intervals far too narrow).  The statistic is
the frame-weighted mean over the resampled clips, i.e. the same mean the summaries report.
Usage: bootstrap_ci.py --out report.md  name=A.csv:B.csv  ...
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np

METRICS = ["lpips", "l1", "psnr", "ssim", "hand_lpips", "hand_l1", "face_l1", "lowfreq_drift", "arm_l1", "disocc_l1"]
HIGHER = {"psnr", "ssim"}


def load(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["key"], int(r["target_row"]))] = r
    return d


def val(r, m):
    try:
        v = float(r.get(m, "nan"))
    except ValueError:
        return math.nan
    return v


def compare(a_path, b_path, n_boot, rng):
    A, B = load(a_path), load(b_path)
    common = sorted(set(A) & set(B))
    clips = sorted({k for k, _ in common})
    ci = {c: i for i, c in enumerate(clips)}
    res = {"frames": len(common), "clips": len(clips)}
    for m in METRICS:
        sa, sb, n = np.zeros(len(clips)), np.zeros(len(clips)), np.zeros(len(clips))
        for k, t in common:
            va, vb = val(A[(k, t)], m), val(B[(k, t)], m)
            if math.isnan(va) or math.isnan(vb):
                continue
            i = ci[k]
            sa[i] += va; sb[i] += vb; n[i] += 1
        if n.sum() == 0:
            continue
        ma, mb = sa.sum() / n.sum(), sb.sum() / n.sum()
        idx = rng.integers(0, len(clips), size=(n_boot, len(clips)))
        cnt = n[idx].sum(1)
        d = (sb[idx].sum(1) - sa[idx].sum(1)) / np.maximum(cnt, 1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = mb - ma
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        better = (delta > 0) if m in HIGHER else (delta < 0)
        sig = lo > 0 or hi < 0
        res[m] = dict(a=ma, b=mb, delta=delta, rel=delta / ma * 100 if ma else math.nan, lo=lo, hi=hi, p=p,
                      verdict=("B better" if better else "B worse") if sig else "n.s.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("pairs", nargs="+", help="name=A.csv:B.csv (delta = B - A)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = ["# Paired clip bootstrap (95 %% CI of B - A, %d resamples)" % a.boot, ""]
    for spec in a.pairs:
        name, paths = spec.split("=", 1)
        pa, pb = paths.split(":", 1)
        try:
            r = compare(pa, pb, a.boot, rng)
        except FileNotFoundError as e:
            lines += ["## %s" % name, "missing: %s" % e, ""]
            continue
        lines += ["## %s" % name, "A=`%s`  B=`%s`  (%d frames, %d clips)" % (pa, pb, r["frames"], r["clips"]), "",
                  "| metric | A | B | B-A | rel | 95 % CI | p | verdict |", "|---|---|---|---|---|---|---|---|"]
        for m in METRICS:
            if m in r:
                x = r[m]
                lines.append("| %s | %.5f | %.5f | %+.5f | %+.1f%% | [%+.5f, %+.5f] | %.3f | %s |" % (
                    m, x["a"], x["b"], x["delta"], x["rel"], x["lo"], x["hi"], x["p"], x["verdict"]))
        lines.append("")
        print(name, {m: r[m]["verdict"] for m in ("lpips", "hand_lpips", "face_l1") if m in r}, flush=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print("written", a.out)


if __name__ == "__main__":
    main()
