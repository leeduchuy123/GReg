"""Post-D6 (A): base frames (GReg run5 or D1) everywhere, D6 frames inside the two hand boxes, feathered.

The boxes come from the *target pose* only (SMPL-X joints -> adaptive hand boxes, exactly the geometry GReg is
conditioned on), never from the target image, so the composite is a legal inference-time method.  Scored with the
same evaluate_reconstruction as every other run (D1 checkout: /workspace/GReg-d1).

Note: hand_l1 / hand_lpips of the composite are ~those of D6 by construction (the composite region covers the metric
hand boxes); what this tests is whether the *global* metrics (LPIPS, L1, face, drift) stay at GReg's level.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/GReg-d1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from greg.data.regions import adaptive_region_boxes, region_visibility  # noqa: E402
from greg.eval import metrics as M  # noqa: E402
from greg.eval.evaluate import evaluate_reconstruction  # noqa: E402
from greg.eval.frame_dir import FrameDirModel  # noqa: E402
from greg.render.body import region_extents_from_joints  # noqa: E402

HANDS = (1, 2)


class ComposeModel(FrameDirModel):
    def __init__(self, base, hands, image_size=256, margin=0.5, feather=9):
        super().__init__(base, image_size)
        self.hands = FrameDirModel(hands, image_size)
        self.margin, self.feather = float(margin), int(feather)

    def extra_inputs(self, batch):
        d = super().extra_inputs(batch)
        d["joints"] = batch["target_joints"]
        return d

    def forward(self, image_source, normal_source, normal_target, centers_target, keys=None, rows=None, joints=None,
                **_):
        base = super().forward(image_source, normal_source, normal_target, centers_target, keys=keys, rows=rows)
        hand = self.hands(image_source, normal_source, normal_target, centers_target, keys=keys, rows=rows)
        b, s = image_source.shape[0], self.image_size
        ext = region_extents_from_joints(joints)
        boxes = adaptive_region_boxes(centers_target, ext, 64, s, s, hand_margin=self.margin, min_side=24.0,
                                      max_side=float(s))
        vis = region_visibility(centers_target, s, s)
        ys = torch.arange(s, device=image_source.device).view(1, s, 1).float()
        xs = torch.arange(s, device=image_source.device).view(1, 1, s).float()
        m = torch.zeros(b, s, s, device=image_source.device)
        for r in HANDS:
            x0, y0, sd = boxes[:, r, 0, None, None], boxes[:, r, 1, None, None], boxes[:, r, 2, None, None]
            inside = (xs >= x0) & (xs < x0 + sd) & (ys >= y0) & (ys < y0 + sd) & (vis[:, r, None, None] > 0)
            m = torch.maximum(m, inside.float())
        m = m[:, None]
        if self.feather > 1:
            k, p = self.feather, self.feather // 2
            m = F.avg_pool2d(F.avg_pool2d(m, k, 1, p, count_include_pad=False), k, 1, p, count_include_pad=False)
        out = m * hand["output"] + (1 - m) * base["output"]
        return {"output": out, "warped": base["warped"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--hands", required=True)
    p.add_argument("--index", required=True, help="index.json of the rows to score (the D5/D6 200-clip rows)")
    p.add_argument("--cache", required=True)
    p.add_argument("--smplx-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--feather", type=int, default=9)
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComposeModel(a.base, a.hands, 256, a.margin, a.feather).to(dev)
    idx = json.loads(Path(a.index).read_text(encoding="utf-8"))["clips"]
    allowed = {k: set(v["rows"]) for k, v in idx.items()}
    rows, summary = evaluate_reconstruction(
        model, a.cache, sorted(allowed), dev, batch_size=16, num_workers=a.num_workers, lpips_metric=M.LPIPSMetric(dev),
        smplx_dir=a.smplx_dir, items_filter=lambda k, s, t: t in allowed.get(k, ()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary.update(base=a.base, hands=a.hands, margin=a.margin, feather=a.feather)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 5) for k, v in summary.items() if k.startswith("mean_") and "@" not in k}))


if __name__ == "__main__":
    main()
