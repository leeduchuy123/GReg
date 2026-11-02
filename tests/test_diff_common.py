"""Shared infrastructure of the diffusion branches: canonical coordinate map, frame-directory evaluation, the new
metrics (hand_lpips geometry, disocc_l1, lowfreq_drift).  CPU only, synthetic data."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from greg.eval import metrics as M
from greg.eval.evaluate import evaluate_reconstruction, hand_lpips
from greg.eval.frame_dir import FrameDirModel
from greg.geometry.canonical import canonical_coordinate_map
from greg.geometry.mesh_flow import correspond


# ----------------------------------------------------------------------------- canonical coordinates
def test_ccm_interpolates_template_coordinates_and_is_zero_on_background():
    # one triangle covering the lower-left half of a 4x4 image; vertex coordinates = unit basis
    faces = torch.tensor([[0, 1, 2]])
    coords = torch.eye(3)
    faces_t = torch.full((1, 4, 4), -1, dtype=torch.long)
    bary = torch.zeros(1, 4, 4, 3)
    faces_t[0, 2, 1] = 0
    bary[0, 2, 1] = torch.tensor([0.2, 0.3, 0.5])
    ccm = canonical_coordinate_map(faces_t, bary, faces, coords)
    assert ccm.shape == (1, 3, 4, 4)
    assert torch.allclose(ccm[0, :, 2, 1], torch.tensor([0.2, 0.3, 0.5]))
    assert ccm.sum() == pytest.approx(1.0)


def test_correspond_returns_target_barycentrics():
    # two identical renders of one big triangle: every covered pixel maps to itself with bary summing to 1
    uvz = torch.tensor([[[0.0, 0.0, 2.0], [7.0, 0.0, 2.0], [0.0, 7.0, 2.0]]])
    faces = torch.tensor([[0, 1, 2]])
    fm = torch.full((1, 8, 8), -1, dtype=torch.long)
    for y in range(8):
        for x in range(8):
            if x + y <= 7:
                fm[0, y, x] = 0
    g = correspond(uvz, fm, uvz, fm, faces)
    b = g["bary_target"]
    assert b.shape == (1, 8, 8, 3)
    assert torch.allclose(b[0][fm[0] >= 0].sum(-1), torch.ones(int((fm[0] >= 0).sum())), atol=1e-5)
    assert float(b[0][fm[0] < 0].abs().sum()) == 0.0
    assert float(g["visibility"][0, 0][fm[0] >= 0].mean()) > 0.9


# ----------------------------------------------------------------------------- metrics
def test_lowfreq_l1_sees_colour_shift_not_texture():
    torch.manual_seed(0)
    base = torch.rand(1, 3, 64, 64)
    textured = (base + 0.3 * (torch.rand(1, 3, 64, 64) - 0.5)).clamp(0, 1)   # high-frequency noise only
    shifted = (base + 0.1).clamp(0, 1)                                          # global colour drift
    assert float(M.lowfreq_l1(textured, base, 8.0)) < float(M.lowfreq_l1(shifted, base, 8.0))
    assert float(M.lowfreq_l1(base, base, 8.0)) == 0.0
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :32, :] = 1
    assert float(M.lowfreq_l1(shifted, base, 8.0, mask)) == pytest.approx(float(M.lowfreq_l1(shifted, base, 8.0)), rel=0.2)


class _MeanLPIPS:
    """Stand-in for the LPIPS metric: mean absolute difference per sample."""
    def __call__(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3))


def test_hand_lpips_uses_hand_boxes_and_skips_absent_hands():
    size = 64
    pred = torch.zeros(2, 3, size, size)
    target = torch.zeros(2, 3, size, size)
    target[:, :, 40:56, 8:24] = 1.0                       # the left hand region differs in both samples
    joints = torch.full((2, 144, 2), 32.0)
    # left hand joints (20, 25..39, tips 66..70) spread over the box [8, 24) x [40, 56)
    lh = [20] + list(range(25, 40)) + list(range(66, 71))
    joints[:, lh, 0] = torch.linspace(9, 22, len(lh))
    joints[:, lh, 1] = torch.linspace(41, 54, len(lh))
    # right hand of sample 1 off-frame (centre outside) -> only the left hand counts; sample 0 both hands
    centers = torch.tensor([[[32.0, 20.0], [16.0, 48.0], [48.0, 32.0]], [[32.0, 20.0], [16.0, 48.0], [-10.0, 32.0]]])
    v = hand_lpips(_MeanLPIPS(), pred, target, centers, joints, size, crop_size=16, upsample=32)
    assert v.shape == (2,)
    assert v[0] > 0.3                                    # the differing hand dominates its crop
    assert v[1] > 0.3                                    # right hand skipped, left hand still scored
    # no evaluable hand -> NaN
    centers_none = torch.tensor([[[32.0, 20.0], [-10.0, 48.0], [-10.0, 32.0]]])
    assert torch.isnan(hand_lpips(_MeanLPIPS(), pred[:1], target[:1], centers_none, joints[:1], size, 16, 32)).all()


# ----------------------------------------------------------------------------- frame directory evaluation
def _fake_cache(root, n_rows=6, size=32):
    for sub in ("frames", "normals", "joints", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    meta = {"clips": {}}
    for key in ("a", "b"):
        (root / "frames" / key).mkdir(exist_ok=True)
        (root / "normals" / key).mkdir(exist_ok=True)
        for r in range(n_rows):
            Image.fromarray(np.full((size, size, 3), 40 * r, np.uint8)).save(root / "frames" / key / f"{r:06d}.png")
            Image.fromarray(np.full((size, size, 4), 128, np.uint8)).save(root / "normals" / key / f"{r:06d}.png")
        np.save(root / "joints" / f"{key}.npy", np.full((n_rows, 144, 2), size / 2, np.float32))
        meta["clips"][key] = {"status": "ok", "num_rows": n_rows, "crop": {"size": size}}
    (root / "meta.json").write_text(json.dumps(meta))
    (root / "splits" / "test.txt").write_text("a\nb\n")
    return root


def test_frame_dir_model_scores_exported_rows_only(tmp_path):
    cache = _fake_cache(tmp_path / "cache")
    frames = tmp_path / "frames"
    index = {"clips": {}}
    for key in ("a", "b"):
        (frames / key).mkdir(parents=True)
        rows = [1, 3, 5]
        index["clips"][key] = {"source": 0, "rows": rows}
        for r in rows:   # generated = target + 10 (uint8) -> L1 = 10/255 ; warp = exact copy of the target
            Image.fromarray(np.full((32, 32, 3), 40 * r + 10, np.uint8)).save(frames / key / f"{r:06d}.png")
            Image.fromarray(np.full((32, 32, 3), 40 * r, np.uint8)).save(frames / key / f"{r:06d}_warp.png")
    (frames / "index.json").write_text(json.dumps(index))
    model = FrameDirModel(frames, 32)
    allowed = {k: set(v["rows"]) for k, v in index["clips"].items()}
    rows, summary = evaluate_reconstruction(model, cache, ["a", "b"], torch.device("cpu"), batch_size=4,
                                            items_filter=lambda k, s, t: t in allowed[k])
    assert summary["num_frames"] == 6 and summary["num_clips"] == 2
    assert summary["mean_l1"] == pytest.approx(10 / 255, abs=1e-6)
    assert summary["mean_warped_l1"] == pytest.approx(0.0, abs=1e-6)
    assert summary["mean_lowfreq_drift"] == pytest.approx(10 / 255, abs=1e-6)
    with pytest.raises(FileNotFoundError):
        evaluate_reconstruction(model, cache, ["a"], torch.device("cpu"), batch_size=4)   # row 2 was not exported
