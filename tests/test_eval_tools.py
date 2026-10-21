import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from greg.data.dataset import HamNoSysCache, SequenceDataset
from greg.data.splits import make_random_split, make_val_split, pick_keys
from greg.eval.evaluate import evaluate_reconstruction, summarize
from greg.eval.fvd import frechet_distance, fvd_from_features, preprocess, uniform_indices


def test_val_split_is_inside_train_and_leaves_test_alone():
    keys = [f"k{i}" for i in range(100)]
    s = make_random_split(keys, 0.8, seed=0)
    v = make_val_split(s["train"], 0.1, seed=0)
    assert len(v["val"]) == 8 and len(v["train_fit"]) == 72
    assert set(v["val"]) | set(v["train_fit"]) == set(s["train"])
    assert not set(v["val"]) & set(v["train_fit"]) and not set(v["val"]) & set(s["test"])
    assert v == make_val_split(s["train"], 0.1, seed=0)


def test_pick_keys_deterministic_subset():
    keys = [str(i) for i in range(50)]
    a = pick_keys(keys, 10, seed=3)
    assert a == pick_keys(keys, 10, seed=3) and len(a) == 10 and set(a) <= set(keys)
    assert pick_keys(keys, None) == sorted(keys) and pick_keys(keys, 99) == sorted(keys)


def test_frechet_distance_known_values():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 8))
    assert fvd_from_features(x, x)["fvd"] < 1e-6
    shifted = x + 3.0                                  # same covariance, mean moved by 3 in 8 dims
    assert frechet_distance(*(x.mean(0), np.cov(x, rowvar=False)), *(shifted.mean(0), np.cov(shifted, rowvar=False))) \
        == pytest.approx(8 * 9.0, rel=1e-6)
    r = fvd_from_features(x, shifted, num_bootstrap=3)
    assert r["fvd_boot_std"] >= 0 and r["num_real"] == 2000


def test_uniform_indices_and_preprocess_range():
    assert uniform_indices(5, 3) == [0, 2, 4] and uniform_indices(2, 4) == [0, 1, 1, 1]
    v = np.zeros((2, 4, 16, 16, 3), dtype=np.uint8)
    v[1] = 255
    x = preprocess(v, size=8)
    assert x.shape == (2, 3, 4, 8, 8) and x[0].max() == pytest.approx(-1, abs=1e-5) and x[1].min() == pytest.approx(1, abs=1e-5)


class _Identity(nn.Module):
    image_size = 32

    def forward(self, Is, Ns, Nt, ct):
        return {"output": Is, "warped": Is}


def _fake_cache(root, n_rows=6, size=32):
    for sub in ("frames", "normals", "joints"):
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
    return root


def test_sequence_dataset_uniform_rows(tmp_path):
    cache = HamNoSysCache(_fake_cache(tmp_path), ["a"])
    first = [t for _, _, t in SequenceDataset(cache, max_rows=3).items]
    spread = [t for _, _, t in SequenceDataset(cache, max_rows=3, uniform_rows=True).items]
    assert first == [1, 2, 3] and spread == [1, 3, 5]


def test_evaluate_reconstruction_identity_model(tmp_path):
    root = _fake_cache(tmp_path)
    rows, s = evaluate_reconstruction(_Identity(), root, ["a", "b"], torch.device("cpu"), batch_size=4,
                                      eval_region_size=8, max_rows=3, uniform_rows=True)
    assert len(rows) == 6 and s["num_clips"] == 2 and s["eval_region_size"] == 8
    # the "model" copies the source (row 0, value 0); target row t has value 40 t / 255
    assert s["mean_l1"] == pytest.approx(np.mean([40 * t / 255 for t in (1, 3, 5)]), abs=1e-5)
    assert s["mean_l1"] == pytest.approx(s["mean_source_copy_l1"], abs=1e-6)
    assert summarize([])["num_frames"] == 0
