"""D4: clip-consistent augmentation and the shared-epoch fix of PairDataset."""
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from greg.data.augment import ClipConsistentAugment, hsv_to_rgb, rgb_to_hsv
from greg.data.dataset import HamNoSysCache, PairDataset, collate_keep_strings


def _cache(root, n_rows=8, size=16, keys=("a", "b", "c", "d")):
    for sub in ("frames", "normals", "joints", "matte"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    meta = {"clips": {}}
    for key in keys:
        for sub in ("frames", "normals", "matte"):
            (root / sub / key).mkdir(exist_ok=True)
        for r in range(n_rows):
            Image.fromarray(np.full((size, size, 3), (10 * r) % 250, np.uint8)).save(root / "frames" / key / f"{r:06d}.png")
            Image.fromarray(np.full((size, size, 4), 128, np.uint8)).save(root / "normals" / key / f"{r:06d}.png")
            m = np.zeros((size, size), np.uint8)
            m[4:12, 4:12] = 255
            Image.fromarray(m).save(root / "matte" / key / f"{r:06d}.png")
        np.save(root / "joints" / f"{key}.npy", np.full((n_rows, 144, 2), size / 2, np.float32))
        meta["clips"][key] = {"status": "ok", "num_rows": n_rows, "crop": {"size": size}}
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_hsv_roundtrip():
    x = torch.rand(3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5)


def test_background_is_the_same_for_source_and_target_and_person_kept(tmp_path):
    root = _cache(tmp_path / "c")
    bg = tmp_path / "bg"
    bg.mkdir()
    Image.fromarray(np.full((16, 16, 3), 250, np.uint8)).save(bg / "0000.png")
    aug = ClipConsistentAugment(root, bg, background_p=1.0, color_p=0.0)
    ds = PairDataset(HamNoSysCache(root), seed=0)
    ds.augment = aug
    it = ds[0]
    for side in ("source", "target"):
        img = it[f"{side}_image"]
        assert abs(float(img[:, 0, 0].mean()) - 250 / 255) < 0.02       # background replaced
        row = int(it[f"{side}_row"])
        assert abs(float(img[:, 8, 8].mean()) - 10 * row / 255) < 0.02   # person untouched
    d1, d2 = aug.draw("a", 3), aug.draw("a", 3)
    assert d1 == d2


def test_shared_epoch_reaches_persistent_workers(tmp_path):
    root = _cache(tmp_path / "c", n_rows=30)

    def pairs(shared):
        ds = PairDataset(HamNoSysCache(root), seed=0, shared_epoch=shared)
        dl = DataLoader(ds, batch_size=4, num_workers=1, persistent_workers=True, collate_fn=collate_keep_strings)
        out = []
        for ep in range(2):
            ds.set_epoch(ep)
            b = next(iter(dl))
            out.append(list(zip(b["source_row"].tolist(), b["target_row"].tolist())))
        return out

    fixed = pairs(False)
    assert fixed[0] == fixed[1]                     # the known issue: same pairs in every epoch
    fresh = pairs(True)
    assert fresh[0] != fresh[1]                     # fixed: the workers see the new epoch
