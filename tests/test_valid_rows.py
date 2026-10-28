import json

import numpy as np
from PIL import Image

from greg.data.dataset import HamNoSysCache, PairDataset, SequenceDataset
from greg.data.frames import fade_trim


def test_fade_trim_keeps_middle_and_flat_clips():
    fade = [0.02, 0.05, 0.1, 0.2] + [0.25] * 20 + [0.1, 0.01]
    assert fade_trim(fade) == (4, 23)
    assert fade_trim([0.49] * 30) == (0, 29)
    assert fade_trim([0.25] * 10 + [0.1] * 2 + [0.25] * 10) == (0, 21)   # dark rows mid-clip are kept


def _fake_cache(root, clips):
    """clips: {key: (num_rows, valid_range or None)} with 8x8 frames/normals and zero joints."""
    meta = {"clips": {}, "config": {}}
    for key, (n, vr) in clips.items():
        meta["clips"][key] = {"status": "ok", "num_rows": n, "crop": {"size": 8}}
        if vr is not None:
            meta["clips"][key]["valid_range"] = list(vr)
        for sub, mode in (("frames", "RGB"), ("normals", "RGBA")):
            (root / sub / key).mkdir(parents=True, exist_ok=True)
            for r in range(n):
                Image.new(mode, (8, 8)).save(root / sub / key / f"{r:06d}.png")
        (root / "joints").mkdir(exist_ok=True)
        np.save(root / "joints" / f"{key}.npy", np.zeros((n, 144, 2), np.float32))
    (root / "meta.json").write_text(json.dumps(meta))
    return HamNoSysCache(root)


def test_valid_rows_restrict_pairs_and_reconstruction(tmp_path):
    cache = _fake_cache(tmp_path, {"gsl_1": (12, (3, 9)), "pjm_1": (5, None), "gsl_2": (6, (2, 2))})
    assert cache.valid_rows("gsl_1") == list(range(3, 10))
    assert cache.valid_rows("pjm_1") == list(range(5))          # caches without valid_range keep every row

    pairs = PairDataset(cache, pairs_per_clip=20)
    assert "gsl_2" not in pairs.keys                            # < 2 usable rows
    for i in range(len(pairs)):
        it = pairs[i]
        if it["key"] == "gsl_1":
            assert 3 <= it["source_row"] <= 9 and 3 <= it["target_row"] <= 9

    seq = SequenceDataset(cache, source_row=0)
    gsl = [(s, t) for k, s, t in seq.items if k == "gsl_1"]
    assert {s for s, _ in gsl} == {3}                           # source = first frame after the fade-in
    assert sorted(t for _, t in gsl) == list(range(4, 10))
    assert {s for k, s, _ in seq.items if k == "pjm_1"} == {0}
    assert not [1 for k, *_ in seq.items if k == "gsl_2"]       # single usable row -> no targets
