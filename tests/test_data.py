import numpy as np
import torch

from greg.data.frames import check_alignment, ffmpeg_rate_args
from greg.data.gradient import gradient_map
from greg.data.regions import (crop_regions, paste_regions, region_boxes, region_gaussians, region_visibility)
from greg.data.signavatars import SMPLX_DIM, SMPLX_SLICES
from greg.data.splits import make_random_split, source_of_key


def test_smplx_slices_cover_182():
    covered = sorted(i for sl in SMPLX_SLICES.values() for i in range(sl.start, sl.stop))
    assert covered == list(range(SMPLX_DIM))


def test_fps_rule():
    assert ffmpeg_rate_args(25.0) == []
    assert ffmpeg_rate_args(24.0) == []
    assert ffmpeg_rate_args(50.0) == ["-r", "25"]


def test_alignment_classes():
    assert check_alignment(77, np.arange(77)) == "exact"
    assert check_alignment(74, np.arange(2, 72)) == "subset_by_total_valid_index"
    assert check_alignment(70, np.arange(0, 88)) == "MISMATCH"


def test_gradient_eq2_constant_image_is_eps():
    x = torch.full((1, 3, 8, 8), 0.5)
    g = gradient_map(x, eps=1e-6)
    # interior: zero differences -> sqrt(eps); first row/col see the zero padding
    assert torch.allclose(g[:, :, 1:, 1:], torch.full((1, 1, 7, 7), 1e-3), atol=1e-6)
    assert g[0, 0, 0, 0] > 0.5


def test_gradient_step_edge():
    x = torch.zeros(1, 1, 4, 4)
    x[:, :, :, 2:] = 1.0  # vertical edge between columns 1 and 2
    g = gradient_map(x)
    assert g[0, 0, 2, 2] > 0.9 and g[0, 0, 2, 3] < 0.01


def test_region_crop_paste_roundtrip():
    x = torch.rand(2, 3, 64, 64)
    centers = torch.tensor([[[10.0, 10.0], [32.0, 32.0], [60.0, 60.0]]] * 2)
    boxes = region_boxes(centers, 16, 64, 64)
    assert boxes.min() >= 0 and boxes.max() <= 48
    crops = crop_regions(x, boxes, 16)
    assert crops.shape == (2, 3, 3, 16, 16)
    canvas = paste_regions(crops[:, :, :1], boxes, 64, 64)
    assert canvas.shape == (2, 3, 64, 64)
    x0, y0 = int(boxes[0, 1, 0]), int(boxes[0, 1, 1])
    assert torch.allclose(canvas[0, 1, y0:y0 + 16, x0:x0 + 16], x[0, 0, y0:y0 + 16, x0:x0 + 16])


def test_region_gaussian_peak_and_visibility():
    centers = torch.tensor([[[5.0, 7.0], [-20.0, 3.0], [15.0, 15.0]]])
    g = region_gaussians(centers, 16, 16, 2.0)
    assert g.shape == (1, 3, 16, 16)
    assert torch.isclose(g[0, 0, 7, 5], torch.tensor(1.0))
    vis = region_visibility(centers, 16, 16)
    assert vis.tolist() == [[1.0, 0.0, 1.0]]


def test_split_is_deterministic_and_disjoint():
    keys = [f"pjm_{i}" for i in range(50)] + [str(10000 + i) for i in range(30)]
    a = make_random_split(keys, 0.8, seed=0)
    b = make_random_split(keys, 0.8, seed=0)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert len(a["train"]) == 64
    assert source_of_key("pjm_1") == "PJM" and source_of_key("10010") == "DGS" and source_of_key("LETTRE") == "LSF"
