"""A1 on the real SMPL-X model: warping a vertex-coloured render of the source pose with the mesh flow must
reproduce the target render wherever V = 1 (docs/A1_B1_kien_truc.md Sec. 9).  Skipped without the ``smplx``
package or the model files (``models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz``)."""
import math
from pathlib import Path

import pytest
import torch

SMPLX_DIR = Path(__file__).resolve().parents[1] / "models_smplx_v1_1" / "models"
pytest.importorskip("smplx")
if not (SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL.npz").exists():
    pytest.skip("SMPL-X model files not available", allow_module_level=True)

from greg.geometry.mesh_flow import MeshCorrespondence  # noqa: E402
from greg.models.tpsmm import warp  # noqa: E402
from greg.render.body import Camera  # noqa: E402
from greg.render.normals import NormalRenderer  # noqa: E402
from greg.render.rasterize import interpolate_vertex_attribute  # noqa: E402

SIZE = 256


def _vec(left_shoulder_z, left_elbow_y):
    v = torch.zeros(1, 182)
    v[0, 0] = math.pi                                   # SMPL-X y-up -> OpenCV y-down (as in SignAvatars)
    v[0, 3 + 15 * 3 + 2] = left_shoulder_z               # joint 16 left shoulder
    v[0, 3 + 16 * 3 + 2] = 1.2                           # joint 17 right shoulder: arm down
    v[0, 3 + 17 * 3 + 1] = left_elbow_y                  # joint 18 left elbow
    v[0, 179:182] = torch.tensor([0.0, 0.35, 2.2])
    return v


def test_mesh_flow_reproduces_target_on_visible_pixels():
    mc = MeshCorrespondence(str(SMPLX_DIR), device="cpu")
    renderer = NormalRenderer(mc.body)
    focal, princpt = torch.tensor([[420.0, 420.0]]), torch.tensor([[128.0, 128.0]])
    vt = mc.body.model.v_template.float()
    colors = ((vt - vt.min(0).values) / (vt.max(0).values - vt.min(0).values))[None]
    src, tgt = _vec(-1.2, 0.0), _vec(-0.3, -2.2)        # arm down -> arm raised, forearm folded
    renders, imgs = [], []
    for v in (src, tgt):
        r = renderer.render(v, Camera(focal, princpt, v[:, 179:182]), SIZE, SIZE, keep_fragments=True)
        renders.append(r)
        imgs.append(interpolate_vertex_attribute(r.frag, colors, mc.faces) * r.mask + 0.2 * (1 - r.mask))
    g = mc(src, focal, princpt, renders[0].frag.face_idx, tgt, focal, princpt, renders[1].frag.face_idx)
    err = (warp(imgs[0], g["grid"]) - imgs[1]).abs().mean(1, keepdim=True)
    body, V = renders[1].mask, g["visibility"]
    vis_body = body * V
    assert float(vis_body.sum() / body.sum()) > 0.9                       # most of the body is visible
    assert float((err * vis_body).sum() / vis_body.sum()) < 0.005         # and warped exactly there
    hidden = body * (1 - V)
    assert float((err * hidden).sum() / hidden.sum()) > 0.03              # the hidden part really differs
    arm = ((g["parts"] >= 3).float())[:, None]
    assert 0.1 < float((arm * (1 - V)).sum() / arm.sum()) < 0.6           # the raised arm reveals new pixels
    assert set(g["parts"].unique().tolist()) == {0, 1, 2, 3, 4, 5, 6}
