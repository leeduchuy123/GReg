"""A1 (analytic mesh flow) and B1 (adaptive boxes, region masks, masked losses) on synthetic data.

No SMPL-X files are needed: ``correspond`` is tested on small quad meshes rasterised with the repo's own
rasteriser, exactly as the training code calls it on SMPL-X renders."""
import json

import numpy as np
import torch
from PIL import Image

from greg.data.dataset import HamNoSysCache, PairDataset
from greg.data.regions import (adaptive_region_boxes, crop_regions, paste_regions, region_boxes, region_masks)
from greg.geometry.mesh_flow import barycentric_2d, correspond, pixel_grid, segment_depth
from greg.losses import GRegLoss
from greg.models.discriminator import RegionalDiscriminators
from greg.models.greg import GReg
from greg.models.vgg import PerceptualLoss, VGG19Features
from greg.render.body import Camera, project
from greg.render.rasterize import rasterize

H = W = 64


def _quad(x0, x1, y0, y1, z):
    v = torch.tensor([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=torch.float32)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def _mesh(*quads):
    vs, fs, n = [], [], 0
    for q in quads:
        v, f = _quad(*q)
        vs.append(v)
        fs.append(f + n)
        n += v.shape[0]
    return torch.cat(vs)[None], torch.cat(fs)


def _render(verts, faces, focal=64.0, cx=32.0, cy=32.0):
    cam = Camera(torch.tensor([[focal, focal]]), torch.tensor([[cx, cy]]), torch.zeros(1, 3))
    uv = project(verts, cam)
    uvz = torch.cat([uv, verts[..., 2:3]], dim=-1)
    frag = rasterize(uv, verts[..., 2], faces, H, W, backend="torch", tile=16)
    return uvz, frag.face_idx


def _q(out):
    return out["source_coords"][0]


def test_barycentric_reconstructs_point():
    tri = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    p = torch.tensor([1.0, 2.0])
    b = barycentric_2d(p, tri)
    assert torch.allclose((b[:, None] * tri).sum(0), p, atol=1e-5) and torch.isclose(b.sum(), torch.tensor(1.0))


def test_identity_frame_maps_to_itself_and_is_visible():
    v, f = _mesh((-0.5, 0.5, -0.5, 0.5, 2.0))
    uvz, fm = _render(v, f)
    out = correspond(uvz, fm, uvz, fm, f, bg_dilate_px=2)
    body = fm[0] >= 0
    p = pixel_grid(H, W)
    assert (_q(out)[body] - p[body]).abs().max() < 1e-3
    assert out["visibility"][0, 0][body].min() == 1
    assert out["visibility"][0, 0, 0, 0] == 1            # far background: visible
    assert out["confidence"][0, 0][body].min() > 0.99   # no magnification
    grid = out["grid"][0]
    assert grid.min() >= -1.001 and grid.max() <= 1.001


def test_translation_is_recovered():
    v, f = _mesh((-0.4, 0.4, -0.4, 0.4, 2.0))
    uvz_s, fm_s = _render(v, f, cx=28.0)
    uvz_t, fm_t = _render(v, f, cx=33.0)          # same body, 5 px to the right in the target
    out = correspond(uvz_s, fm_s, uvz_t, fm_t, f)
    body = fm_t[0] >= 0
    d = _q(out)[body] - pixel_grid(H, W)[body]
    assert torch.allclose(d[:, 0], torch.full_like(d[:, 0], -5.0), atol=1e-3)
    assert d[:, 1].abs().max() < 1e-3
    assert out["visibility"][0, 0][body].min() == 1


def test_disocclusion_and_background_visibility():
    # source: small near quad in front of a big far quad; target: the near quad moved to the right (u ~ 48..61)
    back = (-0.5, 0.5, -0.5, 0.5, 2.0)
    v_s, f = _mesh(back, (-0.1, 0.1, -0.1, 0.1, 1.0))
    v_t, _ = _mesh(back, (0.25, 0.45, -0.1, 0.1, 1.0))
    uvz_s, fm_s = _render(v_s, f)
    uvz_t, fm_t = _render(v_t, f)
    out = correspond(uvz_s, fm_s, uvz_t, fm_t, f, bg_dilate_px=1)
    vis = out["visibility"][0, 0]
    assert fm_t[0, 32, 32] in (0, 1) and vis[32, 32] == 0       # back quad was hidden by the near quad
    assert vis[32, 5] == 1 if fm_t[0, 32, 5] >= 0 else True      # back quad, never hidden
    near = fm_t[0] >= 2
    assert near.any() and vis[near].float().mean() == 1           # the near quad itself is visible in the source
    # background: target bg pixels where the SOURCE had the body are not visible
    v1, f1 = _mesh((-0.5, -0.1, -0.2, 0.2, 2.0))
    v2, _ = _mesh((0.1, 0.5, -0.2, 0.2, 2.0))
    uvz1, fm1 = _render(v1, f1)
    uvz2, fm2 = _render(v2, f1)
    o2 = correspond(uvz1, fm1, uvz2, fm2, f1, bg_dilate_px=0)
    assert fm2[0, 32, 22] < 0 and o2["visibility"][0, 0, 32, 22] == 0   # old body position (u 16..29), now bg
    assert o2["visibility"][0, 0, 2, 2] == 1


def test_magnification_lowers_confidence():
    v, f = _mesh((-0.4, 0.4, -0.4, 0.4, 2.0))
    uvz_s, fm_s = _render(v, f, focal=32.0)        # small in the source
    uvz_t, fm_t = _render(v, f, focal=64.0)        # 2x larger in the target -> 1/4 source pixel per pixel
    out = correspond(uvz_s, fm_s, uvz_t, fm_t, f, stretch_tau=0.5)
    c = out["confidence"][0, 0]
    assert abs(float(c[32, 32]) - 0.5) < 0.02      # 0.25 / 0.5


def test_parts_from_face_labels():
    v, f = _mesh((-0.5, 0.0, -0.5, 0.5, 2.0), (0.0, 0.5, -0.5, 0.5, 2.0))
    uvz, fm = _render(v, f)
    out = correspond(uvz, fm, uvz, fm, f, face_part=torch.tensor([3, 3, 5, 5], dtype=torch.uint8))
    assert out["parts"][0, 32, 24] == 3 and out["parts"][0, 32, 40] == 5 and out["parts"][0, 0, 0] == 0


def test_segment_depth_counts_pixels_inside_the_segment():
    seg = torch.zeros(1, 21, 21, dtype=torch.long)
    seg[0, 3:18, 7:14] = 5                                     # 7 px wide strip
    d = segment_depth(seg, 5)
    assert d[0, 10, 10] == 3 and d[0, 10, 7] == 0 and d[0, 10, 8] == 1 and d[0, 0, 0] == 0


def test_edge_confidence_only_on_the_edges_of_a_moved_limb():
    # a 16 px wide "forearm" (label 5) that moves 10 px to the right between source and target
    v, f = _mesh((-0.25, 0.25, -0.5, 0.5, 2.0))
    fp = torch.tensor([5, 5], dtype=torch.uint8)
    uvz_s, fm_s = _render(v, f, cx=32.0)
    uvz_t, fm_t = _render(v, f, cx=42.0)
    moved = correspond(uvz_s, fm_s, uvz_t, fm_t, f, face_part=fp, edge_px=4)
    c = moved["confidence"][0, 0]
    assert float(c[32, 42]) > 0.99                             # middle of the limb: deep inside -> trusted
    assert float(c[32, 35]) < 0.5                              # 1 px inside the edge -> mostly generated
    assert moved["visibility"][0, 0][fm_t[0] >= 0].min() == 1   # visibility itself is unchanged
    still = correspond(uvz_s, fm_s, uvz_s, fm_s, f, face_part=fp, edge_px=4)
    assert still["confidence"][0, 0].min() > 0.99             # nothing moved: copying is exact, no penalty
    off = correspond(uvz_s, fm_s, uvz_t, fm_t, f, face_part=fp, edge_px=0)
    assert float(off["confidence"][0, 0, 32, 35]) > 0.99      # edge_px = 0 -> v1 behaviour


# ------------------------------------------------------------------------------------------- B1 regions
def test_adaptive_box_with_region_size_equals_fixed_crop():
    x = torch.rand(1, 3, 64, 64)
    centers = torch.tensor([[[20.0, 20.0], [40.0, 30.0], [10.0, 50.0]]])
    fixed = region_boxes(centers, 16, 64, 64)
    adaptive = torch.cat([fixed.float(), torch.full((1, 3, 1), 16.0)], dim=-1)
    assert torch.allclose(crop_regions(x, fixed, 16), crop_regions(x, adaptive, 16), atol=1e-5)


def test_affine_paste_roundtrip_and_zero_outside():
    x = torch.rand(1, 1, 64, 64)
    boxes = torch.tensor([[[10.0, 12.0, 24.0], [30.0, 30.0, 16.0], [0.0, 40.0, 20.0]]])
    crops = crop_regions(x.expand(1, 3, 64, 64)[:, :1], boxes, 24)
    canvas = paste_regions(crops, boxes, 64, 64)
    assert canvas.shape == (1, 3, 64, 64)
    assert torch.allclose(canvas[0, 0, 14:32, 12:30], x[0, 0, 14:32, 12:30], atol=1e-4)   # interior of box 0
    assert canvas[0, 0, 40:, :].abs().sum() == 0 and canvas[0, 0, :, 40:].abs().sum() == 0


def test_adaptive_boxes_fit_hands_and_keep_face():
    centers = torch.tensor([[[100.0, 60.0], [60.0, 200.0], [200.0, 200.0]]])
    extents = torch.tensor([[50.0, 20.0, 200.0]])
    b = adaptive_region_boxes(centers, extents, 64, 256, 256, hand_margin=0.25, min_side=24, max_side=96)
    assert b.shape == (1, 3, 3)
    assert b[0, 0, 2] == 64 and b[0, 1, 2] == 30 and b[0, 2, 2] == 96
    assert torch.isclose(b[0, 1, 0] + b[0, 1, 2] / 2 - 0.5, torch.tensor(60.0))   # centred on the hand


def test_region_masks_take_their_own_part():
    parts = torch.zeros(1, 64, 64, dtype=torch.long)
    parts[0, 5:15, 5:15] = 2          # face
    parts[0, 40:50, 40:50] = 3        # left hand
    boxes = torch.tensor([[[0.0, 0.0, 32.0], [30.0, 30.0, 32.0], [0.0, 32.0, 32.0]]])
    m = region_masks(parts, boxes, 32, dilate_px=0)
    assert m.shape == (1, 3, 1, 32, 32)
    assert m[0, 0].sum() > 50 and m[0, 1].sum() > 50 and m[0, 2].sum() == 0


def test_masked_regional_losses_ignore_the_background():
    torch.manual_seed(0)
    loss = GRegLoss(PerceptualLoss(VGG19Features(pretrained=False), scales=(1.0,)), {}, "hinge", region_size=16,
                    region_perceptual_size=32, region_mask_mode="multiply", struct_bg_weight=0.0)
    boxes = torch.tensor([[[8.0, 8.0, 16.0], [30.0, 30.0, 16.0], [40.0, 4.0, 16.0]]])
    masks = torch.zeros(1, 3, 1, 16, 16)
    masks[..., 4:12, 4:12] = 1
    img = torch.rand(1, 3, 64, 64)
    img2 = img.clone()
    img2[..., 8:12, 8:12] = 0.0      # only the un-masked rim of box 0 changes
    a = loss.region_targets(img, boxes, masks)
    b = loss.region_targets(img2, boxes, masks)
    assert torch.allclose(a["crops"], b["crops"])


# -------------------------------------------------------------------------------------------- model / data
SMALL_MESH = dict(image_size=128, region_size=32, gaussian_sigma=16.0, style_dim=32, style_base=8,
                  flow=dict(type="mesh", smplx_dir="unused-in-this-test",
                            residual=dict(enabled=True, hidden=8, max_px=6.0)), occlusion_refine=True,
                  regions=dict(mode="adaptive", min_side=16, max_side=48),
                  rpg=dict(channels=(8, 16, 16, 16)), gr=dict(base=8, max_channels=32, grm_levels=(16, 32, 64)),
                  generator=dict(base=8, max_channels=32, num_down=4))


def test_greg_mesh_adaptive_forward_backward():
    torch.manual_seed(0)
    m = GReg(SMALL_MESH).freeze_flow()
    assert m.flow is None and not any(n.startswith("flow.") for n, _ in m.named_parameters())
    B = 2
    Is, Ns, Nt, It = (torch.rand(B, 3, 128, 128) for _ in range(4))
    centers = torch.tensor([[[64.0, 30.0], [30.0, 100.0], [100.0, 100.0]]] * B)
    joints = torch.rand(B, 144, 2) * 100 + 14
    ident = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="xy"), -1)
    parts = torch.zeros(B, 128, 128, dtype=torch.long)
    parts[:, 90:110, 20:40] = 3
    body = torch.zeros(B, 1, 128, 128)
    body[:, :, 20:110, 30:100] = 1
    geo = {"grid": ident[None].expand(B, -1, -1, -1), "occlusion": torch.ones(B, 1, 128, 128) * 0.5,
           "visibility": torch.ones(B, 1, 128, 128), "confidence": torch.ones(B, 1, 128, 128), "parts": parts,
           "source_coords": pixel_grid(128, 128).expand(B, -1, -1, -1), "target_body": body}
    out = m(Is, Ns, Nt, centers, geometry=geo, joints_target=joints)
    assert out["boxes"].shape == (B, 3, 3) and out["region_masks"].shape == (B, 3, 1, 32, 32)
    assert torch.allclose(out["occlusion"], geo["occlusion"], atol=1e-6)   # zero-init refiner = geometry
    assert out["flow_residual"].abs().max() == 0                           # zero-init residual = mesh flow
    assert torch.allclose(out["deformation"], geo["grid"])
    D = RegionalDiscriminators(3, 4, base=8, num_layers=2)
    loss = GRegLoss(PerceptualLoss(VGG19Features(pretrained=False), scales=(1.0,)),
                    {"occ": 0.1, "masked_l1": 1.0, "flow_tv": 1.0, "flow_mag": 0.1},
                    "hinge", region_size=32, region_perceptual_size=64, region_mask_mode="multiply",
                    struct_bg_weight=0.1)
    g = loss.generator_loss(out, It, D)
    assert all(k in g for k in ("occ", "masked_l1", "flow_tv", "flow_mag")) and torch.isfinite(g["total"])
    g["total"].backward()
    assert any(p.grad is not None for p in m.occlusion_refiner.parameters())
    last = m.flow_residual.net[-1]
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0  # the image losses reach the residual
    with torch.no_grad():                                                    # the residual stays within max_px
        last.bias.fill_(100.0)
        o2 = m(Is, Ns, Nt, centers, geometry=geo, joints_target=joints)
    assert o2["flow_residual"].abs().max() <= 6.0 + 1e-4
    assert o2["flow_residual"][:, :, :5, :5].abs().max() == 0                 # zero away from the body
    d = loss.discriminator_loss(out, It, D)
    assert torch.isfinite(d["d_total"])


def _fake_mesh_cache(root, rows=6):
    key = "clip_a"
    for sub in ("frames", "normals", "parts", "faces"):
        (root / sub / key).mkdir(parents=True)
    for r in range(rows):
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(root / "frames" / key / f"{r:06d}.png")
        Image.fromarray(np.zeros((16, 16, 4), np.uint8)).save(root / "normals" / key / f"{r:06d}.png")
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(root / "parts" / key / f"{r:06d}.png")
        face = np.zeros((16, 16), np.uint16)
        face[4:8, 4:8] = 20908          # last SMPL-X face (+1): must survive the uint16 round trip
        Image.fromarray(face).save(root / "faces" / key / f"{r:06d}.png")
    (root / "joints").mkdir()
    np.save(root / "joints" / f"{key}.npy", np.zeros((rows, 144, 2), np.float32))
    (root / "mesh").mkdir()
    np.savez(root / "mesh" / f"{key}.npz", smplx=np.arange(rows * 182, dtype=np.float32).reshape(rows, 182),
             focal=np.ones((rows, 2), np.float32), princpt=np.zeros((rows, 2), np.float32))
    meta = {"clips": {key: {"status": "ok", "num_rows": rows, "crop": {"x0": 0, "y0": 0, "side": 16, "size": 16}}}}
    (root / "meta.json").write_text(json.dumps(meta))
    return HamNoSysCache(root)


def test_cache_mesh_fields_and_source_first_sampling(tmp_path):
    cache = _fake_mesh_cache(tmp_path)
    assert cache.has_mesh()
    row = cache.load_row("clip_a", 2, with_mesh=True)
    assert row["faces"].dtype == torch.int32 and row["faces"][0, 0] == -1 and row["faces"][5, 5] == 20907
    assert torch.equal(row["smplx"], torch.arange(2 * 182, 3 * 182, dtype=torch.float32))
    (tmp_path / "mesh" / "clip_a.npz").rename(tmp_path / "moved.npz")    # 02c failed on the only clip
    import pytest
    with pytest.raises(SystemExit):
        HamNoSysCache(tmp_path).keep_keys_with_mesh(max_missing_frac=0.01)
    c2 = HamNoSysCache(tmp_path)
    assert c2.keep_keys_with_mesh(max_missing_frac=1.0) == ["clip_a"] and c2.keys == []
    (tmp_path / "moved.npz").rename(tmp_path / "mesh" / "clip_a.npz")
    ds = PairDataset(cache, pairs_per_clip=20, with_mesh=True, source_first_prob=1.0)
    items = [ds[i] for i in range(len(ds))]
    assert all(it["source_row"] == 0 and it["target_row"] != 0 for it in items)
    assert "source_faces" in items[0] and "target_princpt" in items[0] and "source_joints" in items[0]
    # no native_size in this meta -> the whole 16 x 16 crop is valid
    assert torch.equal(items[0]["source_valid_rect"], torch.tensor([-0.5, 15.5, -0.5, 15.5]))


def test_valid_rect_from_crop_and_frame_size(tmp_path):
    cache = _fake_mesh_cache(tmp_path)
    # a 100 x 60 video, square crop of side 100 starting at y0 = -20: rows -20..0 and 60..80 of the crop are padding
    cache.meta["clip_a"].update(native_size=[100, 60], crop={"x0": 0.0, "y0": -20.0, "side": 100.0, "size": 100})
    r = cache.valid_rect("clip_a")
    assert torch.allclose(r, torch.tensor([-0.5, 99.5, 19.5, 79.5]))


def test_valid_rect_hides_samples_in_the_padding():
    v, f = _mesh((-0.4, 0.4, -0.4, 0.4, 2.0))
    uvz_s, fm_s = _render(v, f, cx=28.0)
    uvz_t, fm_t = _render(v, f, cx=33.0)
    full = correspond(uvz_s, fm_s, uvz_t, fm_t, f)
    # the source video only covers crop columns x <= 30 (the rest is black padding)
    rect = torch.tensor([[-0.5, 30.0, -0.5, 63.5]])
    cut = correspond(uvz_s, fm_s, uvz_t, fm_t, f, valid_rect_s=rect)
    q = cut["source_coords"][0]
    body = fm_t[0] >= 0
    assert full["visibility"][0, 0][body].min() == 1
    assert cut["visibility"][0, 0][body & (q[..., 0] > 29.0)].max() == 0      # sample in the padding (1 px margin)
    assert cut["visibility"][0, 0][body & (q[..., 0] < 27.0)].min() == 1      # sample in the video


def test_generator_sees_only_the_trusted_warp():
    torch.manual_seed(0)
    cfg = dict(SMALL_MESH, gen_input_mask=True, occlusion_refine=False,
               flow=dict(type="mesh", smplx_dir="unused-in-this-test"))
    m = GReg(cfg).eval()
    assert m.generator.stem[0].in_channels == 10
    B = 1
    Is, Ns, Nt = (torch.rand(B, 3, 128, 128) for _ in range(3))
    centers = torch.tensor([[[64.0, 30.0], [30.0, 100.0], [100.0, 100.0]]])
    joints = torch.rand(B, 144, 2) * 100 + 14
    ident = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="xy"), -1)
    occ = torch.ones(B, 1, 128, 128)
    occ[..., 40:90, 40:90] = 0                             # untrusted block
    geo = {"grid": ident[None], "occlusion": occ, "visibility": torch.ones_like(occ), "confidence": occ,
           "parts": torch.zeros(B, 128, 128, dtype=torch.long)}
    seen = []
    hook = m.generator.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    with torch.no_grad():
        a = m(Is, Ns, Nt, centers, geometry=geo, joints_target=joints)
        Is2 = Is.clone()
        Is2[..., 40:90, 40:90] = 0.0                       # change the source only where the warp is untrusted
        b = m(Is2, Ns, Nt, centers, geometry=geo, joints_target=joints)
    hook.remove()
    assert seen[0].shape[1] == 10
    assert torch.allclose(seen[0][:, :3], seen[1][:, :3])   # warp channels: the untrusted pixels are not visible
    assert torch.allclose(seen[0][:, 9:], occ)              # 10th channel = the trust mask O
    assert not torch.allclose(a["warped"], b["warped"])     # although the raw warp did change there
    out = m.train()(Is, Ns, Nt, centers, geometry=geo, joints_target=joints)
    out["output"].mean().backward()
    assert m.generator.stem[0].weight.grad is not None
