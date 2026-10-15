import torch

from greg.render.body import Camera, project
from greg.render.normals import vertex_normals
from greg.render.rasterize import interpolate_vertex_attribute, rasterize


def _square_mesh(z=2.0):
    # two triangles covering x,y in [-0.5, 0.5] at depth z (camera coords)
    v = torch.tensor([[-0.5, -0.5, z], [0.5, -0.5, z], [0.5, 0.5, z], [-0.5, 0.5, z]])[None]
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def test_project_pinhole():
    cam = Camera(torch.tensor([[100.0, 100.0]]), torch.tensor([[32.0, 32.0]]), torch.tensor([[0.0, 0.0, 1.0]]))
    uv = project(torch.tensor([[[0.0, 0.0, 1.0]]]), cam)   # Z = 2 after translation
    assert torch.allclose(uv, torch.tensor([[[32.0, 32.0]]]))


def test_rasterize_square_coverage_and_depth():
    v, f = _square_mesh(z=2.0)
    cam = Camera(torch.tensor([[64.0, 64.0]]), torch.tensor([[32.0, 32.0]]), torch.zeros(1, 3))
    uv = project(v, cam)                      # square spans pixels 16..48
    frag = rasterize(uv, v[..., 2], f, 64, 64, backend="torch", tile=16)
    mask = frag.mask[0]
    assert mask[32, 32] and not mask[5, 5]
    area = mask.float().sum().item()
    assert 30 * 30 <= area <= 34 * 34
    assert torch.allclose(frag.depth[0][mask], torch.full_like(frag.depth[0][mask], 2.0), atol=1e-4)


def test_nearer_triangle_wins():
    v, f = _square_mesh(z=2.0)
    v2, _ = _square_mesh(z=1.0)
    verts = torch.cat([v, v2], dim=1)
    faces = torch.cat([f, f + 4], dim=0)
    cam = Camera(torch.tensor([[64.0, 64.0]]), torch.tensor([[32.0, 32.0]]), torch.zeros(1, 3))
    uv = project(verts, cam)
    frag = rasterize(uv, verts[..., 2], faces, 64, 64, backend="torch")
    assert frag.face_idx[0, 32, 32] >= 2        # a face of the nearer square
    assert torch.isclose(frag.depth[0, 32, 32], torch.tensor(1.0), atol=1e-4)


def test_vertex_normals_and_interpolation():
    v, f = _square_mesh()
    n = vertex_normals(v, f)
    assert torch.allclose(n.abs(), torch.tensor([0.0, 0.0, 1.0]).expand(1, 4, 3))
    cam = Camera(torch.tensor([[64.0, 64.0]]), torch.tensor([[32.0, 32.0]]), torch.zeros(1, 3))
    frag = rasterize(project(v, cam), v[..., 2], f, 64, 64)
    img = interpolate_vertex_attribute(frag, n, f)
    assert img.shape == (1, 3, 64, 64)
    assert torch.allclose(img[0, :, 32, 32].abs(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    assert torch.all(img[0, :, 0, 0] == 0)


def test_cropped_camera_matches_resampled_projection():
    cam = Camera(torch.tensor([[500.0, 500.0]]), torch.tensor([[160.0, 120.0]]), torch.tensor([[0.0, 0.0, 10.0]]))
    pts = torch.randn(1, 10, 3)
    uv = project(pts, cam)
    x0, y0, side, size = 40.0, 10.0, 200.0, 256
    uv_c = project(pts, cam.cropped(x0, y0, side, size))
    expected = (uv - torch.tensor([x0, y0])) * (size / side)
    assert torch.allclose(uv_c, expected, atol=1e-4)
