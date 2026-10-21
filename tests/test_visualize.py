import torch
from PIL import Image

from greg.eval.visualize import clip_gif, error_map, frame_panels, heat, save_gif, to_pil
from greg.models.greg import GReg

SMALL = dict(image_size=128, region_size=32, gaussian_sigma=16.0, style_dim=32, style_base=8,
             flow=dict(num_tps=10, dense_motion=dict(block_expansion=8, max_features=32, num_blocks=5)),
             rpg=dict(channels=(8, 16, 16, 16)), gr=dict(base=8, max_channels=32, grm_levels=(16, 32, 64)),
             generator=dict(base=8, max_channels=32, num_down=4))


def test_heat_and_error_map_range():
    h = heat(torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert h.shape == (3, 1, 3) and h.min() >= 0 and h.max() <= 1
    assert torch.equal(h[:, 0, 0], torch.zeros(3)) and torch.equal(h[:, 0, 2], torch.ones(3))
    a = torch.zeros(3, 4, 4)
    assert error_map(a, a).max() == 0 and error_map(a, a + 0.5, gain=4).min() == 1


def test_frame_panels_and_gif(tmp_path):
    torch.manual_seed(0)
    model = GReg(SMALL).eval()
    Is, It = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    Ns, Nt = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    centers = torch.tensor([[[64.0, 20.0], [30.0, 90.0], [100.0, 90.0]]] * 2)
    with torch.no_grad():
        out = model(Is, Ns, Nt, centers)
    ps = frame_panels(out, 1, Is, Ns, Nt, It, SMALL["region_size"], visible=[1, 1, 0])
    assert isinstance(ps["full"], Image.Image) and ps["full"].width == 6 * 128
    assert ps["regions"].width == 5 * 128 and ps["regions"].height == 3 * (128 + 16)
    frames = [{"normal": Nt[0], "target": It[0], "output": out["output"][0] * k / 3} for k in (1, 2, 3)]   # PIL merges identical frames
    gif = clip_gif(frames, tmp_path / "a" / "clip.gif", fps=5, scale=0.5)
    with Image.open(gif) as g:
        assert g.n_frames == 3
    assert to_pil(torch.rand(1, 8, 8), size=16).size == (16, 16)
    save_gif([to_pil(torch.rand(3, 8, 8))] * 2, tmp_path / "b.gif")
