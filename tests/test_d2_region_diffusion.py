"""D2: flow-matching Regional Experts and their use inside GReg (CPU, tiny channels)."""
import torch

from greg.diffusion.region_diffusion import DiffusionRegionalPriorGenerator, timestep_embedding
from greg.models.greg import GReg

CH = (8, 16, 16, 16)


def test_timestep_embedding_shape_and_distinct():
    e = timestep_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
    assert e.shape == (3, 16) and not torch.allclose(e[0], e[2])


def test_fixed_noise_sampling_is_deterministic_and_in_range():
    g = DiffusionRegionalPriorGenerator(32, 16, CH, steps=3)
    cn, st = torch.rand(2, 3, 3, 16, 16), torch.randn(2, 32)
    a, b = g.sample(cn, st), g.sample(cn, st)
    assert a.shape == (2, 3, 1, 16, 16) and torch.equal(a, b) and float(a.min()) >= 0 and float(a.max()) <= 1
    g.noise_mode = "random"
    assert not torch.equal(g.sample(cn, st), g.sample(cn, st))


def test_flow_matching_loss_learns_a_constant_target():
    torch.manual_seed(0)
    g = DiffusionRegionalPriorGenerator(8, 16, CH, steps=8)
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    cn, st = torch.rand(4, 3, 3, 16, 16), torch.zeros(4, 8)
    struct = torch.full((4, 3, 1, 16, 16), 0.8)
    first = None
    for i in range(150):
        loss = g.training_loss(cn, st, struct)["fm"]
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * first
    s = g.sample(cn, st)
    assert abs(float(s.mean()) - 0.8) < 0.15


def test_greg_with_diffusion_rpg_forward_and_freeze(tmp_path):
    cfg = {"image_size": 64, "region_size": 16, "style_dim": 32, "style_base": 8,
           "flow": {"num_tps": 2, "dense_motion": {"block_expansion": 8, "num_blocks": 2, "max_features": 32,
                                                   "scale_factor": 0.25}},
           "rpg": {"type": "diffusion", "channels": list(CH), "steps": 2},
           "gr": {"grm_levels": [16, 32], "base": 8, "max_channels": 32},
           "generator": {"base": 8, "max_channels": 32, "num_down": 2}}
    m = GReg(cfg)
    torch.save({"rpg": m.rpg.state_dict(), "style_encoder": m.style_encoder.state_dict(), "step": 1}, tmp_path / "re.pth")
    m.load_pretrained_rpg(str(tmp_path / "re.pth"), freeze=True)
    assert not any(p.requires_grad for p in m.rpg.parameters())
    out = m(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64),
            torch.tensor([[[32.0, 16.0], [16.0, 48.0], [48.0, 48.0]]]))
    assert out["output"].shape == (1, 3, 64, 64) and out["prior_patches"].shape == (1, 3, 1, 16, 16)
