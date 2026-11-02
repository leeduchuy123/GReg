"""D1 refiner: skip hooks, conv_in widening, trainable set, save/load, gating, losses.  Tiny randomly initialised
SD-shaped models (no download), CPU."""
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")

from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402

from greg.diffusion.gating import blend, refine_gate, refiner_masks  # noqa: E402
from greg.diffusion.refiner_losses import RefinerLoss  # noqa: E402
from greg.diffusion.turbo_refiner import TurboRefiner, VAESkips, widen_conv_in  # noqa: E402


def tiny_vae():
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 16, 32),
                         layers_per_block=1, latent_channels=4, norm_num_groups=4, sample_size=32)


def tiny_unet():
    return UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                attention_head_dim=8, norm_num_groups=8)


def tiny_refiner(**kw):
    torch.manual_seed(0)
    ac = torch.linspace(0.999, 0.0047, 1000)
    return TurboRefiner(tiny_vae(), tiny_unet(), torch.randn(1, 5, 16), ac, **kw)


def test_widen_conv_in_keeps_output():
    c = torch.nn.Conv2d(4, 8, 3, padding=1)
    w = widen_conv_in(c, 5)
    x = torch.randn(2, 4, 6, 6)
    assert torch.allclose(w(torch.cat([x, torch.randn(2, 5, 6, 6)], 1)), c(x), atol=1e-6)


def test_skips_start_at_zero_and_change_output_when_trained():
    vae = tiny_vae().eval()
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        ref = vae.decode(vae.encode(x).latent_dist.mode()).sample
    sk = VAESkips(vae)
    with torch.no_grad():
        out0 = vae.decode(vae.encode(x).latent_dist.mode()).sample
        assert torch.allclose(out0, ref, atol=1e-5)                 # zero-init skips = plain VAE
        for c in sk.convs:
            c.weight.normal_(0, 0.1)
        out1 = vae.decode(vae.encode(x).latent_dist.mode()).sample
    assert (out1 - ref).abs().max() > 1e-3
    # encoder down-block inputs (8, 8, 8, 16) reversed -> decoder up-block inputs (32, 32, 16, 8)
    assert [tuple(c.weight.shape[:2]) for c in sk.convs] == [(32, 16), (32, 8), (16, 8), (8, 8)]


def test_refiner_forward_trainables_and_extra_channels_start_inert():
    r = tiny_refiner(cond_normal=True, num_masks=4)
    names = [n for n, p in r.named_parameters() if p.requires_grad]
    assert any("lora" in n for n in names) and any("skips.convs" in n for n in names)
    assert any(n.startswith("unet.conv_in") for n in names)
    assert not any(("lora" not in n) and ("skips" not in n) and ("conv_in" not in n) for n in names)
    img, nrm = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    m = torch.rand(2, 4, 32, 32)
    r.eval()
    with torch.no_grad():
        a = r(img, nrm, m)
        b = r(img, torch.rand(2, 3, 32, 32), torch.rand(2, 4, 32, 32))   # different conditions, zero columns
    assert a.shape == (2, 3, 32, 32) and 0 <= float(a.min()) and float(a.max()) <= 1
    assert torch.allclose(a, b, atol=1e-5)
    with torch.no_grad():
        c = r(img, nrm, m)
    assert torch.allclose(a, c)                                      # deterministic (posterior mode)


def test_refiner_gradients_reach_new_columns_and_save_load():
    r = tiny_refiner(cond_normal=True, num_masks=4)
    img, nrm, m = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), torch.rand(1, 4, 32, 32)
    r(img, nrm, m).mean().backward()
    g = r.unet.conv_in.weight.grad
    assert g is not None and g[:, 4:].abs().sum() > 0
    sd = r.trainable_state_dict()
    r2 = tiny_refiner(cond_normal=True, num_masks=4)
    with torch.no_grad():
        for p in r.trainable_parameters():
            p.add_(0.01)
    sd = r.trainable_state_dict()
    r2.load_trainable_state_dict(sd)
    r.eval(), r2.eval()
    with torch.no_grad():
        assert torch.allclose(r(img, nrm, m), r2(img, nrm, m), atol=1e-5)


def test_gate_and_blends():
    parts = torch.zeros(1, 32, 32, dtype=torch.long)
    parts[:, 8:24, 8:24] = 1
    parts[:, 10:14, 10:14] = 3
    vis = torch.ones(1, 1, 32, 32)
    vis[..., 18:22, 18:22] = 0
    g = refine_gate(parts, vis, dilate_px=1)
    assert g[0, 0, 12, 12] == 1 and g[0, 0, 20, 20] == 1 and g[0, 0, 2, 2] == 0
    greg, ref = torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)
    hard = blend(greg, ref, g, "hard")
    assert hard[0, 0, 12, 12] == 1 and hard[0, 0, 2, 2] == 0
    freq = blend(greg + 0.3, ref * 0.3, g, "freq")
    assert abs(float(freq[0, :, 2, 2].mean()) - 0.3) < 0.05     # outside: GReg kept
    assert refiner_masks(parts, vis).shape == (1, 4, 32, 32)


class _L1Map:
    def __call__(self, a, b):
        return (a - b).abs().mean(1, keepdim=True)


def test_refiner_loss_weights_hands_and_temporal_zero_for_static():
    crit = RefinerLoss(_L1Map(), {"l2": 1, "lpips": 1, "hand": 1}, w_hand=9, w_face=9)
    parts = torch.zeros(1, 32, 32, dtype=torch.long)
    parts[:, 4:12, 4:12] = 3
    tgt = torch.zeros(1, 3, 32, 32)
    err_hand, err_bg = tgt.clone(), tgt.clone()
    err_hand[..., 6:10, 6:10] = 1
    err_bg[..., 20:24, 20:24] = 1
    joints = torch.full((1, 144, 2), 8.0)
    centers = torch.tensor([[[16.0, 2.0], [8.0, 8.0], [40.0, 8.0]]])
    vis = torch.ones(1, 1, 32, 32)
    lh = crit(err_hand, tgt, parts, vis, centers, joints)["total"]
    lb = crit(err_bg, tgt, parts, vis, centers, joints)["total"]
    assert lh > lb
    geo = {"grid": torch.stack(torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="xy"), -1)[None],
           "occlusion": torch.ones(1, 1, 32, 32)}
    x = torch.rand(1, 3, 32, 32)
    assert float(RefinerLoss.temporal(x, x, geo)) < 1e-5
