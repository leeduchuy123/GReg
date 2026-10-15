import torch

from greg.losses import GRegLoss, gan_loss_d, gan_loss_g
from greg.models.blocks import make_coordinate_grid
from greg.models.discriminator import RegionalDiscriminators
from greg.models.gradient_refine import GRM, GradientEncoder
from greg.models.greg import GReg
from greg.models.regional_expert import RegionalExpert, RegionalPriorGenerator
from greg.models.style import ModulatedConv2d, StyleEncoder
from greg.models.tpsmm import TPS, FlowEstimator, InpaintingNetwork, warp
from greg.models.vgg import PerceptualLoss, VGG19Features

SMALL = dict(image_size=128, region_size=32, gaussian_sigma=16.0, style_dim=32, style_base=8,
             flow=dict(num_tps=10, dense_motion=dict(block_expansion=8, max_features=32, num_blocks=5)),
             rpg=dict(channels=(8, 16, 16, 16)), gr=dict(base=8, max_channels=32, grm_levels=(16, 32, 64)),
             generator=dict(base=8, max_channels=32, num_down=4))


def test_tps_identity_is_near_identity():
    kp = torch.tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [0, 0]]).view(1, 1, 5, 2).repeat(1, 10, 1, 1)
    grid = TPS("kp", 1, kp_1=kp, kp_2=kp.clone()).transform_frame(torch.zeros(1, 3, 16, 16))
    ident = make_coordinate_grid((16, 16)).view(1, 1, 16, 16, 2)
    assert (grid - ident).abs().max() < 0.05


def test_flow_estimator_shapes_and_warp():
    fe = FlowEstimator(num_tps=10, dense_motion=dict(block_expansion=8, max_features=32, num_blocks=5))
    n = torch.rand(2, 3, 128, 128)   # dense motion runs at 1/4 res with 5 hourglass levels
    out = fe(n, n)
    assert out["deformation"].shape == (2, 32, 32, 2)
    assert out["occlusion"].shape == (2, 1, 32, 32)
    img = torch.rand(2, 3, 128, 128)
    assert warp(img, out["deformation"]).shape == img.shape
    ip = InpaintingNetwork(3, 8, 32, 3)
    g = ip(img, out["dense_motion"])
    assert g["prediction"].shape == img.shape and len(g["warped_encoder_maps"]) == 4


def test_modulated_conv_and_style():
    conv = ModulatedConv2d(4, 6, 3, style_dim=8)
    y = conv(torch.rand(2, 4, 10, 10), torch.rand(2, 8))
    assert y.shape == (2, 6, 10, 10)
    assert StyleEncoder(3, 16, base=4, num_down=3)(torch.rand(2, 3, 32, 32)).shape == (2, 16)


def test_regional_expert_and_prior_map():
    re = RegionalExpert(3, 1, 16, channels=(8, 16, 16, 16))
    assert re(torch.rand(2, 3, 32, 32), torch.rand(2, 16)).shape == (2, 1, 32, 32)
    rpg = RegionalPriorGenerator(16, region_size=32, channels=(8, 16, 16, 16))
    boxes = torch.tensor([[[0, 0], [40, 40], [96, 96]]] * 2)
    out = rpg(torch.rand(2, 3, 128, 128), boxes, torch.rand(2, 16))
    assert out["prior_map"].shape == (2, 3, 128, 128)
    assert out["prior_patches"].shape == (2, 3, 1, 32, 32)
    assert out["prior_map"][0, 0, 40:, :].abs().sum() == 0   # face prior only in its own box


def test_grm_zero_init_is_identity_and_gradient_encoder_levels():
    enc = GradientEncoder(3, base=4, max_channels=16, image_size=64, levels=(8, 16, 32))
    feats = enc(torch.rand(2, 3, 64, 64))
    assert sorted(feats) == [8, 16, 32] and feats[16].shape[-1] == 16
    grm = GRM(8, enc.channels[16], heads=2, zero_init_out=True)
    x = torch.rand(2, 8, 16, 16)
    assert torch.allclose(grm(x, feats[16]), x)


def test_greg_forward_and_losses_backward():
    torch.manual_seed(0)
    m = GReg(SMALL).freeze_flow()
    D = RegionalDiscriminators(3, 4, base=8, num_layers=2)
    B = 2
    Is, Ns, Nt, It = (torch.rand(B, 3, 128, 128) for _ in range(4))
    centers = torch.tensor([[[64.0, 30.0], [20.0, 100.0], [120.0, 120.0]]] * B)
    out = m(Is, Ns, Nt, centers)
    assert out["output"].shape == (B, 3, 128, 128)
    assert out["prior_map"].shape == (B, 3, 128, 128)
    assert out["boxes"].shape == (B, 3, 2)
    loss = GRegLoss(PerceptualLoss(VGG19Features(pretrained=False), scales=(1.0,)), {}, "hinge", region_size=32,
                    region_perceptual_size=64)
    g = loss.generator_loss(out, It, D)
    g["total"].backward()
    assert all(p.grad is None for p in m.flow.parameters())
    assert any(p.grad is not None for p in m.generator.parameters())
    assert any(p.grad is not None for p in m.rpg.parameters())
    d = loss.discriminator_loss(out, It, D)
    d["d_total"].backward()
    assert torch.isfinite(d["d_total"])


def test_gan_losses_all_kinds():
    r, f = [torch.randn(2, 1, 4, 4)], [torch.randn(2, 1, 4, 4)]
    for kind in ("bce", "hinge", "lsgan"):
        assert torch.isfinite(gan_loss_d(r, f, kind)) and torch.isfinite(gan_loss_g(f, kind))
