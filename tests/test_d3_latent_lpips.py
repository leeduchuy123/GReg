"""D3: the latent perceptual loss wrapper and its hook in GRegLoss (tiny VAE, stand-in metric, CPU)."""
import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKL  # noqa: E402

from greg.diffusion.latent_lpips import ELatentLPIPSLoss  # noqa: E402


class _LatentL2(torch.nn.Module):
    def forward(self, a, b, normalize=True, add_l1_loss=True):
        return ((a - b) ** 2).mean(dim=(1, 2, 3), keepdim=True)


def _tiny():
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 16), layers_per_block=1,
                        latent_channels=4, norm_num_groups=4)
    return ELatentLPIPSLoss(vae=vae, metric=_LatentL2())


def test_loss_is_zero_on_identical_images_and_has_gradient_only_through_prediction():
    loss = _tiny()
    x = torch.rand(2, 3, 32, 32)
    assert float(loss(x, x)) == pytest.approx(0.0, abs=1e-7)
    p = torch.rand(2, 3, 32, 32, requires_grad=True)
    v = loss(p, x)
    v.backward()
    assert float(v) > 0 and p.grad is not None and p.grad.abs().sum() > 0
    assert not any(q.requires_grad for q in loss.parameters())
    loss.train()
    assert not loss.vae.training                     # stays in eval mode


def test_grегloss_adds_elpips_terms():
    from greg.losses import GRegLoss

    class _P(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).abs().mean()
    crit = GRegLoss(_P(), {"elpips": 1.0, "elpips_hand": 1.0, "adv": 0}, region_size=16)
    crit.elpips = _tiny()
    crit.elpips_hand_size = 32
    out = {"output": torch.rand(2, 3, 64, 64, requires_grad=True), "boxes": torch.tensor([[[0, 0], [16, 16], [40, 40]]] * 2),
           "prior_patches": torch.rand(2, 3, 1, 16, 16)}
    ls = crit.generator_loss(out, torch.rand(2, 3, 64, 64), None, torch.ones(2, 3))
    assert "elpips" in ls and "elpips_hand" in ls and float(ls["elpips_hand"]) > 0
    ls["total"].backward()
    assert out["output"].grad.abs().sum() > 0
