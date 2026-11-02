"""D6: mask packing, patch-embedding widening, latent codec normalisation and one train / sample step on tiny
Wan-shaped models (CPU, no download)."""
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: E402

from greg.diffusion.wan_video import (WanConditionModel, WanLatentCodec, filtered_mse, flow_schedule,  # noqa: E402
                                      pack_frames, sample, training_loss, widen_patch_embedding)

TINY = dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8, in_channels=36, out_channels=16,
            text_dim=32, freq_dim=16, ffn_dim=32, num_layers=1, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
            eps=1e-6, image_dim=24, added_kv_proj_dim=16, rope_max_seq_len=64)


def test_pack_frames_layout():
    m = torch.zeros(1, 1, 9, 16, 16)
    m[:, :, 0] = 1                              # frame 0 known
    m[:, :, 5] = 1                              # frame 5 -> latent frame 2, channel 0 (pixel frame 4j+k-3)
    p = pack_frames(m, pool=8)
    assert p.shape == (1, 4, 3, 2, 2)
    assert torch.all(p[:, :, 0] == 1)           # the repeated first frame fills latent frame 0
    assert torch.all(p[:, 0, 2] == 1) and p[:, 1:, 2].sum() == 0 and p[:, :, 1].sum() == 0
    with pytest.raises(ValueError):
        pack_frames(torch.zeros(1, 1, 8, 16, 16))


def test_widen_patch_embedding_keeps_output():
    tr = WanTransformer3DModel(**TINY)
    x = torch.randn(1, 36, 1, 4, 4)
    with torch.no_grad():
        a = tr.patch_embedding(x)
        widen_patch_embedding(tr, 32)
        b = tr.patch_embedding(torch.cat([x, torch.randn(1, 32, 1, 4, 4)], 1))
    assert torch.allclose(a, b, atol=1e-5) and tr.config.in_channels == 68
    assert tr.patch_embedding.weight.dtype == torch.float32


def _tiny_vae():
    return AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1, attn_scales=[],
                            temperal_downsample=[False, True, True])


def test_codec_normalises_and_inverts():
    c = WanLatentCodec(_tiny_vae())
    v = torch.rand(1, 3, 5, 16, 16) * 2 - 1
    mean, std = c.encode(v, moments=True)
    assert mean.shape == (1, 16, 2, 2, 2) and (std > 0).all()
    raw = c.vae.encode(v).latent_dist.mean
    assert torch.allclose(mean * c.std + c.mean, raw, atol=1e-4)
    assert c.decode(mean).shape == (1, 3, 5, 16, 16)


def test_filtered_mse_drops_outliers():
    p, t = torch.zeros(10), torch.zeros(10)
    p[0] = 100.0
    p[1] = 1.0
    assert float(filtered_mse(p, t)) == pytest.approx(0.1)


def _window(B=1, Fp=2, h=4):
    z = lambda c: torch.randn(B, c, Fp, h, h)  # noqa: E731
    return {"lat_mean": z(16), "lat_std": torch.full((B, 16, Fp, h, h), 0.1), "lat_masked": z(16),
            "lat_i2v": z(16), "lat_normal": z(16), "lat_ccm": z(16), "lat_skel": z(16),
            "mask": torch.rand(B, 4, Fp, h, h), "mask_i2v": torch.zeros(B, 4, Fp, h, h),
            "region": torch.rand(B, 1, Fp, h, h), "valid": torch.ones(B, 1, Fp, h, h)}


@pytest.mark.parametrize("variant", ["ng", "k", "ng_ones", "base"])
def test_train_step_and_sample(variant):
    torch.manual_seed(0)
    tr = WanTransformer3DModel(**TINY)
    m = WanConditionModel(tr, variant, lora_rank=4)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("lora" in n for n in names) and any("patch_embedding" in n for n in names)
    w, text, clip = _window(), torch.randn(1, 512, 32), torch.randn(1, 257, 24)
    loss = training_loss(m, flow_schedule(5.0), w, text, clip)
    loss.backward()
    assert torch.isfinite(loss) and m.transformer.patch_embedding.weight.grad is not None
    z = sample(m, w, text, clip, steps=2, noise=torch.zeros(1, 16, 2, 4, 4))
    assert z.shape == (1, 16, 2, 4, 4) and torch.isfinite(z).all()
