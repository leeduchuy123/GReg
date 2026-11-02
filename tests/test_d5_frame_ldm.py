"""D5: skeleton drawing, conditions, reference attention and the FrameLDM training / sampling loop on tiny
SD-shaped models (CPU, no download)."""
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("peft")
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel  # noqa: E402

from greg.diffusion.frame_ldm import (CONDITION_CHANNELS, FrameLDM, LHAND_BONES, RHAND_BONES,  # noqa: E402
                                      build_condition, draw_skeleton)


def test_hand_bones_cover_every_hand_joint_once_as_child():
    for bones, wrist, first, tips in ((LHAND_BONES, 20, 25, 66), (RHAND_BONES, 21, 40, 71)):
        children = sorted(b for _, b in bones)
        assert children == sorted(list(range(first, first + 15)) + list(range(tips, tips + 5)))
        assert len(bones) == 20 and all(a == wrist for a, b in bones if b in (first, first + 3, first + 6, first + 9, first + 12))


def test_draw_skeleton_and_conditions():
    j = torch.rand(144, 2) * 200 + 28
    img = draw_skeleton(j, 128)
    assert img.shape == (3, 128, 128) and img.max() > 0.9 and img.mean() < 0.3
    B = 2
    c = build_condition("ng", 64, normal=torch.rand(B, 3, 32, 32), warped=torch.rand(B, 3, 32, 32),
                        vc=torch.rand(B, 1, 32, 32), ccm=torch.rand(B, 3, 32, 32))
    assert c.shape == (B, CONDITION_CHANNELS["ng"], 64, 64)
    assert build_condition("k", 64, joints=torch.rand(B, 144, 2) * 256).shape == (B, 3, 64, 64)


def _tiny(cond_ch=3):
    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 16), layers_per_block=1,
                        latent_channels=4, norm_num_groups=4)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                attention_head_dim=8, norm_num_groups=8)
    cn = ControlNetModel.from_unet(unet, conditioning_channels=cond_ch, conditioning_embedding_out_channels=(8, 16))
    sched = DDPMScheduler(num_train_timesteps=1000)
    return FrameLDM(vae, unet, cn, sched, torch.randn(1, 5, 16), lora_rank=4)


def test_reference_attention_changes_output_and_only_adapters_train():
    m = _tiny()
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any(n.startswith("controlnet.") for n in names) and any("lora" in n for n in names)
    assert not any(n.startswith("unet.") and "lora" not in n for n in names)
    z = torch.randn(1, 4, 8, 8)
    t = torch.tensor([500])
    cond = torch.rand(1, 3, 16, 16)
    m.bank.mode = "off"
    with torch.no_grad():
        e0 = m.eps(z, t, cond)
        m.write_reference(torch.randn(1, 4, 8, 8))
        e1 = m.eps(z, t, cond)
        m.write_reference(torch.randn(1, 4, 8, 8) * 3)
        e2 = m.eps(z, t, cond)
    assert not torch.allclose(e0, e1) and not torch.allclose(e1, e2)


def test_training_loss_backward_and_sampling():
    m = _tiny(cond_ch=CONDITION_CHANNELS["ng"])
    tgt, ref = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    cond = torch.rand(2, CONDITION_CHANNELS["ng"], 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[..., 4:8, 4:8] = 1
    loss = m.training_loss(tgt, ref, cond, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "controlnet" in n)
    out = m.sample(ref, cond, steps=3, noise=torch.zeros(2, 4, 8, 8))
    assert out.shape == (2, 3, 16, 16) and float(out.min()) >= 0 and float(out.max()) <= 1
    sd = m.trainable_state_dict()
    m2 = _tiny(cond_ch=CONDITION_CHANNELS["ng"])
    m2.load_trainable_state_dict(sd)
    with torch.no_grad():
        a = m.sample(ref, cond, steps=2, noise=torch.zeros(2, 4, 8, 8))
        b = m2.sample(ref, cond, steps=2, noise=torch.zeros(2, 4, 8, 8))
    assert torch.allclose(a, b, atol=1e-5)
