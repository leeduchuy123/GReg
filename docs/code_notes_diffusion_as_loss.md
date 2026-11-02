# Ghi chú đọc mã: dùng diffusion model đóng băng làm loss / critic cho generator GAN (hướng C1)

Ngày: 2026-11-02. Mục tiêu: đánh giá việc thêm một loss dựa trên diffusion model **đóng băng** vào generator GReg
(256×256 RGB, điều kiện = ảnh tham chiếu + normal map SMPL-X; hiện huấn luyện bằng VGG-19 pyramid perceptual +
3 PatchGAN vùng, batch 4, xem `greg/losses.py:91-125`, `configs/hamnosys_greg.yaml:30-53`).

Ký hiệu: ✅ = đã kiểm chứng trực tiếp trong mã nguồn (file:line) hoặc tài liệu chính thức; ⚠️ = suy luận / ước lượng của tôi.
Mã được clone vào `C:\Users\LeDucHuy\AppData\Local\Temp\claude\D--Research-Thesis-GReg\4a790883-5e16-41c1-aa85-031ddc6acb39\scratchpad\repos\` (chỉ đọc, không chạy):

| Repo | Commit (`--depth 1`) | Giấy phép |
|---|---|---|
| `DMD2/` https://github.com/tianweiy/DMD2 | `8d8fa55633d47cfb81bbc7a892e7248f9518763f` | CC BY-NC-SA 4.0 ✅ (`DMD2/LICENSE.md:1`) |
| `dift/` https://github.com/Tsingularity/dift | `9421eb2034396c5b66f1aff37f03e540c264e52f` | MIT ✅ (`dift/LICENSE:1`) |
| `dreamsim/` https://github.com/ssundaram21/dreamsim | `db4d16c6948314e36e2e62f0bae2ee23bc974bf3` | MIT ✅ |
| E-LatentLPIPS chính thức https://github.com/mingukkang/elatentlpips (chỉ đọc README qua web, không clone) | – | CC BY-NC ✅ (README) |

---

## 0. Kết luận ngắn

| Hạng (rủi ro cài đặt tăng dần) | Công thức | Thứ bị đóng băng | Chi phí thêm / bước G (batch 4, 256²) ⚠️ | VRAM thêm trên 24 GB ⚠️ |
|---|---|---|---|---|
| 1 (thấp) | **E-LatentLPIPS** (Kang 2024) = loss hồi quy có cặp trong latent SD | SD-VAE encoder + latent-VGG16 | ≈ 1 VAE-encode fwd+bwd (256²→32²) + latent-VGG rất nhỏ; cùng bậc với VGG pyramid hiện tại | ≈ 1–2 GB |
| 2 (trung bình) | **DIFT-feature perceptual** = L1/cos trên đặc trưng `up_blocks[1]` của SD-UNet ở t≈261 | SD-VAE encoder + SD-UNet (SD1.5/2.1/SD-Turbo) | 2 UNet-partial fwd (1 có grad) + 1 bwd; ≈ 3–6× VGG pyramid hiện tại | ≈ 4–7 GB (UNet bf16 1.7 GB + activation) |
| 3 (cao) | **DMD-style distribution matching** (DMD2) | real-score UNet (frozen) **+ fake-score UNet phải huấn luyện** | mỗi bước G: 3 UNet fwd (2 với CFG) + VAE; mỗi bước G kéo theo 5–10 bước huấn luyện fake-UNet (TTUR) | **> 24 GB nếu fine-tune full fake-UNet** (≈14 GB chỉ cho trọng số+grad+AdamW fp32) → bắt buộc LoRA/8-bit Adam/UNet nhỏ |

**Khuyến nghị (mục 7):** làm (1) E-LatentLPIPS trước như một loss bổ sung song song VGG (1–2 ngày công), (2) DIFT-feature chỉ
nếu (1) không cải thiện LPIPS/AKD trên tay–mặt, (3) DMD-style **không** nên làm trong khuôn khổ luận văn trừ khi có GPU 48–80 GB,
vì mô-đun đắt nhất (fake critic) không phải "đóng băng" và câu hỏi nghiên cứu C1 ("frozen diffusion as loss") không còn đúng nghĩa.

---

## 1. DMD2 – đọc mã

### 1.1 Loss distribution matching được tính thế nào ✅

`DMD2/main/sd_guidance.py:168-255` (`SDGuidance.compute_distribution_matching_loss`), bản EDM/ImageNet giống hệt ở `DMD2/main/edm/edm_guidance.py:73-125`:

1. Đầu vào là `latents` = **đầu ra generator trong không gian latent SD** (4×64×64 cho SD1.5) – không phải ảnh RGB.
2. Toàn bộ khối tính "gradient" nằm trong `torch.no_grad()` (dòng 178):
   - `timesteps ~ U[min_step, max_step]` với `min_step = 0.02·1000 = 20`, `max_step = 0.98·1000 = 980` (`sd_guidance.py:84-85`, mặc định `train_sd.py:634-635`).
   - `noise = randn_like(latents)` (dòng 187) – **nhiễu mới, độc lập với nhiễu đầu vào của generator**; `noisy_latents = scheduler.add_noise(latents, noise, t)` (189).
   - `pred_fake_noise = fake_unet(noisy_latents, text, t)` với guidance 1.0 (192-197; assert `fake_guidance_scale == 1` ở dòng 90).
   - `pred_real_noise = real_unet(noisy_latents, text, t)` **có CFG** `real_guidance_scale` (216-229; CFG thực hiện trong `predict_noise`, 9-38: batch nhân đôi uncond/cond).
   - Đổi ε → x̂₀ bằng `get_x0_from_noise` (`DMD2/main/utils.py:279-284`, giả định ε-prediction – ghi chú ở `sd_unified_model.py:286`).
   - `p_real = latents - x̂₀_real`, `p_fake = latents - x̂₀_fake`; `grad = (p_real - p_fake) / mean|p_real|` (chuẩn hoá theo từng mẫu, 235-239).
3. Loss thực sự chỉ là mẹo để autograd đưa `grad` vào generator: `loss = 0.5·MSE(latents, (latents - grad).detach())` (dòng 241). Tức là ∂loss/∂latents = grad/N. **Không có backward nào đi qua real_unet hay fake_unet** (real_unet để bf16 vì "we don't backpropagate through it", 64-67).

Hệ quả quan trọng cho câu hỏi của ta: **loss DMD chỉ là hàm của `latents` (đầu ra generator) + text embedding + nhiễu lấy mẫu tại chỗ**. Nó không cần nhiễu đầu vào của generator. Nhiễu đầu vào chỉ xuất hiện ở phía generator (`sd_unified_model.py:226, 246, 269-272, 287-290`) để tạo `generated_image` từ UNet-generator; với generator CNN bất kỳ thì phần đó thay bằng `x = G(ref, normal)` rồi `latents = vae.encode(x)·scaling_factor` (xem 1.4).

### 1.2 Cái gì được huấn luyện, cái gì đóng băng ✅

| Thành phần | Trạng thái | Nguồn |
|---|---|---|
| `real_unet` (teacher, điểm số thật) | `requires_grad_(False)`, bf16 | `sd_guidance.py:45-50, 66-67` |
| `fake_unet` (critic điểm số giả) | **trainable, full fine-tune fp32** (autocast bf16 khi forward, 143) – bản sao đầy đủ của SD-UNet ≈ 0.86 B tham số | `sd_guidance.py:53-58` |
| `cls_pred_branch` (đầu GAN) | trainable, 3–5 conv trên bottleneck của **fake_unet** | `sd_guidance.py:107-135` |
| `feedforward_model` (generator = SD-UNet nữa) | trainable (hoặc LoRA r=64, chỉ cho SDXL) | `sd_unified_model.py:38-71` |
| text encoder, VAE | đóng băng | `sd_unified_model.py:82-88, 105` |

Tổng cộng DMD2-SD1.5 giữ **3 bản SD-UNet** trên GPU (real/fake/generator) + VAE + CLIP.

### 1.3 GAN loss trên đặc trưng diffusion ✅

- `classify_forward` (`DMD2/main/sd_unet_forward.py:19-251`) là bản sao forward của diffusers UNet; khi `classify_mode=True` nó **dừng sau mid-block** và trả `list(down_block_res_samples) + [sample]` (249-251). DMD2 chỉ dùng phần tử cuối (bottleneck 1280 kênh, 8×8 cho latent 64², `sd_guidance.py:162-164`).
- `compute_cls_logits` (145-165): với `--diffusion_gan` ảnh (latent) được **thêm nhiễu ở t ~ U[0, diffusion_gan_max_timestep=1000)** trước khi vào fake_unet (147-151) – "Diffusion-GAN" augmentation.
- Loss G: `softplus(-D(fake))` (323); loss D: `softplus(D(fake)) + softplus(-D(real))` (391) – non-saturating logistic.
- Lưu ý: đặc trưng mà D dùng là của **fake_unet đang được huấn luyện**, không phải đặc trưng đóng băng. Nên "GAN on diffusion features" của DMD2 ≠ "frozen diffusion as critic".

### 1.4 Vòng lặp huấn luyện, siêu tham số ✅

`DMD2/main/train_sd.py:322-408`:
- Mỗi iteration: generator forward; **chỉ cập nhật G khi `step % dfake_gen_update_ratio == 0`** (330, 367-381); fake_unet + cls head cập nhật **mọi** iteration (386-408) → TTUR: 10 bước critic / 1 bước G (SD1.5), 5 (SDXL, ImageNet).
- Loss G = `loss_dm · dm_loss_weight(1.0) + gen_cls_loss · gen_cls_loss_weight` (369-372).
- Loss critic = `loss_fake_mean` (ε-MSE trên latent generator đã detach, t ~ U[0,1000), `sd_guidance.py:257-310`) + `guidance_cls_loss · guidance_cls_loss_weight` (398-401).
- AdamW β=(0.9,0.999), wd 0.01 cho cả hai (191-202); `max_grad_norm 10`; warm-up 500 bước.

| Cấu hình | lr G / critic | batch/GPU × GPU | real CFG | TTUR | w_gan(G) / w_cls(D) | Nguồn |
|---|---|---|---|---|---|---|
| SD1.5 from scratch | 1e-5 / 1e-5 | 32 × 64 (= 2048) | 1.75 | 10 | – (chưa bật GAN) | `experiments/sdv1.5/..._from_scratch.sh:9-31` |
| SD1.5 resume + GAN | 5e-7 / 5e-7 | 32 × 64 | 1.75 | 10 | 1e-3 / 1e-2, diffusion_gan t<1000 | `..._gan1e-3_noode_resume_fixdata.sh:9-39` |
| SDXL 4-step | 5e-7 / 5e-7 | 2 × 64 (= 128) | 8 | 5 | 5e-3 / 1e-2 | `experiments/sdxl/sdxl_cond999_..._scratch.sh:9-44` |
| ImageNet-64 (EDM) | 2e-6 / 2e-6 | 40 × 7 | – | 5 | 3e-3 / 1e-2 | `experiments/imagenet/imagenet_gan_classifier_..._scratch.sh:6-30` |

Chi phí báo cáo ✅ (README + bài báo arXiv 2405.14867): SD1.5: 64 A100, 39k + 2k iters ≈ 25 + 2 giờ (`experiments/sdv1.5/README.md:5-8`); SDXL: 64 A100, 19k–24k iters ≈ 57–63 giờ (`experiments/sdxl/README.md:5-9`); ImageNet-64: 7 A100, 200k iters ≈ 53 giờ (`experiments/imagenet/README.md:7-11`). **Repo và bài báo không báo cáo VRAM**; SD1.5 chạy với `--use_fp16 --gradient_checkpointing` (script dòng 28, 31), SDXL phải dùng FSDP (`--fsdp`) và README ghi "FSDP for SDXL training is really slow" (`DMD2/README.md:217`).

### 1.5 Điều gì "gãy" khi generator là CNN nhỏ không-diffusion?

- ✅ **Về mặt cơ học thì không gãy**: `compute_distribution_matching_loss` chỉ cần `latents` (có grad) + text embedding. Real-score network không cần nhiễu của generator (mục 1.1). Cần thêm: `latents = vae.encode(G(·)).latent_dist.mean · 0.18215` **có gradient** qua VAE encoder (DMD2 không có bước này vì generator của họ sinh thẳng latent), và một prompt cố định (ví dụ "a person performing sign language in front of a plain background") hoặc uncond-embedding cho cả real/fake.
- ✅ **Cái gãy là ngân sách**: fake critic là SD-UNet đầy đủ, phải được huấn luyện trên phân phối đầu ra của G ở mọi t, với 5–10 bước / 1 bước G. Đây là thành phần đắt nhất và **không đóng băng được** (DMD2 §TTUR: nếu critic không bám kịp G thì huấn luyện bất ổn). ⚠️ Trên 24 GB: fake-UNet fp32 + grad + AdamW ≈ 0.86B×(4+4+8) B ≈ 14 GB, + real-UNet bf16 1.7 GB, + VAE/CLIP ≈ 0.5 GB, + generator GReg/VGG/3 D + activation ⇒ vượt 24 GB. Phải dùng LoRA cho fake critic (DMD2 chỉ có LoRA cho generator, `sd_unified_model.py:44-69`; LoRA cho critic là ⚠️ chưa được repo kiểm chứng) hoặc 8-bit Adam hoặc UNet nhỏ (bk-sdm-small ~0.3B ⚠️).
- ⚠️ **Lệch điều kiện**: real score chỉ thấy text, không thấy normal map / ảnh tham chiếu. DMD khi đó khớp **phân phối biên** p_G(x) với p_SD(x | prompt): nó là một "realism prior" kiểu SDS, không ràng buộc danh tính/tư thế – các loss có cặp hiện tại (VGG, L_struct) vẫn phải giữ vai trò chính; trọng số DMD cần nhỏ hơn nhiều so với `dm_loss_weight=1.0` của DMD2 (ở đó G không có loss nào khác).
- ⚠️ **Lệch độ phân giải**: SD1.5/2.1/SD-Turbo huấn luyện ở 512² (latent 64²); ảnh 256² cho latent 32², ngoài phân phối huấn luyện, điểm số teacher kém hơn. Lựa chọn: upsample G(·) lên 512 trước khi encode (chi phí ×4) hoặc chấp nhận.
- ⚠️ **ε-prediction**: `get_x0_from_noise` giả định ε-pred. SD1.5, SD2.1-base và **SD-Turbo (EulerDiscrete, `prediction_type: epsilon`, scaled_linear, 1000 bước – đã kiểm tra `scheduler_config.json` trên HF ✅)** đều OK; SD2.1-768 là v-pred thì phải đổi công thức.

---

## 2. DIFT (Tang et al., NeurIPS 2023) – đọc mã

### 2.1 Trích đặc trưng ✅ (`dift/src/models/dift_sd.py`)

- `MyUNet2DConditionModel.forward` (14-160) là bản chép forward của diffusers UNet, thêm tham số `up_ft_indices`; chạy đủ `down_blocks` + `mid_block` (97-121) rồi **chỉ chạy `up_blocks` tới chỉ số lớn nhất được yêu cầu** (`if i > np.max(up_ft_indices): break`, 127-128); lưu `sample.detach()` sau up-block `i` (155-156). Không chạy `conv_out`.
- `OneStepSDPipeline.__call__` (162-187): `latents = vae.encode(img)·scaling_factor` (178); `latents_noisy = scheduler.add_noise(latents, randn, t)` với **nhiễu ngẫu nhiên** (180-181); DDIMScheduler (195); `vae.decoder = None` để tiết kiệm bộ nhớ (194).
- Mặc định (`SDFeaturizer.forward`, 210-244; `extract_dift.py:27-40`; README dòng 44-50): model `stabilityai/stable-diffusion-2-1`, ảnh **768×768**, **t = 261**, **up_ft_index = 1** → đặc trưng ở **1/16 độ phân giải ảnh**, prompt rỗng (null) hoặc "a photo of a {cat}", **ensemble_size = 8** (lặp ảnh 8 lần trong batch với 8 nhiễu khác nhau rồi `mean(0)`, dòng 227, 243). Số kênh up_blocks SD: [1280, 1280, 640, 320] ⇒ index 1 = 1280 kênh ⚠️ (suy từ `block_out_channels` mặc định trong docs diffusers).
- Tác vụ khác: HPatches `t=0, up_ft_index=2`; DAVIS `t=51, up_ft_index=2` (README 113-122, 168-180) – tức là đặc trưng cấp thấp hơn (1/8) và t nhỏ cho khớp hình học.
- Bản **ADM pixel-space 256×256** (`dift/src/models/dift_adm.py:14-78`): `guided-diffusion/models/256x256_diffusion_uncond.pt`, chuẩn hoá [-1,1] (46-49), `x_t = q_sample(img, t=101)` (60-61), trích **`output_blocks[4]`** (71-77) – 1/16 độ phân giải, **không cần VAE**, đúng 256² của GReg. Dùng ở SPair với ảnh resize 512 (README 85-92).

### 2.2 Chi phí mỗi forward ⚠️

- SD-UNet (0.86 B) ở latent 32² (ảnh 256²): ≈ ¼ chi phí ở latent 64² (≈ 0.3–0.7 TFLOPs tuỳ nguồn) ⇒ ≈ 0.1–0.2 TFLOPs/ảnh; cắt ở `up_blocks[1]` tiết kiệm ≈ 30–40 % (bỏ 2 up-block độ phân giải cao, phần đắt nhất của UNet). VAE encoder ở 256² ≈ 0.05–0.1 TFLOPs/ảnh. So sánh: VGG-19 pyramid hiện tại (4 scale, 2 ảnh) ≈ 0.05 TFLOPs/mẫu ⇒ DIFT-loss ≈ 3–6× VGG về forward; backward qua UNet-partial ≈ 2× forward nữa.
- Code DIFT là `@torch.no_grad` + `detach()` (210, 156): để làm loss phải bỏ cả hai và không dùng `ensemble_size=8` (×8 chi phí); dùng 1 nhiễu **chung** cho nhánh sinh và nhánh đích để giảm phương sai ⚠️.
- ADM-256 (≈0.55 B, pixel-space, attention ở 32/16/8): forward tới `output_blocks[4]` ≈ 0.5–0.8 TFLOPs/ảnh ⚠️ – nặng hơn SD-latent nhưng không cần VAE và đúng độ phân giải; phụ thuộc repo `openai/guided-diffusion`.

---

## 3. Mã công khai của perceptual loss "kiểu diffusion"

### 3.1 E-LatentLPIPS (Kang et al. 2024, Diffusion2GAN) ✅
- Repo chính thức `mingukkang/elatentlpips` (`pip install elatentlpips`): `ELatentLPIPS(encoder='sd15'|'sd21'|'sdxl'|'sd3'|'flux', augment='b'|'bg'|'bgc'|'bgco')`, gọi `loss = elatentlpips(latent0, latent1, normalize=True).mean()`; ảnh RGB phải ở [-1,1], latent lấy từ `AutoencoderKL.encode(...).latent_dist.sample()`; `normalize=True` tự nhân scaling/shift. CC BY-NC. Có cảnh báo DDP (nâng cấp gói vì `torch.load` dồn về GPU 0).
- Bài báo (arXiv 2405.05967, phụ lục): backbone = **VGG16-bn bỏ 3 max-pool, đầu vào 4 kênh, huấn luyện ImageNet trong latent SD**, hiệu chỉnh tuyến tính trên BAPPS. **LatentLPIPS đơn thuần cho ảnh "wavy, patchy artifacts" và không hội tụ khi overfit 1 ảnh**; ensemble augmentation (b = blit, g = geometric, c = color, o = cutout; tốt nhất "bg+cutout" cho SD-CFG-3) sửa lỗi này. Fig. 4: 1 iteration A100: LPIPS 117 ms / +15.0 GB (79 ms là decode VAE) vs E-LatentLPIPS **12.1 ms / +0.6 GB**; generator của họ **sinh thẳng latent**, nên số liệu **không gồm VAE-encode** – với GReg (sinh RGB) phải cộng VAE-encode fwd+bwd.
- Bản không chính thức: `SNU-VGILab/e-latentlpips` (có mã huấn luyện latent-VGG, báo cáo huấn luyện 8×A6000 ≈ 5 ngày + 10 h hiệu chỉnh), `GunwooHan/E-LatentLPIPS`.

### 3.2 DreamSim ✅ (`dreamsim/dreamsim/model.py`)
- Không phải diffusion: ensemble ViT-B/16 (DINO + CLIP + OpenCLIP, `config.py:3-8`) fine-tune LoRA trên bộ NIGHTS; đầu vào 224², [0,1]; khoảng cách = `1 - cos(embed_a, embed_b)` (105); model `eval().requires_grad_(False)` (295) nhưng **có thể backprop qua input** → dùng được như loss (cộng ≈ 3×17.6 GFLOPs/ảnh ⚠️). Chỉ cho toàn ảnh (CLS/global) → không nhạy chi tiết tay; nêu để loại trừ.

### 3.3 diffusers – hook đặc trưng UNet ✅
- `UNet2DConditionModel.forward(sample, timestep, encoder_hidden_states, ..., return_dict)` **chỉ trả `sample`** (docs v0.42.1); không có `output_hidden_states`. Ba cách lấy đặc trưng: (a) chép forward như DIFT/DMD2 (`dift_sd.py:14-160`, `sd_unet_forward.py:19-251`) – kiểm soát early-exit, tiết kiệm nhất; (b) `unet.up_blocks[i].register_forward_hook(...)` – ngắn nhất nhưng vẫn chạy hết UNet; (c) `diffusers.hooks` (`HookRegistry`, `apply_layerwise_casting`, `apply_group_offloading`) – phục vụ bộ nhớ/offload, không phải trích đặc trưng.
- Bộ nhớ: `enable_gradient_checkpointing()` (DMD2 dùng cho fake_unet, `sd_guidance.py:265-266, 308-309`), `set_attention_slice`, SDPA mặc định; `enable_xformers_memory_efficient_attention` (DIFT dòng 199).
- SD-Turbo (HF model card): fine-tune từ SD2.1 bằng ADD, 512², ≈0.9 B, ε-pred (3.1.5 ở trên) – có thể thay SD2.1 trong DIFT vì cùng kiến trúc UNet; ⚠️ chưa có bằng chứng đặc trưng SD-Turbo tốt hơn cho perceptual loss, nhưng ADD làm đặc trưng ở t nhỏ "sắc" hơn là giả thuyết hợp lý.

---

## 4. Công thức 1 (rủi ro thấp): E-LatentLPIPS làm loss có cặp bổ sung

- **Đóng băng**: SD-VAE encoder (SD1.5, fp16 ≈ 70 MB; không tải decoder) + latent-VGG16-bn trong gói `elatentlpips`. Không có gì trainable thêm.
- **Chỗ cắm**: `GRegLoss.generator_loss` (`greg/losses.py:100-125`): thêm `losses["latent_lpips"] = elpips(vae_enc(out["output"]), vae_enc(It).detach()).mean()` cho toàn ảnh, và tuỳ chọn cho crop vùng 64² đã resize 128² (`losses.py:104-110`; latent 16² – ⚠️ hơi nhỏ, ưu tiên toàn ảnh). Ảnh GReg ở [0,1] → đổi sang [-1,1].
- **Chi phí/bước G, batch 4, 256²** ⚠️: VAE-encode fwd cho 2×4 ảnh + bwd cho 4 ảnh (≈ 0.3–0.6 TFLOPs tổng), latent-VGG trên 4×32² ≈ không đáng kể (bài báo: 12 ms / iter trên A100). Tổng ≈ tương đương 1 lần VGG pyramid hiện tại → thời gian epoch tăng ≈ 10–25 %.
- **VRAM thêm** ⚠️: ≈ 0.6 GB (bài báo) + activation VAE encoder cho bwd batch 4 ở 256² ≈ 0.5–1 GB ⇒ ≈ 1–2 GB.
- **Siêu tham số từ nguồn** ✅: `augment='bg'` (README) / "bg+cutout" (Table A1, SD-CFG-3); trọng số GAN 0.25 so với E-LatentLPIPS 1.0 (D2G) ⇒ trong GReg bắt đầu `latent_lpips: 1.0` song song `perceptual: 1.0`, ⚠️ quét {0.5, 1, 2}.
- **Lỗi đã biết**: ✅ LatentLPIPS không-ensemble ⇒ hoa văn gợn sóng/loang (wavy, patchy) và không hội tụ – phải để `augment` bật; ⚠️ latent SD mã hoá màu/độ sáng toàn cục yếu ⇒ drift màu nhẹ nếu bỏ VGG – giữ VGG hoặc thêm L1 pixel; ⚠️ latent 32² ⇒ mất chi tiết < 8 px (ngón tay) – đây là loss toàn cục, không thay thế loss vùng.
- **Rủi ro cài đặt**: thấp – gói pip, loss có cặp, không đổi vòng lặp D/G, không lệch kiểu loss với VGG.

## 5. Công thức 2 (rủi ro trung bình): DIFT-feature perceptual loss

- **Đóng băng**: SD-VAE encoder + SD-UNet (SD1.5 hoặc SD2.1-base/SD-Turbo, bf16, `requires_grad_(False)`, cắt ở `up_blocks[1]`), text encoder chạy 1 lần để cache null-prompt embedding (`dift_sd.py:200-206`).
- **Cách tính** (chuyển từ `dift_sd.py:162-187, 210-244`): `z_g = enc(G)·sf`, `z_t = enc(I_t)·sf` (no_grad); **cùng** `noise`, `t` cố định hoặc `t ~ U[200, 320]` ⚠️; `f = unet_partial(add_noise(z, noise, t), t, null_emb, up_ft_indices=[1])` (bỏ `detach`); `loss = mean|f_g − f_t|₁` hoặc `1 − cos` theo pixel. Có thể lấy thêm `up_blocks[2]` (1/8, 640 kênh) để có chi tiết hơn (DIFT dùng index 2 cho HPatches/DAVIS).
- **Chi phí/bước G, batch 4, 256²** ⚠️: 2 VAE-encode (1 có grad) + 2 UNet-partial fwd (1 có grad) + 1 UNet-partial bwd ≈ 1.2–2.5 TFLOPs ⇒ ≈ 3–6× VGG pyramid; thời gian epoch tăng ≈ 50–150 %. Với ADM-256 pixel-space: bỏ VAE nhưng UNet nặng ≈ 2× SD-latent.
- **VRAM thêm** ⚠️: UNet bf16 1.7 GB + VAE 0.1 GB + activation cho bwd (batch 4, latent 32², 2 up-block) ≈ 2–4 GB (có thể bật `enable_gradient_checkpointing`) ⇒ ≈ 4–7 GB. Khớp 24 GB cùng GReg hiện tại.
- **Siêu tham số từ nguồn** ✅: `t=261`, `up_ft_index=1`, prompt rỗng, ảnh 768 (SD2.1) – ta dùng 256 → ⚠️ cân nhắc upsample 512 (×4 chi phí) nếu đặc trưng ở 32² quá thô; `ensemble_size=8` → 1.
- **Lỗi đã biết** ⚠️ (suy luận từ bản chất đặc trưng): (i) đặc trưng ở 1/16 độ phân giải, t=261 ⇒ ngữ nghĩa/thô ⇒ **over-smoothing** nếu dùng thay VGG; (ii) **drift màu**: ở t=261 nhiễu đã xoá phần lớn thông tin màu/ánh sáng ⇒ loss gần như mù màu – bắt buộc giữ VGG/L1; (iii) phương sai do nhiễu ⇒ gradient ồn nếu nhiễu khác nhau giữa 2 nhánh; (iv) ảnh 256² ngoài phân phối 512² của SD; (v) không có mode collapse vì là loss có cặp.
- **Rủi ro cài đặt**: trung bình – phải chép forward UNet (DIFT viết cho diffusers cũ: `diffusers.models.unet_2d_condition`, `_encode_prompt` – API đã đổi tên trong diffusers ≥0.25 ⚠️), cần kiểm soát bộ nhớ và dtype.

## 6. Công thức 3 (rủi ro cao): DMD-style distribution matching + Diffusion-GAN head

- **Đóng băng**: real-score UNet (SD1.5/SD-Turbo bf16) + VAE + text encoder. **Phải huấn luyện**: fake-score UNet (full hoặc LoRA ⚠️) + `cls_pred_branch`.
- **Cách tính**: đúng `sd_guidance.py:168-255` với `latents = vae.encode(G)·sf` (có grad); prompt cố định; thêm `compute_loss_fake` + `compute_guidance_clean_cls_loss` vào vòng lặp D của `06_train_greg.py:208-220` với TTUR 5–10 (mỗi bước G chạy 5–10 bước critic trên các batch khác nhau – `train_sd.py:330-335, 386-408`).
- **Chi phí/bước G, batch 4, 256²** ⚠️: nhánh G: VAE-encode (grad) + real-UNet fwd với CFG (batch 8) + fake-UNet fwd (batch 4) + cls head ≈ 1.5–3 TFLOPs; nhánh critic ×(5–10): fake-UNet fwd+bwd trên batch 4 + cls head trên real/fake ≈ 0.6–1.2 TFLOPs mỗi bước ⇒ **≈ 5–15 TFLOPs / bước G**, tức 20–60× VGG hiện tại; thời gian epoch ×5–10.
- **VRAM trên 24 GB** ⚠️: full fine-tune fake-UNet ≈ 14 GB (fp32 + AdamW) + real-UNet 1.7 GB + VAE/CLIP 0.5 GB + GReg (G + VGG + 3 D + activation batch 4) ≈ 4–8 GB ⇒ **vượt 24 GB**. Với LoRA r=64 trên fake-UNet (theo danh sách module `sd_unified_model.py:47-62`) + gradient checkpointing: ≈ 1.7 + 1.7 + 0.5 + ~1 (LoRA/Adam) + activation ≈ 8–12 GB ⇒ vừa, nhưng chưa có bằng chứng critic-LoRA đủ bám G.
- **Siêu tham số từ nguồn** ✅: `t ∈ [20, 980]` cho DM, `t ∈ [0,1000)` cho critic và Diffusion-GAN; real CFG 1.75 (SD1.5) / 8 (SDXL); lr 1e-5 → 5e-7; `w_gan(G)` 1e-3–5e-3, `w_cls(D)` 1e-2; `max_grad_norm 10`; TTUR 10/5; chuẩn hoá grad theo `mean|p_real|`.
- **Lỗi đã biết** ✅ (bài báo DMD2): bỏ regression loss + không TTUR ⇒ **độ sáng và thống kê ảnh dao động không hội tụ**; không có GAN ⇒ **over-saturated, over-smoothed**; chỉ GAN ⇒ bất ổn; giảm nhẹ **đa dạng** (0.61 vs 0.64 teacher); CFG cố định khi train; SDXL 1-step có block artifacts. ⚠️ Với GReg: real-score không thấy normal/ref ⇒ gradient DM có thể kéo tay/mặt về "ảnh người đẹp theo prompt" thay vì tư thế đúng (xung đột với L_struct); 256² ngoài phân phối SD; trọng số DM phải rất nhỏ so với 1.0.
- **Rủi ro cài đặt**: cao – thêm 2 optimizer, TTUR, FSDP/LoRA, prompt engineering; thành phần tốn kém nhất không đóng băng ⇒ lệch khỏi câu hỏi C1.

---

## 7. Khuyến nghị

1. **Làm ngay: E-LatentLPIPS (SD1.5, `augment='bg'`) như loss toàn ảnh bổ sung**, trọng số 1.0, giữ nguyên VGG + 3 PatchGAN. Lý do: chi phí ≈ +15 % thời gian, +1–2 GB, pip-installable, loss có cặp nên không có mode collapse/drift kiểu adversarial; đo bằng LPIPS/FID/AKD tay–mặt trên split val. Nếu muốn "đặc trưng diffusion" theo đúng nghĩa thì đây là phiên bản rẻ nhất (latent của SD-VAE).
2. **Bước 2 nếu (1) không cải thiện vùng tay**: DIFT-feature loss trên crop vùng upsample 128→256 (hoặc toàn ảnh) với `up_blocks[2]` (1/8) + `t≈100–260`, nhiễu chung, SD-Turbo hoặc SD1.5; dự trù +50–150 % thời gian, +4–7 GB. Đây là thí nghiệm "frozen diffusion as perceptual critic" đúng nghĩa cho luận văn.
3. **Không khuyến nghị DMD-style trong luận văn**: cần critic diffusion được huấn luyện (không frozen), > 24 GB nếu không LoRA, ×5–10 thời gian, và tín hiệu "realism theo prompt" không tương thích với bài toán tái tạo có điều kiện (tư thế/danh tính). Nếu vẫn muốn thử một "critic diffusion đóng băng" thật sự, biến thể rẻ hơn là **chỉ dùng phần GAN**: đầu `cls_pred_branch` nhỏ trên bottleneck của **real-UNet đóng băng** (`classify_forward(..., classify_mode=True)`, `sd_unet_forward.py:249-251`) với Diffusion-GAN noise – chỉ huấn luyện head vài trăm nghìn tham số ⚠️ (chưa có bằng chứng trong DMD2, nhưng là thiết kế của Diffusion2GAN/ADD với discriminator khởi tạo từ teacher).

## 8. Nguồn
- DMD2 repo + arXiv 2405.14867 (NeurIPS 2024); DIFT repo + arXiv 2306.03881; Diffusion2GAN arXiv 2405.05967 + https://mingukkang.github.io/Diffusion2GAN/ + https://github.com/mingukkang/elatentlpips; DreamSim repo; diffusers docs v0.42.1 (`api/models/unet2d-cond`, `api/utilities`); HF `stabilityai/sd-turbo` model card và `scheduler/scheduler_config.json`.
