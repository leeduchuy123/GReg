# Ghi chú code: pix2pix-turbo (img2img-turbo) làm refiner 1 bước cho GReg

- Repo: https://github.com/GaParmar/img2img-turbo — commit `86f54146590ffb4543c8cf85b5a36657da670924` (2025-08-01, "Merge pull request #140"), clone `--depth 1`, chỉ đọc, không chạy.
- Clone local (scratchpad, có thể bị xoá): `C:\Users\LeDucHuy\AppData\Local\Temp\claude\D--Research-Thesis-GReg\4a790883-5e16-41c1-aa85-031ddc6acb39\scratchpad\repos\img2img-turbo`
- Bài báo: "One-Step Image Translation with Text-to-Image Models" (Parmar, Park, Narasimhan, Zhu — arXiv 2403.12036). Licence code: MIT (`LICENSE`).
- Ký hiệu: ✅ = đã đọc trực tiếp trong code (có file:line); ⚠️ = suy luận / kiến thức ngoài repo, chưa kiểm chứng bằng cách chạy.

Các file đã đọc toàn bộ: `src/pix2pix_turbo.py` (229 dòng), `src/model.py` (73), `src/train_pix2pix_turbo.py` (307), `src/my_utils/training_utils.py` (409), `src/inference_paired.py` (75), `src/image_prep.py` (12), `requirements.txt`, `environment.yaml`, `README.md`, `docs/training_pix2pix_turbo.md`; đọc lướt `src/cyclegan_turbo.py` (254) và grep `src/train_cyclegan_turbo.py`.

---

## 1. Kiến trúc: SD-Turbo chạy 1 bước như thế nào

### 1.1 Ba module SD-Turbo được nối thành một mạng end-to-end
✅ `Pix2Pix_Turbo.__init__` nạp 4 thành phần từ HF `stabilityai/sd-turbo`: tokenizer + `CLIPTextModel` (`src/pix2pix_turbo.py:32-33`), `AutoencoderKL` (`:36`), `UNet2DConditionModel` (`:45`), scheduler 1 bước (`:34` → `src/model.py:7-11`).

✅ Text encoder bị đóng băng hoàn toàn (`src/pix2pix_turbo.py:163`). Luồng forward "deterministic" (`:197-203`):
1. `encoded_control = vae.encode(c_t).latent_dist.sample() * scaling_factor` (`:198`) — ảnh điều kiện c_t đi thẳng vào VAE encoder (không có noise cộng thêm). **Lưu ý `.sample()`**: lấy mẫu từ posterior N(μ,σ) chứ không phải `.mode()`, nên "deterministic" vẫn có một chút ngẫu nhiên (xem mục 6).
2. `model_pred = unet(encoded_control, timesteps=[999], encoder_hidden_states=caption_enc)` (`:199`, timestep khai báo ở `:162`).
3. `x_denoised = sched.step(model_pred, 999, encoded_control).prev_sample` (`:200`) — scheduler là `DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")` với `set_timesteps(1)` (`src/model.py:8-9`).
4. Decode qua VAE decoder có skip (`:202-203`), `clamp(-1,1)`.

⚠️ Cách "1 bước" hoạt động về mặt toán: scheduler config của sd-turbo trên HF (`scheduler/scheduler_config.json`) là `prediction_type: "epsilon"`, `beta_schedule: scaled_linear`, β∈[0.00085, 0.012], 1000 bước, `clip_sample: false`. Với 1 timestep duy nhất t=999, `DDPMScheduler.step` tính `pred_original_sample = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t` và vì không còn bước trước (prev_t < 0 → ᾱ_prev = 1) nên `prev_sample` chính là x̂₀, không cộng thêm nhiễu. Tức là mạng về bản chất là **UNet dự đoán ε, rồi chuyển thành x̂₀ bằng công thức closed-form**, với x_t = latent của ảnh điều kiện (chứ không phải ảnh nhiễu).
   - Tính từ config: ᾱ₉₉₉ ≈ 0.00466, √ᾱ ≈ 0.0683, √(1−ᾱ) ≈ 0.9977 → **hệ số khuếch đại 1/√ᾱ ≈ 14.6×** lên sai số của UNet. Đây là lý do thiết kế này "hoạt động" (UNet ở t=999 thấy input gần như toàn nhiễu nên có thể học map latent input → latent output) nhưng cũng là điểm nhạy số học (mục 6).
   - Lưu ý `DDPMScheduler.from_pretrained` nạp từ config của `EulerDiscreteScheduler` (class trên HF khác), diffusers chấp nhận vì các trường β chung ⚠️.

### 1.2 Skip connection VAE encoder → decoder ("skip_conv")
✅ Encoder forward bị monkey-patch (`src/pix2pix_turbo.py:37` → `src/model.py:14-27`): trước mỗi `down_block` lưu activation vào `l_blocks`, cuối cùng gán `self.current_down_blocks = l_blocks` (`model.py:19-20, 26`). 4 activation với số kênh lần lượt 128, 128, 256, 512 (theo thứ tự down) ⚠️ (suy ra từ shape của skip_conv bên dưới).

✅ Decoder forward bị patch (`pix2pix_turbo.py:38` → `model.py:30-54`): sau `mid_block`, với mỗi `up_block` idx: `skip_in = skip_convs[idx](incoming_skip_acts[::-1][idx] * gamma)`; `sample = sample + skip_in` rồi mới qua `up_block` (`model.py:36-43`). Có cờ `ignore_skip` để tắt (`:44-46`).

✅ 4 conv 1×1, không bias (`pix2pix_turbo.py:40-43`):
| tên | in→out | tham số |
|---|---|---|
| skip_conv_1 | 512→512 | 262,144 |
| skip_conv_2 | 256→512 | 131,072 |
| skip_conv_3 | 128→512 | 65,536 |
| skip_conv_4 | 128→256 | 32,768 |
Tổng 491,520 (tính tay). Khi train từ đầu, cả 4 được init hằng `1e-5` (`:133-136`) — tức "Zero-Conv" kiểu ControlNet, ban đầu decoder ≈ VAE gốc.

✅ Kết nối giữa encoder và decoder được làm **bằng side-effect**: `self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks` (`:202`, `:216`) — phải encode trước rồi decode ngay trong cùng forward; `gamma` nhân vào skip (`:161` = 1; mode stochastic đặt `= r` ở `:217`).

### 1.3 LoRA: rank và target modules
✅ Mặc định `lora_rank_unet=8`, `lora_rank_vae=4` (`pix2pix_turbo.py:30`; CLI `training_utils.py:47-48`). `init_lora_weights="gaussian"` (`:141, :148`).

✅ VAE target (`:137-140`), adapter tên `"vae_skip"` (`:143`):
`conv1, conv2, conv_in, conv_shortcut, conv, conv_out, skip_conv_1..4, to_k, to_q, to_v, to_out.0`
→ LoRA phủ cả encoder lẫn decoder VAE (vì peft match theo hậu tố tên module, cả hai nhánh đều có `conv1/conv2/...`) ⚠️.

✅ UNet target (`:144-147`), adapter tên mặc định `"default"` (`:151`, được tham chiếu ở `:206`):
`to_k, to_q, to_v, to_out.0, conv, conv1, conv2, conv_shortcut, conv_out, proj_in, proj_out, ff.net.2, ff.net.0.proj`
→ attention (self + cross), feed-forward GEGLU, các conv của ResNet block, conv của Down/Upsample (`conv`), `conv_out`. **Không** LoRA `conv_in` của UNet vì nó được train full (mục 1.4).

⚠️ `LoraConfig` không truyền `lora_alpha` → peft mặc định `lora_alpha=8` ⇒ scale = α/r = 1.0 cho UNet (r=8) nhưng = 2.0 cho VAE (r=4). Nếu đổi rank mà không đặt `lora_alpha` thì scale thay đổi ngầm. CycleGAN-Turbo thì có đặt `lora_alpha=rank` tường minh (`src/cyclegan_turbo.py:66-68`) — cho thấy tác giả có ý thức về điều này ở nhánh kia.

### 1.4 Tham số nào được train
✅ `set_train()` (`pix2pix_turbo.py:171-184`): mọi param tên có `"lora"` của UNet & VAE, **toàn bộ `unet.conv_in`** (`:177`, Conv2d 4→320, k=3: 11,840 tham số, tính tay), 4 `skip_conv_*` (`:181-184`). Optimizer gom đúng tập này (`train_pix2pix_turbo.py:83-96`; chú ý `:90` chỉ lấy LoRA VAE có `"vae_skip"` trong tên).

✅ Đóng băng: text encoder (`pix2pix_turbo.py:163`), base weights UNet/VAE (peft tự `requires_grad=False` khi add adapter ⚠️), LPIPS (`train:80`), CLIP (`train:77`), backbone của discriminator `cv_ensemble` (`train:72`).

✅ `save_model` (`pix2pix_turbo.py:221-229`) chỉ lưu: rank, target list, state-dict UNet có `"lora"` hoặc `"conv_in"`, state-dict VAE có `"lora"` hoặc `"skip"`. Khi load (`:116-129`) tạo adapter theo rank/target lưu trong pkl rồi `load_state_dict` đè lên.

⚠️ Số tham số trainable: checkpoint `edge_to_image_loras.pkl` trên server CMU có `Content-Length` 38,261,681 byte (HEAD request). Pkl chứa gần như chỉ tensor fp32 của đúng tập trainable ⇒ ≈ **9.57 M tham số** (rank 8/4, giả định checkpoint này dùng rank mặc định). So với UNet SD2.1 ≈ 865 M + VAE ≈ 83.7 M ⇒ ~1 % tham số. `sketch_to_image_stochastic_lora.pkl` là 524,833,285 byte ≈ 131 M fp32 ⇒ chắc chắn rank lớn hơn nhiều (có thể 128 như CycleGAN) — không xác nhận được nếu không nạp file.

### 1.5 Mode stochastic (sketch) — chỉ dùng khi inference
✅ `forward(..., deterministic=False, r, noise_map)` (`pix2pix_turbo.py:204-218`):
- scale toàn bộ LoRA theo r: `unet.set_adapters(["default"], weights=[r])`, `set_weights_and_activate_adapters(vae, ["vae_skip"], [r])` (`:206-207`);
- `unet_input = encoded_control * r + noise_map * (1 − r)` (`:210`);
- `unet.conv_in` là `TwinConv` (`:16-26`): `out = conv_in_pretrained(x).detach()·(1−r) + conv_in_curr(x)·r` — chỉ được tạo khi nạp model `sketch_to_image_stochastic` (`:100-101`);
- skip VAE nhân `gamma = r` (`:217`).
⇒ r=1: hoàn toàn theo điều kiện; r→0: trở về SD-Turbo gốc text-to-image từ nhiễu. ✅ Script train **không** có nhánh này (chỉ gọi `deterministic=True`, `train:175, :198`) ⇒ muốn train stochastic phải tự thêm.

---

## 2. Training (`src/train_pix2pix_turbo.py`, args ở `training_utils.py:12-101`)

### 2.1 Loss và trọng số mặc định
✅ CLI defaults (`training_utils.py:22-27`):
| loss | λ mặc định | code |
|---|---|---|
| L2 (MSE pixel, ảnh [-1,1]) | `lambda_l2 = 1.0` | `train:177` |
| LPIPS (VGG) | `lambda_lpips = 5.0` | `train:75, :178` |
| CLIP-sim (ViT-B/32) | `lambda_clipsim = 5.0` | `train:76, :181-187` |
| GAN (vision-aided, CLIP) | `lambda_gan = 0.5` | `train:64-66, :199, :211, :219` |
- GAN: `vision_aided_loss.Discriminator(cv_type='clip', loss_type='multilevel_sigmoid_s')` (`train:66`, default `training_utils.py:22-23`); backbone CLIP đóng băng, chỉ head train (`train:71-73`). Discriminator **không có điều kiện** — chỉ nhìn ảnh output/ảnh thật (`:199, :211, :219`), không ghép với input như pix2pix cổ điển.
- CLIP-sim: `1 − cos_sim(img, caption)/100` (CLIP logit scale 100) (`:186`); ảnh được resize 224 + renorm CLIP (`:125, :182-183`); caption lấy từ `batch["caption"]` tokenised bằng `clip.tokenize` (`:184`).
- ✅ Mỗi iteration làm **3 lượt backward + 2 forward generator**: (i) L2+LPIPS+CLIPsim → step G (`:175-193`); (ii) forward lại, loss_G adversarial → step G lần 2 (`:198-205`); (iii) D real rồi D fake, 2 step D riêng (`:211-224`). Cả hai lần step G đều gọi `lr_scheduler.step()` (`:192, :204`) ⇒ scheduler G tiến 2 bước/iter.

### 2.2 Optimizer / LR / steps / batch / độ phân giải
✅ AdamW cho cả G và D (`train:98-112`): `lr=5e-6` (mặc định `training_utils.py:61`), β=(0.9, 0.999), weight_decay 1e-2, eps 1e-8 (`:75-78`); `lr_scheduler="constant"`, `lr_warmup_steps=500` (vô nghĩa với constant) (`:62, :68`); `max_grad_norm=1.0` (`:79`), clip trên `layers_to_opt` (`train:190, :202`).
✅ `train_batch_size=4` (`training_utils.py:55`), doc khuyên `--train_batch_size=2` ở 512 (`docs/training_pix2pix_turbo.md:51`). `max_train_steps=10_000`, `num_training_epochs=10` (`:56-57`), `gradient_accumulation_steps=1` (`:59`), checkpoint mỗi 500 (`:58`), eval mỗi 100 trên 100 mẫu (`:35, :37`).
✅ Độ phân giải thực tế do `--train_image_prep` quyết định, mặc định `resized_crop_512` = Resize(512, LANCZOS) + CenterCrop(512) (`training_utils.py:31-32, :194-198`). `--resolution=512` (`:54`) **chỉ** dùng cho transform khi tính FID (`train:156-159`), không ảnh hưởng dataset.
✅ Mixed precision: `--mixed_precision {no,fp16,bf16}` mặc định None (`:92`); khi bật, toàn bộ `net_pix2pix`, disc, lpips, clip bị `.to(dtype=weight_dtype)` (`train:126-136`) — tức cast thẳng tham số (kể cả LoRA đang train) sang half, không phải autocast (xem mục 6).
✅ Gradient checkpointing: `--gradient_checkpointing` → `unet.enable_gradient_checkpointing()` (`train:58-59`), chỉ UNet. xformers: `--enable_xformers_memory_efficient_attention` (`:52-56`), raise nếu không có. `--allow_tf32` (`:61-62`).
⚠️ **VRAM không được báo cáo** ở README/docs. README chỉ nêu latency inference 512²: 0.29 s (A6000) / 0.11 s (A100) (`README.md:17`). Ước lượng: UNet 865 M fp32 + grad/Adam chỉ cho ~10 M param, nhưng activation của UNet+VAE ở 512² × 2 forward/iter + VGG-LPIPS + CLIP disc ⇒ batch 2 ở 512 có lẽ cần ≥24 GB nếu không có xformers/grad-ckpt; ở 256² giảm ~4× activation.

### 2.3 Dataset và prompt
✅ `PairedDataset` (`training_utils.py:218-308`): thư mục `dataset_folder/{train_A, train_B, test_A, test_B}` + `train_prompts.json`, `test_prompts.json` (`:237-244`; mô tả ở `docs/training_pix2pix_turbo.md:12-35`). JSON là dict `{tên_file: caption}`; **danh sách ảnh = keys của JSON** (`:247`), cùng tên file ở A và B (`:286-287`).
✅ Chuẩn hoá: input A → `to_tensor` **[0,1]** (`:291-292`); target B → **[-1,1]** (`:294-296`). Prompt tokenised bằng CLIP tokenizer của SD-Turbo, pad 77 (`:298-301`); batch trả `output_pixel_values, conditioning_pixel_values, caption, input_ids` (`:303-308`).
✅ Cùng một transform `T` áp **riêng rẽ** cho A và B (`:291, :294`) — chỉ an toàn vì các prep mặc định là deterministic (mục 6).
✅ Ảnh không `.convert("RGB")` (`:286-287`) khác với `UnpairedDataset` (`:396-397`).
✅ Eval: batch 1 bắt buộc (`train:117, :269`), tính L2/LPIPS/CLIPsim và (tuỳ chọn) clean-FID với `test_B` làm reference (`train:153-163, :259-301`).

---

## 3. Inference (`src/inference_paired.py`)
✅ 1 lần gọi `model(c_t, prompt)` = 1 lần VAE-enc + 1 UNet + 1 VAE-dec (`pix2pix_turbo.py:197-203`); không có vòng lặp denoise. Latency: README 0.29 s/0.11 s cho 512² trên A6000/A100 (`README.md:17`) ⚠️ (không nêu fp16 hay fp32).
✅ `--use_fp16` → `model.half()` và cast input (`inference_paired.py:34-36, :51-52`). Input bị resize về bội số 8 (`:39-41`). Output `*0.5+0.5` → PIL (`:72`).
✅ Prompt: tokenize + text encoder mỗi lần gọi (`pix2pix_turbo.py:190-194`), đưa vào UNet qua cross-attention `encoder_hidden_states` (`:199`). Prompt là **bắt buộc** (`:188` assert; CLI `required=True` ở `inference_paired.py:14`). Với refiner ta có thể dùng 1 prompt cố định và cache `caption_enc` (text encoder đóng băng) ⚠️.
✅ Deterministic vs stochastic:
- `edge_to_image` / `model_path`: `deterministic=True` (`inference_paired.py:46-53, :66-70`) — nhưng vẫn có `latent_dist.sample()` (`pix2pix_turbo.py:198`) ⇒ không bit-exact giữa 2 lần chạy trừ khi seed ⚠️.
- `sketch_to_image_stochastic`: input nhị phân `to_tensor(img) < 0.5` (`inference_paired.py:56-57`), `noise ~ N(0,1)` shape (1,4,H/8,W/8) với `torch.manual_seed(seed)` (`:58-60`), `r = --gamma` mặc định 0.4 (`:20, :64`); nội suy latent/LoRA/skip theo r (mục 1.5). Gradio demo dùng cùng API (`gradio_sketch2image.py:82-83`).

---

## 4. Trọng số pretrained cần tải và licence
✅ Code chỉ gọi `from_pretrained("stabilityai/sd-turbo", subfolder=...)` cho tokenizer, text_encoder, vae, unet, scheduler (`pix2pix_turbo.py:32-45`, `model.py:8`) — **không** truyền `variant="fp16"` ⇒ tải bản fp32.
⚠️ Kích thước file trên HF (từ API `huggingface.co/api/models/stabilityai/sd-turbo`, 2026-11-02):
| file | fp32 | fp16 |
|---|---|---|
| `unet/diffusion_pytorch_model` | 3,463.7 MB | 1,731.9 MB |
| `text_encoder/model` (OpenCLIP ViT-H text) | 1,361.6 MB | 680.8 MB |
| `vae/diffusion_pytorch_model` | 334.6 MB | 167.3 MB |
| `sd_turbo.safetensors` (đóng gói, không cần) | 5,214.6 MB | — |
⇒ ~5.2 GB tải về cho fp32 (cache HF).
⚠️ Thêm cho training: LPIPS VGG16 (torchvision, ~528 MB), CLIP ViT-B/32 cho CLIP-sim (`train:76`, ~338 MB), `vision_aided_loss` với `cv_type='clip'` (tải CLIP riêng qua gói đó), clean-fid InceptionV3 (~100 MB) nếu `--track_val_fid`.
⚠️ LoRA pretrained của tác giả (không cần cho bài toán refiner): `edge_to_image_loras.pkl` 38.3 MB, `sketch_to_image_stochastic_lora.pkl` 524.8 MB, host trên `www.cs.cmu.edu/~img2img-turbo/models/` (`pix2pix_turbo.py:48, :82`). CycleGAN: day2night/night2day/clear2rainy/rainy2clear (`cyclegan_turbo.py:127-145`).
⚠️ Licence: code MIT (`LICENSE`). SD-Turbo: README repo trỏ đến HF LICENSE (`README.md:228`); file hiện hành trên HF là `LICENSE.md` = **"STABILITY AI COMMUNITY LICENSE AGREEMENT" (cập nhật 2024-07-05)** — miễn phí cho nghiên cứu/phi thương mại và doanh thu < 1 M USD/năm, cần ghi "Powered by Stability AI"; model card ghi "For commercial use, please refer to https://stability.ai/license". Dùng cho luận văn là OK. CLIP (OpenAI) MIT; LPIPS BSD-2; vision-aided-loss MIT ⚠️.
⚠️ Phiên bản pin trong `environment.yaml`: `diffusers==0.25.1`, `transformers==4.35.2`, `peft` không pin, `torch>=2.0.1`, python 3.10. `requirements.txt` không pin gì cả (`requirements.txt:1-28`). Monkey-patch `my_vae_encoder_fwd/decoder_fwd` sao chép nội dung `Encoder.forward/Decoder.forward` của diffusers 0.25 ⇒ diffusers mới hơn có thể đổi chữ ký (`latent_embeds`, `upscale_dtype`) → nên pin 0.25.1.

---

## 5. Cần sửa gì cho refiner GReg

### (a) Điều kiện thêm kênh (normal map, visibility mask) ngoài RGB
Hiện tại input c_t phải là **3 kênh** vì đi qua VAE encoder `conv_in` 3→128 (`pix2pix_turbo.py:198`, `model.py:15`); UNet nhận latent 4 kênh qua `conv_in` 4→320 (full-train, `:177`).
Phương án (⚠️ thiết kế, chưa có trong code):
1. **Encode riêng rồi concat ở latent** (ít xâm lấn nhất, giống ControlNet/InstructPix2Pix): `z_rgb = vae.encode(rgb)`, `z_cond = vae.encode(normal_map_3ch)` (cùng VAE, có thể đóng băng), mask downsample 8× bằng `F.interpolate` (area) → `unet_in = cat([z_rgb, z_cond, mask_lat], 1)` (4+4+1 = 9 kênh). Mở rộng `unet.conv_in` thành `Conv2d(9, 320, 3)`: copy 4 kênh đầu từ pretrained, kênh mới init 0 (zero-init như ControlNet). `conv_in` đã nằm trong `layers_to_opt` (`train:88`) nên tự được train. Phải sửa `sched.step(model_pred, t, encoded_control)` (`pix2pix_turbo.py:200`) để `sample` vẫn là 4 kênh `z_rgb` chứ không phải tensor 9 kênh. `save_model` lọc `"conv_in"` nên conv_in mới vẫn được lưu (`:227`), nhưng khi load (`:116-129`) phải tạo lại conv_in 9 kênh **trước** `load_state_dict`.
2. Mở rộng `vae.encoder.conv_in` (3→128) thành (3+3+1)→128: skip activation (`model.py:19`) sẽ mang cả thông tin normal/mask xuống decoder — hữu ích vì skip quyết định chi tiết pixel; nhưng VAE encoder chỉ có LoRA rank 4 trên `conv_in` (`:137`), phải full-train `conv_in` mới (thêm vào `set_train`/`layers_to_opt`) và giữ 3 kênh đầu từ pretrained.
3. Cách rẻ nhất để thử nhanh: đóng gói (normal, mask) thành ảnh RGB 3 kênh và chạy 2 lần encoder (không sửa kiến trúc), chỉ sửa concat ở UNet như (1).
Dataset: `PairedDataset.__getitem__` (`training_utils.py:258-308`) cần trả thêm `normal`, `mask` (cùng transform với A/B); file lưu theo cùng tên trong thư mục `train_N`, `train_M` ⚠️.
Lưu ý thang giá trị: input RGB hiện đưa vào VAE ở **[0,1]** (`training_utils.py:292`) trong khi VAE SD được train với [-1,1]; với refiner mà input ≈ output nên chuẩn hoá input về [-1,1] cho nhất quán (mục 6).

### (b) Giới hạn loss / chỉnh sửa trong vùng mask
- L2: thay `F.mse_loss(pred, tgt)` (`train:177`) bằng `((pred−tgt)²·m).sum()/(m.sum()·C + eps)`, hoặc có trọng số `w_in·m + w_out·(1−m)`.
- LPIPS: `lpips.LPIPS` trả map (B,1,H,W) nếu gọi với `spatial=True` khi khởi tạo (`lpips.LPIPS(net='vgg', spatial=True)`) ⇒ có thể nhân mask rồi mean ⚠️; hoặc crop bbox quanh tay rồi LPIPS trên crop (tốt hơn cho tay nhỏ ở 256²).
- GAN: discriminator hiện nhìn toàn ảnh (`train:199, :211, :219`) — để tập trung vào tay, đưa crop bbox tay (resize về 224 vì backbone CLIP) hoặc ảnh đã composite `pred·m + tgt·(1−m)`. CLIP-sim: với refiner nên đặt `--lambda_clipsim 0` (`training_utils.py:27`), vì caption video ký hiệu không mang nghĩa pixel.
- "Edit chỉ trong mask" ở output: composite sau decode `out = pred·m + input·(1−m)` (như inpainting) — làm ở `forward` (`pix2pix_turbo.py:203`) hoặc ngoài; nếu composite trước loss thì gradient ngoài mask = 0 tự nhiên.
- Mask ở latent (cho skip/UNet): không bắt buộc; nhưng nếu muốn UNet "biết" vùng cần sửa thì đưa mask làm kênh điều kiện như (a).

### (c) Loss thời gian giữa 2 frame liên tiếp
- Dataset: trả cặp (frame_t, frame_{t+1}) cho cả A và B (sửa `PairedDataset` thành dataset theo clip: `__getitem__` nạp `name_t` và `name_{t+1}` theo quy ước tên file `clip_XXXX_fYYYY.png`) ⚠️.
- Forward 2 frame trong cùng batch (stack theo batch dim: `x_src = cat([a_t, a_t1])`) để dùng nguyên `net_pix2pix(x_src, ...)` (`train:175`), rồi tách. Chú ý skip-activation side-effect (`pix2pix_turbo.py:202`) hoạt động theo batch nên an toàn khi gộp batch, **không** an toàn nếu gọi encode 2 lần rồi decode 2 lần xen kẽ.
- Loss: (i) đơn giản: `‖(ŷ_{t+1} − ŷ_t) − (y_{t+1} − y_t)‖₁` (temporal-difference matching); (ii) warped: `‖ŷ_{t+1} − warp(ŷ_t, flow_{t→t+1})‖·occ_mask` với flow từ mesh (đã có trong GReg — xem memory "arm-warp direction"), chỉ trong vùng mask tay. Thêm `--lambda_temporal` vào `parse_args_paired_training` (`training_utils.py:21-27`) và cộng vào `loss` ở `train:179`.
- Để loại nguồn nhiễu khiến frame nhấp nháy: thay `latent_dist.sample()` bằng `.mode()` (`pix2pix_turbo.py:198`) — xem mục 6.
- Chi phí: batch hiệu dụng ×2; ở 256² chấp nhận được ⚠️.

### (d) Chạy 256×256 thay vì 512
- ✅ Không có hard-code 512 trong model; chỉ cần `--train_image_prep resize_256 --test_image_prep resize_256` (đã có sẵn, `training_utils.py:205-208`) hoặc `no_resize` (`:213-214`) nếu dữ liệu đã 256. Đặt `--resolution 256` để FID transform khớp (`train:158`).
- Latent 32×32×4; skip activation 256/128/64/32 ⚠️. UNet SD2.1-base được train ở 512 (latent 64) — ở latent 32 chất lượng prior có thể giảm nhưng đây là fine-tune có LoRA nên chấp nhận được; SD-Turbo chính thức là model 512 ⚠️.
- LPIPS VGG và discriminator CLIP (224 input) vẫn chạy; CLIP-sim đã resize về 224 (`train:183`).
- Nếu muốn giữ prior tốt hơn: upsample 256→512 trước model rồi downsample output (latency ×4) — không khuyến nghị cho real-time ⚠️.

---

## 6. Bẫy / điểm dễ hỏng âm thầm
1. ✅ **Input ở [0,1], target ở [-1,1]** (`training_utils.py:290-296`). VAE SD được train với ảnh [-1,1]; đưa [0,1] vào vẫn chạy (mạng học bù) nhưng skip activation/latent lệch phân phối. Với refiner (input ≈ output) nên chuẩn hoá cả hai về [-1,1]. Khi inference cũng phải dùng đúng thang đã train (`inference_paired.py:50, :67` dùng `to_tensor` = [0,1]).
2. ✅ **`latent_dist.sample()` trong nhánh deterministic** (`pix2pix_turbo.py:198`, cả `:208`) ⇒ nhiễu posterior VAE lọt vào mỗi forward; gây nhấp nháy giữa frame video và làm 2 forward trong 1 iteration (`train:175` vs `:198`) khác nhau. Dùng `.mode()` hoặc `.mean`.
3. ⚠️ **Khuếch đại 14.6× ở t=999** (mục 1.1): mọi sai số ε̂ của UNet được nhân ~14.6 khi đổi sang x̂₀; fp16 ở đây dễ tràn/mất chính xác. Kết hợp với mục 4 bên dưới (cast tham số sang fp16) là rủi ro NaN.
4. ✅ **Mixed precision = cast thẳng tham số** (`train:133-136`): `net_pix2pix.to(dtype=fp16)` biến cả LoRA/skip_conv đang train thành fp16 rồi AdamW cập nhật trực tiếp trên fp16 (không master weight fp32) ⇒ với lr 5e-6 các update nhỏ hơn eps fp16 có thể bị nuốt (update ≈ 0). Dùng `bf16` hoặc giữ fp32 cho tham số train và chỉ autocast forward.
5. ✅ **Transform ngẫu nhiên tách rời A/B**: `self.T` gọi riêng cho input và output (`training_utils.py:291, :294`). Nếu chọn `--train_image_prep resize_286_randomcrop_256x256_hflip` (`:199-204`) cho paired training thì crop/flip của A và B **khác nhau** ⇒ cặp bị lệch mà không báo lỗi. Chỉ dùng prep deterministic, hoặc viết transform dùng chung seed.
6. ✅ **Không `.convert("RGB")`** ở `PairedDataset` (`training_utils.py:286-287`): PNG RGBA/grayscale/palette sẽ cho 4/1 kênh → lỗi shape ở VAE, hoặc tệ hơn là ảnh palette bị đọc sai. (UnpairedDataset có convert, `:396-397`.)
7. ✅ **`max_train_steps` không dừng vòng lặp**: không có `break` khi `global_step >= max_train_steps` (`train:167-302`); training chạy hết `num_training_epochs` (mặc định 10). Progress bar chỉ là range (`:144`). Với lr constant không sao, nhưng với cosine/linear thì lr về 0 rồi vẫn train.
8. ✅ **`lr_scheduler.step()` 2 lần/iter cho G** (`train:192, :204`) ⇒ các scheduler không-constant decay nhanh gấp đôi so với `max_train_steps`.
9. ✅ **Checkpoint/eval/viz ở `% N == 1`** (`train:244, :254, :259`) ⇒ eval ở step 1 (trước khi học gì) và checkpoint `model_1.pkl`; file cuối cùng không nhất thiết là step cuối. Khi `checkpointing_steps=1` không bao giờ lưu (1 % 1 = 0).
10. ✅ **Side-effect skip**: encoder gán `current_down_blocks` (`model.py:26`) và forward gán `incoming_skip_acts` (`pix2pix_turbo.py:202`). Gọi `vae.decode` mà không encode ngay trước (ví dụ decode latent khác, hoặc với `torch.no_grad` ở nơi khác) sẽ dùng activation cũ → ảnh sai mà không lỗi. Với gradient checkpointing/DDP cũng cần cẩn thận vì activation được giữ tham chiếu.
11. ✅ **`sys.path.append("src/")`** (`pix2pix_turbo.py:11-12`, `cyclegan_turbo.py:10-11`) ⇒ phải chạy từ thư mục gốc repo; `train_pix2pix_turbo.py` import `from pix2pix_turbo import` nên chạy bằng `python src/train_...` (`docs/training_pix2pix_turbo.md:46`).
12. ✅ **`save_model` lọc theo substring** (`pix2pix_turbo.py:227-228`): `"conv_in"` sẽ bắt mọi key UNet chứa `conv_in` (hiện chỉ có top-level, OK), nhưng nếu thêm module mới có tên chứa `lora`/`skip`/`conv_in` thì sẽ được lưu kèm (tốt) hoặc nếu đặt tên khác thì **không được lưu** (ví dụ thêm `cond_proj` cho normal map → mất khi load). Phải sửa cả `save_model` lẫn 3 nhánh load (`:116-129`).
13. ✅ **Load checkpoint chỉ qua `torch.load` pickle** (`:66, :102, :117`) — không `weights_only=True`; torch ≥2.6 mặc định `weights_only=True` sẽ fail nếu pkl có object không phải tensor (có list string `target_modules`) ⚠️ → cần `weights_only=False`.
14. ✅ **Adapter name UNet là `"default"`** (`:151`, dùng ở `:206`); nếu đổi tên khi `add_adapter` thì nhánh stochastic hỏng. VAE lọc LoRA theo `"vae_skip"` (`train:90`) — đổi tên adapter sẽ khiến optimizer **bỏ qua** LoRA VAE mà `assert` ở `:91` không bắt được (vòng for không vào).
15. ✅ **Discriminator không điều kiện + fused_attn tắt thủ công** (`train:147-150`): backbone CLIP trong `vision_aided_loss` phải tắt fused attention; nếu phiên bản timm/CLIP đổi tên module `"attn"` thì vòng lặp không tác dụng → có thể lỗi/khác biệt im lặng ⚠️.
16. ✅ **`--mixed_precision` default None** (`training_utils.py:92`) ⇒ accelerate dùng config `accelerate config` của máy; ở `train:126-130` chỉ đọc `accelerator.mixed_precision`. Nếu `accelerate config` đặt fp16 thì rơi vào bẫy số 4 dù CLI không truyền gì.
17. ⚠️ `accelerator.prepare(net_clip, net_lpips)` (`train:123`) bọc cả model đóng băng; multi-GPU DDP sẽ phàn nàn vì không có param cần grad; code chủ yếu được thử 1 GPU.
18. ✅ `clip_grad_norm_` trên `layers_to_opt` sau `accelerator.backward` (`:189-190`); với `gradient_accumulation_steps>1`, loss không chia cho số bước tích luỹ thủ công (accelerate tự scale) nhưng loss D cũng bị tích luỹ theo `accumulate(*l_acc)` (`:169-170`) — OK, chỉ cần biết optimizer_disc step cũng bị gate.
19. ✅ `requirements.txt` không pin version và kéo `xformers`, `gradio`, `streamlit-keyup`, `torchdata==0.6.1` (environment.yaml) — `torchdata 0.6.1` không tương thích torch mới; nên cài tay: torch/torchvision, diffusers==0.25.1, transformers==4.35.2, peft (≤0.10 để match diffusers 0.25 ⚠️), lpips, clean-fid, vision-aided-loss, openai-clip, accelerate, wandb.
20. ✅ wandb là mặc định (`training_utils.py:86`); không có `--report_to none` → cần `WANDB_MODE=offline` hoặc đổi sang `tensorboard`.

---

## 7. Tóm tắt áp dụng cho GReg (⚠️ đề xuất)
- Dùng nhánh deterministic, `.mode()` thay `.sample()`, input/target đều [-1,1], 256², `--lambda_clipsim 0`, bf16 hoặc fp32, `--train_image_prep resize_256`.
- Điều kiện: concat latent (RGB thô từ GReg | normal map | mask↓8) → `unet.conv_in` 9→320 zero-init kênh mới; cân nhắc thêm normal/mask vào `vae.encoder.conv_in` để skip mang thông tin.
- Loss: L2+LPIPS có mask tay (crop bbox), GAN trên crop tay, + temporal (warp theo mesh flow) với λ nhỏ (0.1–1).
- Phải tự thêm: dataset theo clip, `--lambda_temporal`, hàm mask-loss, sửa `save_model`/load cho module mới, `break` theo `max_train_steps`, `weights_only=False`.
