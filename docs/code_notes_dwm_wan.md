# Ghi chú đọc mã: DWM (Dexterous World Models) nhánh Wan2.1 → hướng dẫn port cho D6 (Wan2.1-Fun-V1.1-1.3B-InP + LoRA + kênh zero-init)

Ngày: 2026-11-02. Mục tiêu: port công thức huấn luyện **Wan2.1-Fun-InP + LoRA + patch-embedding mở rộng zero-init** của DWM
(CVPR 2026) sang bài toán D6 của GReg: điều kiện = `I_st` (ảnh nguồn warp theo tương ứng mesh tới từng khung đích),
`V·C` ∈ [0,1] (1 = giữ/đã biết, 0 = phải sinh), **mới**: video normal-map SMPL-X đích `N_t` và video CCM (RGB);
khung 0 = khung nguồn thật; cửa sổ 49 khung (4n+1), 25 fps, crop vuông 320² rồi 480², prompt cố định, loss vùng tay/mặt.

Ký hiệu: ✅ = đã kiểm chứng trực tiếp trong mã (file:line) hoặc trang chính thức; ⚠️ = suy luận / ước lượng của tôi.

| Nguồn | Phiên bản | Giấy phép |
|---|---|---|
| `dwm/` https://github.com/snuvclab/dwm (clone `--depth 1`, chỉ đọc, không chạy) — thư mục `C:\Users\LeDucHuy\AppData\Local\Temp\claude\D--Research-Thesis-GReg\4a790883-5e16-41c1-aa85-031ddc6acb39\scratchpad\repos\dwm` | `19de79b879e662a133761e9f9afb60d4d92a618c` | Apache-2.0 ✅ (`LICENSE`) |
| HF `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP` (config.json + cây file qua API, model card) | main @ 2026-11-02 | Apache-2.0 ✅ (front-matter model card) |
| diffusers `v0.42.1` (`transformer_wan.py`, `pipeline_wan_i2v.py`, `single_file_utils.py`, đọc qua web) | 0.42.1 = bản mới nhất PyPI ✅ | Apache-2.0 |
| VideoX-Fun `scripts/wan2.1_fun/train.py` (đọc qua web, để biết Fun-InP được pretrain thế nào) | main | Apache-2.0 |
| Bài báo arXiv 2512.17907 (HTML v1) | v1 | – |

Đường dẫn file bên dưới là tương đối với gốc repo `dwm/`.

---

## 0. Kết luận ngắn

1. **Nhánh Wan của DWM dựng trên VideoX-Fun (aigc-apps)** — mã model là bản vendored của VideoX-Fun/Wan2.1 (`training/wan/models/wan_transformer3d.py:1` "Modified from Wan2.1 model.py"; `train_dwm_wan.py:2` "VideoX-Fun style"; LoRA "VideoX-Fun style" `train_dwm_wan.py:2319`; lời cảm ơn `README.md:143`) ✅. Bài báo dùng **CogVideoX-Fun-V1.5-5B-InP** làm model chính (Appendix A: LoRA r64/α64, lr 1e-4, batch hiệu dụng 56, ~10 ngày trên 4×A100) ✅; nhánh Wan là phần phát hành thêm (config ghi ngày 2026-04-01, `training/wan/configs/examples/dwm_wan_14b_lora.yaml:4`) ✅.
2. Checkpoint DWM nạp: **`Wan2.1-Fun-V1.1-14B-InP`** (`training/wan/README.md:27-47`, `dwm_wan_14b_lora.yaml:9`) ✅; nhưng file pipeline mặc định trỏ **`PAI/Wan2.1-Fun-V1.1-1.3B-InP`** (`configs/pipelines/wan2.1_fun_inp_hand_concat.yaml:4`) ✅ → code đã được thiết kế chạy với 1.3B. LoRA đã train: HF `byungjun-kim/DWM-Wan2.1-Fun-14b-LoRA` (`training/wan/README.md:89`) ✅ (14B, không dùng được cho 1.3B).
3. **1.3B-InP có đúng layout 36 kênh** (`in_dim: 36`, `model_type: "i2v"`, `dim 1536`, 30 lớp, 12 head, ffn 8960) và **có nhánh CLIP ảnh** (`model_type=i2v` ⇒ `img_emb = MLPProj(1280, dim)` + cross-attn có `k_img/v_img`), **không có** `add_ref_conv`/control adapter (không có khoá đó trong config ⇒ mặc định False) ✅ (§3).
4. **Có thể bỏ VideoX-Fun, dùng diffusers 0.42.1 thuần**: `WanTransformer3DModel`, `AutoencoderKLWan`, `WanImageToVideoPipeline` đều có, layout đầu vào i2v của diffusers trùng (noisy 16 | mask 4 | latent điều kiện 16) ✅. Việc cần làm: tự chuyển trọng số 1.3B-InP sang định dạng diffusers (bộ tự nhận diện single-file của diffusers **nhận nhầm** checkpoint này là `wan-t2v-1.3B`) ✅, và **tự chuẩn hoá latent** bằng `latents_mean/latents_std` (VAE diffusers không tự làm, VAE VideoX-Fun thì có) ✅.
5. Phát hiện quan trọng (bẫy) — ✅ code / ⚠️ hệ quả: trong chế độ `dwm` DWM huấn luyện với **`clip_fea=None`** (`train_dwm_wan.py:3323-3333`), nhưng lúc validation/inference truyền **CLIP toàn 0** (`pipeline_wan_fun_inpaint_hand_concat.py:439-446`). Với `model_type=i2v`, cross-attn luôn cắt `context[:, :257]` làm token ảnh (`wan_transformer3d.py:599-600`) ⇒ khi train, 257 token *văn bản* đầu đi vào nhánh `k_img/v_img`, nhánh văn bản chỉ thấy embedding của padding; khi infer thì ngược lại ⇒ **lệch train/infer**. Khi port: **luôn truyền cùng một loại token ảnh khi train và infer** (khuyến nghị: CLIP của khung nguồn thật, đúng như Fun-InP được pretrain).
6. Ước lượng tài nguyên (1.3B, LoRA r32 mọi Linear, gradient checkpointing, bf16, latent/T5/CLIP cache sẵn, batch 1) ⚠️: **320²×49 ≈ 6–9 GB, 480²×49 ≈ 9–13 GB**; ≈ 1.5–2.5 s/bước (320², 4090), ≈ 2–3 s/bước (480², A100) (§6). 4090 24 GB đủ cả hai độ phân giải với batch 1; A100 80 GB cho batch 4–8 ở 480².

---

## 1. Danh sách file đã đọc

Đọc toàn bộ: `training/wan/models/wan_transformer3d_with_conditions.py` (269 dòng), `training/wan/pipeline/pipeline_wan_fun_inpaint_hand_concat.py` (554), `training/wan/pipeline/pipeline_wan_fun_inpaint.py` (772, lớp cha), `training/wan/train_dwm_wan.py` (3558), `training/wan/args.py` (239), `training/wan/utils/discrete_sampler.py` (51), `training/wan/utils/lora_utils.py` (530), `training/wan/utils/cfg_optimization.py`, `training/wan/utils/checkpoint_utils.py`, `training/wan/models/wan_transformer3d_i2v_with_conditions.py` (245, biến thể **diffusers** của DWM), `training/wan/models/__init__.py`, `training/wan/README.md`, `training/wan/configs/examples/dwm_wan_14b_lora.yaml`, `configs/pipelines/wan2.1_fun_inp_hand_concat.yaml`, `training/wan/examples/train_static_hand_concat.sh`, `.../infer_static_hand_concat.sh`, `training/wan/diffusers_compat.py`, `data_processing/encode_with_wan_impl.py` (419), `requirements.txt`.
Đọc phần liên quan: `training/wan/models/wan_transformer3d.py` (1–1698: attention, RoPE, block, cross-attn I2V, `__init__`, `forward`, `from_pretrained`; bỏ qua `Wan2_2Transformer3DModel`), `training/wan/models/wan_vae.py` (490–710: encode/decode/chuẩn hoá), `training/wan/models/wan_image_encoder.py` (500–556), `training/wan/models/wan_text_encoder.py` (lướt cấu trúc), `training/wan/dataset.py` (1–1200; phần 1200–1756 là lớp multiview/human-motion/BucketSampler **không nằm trên đường DWM**: train dùng `VideoDatasetWithConditionsAndResizing` + `DataLoader(shuffle=True)`, `train_dwm_wan.py:2589,2622-2630`), `training/wan/inference.py` (290–530, 880–980, 1340–1430, 1595–1631).
Không đọc (không thuộc đường DWM-Wan2.1-concat): `dist/*` (xfuser/FSDP), VACE, multiview, causal, Wan2.2 (`wan_vae3_8.py`, pipeline 2.2), `lightx2v_compat.py`, `fm_solvers*.py` (chỉ dùng khi đổi scheduler).

---

## 2. Pipeline tensor của một bước huấn luyện (DWM, chế độ `conditioning_mode="dwm"`)

Ký hiệu kích thước cho DWM gốc 480×720×49; cột cuối là con số cho D6 ở 320² / 480².

### 2.1 Dữ liệu → pixel ✅
- Video đọc bằng decord, cửa sổ `max_num_frames=49` khung liên tiếp, điểm bắt đầu ngẫu nhiên (`dataset.py:274-291`); thiếu khung thì **lặp khung cuối** (`dataset.py:318-320`). Resize về bucket gần nhất (`dataset.py:956-972`, bucket 480×720 ở `dwm_wan_14b_lora.yaml:55-57`).
- Chuẩn hoá: `/255` rồi `Normalize(0.5, 0.5)` ⇒ **[-1, 1]** (`dataset.py:153-161`). Batch: `[B, F, C, H, W]` → `rearrange "b f c h w -> b c f h w"` (`train_dwm_wan.py:3068`).
- Thực tế DWM dùng `load_tensors: true` (`dwm_wan_14b_lora.yaml:39`): latent mã hoá trước bằng `encode_with_wan_impl.py:103-118` (video [-1,1], **VAE bf16** `:117,:348`, lưu `latent_dist.sample()` — tức **một mẫu cố định** cho mỗi clip). Khi đó VAE và T5 bị xoá khỏi GPU lúc train (`train_dwm_wan.py:2690-2701`).

### 2.2 VAE → latent ✅
- VAE VideoX-Fun `AutoencoderKLWan` (`wan_vae.py:624-710`): nén **thời gian ×4, không gian ×8**, `z_dim=16`. Encoder chạy theo cụm khung **1, 4, 4, 4, …** (`wan_vae.py:527-543`, `iter_ = 1 + (t-1)//4`) ⇒ **F = 4n+1 bắt buộc**; 49 khung → **13** latent frame; latent frame 0 chỉ chứa pixel frame 0 (VAE nhân quả).
- **Chuẩn hoá nằm *bên trong* encode**: `mu = (mu - mean) * (1/std)` với 16 giá trị mean/std cố định (`wan_vae.py:545-549`, hằng số `:634-644`); decode làm ngược lại (`:558-562`) và `clamp(-1,1)` (`:673`). Lưu ý: chỉ `mu` được chuẩn hoá, `log_var` thì không ⇒ `.sample()` thêm nhiễu với std chưa chia (⚠️ ảnh hưởng không đáng kể vì log_var rất âm).
- `latents`: `[B,16,13,60,90]` (DWM) — D6: **`[B,16,13,40,40]` (320²) / `[B,16,13,60,60]` (480²)**.
- Encode trong train: `vae.encode(x)[0].sample()` (`train_dwm_wan.py:2988-2996`); trong pipeline thì masked video dùng `.mode()` (`pipeline_wan_fun_inpaint.py:391-400`), video điều kiện dùng `.sample(generator)` (`pipeline_wan_fun_inpaint_hand_concat.py:135`).

### 2.3 Mask 4 kênh và masked-video latent ✅
Quy ước **pixel** (VideoX-Fun / pipeline): `mask_video` = 1 (hoặc 255) ở chỗ **phải sinh**, 0 ở chỗ **giữ**. Quy ước **latent**: kênh mask = **`1 − mask`** ⇒ 1 = giữ/đã biết.
Đường đi trong pipeline (`pipeline_wan_fun_inpaint_hand_concat.py:328-370`):
1. Nếu `mask_video` toàn 255 ⇒ mask latent = 0 và masked latent = 0 (chế độ t2v) (`:329-333`).
2. `mask_condition = mask_processor.preprocess(...)`, với `do_binarize=True, do_convert_grayscale=True` (`pipeline_wan_fun_inpaint.py:193-195`) ⇒ **mask bị nhị phân hoá ở 0.5**.
3. `masked_video = init_video * (mask < 0.5)` ⇒ vùng phải sinh = **0** (xám trong [-1,1]) (`:344`); encode bằng VAE ⇒ `masked_video_latents [B,16,13,h,w]` (`:345-356`).
4. **Lặp khung đầu 4 lần**: `cat([repeat_interleave(mask[:,:,0:1], 4, dim=2), mask[:,:,1:]], dim=2)` ⇒ 49 → 52 khung (`:358-363`).
5. `view(bs, 52//4=13, 4, H, W).transpose(1,2)` ⇒ `[B, 4, 13, H, W]`: kênh k của latent frame j ↔ pixel frame 4j+k−3 (frame 0 lặp cho j=0) (`:364-365`).
6. `mask_latents = resize_mask(1 − mask_condition, masked_video_latents, True)` — nội suy **trilinear** (khung 0 riêng, các khung còn lại riêng) xuống `[B,4,13,h,w]` (`:366`, `resize_mask` ở `pipeline_wan_fun_inpaint.py:98-132`; bản sao `train_dwm_wan.py:325-360`).
7. `y = cat([mask_latents(4), masked_video_latents(16)], dim=1)` = **20 kênh** (`:486`).

Trong **train DWM** không có bước 2–6: chế độ `dwm` đặt thẳng `mask_latents = ones(B,4,F',h,w)` (toàn bộ "đã biết") rồi `inpaint_latents = cat([mask_latents, static_latents])` (`train_dwm_wan.py:3161-3165`); khi infer truyền `mask_video = zeros` (`inference.py:907-914`, validation `train_dwm_wan.py:1506`) ⇒ sau `1 − 0` cũng ra toàn 1 ⇒ **nhất quán** ✅. Chế độ `i2v`: mask = 0, chỉ latent frame 0 = 1 (`:3245-3250`).
Pretrain Fun-InP (VideoX-Fun `train.py`) ✅: mask ngẫu nhiên nhị phân, `mask_pixel_values = pixels * (1 - mask)` (điền 0), `mask = resize_mask(1 - mask, latents)`; mẫu toàn-mask (t2v) bị **zero cả `inpaint_latents`** với xác suất 0.9; CLIP khung đầu, **zero CLIP 10%**.

### 2.4 Kênh điều kiện thêm (concat) ✅
- Video tay (`videos_hands`) → VAE → `hand_latents [B,16,13,h,w]` (`train_dwm_wan.py:3077-3088`); tuỳ chọn disparity tĩnh/tay cũng qua VAE (`:3090-3122`).
- Thứ tự ghép: `condition_latents = cat([static_disparity?, hand, hand_disparity?], dim=1)` (`:3254-3259`), kiểm tra số kênh khớp `transformer.condition_channels` (`:3261-3273`). DWM config: **16 kênh** (chỉ tay) (`dwm_wan_14b_lora.yaml:13`).
- Trong `forward`: `y = cat([y, condition_latents], dim=1)` (`wan_transformer3d_with_conditions.py:231-251`) rồi lớp cha `x = cat([x, y], dim=0)` theo từng mẫu (`wan_transformer3d.py:1108-1109`).
- **Thứ tự kênh vào `patch_embedding`** (✅): `[noisy x (16) | mask (4) | masked/static latent (16) | condition (16)]` = 36 + 16 = **52**. (Docstring ghi "17" ở `wan_transformer3d_with_conditions.py:45-49` và `pipeline_wan_fun_inpaint_hand_concat.py:74-78` là **sai**, sót từ CogVideoX — số đúng là 4+16=20.)

### 2.5 Text và CLIP token ✅
- Tokenizer `google/umt5-xxl`, `max_length=512`, `padding="max_length"`; T5 encoder `models_t5_umt5-xxl-enc-bf16.pth` (24 lớp, dim 4096, ffn 10240, 64 head, `shared_pos=False`) (`configs/pipelines/wan2.1_fun_inp_hand_concat.yaml:21-33`); embedding cắt về độ dài thật `u[:v]` (`train_dwm_wan.py:3307-3319`) hoặc cache `[L,4096]` rồi pad 0 tới 512 (`dataset.py:576-587`).
- Trong model: pad 0 tới `text_len=512` → `text_embedding` (Linear 4096→dim, GELU, Linear) ⇒ `[B,512,dim]` (`wan_transformer3d.py:1164-1169`). **Không có attention mask cho text** (`context_lens=None`, `:1163`) ⇒ các vị trí padding (embedding của vector 0) vẫn được attend.
- CLIP: `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, ảnh resize bicubic 224, chuẩn hoá mean/std CLIP, lấy đầu ra **31 block đầu** (`use_31_block=True`) ⇒ `[B,257,1280]` (`wan_image_encoder.py:516-531`). `img_emb = MLPProj(1280, dim)` (LN–Linear–GELU–Linear–LN) (`wan_transformer3d.py:785-797, 940-941`) ⇒ `[B,257,dim]`, ghép **trước** text: `context = cat([clip, text])` ⇒ `[B,769,dim]` (`:1171-1173`).
- Cross-attn I2V: `context_img = context[:, :257]`, `context = context[:, 257:]`, hai attention riêng (`k_img/v_img/norm_k_img`) rồi cộng (`wan_transformer3d.py:577-631`).
- **DWM chế độ `dwm` không truyền CLIP khi train** (`clip_context=None` trừ khi `i2v`, `train_dwm_wan.py:3323-3333`) → xem bẫy §5.1.

### 2.6 Timestep, shift, sampler phân tầng ✅
- `noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0, use_dynamic_shifting=False)` (`train_dwm_wan.py:2116-2118`; tham số ở `configs/pipelines/wan2.1_fun_inp_hand_concat.yaml:35-43`, shift ở `:38`).
- `uniform_sampling` **luôn True** (argparse `store_true, default=True`, `args.py:135-136`) ⇒ `indices = DiscreteSampling(1000, uniform_sampling=True)(bsz)` (`train_dwm_wan.py:2980, 3350-3353`), `timesteps = noise_scheduler.timesteps[indices]` (`:3355`).
- `DiscreteSampling` (`utils/discrete_sampler.py:5-52`): 1 GPU ⇒ `randint(0,1000)` đều; nhiều GPU ⇒ chia [0,1000) thành `group_num` khoảng liền kề, mỗi nhóm rank lấy trong khoảng của mình (**phân tầng giữa các GPU**, mỗi bước toàn cục phủ đủ dải nhiễu) (`:12-35, 38-46`).
- ⚠️ (từ mã diffusers, không nằm trong repo): lịch ở `__init__` là `s = linspace(1, 1/1000, 1000)`, `σ = shift·s / (1 + (shift−1)·s)`, `timesteps = σ·1000` ⇒ index đều trên lịch đã shift 5 ⇒ P(σ > 0.5) ≈ 83 %: **huấn luyện thiên mạnh về nhiễu cao**.
- `sigmas = get_sigmas(...)` tra theo timestep (`train_dwm_wan.py:363-373, 3358`).

### 2.7 Flow matching và loss ✅
- `noise = randn(latents.shape, dtype=bf16)` (`:3337`); `noisy = (1 − σ)·latents + σ·noise` (`:3359`); **target = noise − latents** (velocity) (`:3371`; nhánh hand-dropout thay `latents` bằng `static_latents` cho mẫu bị drop, `:3364-3369`, DWM đặt `hand_dropout_prob: 0.0` ⇒ không dùng).
- `seq_len = ceil((h/2)·(w/2)·(F'/1))` (`:3374-3376`) — D6: **13·20·20 = 5 200 token (320²)**, **13·30·30 = 11 700 (480²)**; DWM 480×720: 17 550.
- Forward dưới `autocast(bf16)` (`:3385-3406`). Patch-embed Conv3d k=s=(1,2,2) ⇒ `[B,dim,13,h/2,w/2]` → flatten → `[B,L,dim]` (`wan_transformer3d.py:1112, 1121`), pad tới `seq_len` (`:1139-1142`). Time embedding sinusoidal 256 → MLP → `[B,dim]`, `time_projection` → `[B,6,dim]` tính trong **fp32 autocast** (`:1145-1160`). 30 block (1.3B) → `head` → `[B,L,64]` → unpatchify → `[B,16,13,h,w]` (`:1290-1316`).
- `weighting = compute_loss_weighting_for_sd3("none")` = 1 (`args.py:137-139`, `train_dwm_wan.py:3432`).
- **`custom_mse_loss`**: MSE fp32, **bỏ các phần tử có |pred − target| > 50** rồi mean trên toàn bộ phần tử (kể cả phần tử bị bỏ ở mẫu số) (`train_dwm_wan.py:376-387, 3433`). Giống hệt VideoX-Fun upstream ✅. Với velocity chuẩn hoá (~N(0,2)) ngưỡng 50 gần như không bao giờ kích hoạt ⇒ chỉ là chốt an toàn ⚠️.
- Không có loss vùng, không có loss trên x0 trong DWM.

---

## 3. Mở rộng patch-embedding, LoRA, optimizer, precision ✅

### 3.1 Mở rộng `patch_embedding` (zero-init)
- `WanTransformer3DModelWithConcat.__init__(…, condition_channels=16)` gọi lớp cha rồi `_extend_patch_embedding` (`wan_transformer3d_with_conditions.py:99-133`): tạo `Conv3d(in+cond, dim, k=(1,2,2), s=(1,2,2))`, chép trọng số cũ vào `[:, :in]`, **đặt `[:, in:] = 0`**, giữ bias (`:152-169`), ghi `in_channels/original_in_channels/condition_channels` vào config (`:172-176`).
- Khi `from_pretrained`: model dựng từ config (đã mở rộng), checkpoint 36 kênh được chép vào 36 kênh đầu và **phần còn lại đặt 0 lần nữa** (`wan_transformer3d.py:1671-1674`; nhánh low-mem `_expand_patch_embedding_tensor` `:77-98`). `dict_mapping` ánh xạ `in_dim→in_channels`, `dim→hidden_size` (`configs/pipelines/...yaml:11-14`, `wan_transformer3d.py:1392-1394`).
- Train: `trainable_parameter_patterns: ["patch_embedding"]` (`dwm_wan_14b_lora.yaml:32-33`) ⇒ **toàn bộ** `patch_embedding.weight` (cả 36 kênh cũ) + `bias` được mở khoá (`train_dwm_wan.py:2333-2341`), cùng lr với LoRA (`:2349-2353`).
- Lưu: `non_lora_weights.safetensors` chứa mọi tham số khớp regex (`train_dwm_wan.py:1722-1735`); LoRA ở `lora_diffusion_pytorch_model.safetensors` (`:1718-1720`, ép dtype bf16 khi lưu). Nạp lại khi infer: `merge_lora(multiplier=1.0)` rồi chép non-LoRA vào state dict (`inference.py:484-510`).
- Biến thể diffusers của chính DWM: `WanI2VTransformer3DModelWithConcat` (lớp con của `diffusers.WanTransformer3DModel`) làm y hệt trên `self.patch_embedding` (`wan_transformer3d_i2v_with_conditions.py:77-102`), nạp base rồi mở rộng (`:104-167`), ghép kênh `hidden_states | y | condition_latents` (`:205-208`) ⇒ **bằng chứng DWM đã chạy được đường diffusers**.

### 3.2 LoRA
- Mạng LoRA kiểu kohya/VideoX-Fun (`utils/lora_utils.py`): bọc **mọi `nn.Linear`** (và Conv2d) nằm trong module có tên lớp ∈ {`WanTransformer3DModel`, `WanTransformer3DModelWithConcat`, …} (`:158-166, 204-240`) ⇒ q/k/v/o self-attn, q/k/v/o/`k_img`/`v_img` cross-attn, `ffn.0/ffn.2`, **cả** `text_embedding`, `time_embedding`, `time_projection`, `img_emb.proj`, `head.head`. **Không** bọc `patch_embedding` (Conv3d). `lora_up` init 0, `lora_down` kaiming (`:66-68`); scale = α/r (`:60-63`). LoRA text encoder bị tắt (`apply_to(..., args.train_text_encoder=False, True)`, `train_dwm_wan.py:2329`).
- DWM: **r = 64, α = 64** (`dwm_wan_14b_lora.yaml:30-31`); default args cũng 64/64 (`args.py:49-50`).
- Tham số LoRA được tạo **fp32** (nn.Linear mặc định), transformer gốc **bf16** (`train_dwm_wan.py:2665`); forward LoRA ép input về dtype của `lora_down` (`lora_utils.py:90`) ⚠️ ⇒ LoRA tính fp32 dưới autocast.

### 3.3 Optimizer, LR, precision, checkpoint
- `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)` — **betas/wd viết cứng** (`train_dwm_wan.py:2512-2518`); các khoá `optimizer/beta1/beta2/weight_decay` trong YAML (`dwm_wan_14b_lora.yaml:21-28`) **bị bỏ qua** ✅.
- Nhóm tham số: LoRA transformer lr = lr; patch-embedding lr = lr (`:2343-2353`).
- LR: `cosine_with_restarts`, warmup 100, 1 chu kỳ, `max_train_steps=10000` (cả warmup và tổng đều nhân `num_processes`) (`:2534-2540`, YAML `:16-26`). Batch 1/GPU, grad-accum 1, ví dụ launch 2 GPU (`train_static_hand_concat.sh:11`).
- **Clip grad norm 1.0 chỉ trên `network.parameters()`** — `patch_embedding` **không bị clip** (`train_dwm_wan.py:3443-3445`) ✅.
- `mixed_precision: bf16` (`dwm_wan_14b_lora.yaml:37`); transformer `.to(bf16)` ⇒ **patch_embedding được AdamW cập nhật trực tiếp trên trọng số bf16** (không có master fp32) ✅ (`:2665` + `:2338`).
- `gradient_checkpointing: true` (`:38`) ⇒ checkpoint từng block và head, `use_reentrant=False` (`wan_transformer3d.py:1253-1297`). `allow_tf32: true`. Attention: FlashAttention 3/2 nếu cài, nếu không rơi về `F.scaled_dot_product_attention` (bỏ qua `k_lens`) (`wan_transformer3d.py:240-301`, biến môi trường `VIDEOX_ATTENTION_TYPE` `:256`). DDP `find_unused_parameters=True` (`train_dwm_wan.py:2028`).
- Checkpoint mỗi 1000 bước (`dwm_wan_14b_lora.yaml:45`), giữ full accelerator state cho checkpoint mới nhất (`checkpoint_utils.py:129-157`).
- **VRAM/compute cho nhánh Wan: repo không báo**. Chỉ có "the default 5B training path typically needs an 80 GB-class GPU" (CogVideoX, `README.md:137`) và bài báo: CogVideoX-5B, ~10 ngày trên 4×A100, batch hiệu dụng 56 ✅.

---

## 4. Vòng lặp inference ✅

- Scheduler: `FlowMatchEulerDiscreteScheduler(**scheduler_kwargs)` — cùng config shift 5.0 (`inference.py:440-443`, validation `train_dwm_wan.py:1033-1035`). `set_timesteps(50, mu=1)` (`pipeline_wan_fun_inpaint_hand_concat.py:271-272`); vì `use_dynamic_shifting=False` nên `mu` bị bỏ qua và shift = **5.0** được áp; tham số `shift=3.0` của `__call__` (`:178`) **chỉ có tác dụng với UniPC/DPM** (`:273-282`) ⚠️ (theo mã diffusers).
- 50 bước, **CFG 6.0**, negative prompt **"bad detailed"**, seed 42 (`inference.py:891-894, 1364-1375, 1413-1423`; validation `train_dwm_wan.py:1118, 1516`). CFG kiểu batch đôi: `in_prompt_embeds = negative + prompt` (nối list), `cat([latents]*2)`, `uncond + s·(cond − uncond)` (`pipeline_wan_fun_inpaint_hand_concat.py:265-268, 477-518`). Toàn bộ điều kiện (mask, masked video, hand, CLIP) **giống nhau** ở hai nửa CFG ⇒ CFG chỉ trên văn bản.
- Bước: `latents = scheduler.step(v̂, t, latents)` (`:521`), decode VAE (bf16, clamp) → `[0,1]` (`pipeline_wan_fun_inpaint.py:407-412`).
- Lưu ý: DWM **không huấn luyện dropout prompt** (`prompt_mode=normal`, `prompt_dropout_prob: 0.0`, `train_dwm_wan.py:145-172`) nhưng vẫn dùng CFG 6 khi infer — dựa vào khả năng CFG sẵn có của base.
- `model_fps` (RoPE thời gian) mặc định 16 ⇒ không scale (`inference.py:381-389`, `wan_transformer3d.py:879-883, 931-938`). Đây là **sửa đổi riêng của DWM** (pos_scale = 16/fps); diffusers không có.

---

## 5. Những mảnh cần tái tạo, phụ thuộc, và diffusers 0.42

### 5.1 Tối thiểu phải tái tạo (≈ 300 dòng tự viết nếu đi đường diffusers)
| # | Mảnh | Nguồn DWM | Ghi chú port |
|---|---|---|---|
| 1 | Mở rộng `patch_embedding` zero-init + lưu/nạp non-LoRA | `wan_transformer3d_with_conditions.py:135-178`; bản diffusers `wan_transformer3d_i2v_with_conditions.py:77-167`; lưu `train_dwm_wan.py:1714-1735` | trên `diffusers.WanTransformer3DModel.patch_embedding` |
| 2 | Đóng gói mask 4 kênh/latent frame + `1 − mask` + resize | `pipeline_wan_fun_inpaint_hand_concat.py:335-366`, `resize_mask` `pipeline_wan_fun_inpaint.py:98-132` | §7.2: dùng avg-pool thay trilinear cho V·C mềm |
| 3 | Bước train flow-matching: sampler đều trên lịch shift, `noisy`, `target = ε − z0`, `custom_mse_loss` | `train_dwm_wan.py:3335-3434`, `utils/discrete_sampler.py` | thêm trọng số vùng qua tham số `weighting` |
| 4 | LoRA mọi Linear trong DiT | `utils/lora_utils.py` | thay bằng PEFT `transformer.add_adapter(LoraConfig(...))` (diffusers có `PeftAdapterMixin` ✅) |
| 5 | Vòng sampling inpaint + concat | `pipeline_wan_fun_inpaint_hand_concat.py:466-521` | tự viết ~60 dòng, hoặc tận dụng `WanImageToVideoPipeline` với `condition` tự dựng |
| 6 | Cache latent/T5/CLIP | `encode_with_wan_impl.py` | prompt cố định ⇒ T5 chạy **một lần** |

Không cần: `dist/` (xfuser/USP), TeaCache, cfg-skip, RIFLEx, VACE, multiview, causal, Wan2.2, LightX2V, DeepSpeed/FSDP.

### 5.2 Phụ thuộc của DWM ✅
`requirements.txt`: `torch==2.6.0`, `diffusers>=0.32.2`, `transformers==4.51.3`, `peft>=0.13.0`, `accelerate>=1.2.1`, `decord`, `deepspeed`, `vllm==0.8.5` (chỉ cho caption). Mã model là **bản vendored VideoX-Fun** (không `pip install` VideoX-Fun), flash-attn tuỳ chọn. Đường "i2v diffusers backend" dùng `diffusers.AutoencoderKLWan`, `UMT5EncoderModel`, `CLIPVisionModel` (`train_dwm_wan.py:2121-2141, 2181-2211`).
GReg hiện ghim `diffusers==0.42.1`, `transformers==5.22.1`, `peft==0.21.2` (`requirements-diffusion.txt:4-6`) ⇒ vendoring VideoX-Fun (viết cho transformers 4.51) có rủi ro tương thích; **đi đường diffusers**.

### 5.3 diffusers 0.42.1 có đủ không? ✅ (đọc mã nguồn tag v0.42.1)
- `WanTransformer3DModel(patch_size, num_attention_heads, attention_head_dim, in_channels, out_channels, text_dim, freq_dim, ffn_dim, num_layers, cross_attn_norm, qk_norm, eps, image_dim, added_kv_proj_dim, rope_max_seq_len, pos_embed_seq_len)`; `self.patch_embedding = nn.Conv3d(...)`; `forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None, return_dict, attention_kwargs)`; ảnh ghép **trước** text; processor tách `image_context_length = encoder_hidden_states.shape[1] - 512` ⇒ **text phải pad đúng 512**; `_supports_gradient_checkpointing = True`; kế thừa `PeftAdapterMixin`, `FromOriginalModelMixin`; attention qua `dispatch_attention_fn`.
- `WanImageToVideoPipeline.prepare_latents`: `latent_condition = (latent_condition - latents_mean) * latents_std` (std nghịch đảo), mask = 1 ở khung biết, lặp khung đầu `vae_scale_factor_temporal` lần, `view(..., 4, ...)`, `transpose`, `concat([mask, latent_condition])`; `latent_model_input = cat([latents, condition])` ⇒ **cùng layout và cùng quy ước "1 = biết"** với Fun-InP. CLIP: `hidden_states[-2]` của `CLIPVisionModel` (⚠️ tương đương `use_31_block` của ViT-H 32 lớp). VAE encode dùng `argmax` (= mode). Mặc định CFG 5.0, 50 bước, `max_sequence_length=512`.
- `AutoencoderKLWan` (diffusers) **không chuẩn hoá trong encode** ⇒ phải tự `(z − mean)·(1/std)` khi encode và đảo lại trước decode (dùng `vae.config.latents_mean/latents_std`) ⚠️ (theo cách pipeline diffusers làm).
- **Nạp trọng số 1.3B-InP**: `single_file_utils.infer_diffusers_model_type` chỉ phân biệt theo `patch_embedding.shape[0]==1536` ⇒ gán **`wan-t2v-1.3B`** (in_channels 16, không image_dim) — **sai** cho InP; phải tự chuyển: gọi `convert_wan_transformer_to_diffusers(state_dict)` (đổi tên `self_attn→attn1`, `cross_attn→attn2`, `.q.→.to_q.`, `k_img→add_k_proj`, `v_img→add_v_proj`, `norm_k_img→norm_added_k`, `img_emb.proj.*→condition_embedder.image_embedder.*`, `modulation→scale_shift_table`, `head.head→proj_out`, hoán `norm2/norm3`) rồi `WanTransformer3DModel(**cfg).load_state_dict(strict=True)` với cfg ⚠️:
  `{patch_size:[1,2,2], num_attention_heads:12, attention_head_dim:128, in_channels:36, out_channels:16, text_dim:4096, freq_dim:256, ffn_dim:8960, num_layers:30, cross_attn_norm:true, qk_norm:"rms_norm_across_heads", eps:1e-6, image_dim:1280, added_kv_proj_dim:1536, rope_max_seq_len:1024}`.
  DWM ghi chú: một số checkpoint diffusers 1.3B chứa `added_kv_proj_dim=5120` sai, phải ép = inner_dim (`wan_transformer3d_i2v_with_conditions.py:42-46`) ✅. Có bản chuyển cộng đồng `engineerA314/Wan2.1-Fun-V1.1-1.3B-InP-Diffusers` (tự báo 983 khoá, `in_channels=36`) — chỉ dùng để **đối chiếu**, không tin mù.
- VAE / T5 / CLIP dạng diffusers: VAE + `UMT5EncoderModel` + tokenizer từ `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (mọi Wan2.1 dùng chung VAE — chú thích trong `single_file_utils` ✅), `CLIPVisionModel` + `image_processor` từ `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` ⚠️ (cùng tháp ViT-H của file `.pth`).
- Test bắt buộc sau chuyển đổi ⚠️: cùng input (latent ngẫu nhiên, t, text, CLIP) → so output VideoX-Fun (`wan_transformer3d.py` vendored, chỉ forward, CPU/GPU) với diffusers, `max|Δ| < 1e-2` ở bf16.

### 5.4 Checkpoint HF 1.3B-InP ✅
`config.json`: `{"_class_name":"WanModel","dim":1536,"eps":1e-06,"ffn_dim":8960,"freq_dim":256,"in_dim":36,"model_type":"i2v","num_heads":12,"num_layers":30,"out_dim":16,"text_len":512}` (không `patch_size` ⇒ mặc định (1,2,2)). File: `diffusion_pytorch_model.safetensors` 3.13 GB (≈1.56 B tham số bf16 ⚠️), `Wan2.1_VAE.pth` 0.51 GB, `models_t5_umt5-xxl-enc-bf16.pth` 11.36 GB, `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` 4.77 GB, tokenizer `google/umt5-xxl/`, `xlm-roberta-large/`. Giấy phép Apache-2.0. Model card: "trained at multiple resolutions, supports start-end image prediction"; Control-variants "81 frames at 16 FPS"; không nêu shift/CFG khuyến nghị ✅.

---

## 6. Bẫy làm hỏng huấn luyện một cách im lặng

1. **Dấu mask** ✅: pixel `mask_video` = 1 ⇔ *phải sinh*; latent = `1 − mask` ⇔ 1 = *biết*. `V·C` của ta đã là "1 = biết" ⇒ **kênh mask latent = V·C (đóng gói), KHÔNG phải 1 − V·C**; masked video = `I_st · 1[V·C ≥ 0.5]` (vùng phải sinh = 0 trong [-1,1]). Đặt nhầm dấu: model vẫn hội tụ "đẹp" trên val dễ nhưng dạy ngược nghĩa của pretrain → kiểm tra bằng zero-shot (ablation 0) phải tái tạo đúng vùng biết.
2. **Chuẩn hoá latent** ✅/⚠️: VAE VideoX-Fun tự chuẩn hoá; VAE diffusers thì không. Trộn hai thứ (vd. cache latent bằng một VAE, decode bằng VAE kia) ⇒ lệch thống kê 16 kênh, model vẫn train nhưng ảnh xỉn/sai màu. Kiểm tra: latent chuẩn hoá phải có mean≈0, std≈1 theo kênh trên vài clip.
3. **F = 4n+1** ✅: encoder cắt theo cụm 1+4k (`wan_vae.py:528`); 50 khung ⇒ khung cuối bị bỏ lặng lẽ, và `view(bs, F//4, 4, …)` của mask sẽ lệch/hỏng (`pipeline_wan_fun_inpaint_hand_concat.py:364`). Khẳng định `F % 4 == 1` ở dataloader.
4. **CLIP None vs CLIP 0** ✅ code / ⚠️ hệ quả: §0.5. Ở diffusers, nếu `image_dim` có mà không truyền `encoder_hidden_states_image` thì `image_context_length = L_text − 512 = 0` ⇒ nhánh ảnh attention trên 0 token (⚠️ có thể NaN/hành vi không định nghĩa). ⇒ Luôn truyền `[B,257,1280]`, cùng loại ở train & infer.
5. **Text padding đúng 512** ✅ (diffusers cắt cứng `-512`); prompt cố định ⇒ cache một tensor `[512,4096]` (pad 0).
6. **dtype VAE** ✅/⚠️: DWM encode và decode VAE bằng **bf16** (`encode_with_wan_impl.py:117,348`, `inference.py:306`). Encode bf16 chấp nhận được; decode để đo metric nên dùng **fp32** (diffusers khuyên VAE fp32 cho Wan) ⚠️.
7. **umT5-XXL** ✅: file encoder 11.4 GB bf16 (≈5.7 B tham số ⚠️) — không bao giờ để trên GPU lúc train; với prompt cố định chỉ chạy một lần (CPU cũng được).
8. **patch_embedding bf16 + AdamW** ✅: DWM cập nhật thẳng trọng số bf16 và không clip grad của nó ⇒ cập nhật nhỏ (lr 1e-4 × ~1) có thể bị làm tròn mất ⚠️. Port: giữ `patch_embedding` (ít nhất phần kênh mới) ở **fp32** và đưa vào clip-grad.
9. **betas/wd trong YAML bị bỏ qua** ✅ (`train_dwm_wan.py:2515-2516`) — đừng "tái lập" DWM bằng beta2 = 0.95.
10. **RandomHorizontalFlip áp theo từng khung** ✅ (`dataset.py:155-157` + `:968`): nếu bật `random_flip` thì các khung trong một clip lật ngẫu nhiên độc lập. DWM tắt; ta: **không lật** (ngôn ngữ ký hiệu có tay thuận).
11. **Hằng số 257 và 512** ✅: VideoX-Fun cắt `[:257]`, diffusers cắt `-512`; đổi CLIP (số token khác) hay text_len là vỡ.
12. **RoPE thời gian / fps** ✅: DWM có `fps` (pos_scale=16/fps), diffusers không. 25 fps với 49 khung ≈ 1.96 s (base quen 16 fps). Không scale RoPE; xét stride 2 (12.5 fps) như ablation ⚠️.
13. **Mask mềm** ✅/⚠️: Fun-InP pretrain với mask nhị phân và pipeline nhị phân hoá ở 0.5 ⇒ đưa V·C mềm thẳng vào 4 kênh mask là ngoài phân phối (LoRA sẽ phải học lại). Đề xuất §7.2.
14. **`resize_mask` trilinear** ✅/⚠️: hạ 8× bằng trilinear `align_corners=False` chỉ lấy mẫu vài pixel ⇒ ngón tay mảnh bị răng cưa/mất. Dùng `avg_pool` (area) theo không gian.
15. **`.sample()` cố định khi cache** ✅: DWM lưu một mẫu duy nhất ⇒ mất augmentation nhiễu VAE; cache **moments** (μ, logσ²) và lấy mẫu mỗi bước (kế hoạch đã ghi) ⚠️ nhớ chia std nhiễu cho std kênh nếu chuẩn hoá thủ công.
16. **Shift 5 cho train** ✅: 83 % mẫu ở σ>0.5. Với điều kiện rất mạnh (I_st đã gần đúng) phần lớn tín hiệu học nằm ở σ thấp–trung bình ⚠️ ⇒ cân nhắc shift 3 hoặc logit-normal làm ablation phụ; giữ shift 5 làm mặc định để trùng DWM/VideoX-Fun.
17. **Không dropout điều kiện** ✅: DWM không drop prompt/điều kiện ⇒ không có CFG theo điều kiện; prompt cố định ⇒ CFG văn bản gần như vô nghĩa ⚠️. Thử CFG ∈ {1, 3, 6} trên val; CFG=1 giảm nửa thời gian infer.

---

## 7. Kế hoạch port cụ thể cho D6 (1.3B)

### 7.1 Layout kênh đề xuất
| Khối | Kênh | Nội dung | Khởi tạo |
|---|---|---|---|
| noisy `z_t` | 16 | latent đích | pretrain |
| mask | 4 | `pack(M)` với `M = 1[V·C ≥ 0.5]`, khung 0 = 1 toàn ảnh | pretrain |
| masked video | 16 | `VAE(I_st · M)`, khung 0 = khung nguồn thật | pretrain |
| `N_t` | 16 | `VAE(normal RGB [-1,1])`, nền = −1 (như video tay DWM) | **0** |
| CCM | 16 | `VAE(CCM RGB [-1,1])`, nền = −1 | **0** |
| (tuỳ chọn) V·C mềm | 4 | `pack(V·C)` avg-pool | **0** |

⇒ `in_channels = 36 + 16 + 16 (+4) = 68 (72)`. Thứ tự giữ y như DWM: `[z_t | mask | masked | N_t | CCM | V·C]`. Ablation 2 "mask toàn 1" = đúng chế độ `dwm` của DWM (`train_dwm_wan.py:3161`).
Khung 0: nếu cửa sổ 49 khung bắt đầu bằng khung nguồn ⇒ latent frame 0 = khung nguồn chính xác (VAE nhân quả) và mask = 1 — y như i2v Fun-InP. Nếu khung nguồn nằm ngoài cửa sổ ⇒ **chèn** khung nguồn làm khung 0 của *cả* đích và điều kiện (cửa sổ = nguồn + 48 khung đích) và loại latent frame 0 khỏi loss ⚠️.
CLIP: `CLIPVisionModel(source_frame).hidden_states[-2]` ⇒ `[257,1280]`, cache theo clip (≈0.66 MB bf16).

### 7.2 Bước train (pseudo, shape cho 320²; 480² thay 40→60, 5 200→11 700 token)
```
z0   = norm(VAE.enc(x[-1,1] : [B,3,49,320,320]))      -> [B,16,13,40,40]   (cache μ,logσ²; sample mỗi bước)
mlat = norm(VAE.enc(I_st*M)); nlat = norm(VAE.enc(N_t)); clat = norm(VAE.enc(CCM))   -> mỗi cái [B,16,13,40,40]
Mp   = cat([M[:,:,:1].repeat(1,1,4,1,1), M[:,:,1:]],2)  : [B,1,52,320,320]
Mp   = Mp.view(B,13,4,320,320).transpose(1,2)           : [B,4,13,320,320]
Mlat = avg_pool3d(Mp, (1,8,8))                           : [B,4,13,40,40]    (giá trị 1 = biết)
idx ~ U{0..999}; t = sched.timesteps[idx]; σ = sched.sigmas[idx]   (shift 5; nhiều GPU: phân tầng như DiscreteSampling)
z_t  = (1-σ) z0 + σ ε ;   v = ε - z0
inp  = cat([z_t, Mlat, mlat, nlat, clat], 1)             : [B,68,13,40,40]
v̂    = DiT(inp, t, text[B,512,4096], clip[B,257,1280])   : [B,16,13,40,40]
W    = 1 + λ_h·pack(hand) + λ_f·pack(face)  (avg-pool cùng kiểu Mp, [B,1→16,13,40,40])
L    = mean(W · (v̂-v)² · 1[|v̂-v|≤50])
x̂0   = z_t - σ v̂   (nếu cần loss vùng trên x0_hat / decode một phần)
```
- LoRA: PEFT `r=32, α=32`, `target_modules` = các Linear trong block (`attn1.to_q/k/v/out.0`, `attn2.to_q/k/v/out.0`, `attn2.add_k_proj/add_v_proj`, `ffn.net.0.proj`, `ffn.net.2`); có thể thêm `condition_embedder.*` và `proj_out` để trùng DWM. ≈ 50 M tham số ở r32 ⚠️ (1.66 M/block × 30).
- Trainable khác: `patch_embedding` (fp32, lr 1e-4, clip grad cùng LoRA).
- Optimizer: AdamW(0.9, 0.999, wd 0.01, eps 1e-8), lr 1e-4, warmup 100, cosine; bf16 autocast; gradient checkpointing; TF32.
- Infer: `FlowMatchEulerDiscreteScheduler(shift=5)`, 50 bước (thử 25), CFG 1/3/6, negative "bad detailed"; VAE decode fp32.

### 7.3 Ước lượng VRAM (batch 1, latent/T5/CLIP đã cache, ckpt gradient) ⚠️
| Thành phần | 320²×49 (L=5 200) | 480²×49 (L=11 700) |
|---|---|---|
| DiT bf16 frozen (1.56 B) | 3.1 GB | 3.1 GB |
| LoRA r32 fp32 + grad + AdamW (≈50 M × 16 B) + patch-embed | 0.8 GB | 0.8 GB |
| Đầu vào block lưu cho checkpoint (30 × L × 1536 × 2 B) | 0.5 GB | 1.1 GB |
| Đỉnh recompute 1 block (q/k/v, RoPE fp32, FFN 8960, grad) | 0.6–1.2 GB | 1.5–3 GB |
| CUDA context + phân mảnh | 1.5–2.5 GB | 1.5–2.5 GB |
| **Tổng** | **≈ 6–9 GB** | **≈ 9–13 GB** |
| + mỗi mẫu batch thêm | +1–1.5 GB | +2.5–4 GB |
| Nếu encode VAE online (bf16) | +3–6 GB | +6–12 GB |
| Nếu T5 trên GPU | +11.4 GB | +11.4 GB |
Không ckpt gradient: ≈ +15–20 GB ở 320², > 40 GB ở 480² ⇒ luôn bật.
⇒ 4090 24 GB: 320² batch 2–4 hoặc 480² batch 1–2; A100 80 GB: 480² batch 6–12 (hoặc batch 4 + grad-accum).

### 7.4 Ước lượng thời gian ⚠️
FLOPs forward ≈ 2·1.53 B·L + 4·L²·1536·30 (+ cross-attn 769 token): 320² ≈ 22 TFLOP, 480² ≈ 63 TFLOP; một bước train (fwd + recompute + bwd, trọng số gốc frozen) ≈ 3.6× fwd.
| | 4090 (≈60–70 TFLOPS hiệu dụng) | A100 80 GB (≈120–140 TFLOPS hiệu dụng) |
|---|---|---|
| Train 320²×49, B=1 | 1.5–2.5 s/bước | 0.7–1.2 s/bước |
| Train 480²×49, B=1 | 4–6 s/bước | 2–3 s/bước |
| Infer 480²×49, 50 bước, CFG 6 | ~2–3 phút/clip | 60–80 s/clip (CFG 1: 30–40 s) |
Ví dụ ngân sách: 10 k bước × batch 4 ở 480² trên A100 ≈ 10 k × (4 × 2.5 s) ≈ 28 h (≈ 40–50 USD ở 1.5 USD/h) ⚠️ — khớp khung 60–120 h của kế hoạch D6 cho nhiều ablation.

### 7.5 Thứ tự làm
1. Chuyển trọng số 1.3B-InP → diffusers, test cân bằng output với forward VideoX-Fun vendored (§5.3). Kiểm tra `in_channels=36`, `image_dim=1280`, 983 khoá.
2. Script cache: latent (μ, logσ²) của đích, `I_st·M`, `N_t`, CCM; `M` và hand/face mask đóng gói sẵn `[4,13,h,w]`; CLIP khung nguồn; 1 tensor T5 cho prompt cố định. Khẳng định `F%4==1`, mean/std latent ≈ 0/1.
3. **Ablation 0 (zero-shot)** trên 4090 ở 320²: pipeline Fun-InP thuần (36 kênh) với `I_st`, `M`, CLIP nguồn — kiểm chứng dấu mask (vùng `M=1` phải tái tạo gần đúng `I_st`).
4. Mở rộng patch-embed (zero-init) → kiểm tra output **bit-identical** với base khi kênh mới = bất kỳ (vì trọng số 0).
5. Smoke train 320² 200–500 bước (loss giảm, không NaN), rồi chuyển A100 480² cho ablation 1–4, 9.
