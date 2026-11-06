# Báo cáo D6: Video DiT "warp rồi diffuse" (Wan2.1-Fun-1.3B-InP + LoRA + kênh điều kiện từ mesh)

Nhánh: `feat/diff-d6-video-dit` (@`375fc83`). Bản thực sự chạy trên máy vast 58802139 là nhánh này cộng hai bản sửa của D5 (`fix-d6-tmp`: `2da9e2d`, `acb2144`). Cả hai không đổi mô hình D6; cần chúng vì `32_cache_wan_windows.py` import `draw_skeleton` từ `greg/diffusion/frame_ldm.py`.
Ngày chạy: 2026-11-04 17:19 UTC → 2026-11-05 09:04 UTC.

Quy ước: ✅ đã kiểm chứng (mã nguồn `path:dòng`, hoặc số đo đọc từ `summary.json` / `bootstrap.md` trên máy, đã báo trong hội thoại); ⚠️ suy luận hoặc ước lượng, chưa đo.
Mọi đường dẫn mã dưới đây là trên nhánh `feat/diff-d6-video-dit` (đọc bằng `git show feat/diff-d6-video-dit:<path>`).

---

## 0. Tóm tắt

- **Làm gì:** thay hẳn bộ sinh GReg bằng một mô hình video diffusion có sẵn (Wan2.1-Fun-V1.1-1.3B-InP).
  - Mô hình nhận một video 49 khung. Ở những pixel mà mesh SMPL-X cho biết nhìn thấy được trong ảnh nguồn, video là ảnh nguồn được warp theo tương ứng mesh; phần còn lại để trống. Kèm theo là video normal map và video CCM của tư thế đích.
  - Mô hình học cách tô phần còn trống. Fine-tune bằng LoRA r32 và các kênh vào khởi tạo bằng 0, theo công thức DWM.
- **So sánh có kiểm soát:** bốn biến thể cùng backbone, cùng số bước train, chỉ đổi điều kiện: `ng` (warp + mặt nạ V·C + normal + CCM), `ng_ones` (như `ng` nhưng mặt nạ toàn 1, kiểu DWM), `n` (chỉ khung nguồn + normal), `k` (chỉ khung nguồn + skeleton 2D).
- **Kết quả** (200 clip test × 16 khung, ghép cặp với GReg run5):
  - Tay tốt nhất trong toàn chương trình: hand LPIPS **0.236 → 0.184 (−22 %)**, hand L1 **0.113 → 0.091 (−19 %)**.
  - Toàn khung kém hơn GReg: LPIPS **0.0476 → 0.0527 (+11 %)**, face L1 **+20 %**, trôi màu **+34 %**, PSNR −1.4 dB.
  - Thứ tự toàn khung: `ng` > `ng_ones` > `n` > `k`. Ở tay, bốn biến thể gần như bằng nhau (0.184–0.187).
- **Kết luận:**
  - D6 một mình **không** thay được GReg.
  - D6 là nguồn tay tốt nhất ta có, và chính điều này sinh ra phương pháp ghép "GReg + tay D6" (báo cáo riêng `GReg_plus_D6_hands.md`). Phương pháp đó tốt hơn GReg ở mọi chỉ số.
  - Điều kiện mesh (warp + V·C) giúp giữ danh tính và màu. Nó **không** phải nguồn gốc của cải thiện ở tay.

---

## 1. Bối cảnh

- **GReg run5 là mốc so sánh.** Kiến trúc là TPSMM warp + generator GAN + Regional Expert, điều kiện là normal map SMPL-X. Trên toàn bộ tập test: L1 0.01728, LPIPS 0.04851 ✅.
- **Điểm yếu đã đo được của GReg:**
  - Tay: hand L1 0.120, hand LPIPS 0.262 trên toàn bộ test ✅.
  - Vùng mà ảnh nguồn không thấy: ở chuyển động lớn, 39 % pixel tay và cánh tay phải "vẽ từ không" (`docs/diffusion_6_huong_ke_hoach.md:37`) ✅.
- **Vị trí của D6 trong chương trình D1–D6:** D6 là hướng "thay bộ sinh" ở dạng video.
  - Bản theo từng khung là D5 (SD1.5 + ControlNet). D5 thất bại nặng: LPIPS 0.383 trên cùng 200 clip, mô hình bỏ qua tư thế điều kiện ✅.
  - D6 kiểm tra xem một backbone video lớn, có prior chuyển động và có nhánh inpainting sẵn, có cứu được hướng này không.
- **Thay đổi so với kế hoạch** (`docs/diffusion_6_huong_ke_hoach.md:290-298`):
  - Kế hoạch dùng A100 80 GB ở 480² và 60–120 GPU-giờ.
  - Thực tế chạy trên RTX 4090 24 GB ở 320² × 49 khung, 3 000 bước mỗi biến thể. Lý do: ngân sách còn khoảng $8 lúc bắt đầu D6 ✅.
  - Các ablation 5–11 trong kế hoạch (TPSMM warp thay mesh warp, 480², …) **không chạy**.

---

## 2. Ý tưởng và điểm đặc sắc

### 2.1 Ý tưởng: "warp rồi diffuse" có mặt nạ tin cậy từ mesh

Ta biết chính xác pixel nào của khung đích có thể chép từ ảnh nguồn, và chép từ đâu:
- Tương ứng mesh SMPL-X (`MeshCorrespondence`) cho lưới warp `I_s → I_t`.
- Nó cũng cho `V·C`: phần nhìn thấy trong nguồn nhân độ tin cậy.

Vì vậy bài toán sinh được viết lại thành **inpainting video**:
- Vùng `V·C ≥ 0.5`: mô hình nhận pixel warp, gần như chỉ cần "làm sạch".
- Vùng còn lại: tay đổi hình dạng, vùng bị che, mặt sau cánh tay. Mô hình phải tự vẽ, dựa vào prior video, normal map và CCM của đích.

Mô hình Wan2.1-Fun-**InP** được pretrain đúng cho kiểu đầu vào "mặt nạ + video bị che", nên chỉ cần đổi nghĩa của mặt nạ.

### 2.2 Những gì mượn

| Thành phần | Nguồn |
|---|---|
| Backbone video DiT 1.3B, VAE 3D, umT5, CLIP ViT-H | Wan2.1 / Wan2.1-Fun-V1.1-1.3B-InP (Apache-2.0) |
| Ghép kênh điều kiện vào `patch_embedding` với cột mới khởi tạo bằng 0; LoRA; lr 1e-4; flow matching shift 5; MSE bỏ phần tử có \|lỗi\| > 50 | DWM (Dexterous World Models), nhánh Wan, commit `19de79b`, đọc từng dòng ở `docs/code_notes_dwm_wan.md` |
| Layout mặt nạ 4 kênh / latent frame (khung 0 lặp 4 lần) | Wan-Fun / DWM (`docs/code_notes_dwm_wan.md:56-67`) |
| Trọng số loss ×10 ở vùng tay và mặt | SignRefine Eq. 4 (`docs/diffusion_6_huong_ke_hoach.md:258`) |

### 2.3 Những gì riêng của luận văn 🛠

1. **Mặt nạ "đã biết" lấy từ mesh.**
   - DWM huấn luyện với mặt nạ **toàn 1**, coi cả video tĩnh là "đã biết" (`docs/code_notes_dwm_wan.md:67`).
   - D6 `ng` dùng mặt nạ `1[V·C ≥ 0.5]` thật, theo từng khung (`scripts/32_cache_wan_windows.py:129`). Mô hình vì thế biết chính xác chỗ nào phải vẽ.
   - Biến thể `ng_ones` tái lập đúng chế độ DWM để đo riêng đóng góp của mặt nạ này.
2. **Hai kênh hình học mới**, mỗi kênh 16 latent channel, khởi tạo bằng 0:
   - Video normal map SMPL-X của đích: chính điều kiện của GReg.
   - Video **CCM**: toạ độ T-pose của từng pixel đích, tô thành RGB (`scripts/32_cache_wan_windows.py:131`). Kênh này cho mô hình biết "pixel này thuộc điểm nào trên cơ thể".
3. **So sánh có kiểm soát giữa bốn loại điều kiện** trên cùng backbone, cùng số bước, cùng dữ liệu (`greg/diffusion/wan_video.py:37`, `:166-176`):
   - mesh đầy đủ (`ng`)
   - mesh kiểu DWM (`ng_ones`)
   - chỉ normal (`n`)
   - chỉ keypoint 2D (`k`), kiểu SignRefine. Keypoint lấy từ **cùng** fit SMPL-X chiếu xuống 2D, nên chỉ cách biểu diễn khác nhau.
4. **Sửa ba điểm sai hoặc yếu của DWM khi port:**
   - **CLIP lệch giữa train và infer.** DWM train với CLIP = None nhưng infer với CLIP toàn 0. Với `model_type=i2v`, điều này khiến 257 token văn bản đầu chui vào nhánh ảnh lúc train (`docs/code_notes_dwm_wan.md:28`).
     - D6 luôn dùng CLIP `hidden_states[-2]` của khung nguồn thật, cả lúc train và infer (`greg/diffusion/wan_video.py:15-17`; `scripts/32_cache_wan_windows.py:150-151`).
   - **Hạ mẫu mặt nạ.** DWM resize mặt nạ bằng trilinear, gây aliasing ở ngón tay. D6 dùng **average pooling** (`greg/diffusion/wan_video.py:11-12`, `:70-80`).
   - **`patch_embedding`.** DWM cập nhật nó trực tiếp ở bf16 và không clip gradient. D6 giữ nó ở **fp32** và clip gradient cùng LoRA (`greg/diffusion/wan_video.py:6-8`, `:56-67`; `scripts/33_train_wan.py:127`).
   - Ngoài ra: VAE của diffusers không chuẩn hoá latent. D6 tự chuẩn hoá theo `latents_mean/std` (`greg/diffusion/wan_video.py:83-107`). Thiếu bước này thì latent lệch phân phối pretrain.
5. **Chuyển trọng số tự viết.** Bộ nạp single-file của diffusers nhận nhầm checkpoint 1.3B-InP thành t2v-1.3B (16 kênh, không có nhánh ảnh). D6 tự dựng `WanTransformer3DModel` với cấu hình 36 kênh và nạp strict, báo lỗi nếu thiếu hoặc thừa khoá (`greg/diffusion/wan_video.py:30-53`).

---

## 3. Kiến trúc chi tiết

### 3.1 Sơ đồ một cửa sổ 49 khung (320², biến thể `ng`)

```
Cache 256² (ảnh, normal, faces, part map, SMPL-X) của 1 clip
 │  cửa sổ = [khung nguồn s0] + 48 khung đích liên tiếp  → T = 49 = 4·12 + 1      (32_cache:10-11, 68-84)
 │  mọi video phóng bicubic 256² → 320² (work size, bội của 16)                    (32_cache:21, 87-90, 134)
 ▼
MeshCorrespondence(s0 → t) ── grid ──► I_st = warp(I_s0, grid)      (49, 3, 320, 320)   (32_cache:125-128)
                          └─ occlusion ─► M = 1[V·C ≥ 0.5]          (49, 1, 320, 320)   (32_cache:129)
khung 0: M = 1, I_st = I_s0 thật                                                         (32_cache:130)
CCM  = canonical_coordinate_map(faces_t, bary_t)  (49, 3, ...)                           (32_cache:131)
skel = draw_skeleton(joints_2d)  (chỉ dùng cho biến thể k)                               (32_cache:132)
region = part_mask(parts, {mặt, tay T, tay P}, giãn 3 px)                                (32_cache:133)

──────────── VAE 3D Wan2.1 (đóng băng; nén thời gian ×4, không gian ×8; 16 kênh) ────────────
 video thật I_t            → (lat_mean, lat_std)  (16, 13, 40, 40)   posterior, lấy mẫu lại mỗi bước train
 I_st · M  ∈ [-1,1]        → lat_masked           (16, 13, 40, 40)
 [I_s0, 0, 0, …]           → lat_i2v              (16, 13, 40, 40)   (chỉ cho n, k)
 normal / CCM / skel video → lat_normal / lat_ccm / lat_skel (16, 13, 40, 40) mỗi cái
 M        → pack_frames: khung 0 lặp ×4 → (4, 13, 320, 320) → avg-pool ×8 → mask (4, 13, 40, 40)
 region   → pack_frames + amax 4 kênh → (1, 13, 40, 40);  valid (1, 13, 40, 40) (khung đệm = 0)
 CLIP ViT-H (ảnh nguồn) hidden_states[-2] → clip (257, 1280)
 umT5 (prompt cố định, cắt theo độ dài thật, pad 0 tới 512) → text (512, 4096)
                                                                     (32_cache:13-19, 52-65, 136-151)
───────────────────────────────── Transformer ─────────────────────────────────
 z_t (16) ‖ mask (4) ‖ lat_masked (16) ‖ lat_normal (16) ‖ lat_ccm (16)  = 68 kênh × (13, 40, 40)
          │  patch_embedding Conv3d k = s = (1, 2, 2): 36 cột gốc + 32 cột mới = 0 (fp32, train toàn bộ)
          ▼
 5 200 token (13 × 20 × 20) × dim 1536
          │  30 khối DiT (12 head × 128), RoPE 3D, self-attn toàn cửa sổ
          │  cross-attn: 257 token CLIP (nhánh add_k/add_v) + 512 token văn bản
          │  LoRA r = 32, α = 32 trên q, k, v, o, add_k_proj, add_v_proj, ffn.0, ffn.2
          ▼
 v̂ (16, 13, 40, 40)   ≈ ε − z0  (velocity của flow matching)
───────────────────────────────── Lấy mẫu ─────────────────────────────────
 z ~ N(0, I) cố định theo clip → 25 bước Euler (FlowMatchEuler, shift 5, không CFG)
 → VAE decode (fp32) → (3, 49, 320, 320) → bỏ khung 0 và khung đệm → area ↓ 256² → PNG
                                                                     (34_sample:69-93; wan_video:198-207)
```

Kích thước latent và token: 320/8 = 40, 49 khung → 13 latent frame, patch (1, 2, 2) → 13·20·20 = 5 200 token (`docs/code_notes_dwm_wan.md:53`, `:92`) ✅.

### 3.2 Bốn biến thể

Hàm ghép kênh: `greg/diffusion/wan_video.py:166-176`. Bảng kênh thêm: `:37`.

| Biến thể | Kênh "mặt nạ" (4) | Kênh "video bị che" (16) | Kênh thêm | Số kênh vào | Ý nghĩa |
|---|---|---|---|---|---|
| `ng` | `1[V·C ≥ 0.5]` thật, từng khung | VAE(I_st · M) | normal + CCM (32) | 68 | mesh đầy đủ: pixel chép được + chỗ phải vẽ + hình học đích |
| `ng_ones` | toàn 1 | VAE(I_st · M) | normal + CCM (32) | 68 | chế độ DWM: warp là điều kiện tĩnh, mô hình không được báo chỗ nào là lỗ |
| `n` | 1 ở khung 0, 0 ở các khung khác (i2v) | VAE([I_s0, 0, …]) | normal (16) | 52 | chỉ ảnh nguồn + normal map (điều kiện của GReg) |
| `k` | như `n` | như `n` | skeleton 2D (16) | 52 | chỉ ảnh nguồn + keypoint 2D (kiểu SignRefine), keypoint lấy từ cùng fit SMPL-X |
| `base` (zero-shot) | `1[V·C ≥ 0.5]` | VAE(I_st · M) | không | 36 | Fun-InP gốc, không LoRA (`34_sample:42`, `:57`) |

### 3.3 Phần train / đóng băng, số tham số

- **Đóng băng:** toàn bộ transformer gốc (bf16), VAE, umT5, CLIP (`greg/diffusion/wan_video.py:137-138`, `:87-89`). Latent, văn bản và CLIP được **cache trước**, nên khi train chỉ transformer nằm trên GPU (`scripts/32_cache_wan_windows.py`).
- **Train:**
  - LoRA r32 trên 8 loại Linear của cả 30 khối (`greg/diffusion/wan_video.py:34`, `:139-142`).
  - **Toàn bộ** `patch_embedding` ở fp32: 36 cột gốc và các cột mới (`:143-148`).
- **Số tham số:**
  - Transformer gốc khoảng 1.3–1.56 B. ⚠️ Con số 1.56 B ước từ kích thước file 3.13 GB ở bf16 (`docs/code_notes_dwm_wan.md`, §5.4). Log in số chính xác không còn lưu.
  - Phần train khoảng **50 M** ⚠️, tính tay:
    - LoRA mỗi khối: 4 × r·(1536+1536) cho self-attn, 6 × r·(1536+1536) cho cross-attn kể cả `add_k/add_v`, 2 × r·(1536+8960) cho FFN. Cộng lại khoảng 1.65 M, nhân 30 khối ≈ 49.6 M.
    - `patch_embedding`: 1536 × 68 × 4 ≈ 0.42 M (`ng`).
    - Log `trainable …M` của `33_train_wan.py:85` không còn lưu.

### 3.4 Loss và lịch huấn luyện

- **Flow matching:**
  - `z_t = (1 − σ)·z0 + σ·ε`, mục tiêu `ε − z0`.
  - `z0` được **lấy mẫu lại từ posterior đã cache ở mỗi bước** (`lat_mean + lat_std·N`).
  - Chỉ số timestep đều trong [0, 1000) trên lịch shift 5. Lịch này thiên mạnh về nhiễu cao: P(σ > 0.5) ≈ 83 % ⚠️ (`docs/code_notes_dwm_wan.md:87`).
  - Mã: `greg/diffusion/wan_video.py:184-195`.
- **Loss theo vùng:**
  - `filtered_mse(v̂, ε − z0, w)` với `w = (1 + 9·region)·valid` (`:115-124`, `:194`).
  - Ô latent phủ tay hoặc mặt nặng tới ×10. Khung đệm cuối clip không tính.
- **Siêu tham số** (`configs/diffusion/d6_wan.yaml:9-26`; `scripts/33_train_wan.py:87-89`, `:120-128`):

| Tham số | Giá trị |
|---|---|
| Số bước | 3 000 |
| Batch | 2 cửa sổ × 49 khung |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), wd 0.01 |
| Lịch lr | warmup 100 bước rồi cosine về 0 |
| Clip gradient | 1.0 |
| Precision | bf16 autocast, gradient checkpointing |
| Seed | 0 |

- **Dữ liệu train:**
  - **Một** cửa sổ 49 khung cố định cho mỗi clip của split `train_fit`, điểm bắt đầu chọn ngẫu nhiên theo seed và clip (`scripts/32_cache_wan_windows.py:68-84`; `scripts/vast/stages_d6.sh:56-57`).
  - 3 000 bước × 2 = 6 000 cửa sổ, tức khoảng 1.5 lượt qua tập cửa sổ train ⚠️. Số cửa sổ train chính xác không còn lưu.
- **Lấy mẫu** (`scripts/34_sample_wan.py:69-93`):
  - 25 bước Euler, **không CFG** (guidance 1).
  - Nhiễu khởi tạo theo `manual_seed(clip_id)`: **cùng một nhiễu cho mọi cửa sổ của một clip**.
  - Cửa sổ test **không chồng lấn**: mỗi cửa sổ là `[nguồn] + 48 khung đích kế tiếp` (`32_cache:73-74`). Các cửa sổ được sinh **độc lập** rồi nối lại; không có cơ chế làm mượt chỗ nối giữa hai cửa sổ.

### 3.5 Chi phí đo được (RTX 4090) ✅

| Hạng mục | Thời gian |
|---|---|
| Tải trọng số | 6 phút (17:19–17:25) |
| Cache văn bản | 1 phút |
| Cache cửa sổ test (200 clip, mọi cửa sổ) | 26 phút |
| Zero-shot (60 cửa sổ) + chấm | 7 phút |
| Cache cửa sổ train | **3 h 48 phút** (17:59–21:47) |
| Smoke 20 bước | 1 phút |
| Train mỗi biến thể | 3 000 bước × **1.78–1.79 s/bước** = **89.4 phút**, VRAM đỉnh **5.6 GB** (batch 2) |
| Lấy mẫu mỗi biến thể | 17 113 khung × **0.186 s/khung** (25 bước) ≈ 53 phút |
| Chấm mỗi biến thể | khoảng 1 phút |
| **Tổng D6** | khoảng 14.5 GPU-giờ, khoảng **$9** ở $0.62/h (kể cả khoảng 75 phút GPU rỗi do lỗi `--index`, §8) |

So sánh tốc độ suy luận: GReg khoảng 0.02 s/khung (`docs/diffusion_6_huong_ke_hoach.md:323`), nên D6 chậm hơn **khoảng 9 lần**.

---

## 4. Giao thức đánh giá

- **Tập s200:** 200 clip test, 16 hàng mỗi clip, tổng **3 200 khung** ✅ (log: "3200 rows in 200 clips").
  - Dùng **đúng các hàng mà D5 đã sinh** (`index.json` của D5, `scripts/vast/stages_d6.sh:10`, `:31-35`).
  - Khung nguồn là khung hợp lệ đầu tiên của clip, đúng giao thức tái tạo của GReg.
- **Mốc GReg:** GReg run5 (epoch 100) chấm trên đúng 3 200 khung đó (`d5/greg_run5_eval_test`).
- **Bộ chấm:** `scripts/21_eval_frames.py` (bản trên nhánh D1, có `--index`), ở 256².
  - Chỉ số: L1, PSNR, SSIM, LPIPS (AlexNet), region L1 (ô 64×64), hand L1, hand LPIPS (crop tay vừa khớp khớp tay, phóng lên), face L1, `lowfreq_drift` (L1 sau Gaussian σ = 8 px, **ngoài** tay), `arm_l1`, `disocc_l1` (pixel thân mà nguồn không thấy).
  - Định nghĩa: `docs/diffusion_6_huong_ke_hoach.md:78-84`.
- **Kiểm định:** bootstrap ghép cặp theo **clip**. Ghép theo `(clip, hàng)`, lấy mẫu lại 200 clip có hoàn lại, 2 000 lần, khoảng tin cậy 95 % theo phân vị (`docs/bao_cao_diffusion/code/bootstrap_ci.py:35-62`). "Có ý nghĩa" khi khoảng tin cậy không chứa 0.
- **Zero-shot:** chấm trên tập **khác**: 60 cửa sổ đầu, mọi hàng của các cửa sổ đó, không dùng index (`scripts/vast/stages_d6.sh:52-55`). Chỉ so được với s200 một cách thô.
- **Không đo:** FVD, AKD, `ccm_tc` và `hkd` (độ nhất quán thời gian và keypoint tay), độ nhấp nháy ở chỗ nối cửa sổ.

---

## 5. Kết quả

### 5.1 Toàn bộ chỉ số trên s200 (3 200 khung, 200 clip) ✅

| Mô hình | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | region L1 ↓ | hand L1 ↓ | hand LPIPS ↓ | face L1 ↓ | drift ↓ | disocc L1 ↓ | arm L1 ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GReg run5 | **0.01648** | **29.61** | **0.9179** | **0.04758** | 0.04369 | 0.1127 | 0.2365 | **0.03040** | **0.00774** | 0.0545 | 0.0516 |
| D6 `ng` | 0.01913 | 28.22 | 0.9128 | 0.05274 | **0.04329** | **0.0909** | **0.1841** | 0.03647 | 0.01041 | **0.0503** | **0.0487** |
| D6 `ng_ones` | 0.01936 | 28.16 | 0.9119 | 0.05332 | 0.04355 | 0.0915 | 0.1856 | 0.03666 | 0.01063 | không còn lưu | không còn lưu |
| D6 `n` | 0.01917 | 27.85 | 0.9107 | 0.05435 | 0.04604 | 0.0921 | 0.1857 | 0.04174 | 0.01020 | không còn lưu | không còn lưu |
| D6 `k` | 0.01944 | 27.71 | 0.9090 | 0.05531 | 0.04660 | 0.0929 | 0.1874 | 0.04218 | 0.01039 | không còn lưu | không còn lưu |
| (tham khảo) D5 `ng` | 0.17173 | 13.27 | 0.5661 | 0.38294 | 0.17284 | 0.2215 | 0.5288 | 0.14377 | 0.15593 | – | – |

Nguồn: bảng `summary.json` đọc trên máy ngày 2026-11-05 và cột B của `bootstrap.md`. `disocc_l1` và `arm_l1` của `n`, `k`, `ng_ones` có trong `bootstrap.md` nhưng chưa được trích ra trước khi xoá máy.

### 5.2 Bootstrap ghép cặp: D6 so với GReg run5 trên s200 (B − A, khoảng tin cậy 95 %, 2 000 lần) ✅

| Biến thể | LPIPS | L1 | PSNR (dB) | hand LPIPS | face L1 | drift |
|---|---|---|---|---|---|---|
| `ng` | +10.8 % [+0.00384, +0.00647] | +16.1 % | −1.38 [−1.55, −1.21] | **−22.1 %** [−0.0591, −0.0455] | +20.0 % | +34.4 % |
| `ng_ones` | +12.1 % [+0.00444, +0.00703] | +17.5 % | −1.45 [−1.62, −1.28] | **−21.5 %** [−0.0582, −0.0441] | +20.6 % | +37.3 % |
| `n` | +14.2 % [+0.00531, +0.00818] | +16.4 % | −1.76 [−1.94, −1.56] | **−21.5 %** [−0.0577, −0.0436] | +37.3 % | +31.7 % |
| `k` | +16.3 % [+0.00622, +0.00921] | +17.9 % | −1.89 [−2.08, −1.70] | **−20.8 %** [−0.0559, −0.0420] | +38.7 % | +34.2 % |

Mọi ô trong bảng đều có ý nghĩa thống kê (p < 0.001): **tay tốt hơn, mọi chỉ số toàn khung kém hơn**.

### 5.3 Zero-shot (Fun-InP gốc, không LoRA, 36 kênh, mặt nạ V·C; 60 cửa sổ, mọi hàng) ✅

| L1 | PSNR | SSIM | LPIPS | hand L1 | hand LPIPS | face L1 | disocc L1 | drift | arm L1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0327 | 21.71 | 0.8418 | **0.1441** | 0.2064 | 0.3310 | 0.0430 | 0.2435 | 0.0184 | 0.1148 |

### 5.4 Đường loss train ✅ (một vài điểm còn trong log)

- Biến thể `n`: bước 160: 0.0927; bước 820: 0.0787; bước 880: 0.0666.
- Hai dòng khác (không còn rõ của biến thể nào): bước 2 240: 0.0888; bước 3 000: 0.0707.
- Loss flow matching nhiễu mạnh theo timestep, nên các điểm lẻ này không đủ để kết luận về hội tụ ⚠️.

---

## 6. Phân tích nguyên nhân và hiện tượng

### 6.1 Vì sao tay tốt hơn GReg 21–22 %

1. **Prior video về bàn tay.** Wan được pretrain trên lượng lớn video người, nên biết hình dạng ngón tay và cách bàn tay chuyển động.
   - GReg là mạng hồi quy, gặp tư thế tay mơ hồ thì cho ra **trung bình có điều kiện**, tức ngón tay mờ hoặc dính nhau.
   - Bằng chứng ✅: chỉ sau 3 000 bước LoRA, cả bốn biến thể đều đạt hand LPIPS 0.184–0.187, kể cả `k` chỉ có skeleton 2D. Một mô hình học từ đầu không thể có prior này sau 6 000 cửa sổ ⚠️.
2. **Fine-tune là bắt buộc.** Zero-shot cho hand LPIPS 0.331 và LPIPS 0.144 ✅. Prior chỉ thành tay đúng tư thế khi mô hình học đọc kênh điều kiện mới và nghĩa mới của mặt nạ.
3. **Loss vùng ×10.** Mỗi ô latent phủ tay hoặc mặt nặng gấp 10 lần ô thường, nên mô hình bị ép dồn năng lực vào đó. Chưa có ablation `w_region = 0` nên chưa tách được đóng góp này ⚠️.
4. **Ngữ cảnh 49 khung.** Self-attention phủ cả cửa sổ, nên mô hình thấy bàn tay ở nhiều khung lân cận. Khi tay bị mờ chuyển động ở một khung, nó có thể dựa vào khung khác. GReg xử lý từng khung riêng ⚠️ (chưa có ablation 1 khung).
5. **Độ phân giải làm việc 320².** Lưới latent 40×40 thay vì 32×32, nên bàn tay chiếm nhiều ô latent hơn khoảng 1.56 lần ⚠️. Tuy vậy, video 320 chỉ là 256 phóng bicubic, **không có thêm chi tiết thật**.
6. **Vùng mà nguồn không thấy cũng tốt hơn** (`ng`): `disocc_l1` 0.0545 → 0.0503 (−7.7 %), `arm_l1` 0.0516 → 0.0487 (−5.6 %) ✅ (ước lượng điểm, chưa kiểm định bootstrap riêng). Điều này khớp giả thuyết ban đầu: diffusion có ích đúng ở chỗ phải "vẽ từ không".

### 6.2 Vì sao toàn khung kém hơn GReg

1. **Diffusion vẽ lại mọi pixel, kể cả pixel đã biết.**
   - Ở vùng V·C = 1, mô hình nhận pixel warp nhưng vẫn phải đi qua VAE encode, khử nhiễu 25 bước rồi decode. Không có bước dán lại pixel warp ở vùng tin cậy (ablation "K4" trong kế hoạch, `docs/diffusion_6_huong_ke_hoach.md:286`, chưa chạy).
   - GReg thì chép trực tiếp ngoại hình từ nguồn qua warp, đúng thứ L1/LPIPS thưởng.
2. **VAE 3D nén mạnh.** Nén ×8 không gian và ×4 thời gian xuống 16 kênh. Chi tiết nhỏ như mắt, miệng, chữ trên áo bị mất ngay ở bước mã hoá.
   - Hệ quả: face L1 +20 %, PSNR −1.4 dB ✅.
   - ⚠️ Trần chất lượng của riêng VAE (encode → decode video thật, rồi chấm) **chưa đo**. Đây là thí nghiệm rẻ nhất để tách lỗi VAE khỏi lỗi khử nhiễu.
3. **Trôi màu tần số thấp: +31–37 % ở cả bốn biến thể** ✅. Kể cả `ng`, có pixel warp ở phần lớn khung, vẫn trôi +34 %.
   - Nguyên nhân khả dĩ ⚠️: lịch shift 5 dồn huấn luyện vào nhiễu cao; flow matching với 25 bước Euler không CFG; tông màu toàn khung được quyết định ở các bước nhiễu cao, nơi điều kiện warp còn yếu.
   - `ng` và `n` trôi gần như bằng nhau (0.0104 vs 0.0102). Như vậy pixel warp **không** chống được trôi màu ở mức này. Màu chủ yếu đến từ CLIP + khung 0 ⚠️.
4. **Mô hình được huấn luyện ít.** Chỉ 3 000 bước × 2 cửa sổ, so với DWM khoảng 10 ngày trên 4×A100 (`docs/code_notes_dwm_wan.md:24`). LoRA chưa đủ để mô hình "nghe lời" pixel warp tới mức chép y nguyên ⚠️.
5. **Phép đo và chất lượng cảm nhận.** L1, PSNR và LPIPS là khoảng cách tới **đúng** ảnh thật. Diffusion sinh chi tiết hợp lý nhưng không trùng pixel, nên bị phạt theo đánh đổi perception–distortion (Blau & Michaeli 2018) ⚠️. Ta không đo FID/FVD, loại chỉ số thường có lợi cho diffusion.

### 6.3 Thứ tự `ng` > `ng_ones` > `n` > `k` nói gì về giả thuyết "mesh giúp"

| So sánh | Khác biệt về điều kiện | LPIPS | face L1 | region L1 | hand LPIPS |
|---|---|---|---|---|---|
| `n` → `ng` | + ảnh nguồn warp theo mesh ở mọi khung + mặt nạ V·C + CCM | 0.0544 → **0.0527** | 0.0417 → **0.0365 (−12.6 %)** | 0.0460 → **0.0433** | 0.1857 → 0.1841 |
| `ng_ones` → `ng` | mặt nạ toàn 1 → mặt nạ V·C thật | 0.0533 → **0.0527** | 0.0367 → 0.0365 | 0.0436 → 0.0433 | 0.1856 → 0.1841 |
| `k` → `n` | skeleton 2D → normal map | 0.0553 → **0.0544** | 0.0422 → 0.0417 | 0.0466 → 0.0460 | 0.1874 → 0.1857 |

1. **Mesh giúp rõ nhất ở danh tính và mặt, không phải ở tay.**
   - Bước nhảy lớn nhất là `n` → `ng`: face L1 −12.6 %, PSNR +0.37 dB.
   - Ở `n` và `k`, mô hình chỉ thấy khung nguồn ở vị trí 0 và phải tự "kéo" ngoại hình qua 48 khung. Ở `ng`, mỗi khung có sẵn pixel nguồn được warp đúng chỗ, nên mặt và quần áo giữ tốt hơn.
   - ⚠️ Bước này gộp ba yếu tố (warp, mặt nạ, CCM). Chưa có biến thể "warp không CCM" để tách riêng.
2. **Mặt nạ V·C thật tốt hơn mặt nạ toàn 1 kiểu DWM** ở mọi chỉ số, nhưng chênh lệch nhỏ: LPIPS −1.1 %, drift −2 %, hand LPIPS −0.8 %.
   - Báo cho mô hình biết "chỗ này là lỗ" giúp nó không tin vào pixel warp sai, ví dụ mép tay hoặc vùng bị che.
   - ⚠️ Chưa kiểm định trực tiếp `ng` vs `ng_ones`. Bootstrap chỉ so từng biến thể với GReg, và hai khoảng tin cậy chồng nhau nhiều.
3. **Normal map tốt hơn keypoint 2D một chút** (LPIPS −1.7 %, region L1 −1.2 %). Hướng khớp giả thuyết "điều kiện dày tốt hơn điều kiện thưa", nhưng biên độ nhỏ ⚠️ (chưa kiểm định trực tiếp).
4. **Ở tay, bốn biến thể không phân biệt được.**
   - hand LPIPS 0.1841 / 0.1856 / 0.1857 / 0.1874; khoảng tin cậy của hiệu số so với GReg là [−0.059, −0.045] (`ng`) và [−0.056, −0.042] (`k`), chồng nhau gần hết ✅.
   - Như vậy, ở ngân sách này, **cải thiện tay đến từ prior video + loss vùng, gần như không phụ thuộc loại điều kiện**.
   - Đây là kết luận phải ghi trung thực: kết quả D6 **không** ủng hộ luận điểm "mesh giúp tay" của SignRefine Table 5 (`docs/diffusion_6_huong_ke_hoach.md:258`). Nó chỉ ủng hộ "mesh giúp giữ danh tính và độ trung thực toàn khung".

### 6.4 Zero-shot 0.144: mô hình gốc không dùng được

- LPIPS 0.144, gấp khoảng 3 lần GReg. `disocc_l1` 0.2435, gấp khoảng 4.5 lần GReg. Mô hình gốc **tô sai các lỗ** dù khung có pixel warp ✅.
- L1 0.0327 chỉ gấp khoảng 2 lần GReg, vì phần lớn khung là pixel warp giữ được.
- Kết luận về đường ống ⚠️: phép chuyển trọng số và quy ước dấu mặt nạ không sai hoàn toàn. Nếu mặt nạ bị đảo dấu, mô hình sẽ xoá phần đã biết và L1 sẽ cao hơn nhiều. Tuy vậy, không có lần chạy dấu ngược để đối chiếu.
- Sau 3 000 bước LoRA, LPIPS giảm từ 0.144 xuống 0.0527 (khoảng −63 %, trên hai tập khác nhau nên chỉ là ước lượng thô).

### 6.5 Hệ quả: tách vai trò

D6 giỏi tay và vùng phải vẽ, nhưng kém ở mặt, màu và độ trung thực pixel. GReg thì ngược lại. Hai lỗi gần như bổ sung cho nhau theo **vùng ảnh**. Đây là lý do trực tiếp của phương pháp ghép "GReg + tay D6" (`docs/bao_cao_diffusion/GReg_plus_D6_hands.md`): LPIPS −9 % so với GReg, mặt không đổi.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế:**
1. **Chỉ 200 trong 1 146 clip test, 16 khung mỗi clip.** Chưa có số trên toàn bộ test để đặt cạnh run5 (0.04851). Chạy một biến thể trên toàn bộ test cần khoảng 7.5 GPU-giờ, khoảng $4.7.
2. **Một seed, một lần train mỗi biến thể, 3 000 bước.** Chưa biết độ dao động giữa các seed. So sánh *giữa* các biến thể chưa kiểm định trực tiếp.
3. **Chưa có chỉ số thời gian:** FVD, nhấp nháy, chỗ nối giữa các cửa sổ 48 khung (sinh độc lập, chỉ chung nhiễu và khung nguồn).
4. **Độ phân giải 320** chỉ là 256 phóng to. Kế hoạch 480² và cache mới (`docs/diffusion_6_huong_ke_hoach.md:294`) chưa làm.
5. **Chi phí suy luận** khoảng 9 lần GReg (0.186 so với khoảng 0.02 s/khung).
6. **Ablation còn thiếu:**
   - TPSMM warp thay mesh warp (ablation 9)
   - `w_region = 0`
   - warp không CCM
   - dán lại pixel ở V·C = 1 (K4)
   - CFG

**Nên làm tiếp (xếp theo lợi ích trên chi phí):**
1. **Trần VAE:** encode → decode video thật của s200 rồi chấm. Không cần train, chỉ vài phút GPU. Kết quả cho biết bao nhiêu phần lỗi face/PSNR là do VAE.
2. **Dán lại pixel warp ở vùng V·C = 1 cao** (K4), sau khi khớp màu. Chỉ là hậu xử lý, có thể giảm trôi màu và lỗi mặt.
3. **Train lại `ng`** (khoảng 2.5 h, khoảng $1.5) để có lại checkpoint, rồi chạy trên toàn bộ tập test.
4. **D6 chỉ cho vùng tay:** crop video quanh tay ở độ phân giải cao hơn. Rẻ hơn sinh cả khung, và phù hợp với phát hiện ở §6.3–6.5.
5. **Kiểm định ghép cặp trực tiếp giữa các biến thể** (`ng` vs `n`, `ng` vs `ng_ones`) bằng `bootstrap_ci.py`, khi có lại `per_frame.csv`.

---

## 8. Trạng thái dữ liệu

Máy vast 58802139 đã bị **xoá** ngày 2026-11-05 khoảng 11:20 UTC. Google Drive đã đầy từ 2026-11-04 02:43 UTC, nên mọi thứ D6 sinh ra **chưa từng được đồng bộ**.

| Hạng mục | Trạng thái |
|---|---|
| 4 checkpoint LoRA + `patch_embedding` (`outputs/d6/{ng,n,k,ng_ones}/checkpoints/final.pt`) | **mất** |
| Cache cửa sổ train/test (latent, CLIP, text) | **mất** (tạo lại: khoảng 4.5 h) |
| Khung sinh ra (`frames_test`, 17 113 PNG mỗi biến thể) và ảnh zero-shot | **mất**: không còn ảnh minh hoạ |
| `per_frame.csv`, `summary.json`, `results_test.md` của D6 | **mất** |
| `bootstrap.md` (toàn bộ khoảng tin cậy, kể cả SSIM, hand L1, arm, disocc theo cặp) | **mất**. Chỉ còn các dòng LPIPS / L1 / PSNR / hand LPIPS / face L1 / drift đã trích trong hội thoại (§5.2) |
| Log loss đầy đủ (`logs/loss.csv`) | **mất**. Chỉ còn vài dòng (§5.4) |
| Mã nguồn | **còn**: nhánh `feat/diff-d6-video-dit` (đã push) và `fix-d6-tmp` (chưa push) |

Số liệu trong báo cáo này lấy từ các bảng đã đọc trên máy và báo trong hội thoại ngày 2026-11-05 (transcript phiên `4a790883…`, memory `diffusion-programme.md`).

**Lưu ý về mã chạy thật trên máy:**
- `scripts/vast/stages_d6.sh` đã được sửa bằng `sed` để gọi `/workspace/GReg-d1/scripts/21_eval_frames.py`. Nhánh D6 thiếu `--index`, đây là lỗi làm hỏng bước chấm `d6_ng` lúc 00:18 UTC và để GPU rỗi khoảng 75 phút.
- Bản sửa này **chỉ tồn tại trên máy đã xoá**. Cần port `--index` vào nhánh D6 trước khi chạy lại.
