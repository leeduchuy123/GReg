# Báo cáo D2: Regional Expert bằng diffusion (flow matching)

Nhánh `feat/diff-d2-diffusion-re`. Code đọc ở `origin/feat/diff-d2-diffusion-re` @`c4ea69c`; nhánh local cùng tên đang thiếu commit sửa lỗi này. Chạy trên vast.ai instance 58802139 (RTX 4090) ngày 2026-11-03.

Ký hiệu: ✅ đã kiểm chứng (code, log hoặc số đo) · ⚠️ giả thuyết hoặc suy luận, chưa đo riêng.

---

## 0. Tóm tắt

- **Làm gì:** GReg có một module Regional Expert (RE). RE dự đoán một "prior cấu trúc" (bản đồ gradient 64×64) cho mặt và hai bàn tay bằng **hồi quy L1**. D2 giữ nguyên mạng RE, chỉ đổi mục tiêu huấn luyện sang **flow matching**, tức một mô hình diffusion có điều kiện. Prior lúc chạy được **lấy mẫu** (Euler 8 bước) thay vì dự đoán thẳng. Bộ sinh và loss của GReg giữ nguyên.
- **Giai đoạn A (chỉ train RE):** sai số prior của RE diffusion là 0.0325, kém RE hồi quy của run5 (0.0238) **37 %** ✅.
- **Giai đoạn B (GReg 15 epoch, val 100 clip × 16 khung):** cả hai cấu hình đều kém `base` ✅.
  - `d2_fixed` (nhiễu cố định): LPIPS 0.0617 → 0.0745 (**+20.7 %**), hand LPIPS +12.2 %.
  - `d2_random` (nhiễu ngẫu nhiên): LPIPS +4.1 %, L1 bằng `base`.
- **Kết luận:** D2 bị loại. Với GReg, prior cấu trúc là **tín hiệu điều kiện cho bộ sinh**, không phải ảnh đầu ra. Ở vai trò đó, trung bình có điều kiện ổn định của hồi quy tốt hơn một mẫu ngẫu nhiên có độ lệch cao. Diffusion nên đặt ở đầu ra (như D1 và D6), không đặt ở tầng prior.

---

## 1. Bối cảnh

### 1.1 GReg run5 làm gì

GReg (baseline run5, `greg/models/greg.py:1-17`) sinh khung đích `Î` từ ảnh nguồn `I_s` và normal map SMPL-X của tư thế đích `N_t`:

1. **Flow:** TPSMM đóng băng (49.2 M tham số) ước lượng biến dạng và bản đồ che khuất `O` từ `(N_s, N_t)`. Từ đó có ảnh warp `I_st = warp(I_s, F)`.
2. **Regional Prior Generation (RPG):** cắt 3 vùng 64×64 (mặt, tay trái, tay phải) từ `N_t`. RE nhận crop normal và mã style của `I_s`, trả về **prior cấu trúc 1 kênh** cho mỗi vùng. Các prior được dán lại thành `prior_map` (B, 3, 256, 256).
3. **Gradient Regularization:** bản đồ gradient vùng của `N_t` đi qua GradientEncoder và GRM (attention, `qk_norm`) để điều biến bộ sinh.
4. **Bộ sinh:** đầu vào `cat[I_st, N_t, prior_map]` (9 kênh), ra `I_gen`. Ảnh cuối `Î = O·I_st + (1−O)·I_gen` (`greg.py:316-318`, nhánh D2).
5. **Loss:** VGG pyramid toàn ảnh, VGG trên crop vùng (×3), `L_struct` (L1 giữa prior và gradient của crop đích, ×10), và 3 PatchGAN theo vùng.

Số tham số train của GReg là 34.1 M, discriminator 2.0 M (log ngày 2026-11-03) ✅.

### 1.2 D2 thay phần nào

D2 chỉ thay **RPG**, tức hai RE: một cho mặt, một dùng chung cho hai tay. Mọi phần khác giữ nguyên: TPSMM, GradientEncoder, GRM, bộ sinh, discriminator và loss. Giao diện `forward(normal_target, boxes, style) -> {prior_patches, prior_map, normal_crops}` cũng giữ nguyên (`region_diffusion.py:125-130`). Lựa chọn được bật qua khoá `model.rpg.type: regression | diffusion` (`greg.py:138-151`).

### 1.3 Vì sao chọn hướng này

- **Tay bị che hoặc bắt chéo là chỗ GReg yếu nhất.** Ở cử động lớn, khoảng 39 % pixel tay và cánh tay không có trong ảnh nguồn (`disocc_frac@large` = 0.388 ✅, `docs/diffusion_6_huong_ke_hoach.md` §0). Chỉ dựa vào `(N_crop, style)`, cấu trúc thật của bàn tay ở các trường hợp này là **đa mode**: ngón có thể gập hay duỗi, nằm trước hay sau. Hồi quy L1 trả về trung vị, tức một prior mờ.
- **Giả thuyết:** một prior *sắc* (một mẫu hợp lý) giúp bộ sinh vẽ ngón tay rõ hơn một prior *mờ* (trung bình).
- **Chi phí thấp:** crop chỉ 64², nên chạy 8 bước mỗi crop là không đáng kể so với bộ sinh 256². Ước tính trong kế hoạch là khoảng 6 USD.

---

## 2. Ý tưởng và điểm mới

### 2.1 Ý tưởng

RE được thay bằng **mô hình sinh có điều kiện** `p(prior | N_crop, style)` thay vì hàm tất định `f(N_crop, style)`. Prior được lấy mẫu rồi đưa vào GReg đúng chỗ prior hồi quy đi vào trước đây.

### 2.2 Điểm đặc sắc

1. **"Cùng mạng, chỉ đổi mục tiêu" (so sánh có kiểm soát).** Kế hoạch ban đầu viết một UNet mới khoảng 6 M tham số. Bản hiện thực dùng **đúng lớp `RegionalExpert` của GReg**: UNet modulated-conv kiểu StyleGAN2 với kênh 64-128-256-256 (`regional_expert.py:27-55`). Chỉ có ba thay đổi:
   - thêm 1 kênh đầu vào `x_t`, từ 3 lên 4;
   - thêm nhúng thời gian vào vector style, từ 256 lên 256+128;
   - bỏ sigmoid ở đầu ra (`out_activation="none"`, `regional_expert.py:29-31, 54-55`).

   Vì vậy, khác biệt kết quả nằm ở **mục tiêu học** (hồi quy so với flow matching), không ở dung lượng mạng.
2. **Lần đầu đưa diffusion vào tầng prior của GReg** ⚠️ (theo những gì nhóm đã đọc). Các công trình sinh video ký hiệu dùng diffusion thường thay toàn bộ bộ sinh, như SignViP hay D5/D6 ở đây. D2 đặt diffusion vào một thành phần trung gian 64² rất rẻ.
3. **Nhiễu cố định để tất định và ổn định theo thời gian.** Chế độ `fixed` dùng **một** tensor nhiễu (seed 0) cho mọi mẫu và mọi khung (`region_diffusion.py:78-84`). Hai khung liên tiếp chỉ khác nhau ở điều kiện, nên prior không nhấp nháy do lấy mẫu lại. Chế độ `random` là đối chứng.
4. **Đo trực tiếp mức "đa mode".** Validation của giai đoạn A tính ba đại lượng (`26_train_region_diffusion.py:78-125`):
   - `diff_l1`: sai số của mẫu nhiễu cố định, đúng thứ GReg sẽ thấy;
   - `diff_mean_l1`: sai số của trung bình 8 mẫu;
   - `diff_std`: độ lệch chuẩn giữa 8 mẫu, đo mức mô hình coi vùng đó là đa mode.

   Cả ba được so trên **cùng crop** với RE hồi quy của run5 epoch_100.

### 2.3 Mượn từ đâu, phần nào là của nhóm

| Thành phần | Nguồn |
|---|---|
| Tham số hoá rectified flow: `x_t = (1−t)x0 + tε`, mục tiêu `v = ε − x0`, Euler từ t=1 về 0 | Wan / DWM (`docs/bai_hoc_diffusion_mesh_cho_greg.md` §2.2) ✅ |
| Nhúng thời gian sinusoidal (t × 1000, giống timestep DDPM) | Chuẩn DDPM |
| EMA 0.999, warmup tuyến tính 500 bước, clip gradient 1.0 | Thực hành chuẩn |
| Kiến trúc RE, StyleEncoder, modulated conv, khung hai giai đoạn "train RE riêng" + `detach_prior` | GReg (paper và code tái hiện) |
| **Của nhóm:** thay mục tiêu RE bằng flow matching trong GReg, nhiễu cố định theo chương trình, giao diện thay thế trực tiếp, bộ đo `diff_l1 / diff_mean_l1 / diff_std` so với RE hồi quy, thí nghiệm `fixed` / `random` | 🛠 |

---

## 3. Kiến trúc chi tiết

### 3.1 RE hồi quy gốc (bị thay thế)

```
N_crop (B,3,64,64) ─┐
                    ├─ ModulatedConv2d(3→64, style 256) ─ LeakyReLU          64²
style s (B,256) ────┘   down: StyledBlock 64→128 (32²) → 128→256 (16²) → 256→256 (8²)
                        mid : StyledBlock 256→256 (8²)
                        up  : 3 × [bilinear ×2, concat skip, StyledBlock] → 64 kênh, 64²
                        outc: Conv 3×3 64→1 → sigmoid → prior (B,1,64,64) ∈ [0,1]
Loss: L_struct = Σ_r w_r · |prior_r − G(crop_r(I_t))|₁ / Σ_r w_r   (λ = 10, losses.py:148-158)
```

- `G(·)` là độ lớn gradient trên ảnh xám: `sqrt(dx² + dy² + 1e-6)` với sai phân lùi (`greg/data/gradient.py`).
- `w_r` là trọng số theo độ hiển thị của vùng.
- Hai expert: mặt riêng, hai tay dùng chung (`regional_expert.py:58-80`).
- RE được train **cùng lúc** với GReg trong 100 epoch, nhưng `detach_prior: true` (`greg.py:103, 306`) chặn gradient từ bộ sinh. RE chỉ học từ `L_struct`.

### 3.2 RE diffusion (D2): `FlowExpert` và `DiffusionRegionalPriorGenerator`

```
                         t ∈ [0,1] ─ sinusoidal(128, t×1000) ─ MLP(128→128, SiLU, 128→128)
                                                                     │
style s (B,256) ───────────────────────────── concat ─────────────── cond (B,384)
                                                                     │
x_t (B,1,64,64) ─┐                                                   ▼
                 ├─ concat (B,4,64,64) ─► RegionalExpert(in=4, out=1, style_dim=384,
2·N_crop−1 ──────┘                         kênh 64-128-256-256, out_activation=none)
 (B,3,64,64)                                    │
                                                ▼
                                    v̂ (B,1,64,64)  ≈ ε − x0   (vận tốc)
```

`region_diffusion.py:43-55`. Mỗi `DiffusionRegionalPriorGenerator` có `face_expert` và `hand_expert`; tay phải dùng chung `hand_expert` (`region_diffusion.py:58-75`).

**Số tham số:**
- Hai expert (gồm MLP thời gian): **14.44 M**; StyleEncoder: **1.64 M** (log của script giai đoạn A) ✅.
- RE hồi quy có cùng số kênh nhưng không có MLP thời gian, và modulation của nó 256 chiều thay vì 384, nên ít tham số hơn một chút. Số chính xác không được log riêng.

### 3.3 Giai đoạn A: train RE diffusion riêng

`scripts/26_train_region_diffusion.py`, chạy qua `stages_d2.sh:15-18`.

- **Dữ liệu:** các cặp (nguồn, đích) của `train_fit`, `pairs_per_clip=8`. Mỗi cặp cho ba thứ:
  - crop `N_t` ở 3 hộp cố định 64×64;
  - style của khung nguồn;
  - mục tiêu `struct = G(crop(I_t))` (`26_train_region_diffusion.py:67-75`).
- **Chuẩn hoá:** `x0 = 2·clamp(struct, 0, 1) − 1`.
- **Loss** (`region_diffusion.py:87-105`). Mỗi vùng lấy một t riêng, `t ~ U(0,1)`, kích thước (B,3):

  ```
  x_t^r = (1 − t^r)·x0^r + t^r·ε^r,     ε ~ N(0, I)
  L_FM  = Σ_b Σ_r w_{b,r} · mean_pixel( v_θ(x_t^r, N^r, s, t^r) − (ε^r − x0^r) )²  /  Σ w
  ```

  `w` là độ hiển thị của vùng (`region_visibility`). Style bị dropout với p = 0.1, nhưng lúc lấy mẫu **không dùng CFG**.
- **Siêu tham số** (`26_train_region_diffusion.py:49-63`):
  - 30 000 bước, batch 32 cặp (96 crop);
  - AdamW lr 2e-4, weight decay 0, warmup 500 bước;
  - autocast bf16, clip gradient 1.0, EMA 0.999 (checkpoint lưu bản EMA);
  - validation mỗi 5 000 bước trên 60 clip val × 8 khung, 8 mẫu.
- **Chi phí:** 0.141 s/bước. 30 000 bước mất khoảng 1 h 15 phút trên 4090, xong lúc 13:10 UTC ngày 03/11 ✅.
- **Lượng dữ liệu đã thấy:** 30 000 × 32 = 960 000 cặp. RE hồi quy của run5 thấy khoảng 100 epoch × ~4 100 cặp ≈ 410 000 cặp (suy từ cấu hình `pairs_per_clip: 1`, batch 4). RE diffusion **không thiếu dữ liệu** so với hồi quy ⚠️ (số cặp suy từ cấu hình, không đếm trực tiếp).

### 3.4 Lấy mẫu (lúc train giai đoạn B và lúc suy luận)

`region_diffusion.py:107-123`:

```
x ← ε_fixed (1,3,1,64,64), seed 0, expand theo batch      [hoặc randn ở chế độ random]
ts = linspace(1, 0, 9)
với mỗi vùng r, i = 0..7:   x_r ← x_r + (ts[i+1] − ts[i]) · v_θ(x_r, N^r, s, ts[i])
prior_r = clamp((x_r + 1)/2, 0, 1)                                   (B,1,64,64)
```

- Toàn bộ chạy trong `@torch.no_grad()`, gồm 8 bước × 2 mạng (tay phải dùng chung mạng tay) × 3 vùng.
- Prior được dán vào canvas bằng `paste_regions`, cho `prior_map` (B,3,256,256). Phần sau giống hệt GReg.

### 3.5 Giai đoạn B: GReg dùng RE diffusion đóng băng

```
I_s (B,3,256,256) ─► StyleEncoder [ĐÓNG BĂNG, nạp từ giai đoạn A] ─► s (B,256)
N_t (B,3,256,256) ─► crop 3 hộp ─► N_crop (B,3,3,64,64)
                         │
                         ▼
        DiffusionRPG [ĐÓNG BĂNG]: 8 bước Euler × 3 vùng ─► prior_patches (B,3,1,64,64)
                         │ paste
                         ▼
                    prior_map (B,3,256,256) ─ detach ─┐
I_st = warp(I_s, F_TPSMM) (B,3,256,256) ──────────────┼─► Generator(9 kênh; GRM(GradEnc(G_R(N_t))))
N_t ──────────────────────────────────────────────────┘          │
                                                                  ▼
                                               Î = O·I_st + (1−O)·I_gen
```

- **Nạp và đóng băng:** `GReg.load_pretrained_rpg` nạp trạng thái EMA của RPG và StyleEncoder rồi đặt `requires_grad=False` (`greg.py:171-181`). `06_train_greg.py:110-118` chỉ nạp khi `model.rpg.type == diffusion` (commit sửa `c4ea69c`, xem §8).
- **Loss của GReg không đổi.** `L_struct` vẫn được tính nhưng **không có gradient**, vì prior lấy mẫu dưới `no_grad` và RE đã đóng băng. Nó chỉ còn là con số trong log, khoảng 0.03 ✅.
- **Cấu hình sweep** (`configs/sweeps/d2_diffusion_re.yaml`):
  - kế thừa `configs/sweep_greg.yaml`: 15 epoch, milestone [10, 13], batch 4, lr 2e-4;
  - cộng `model.gr.qk_norm=true` và `loss.weights.regional_perceptual=3`, đúng cấu hình run5;
  - hai biến thể: `d2_fixed` (`rpg.type=diffusion, freeze=true, noise_mode=fixed`) và `d2_random` (như trên nhưng `noise_mode=random`);
  - `base` là checkpoint `run5/outputs/sweep_clean/ref` (run5 train 15 epoch), **không train lại**, chỉ chấm lại cùng cách (`stages_d2.sh:20-25`).
- **Chi phí:** 0.27 s/bước, VRAM tối đa 3.9 GB ✅. Khoảng 70 phút mỗi run. Toàn bộ sweep (chấm `base` + 2 run) chạy từ 13:24 đến 15:47 UTC, khoảng 2 h 23 phút ✅.
- **Tổng GPU của D2:** khoảng 1.25 h + 2.4 h ≈ 3.7 h, tức khoảng 2.3 USD ở 0.62 USD/h ⚠️ (tính từ mốc thời gian trong log).

---

## 4. Giao thức đánh giá

| Giai đoạn | Tập | Cách đo |
|---|---|---|
| A (chỉ RE) | 60 clip val × 8 khung lấy đều, nguồn = khung 0. Chỉ tính vùng hiển thị | L1 giữa prior và `G(crop(I_t))` trong [0,1], theo từng vùng. RE hồi quy = RE + StyleEncoder của run5 epoch_100, chạy trên **cùng crop** |
| B (GReg) | **sweep15**: GReg train 15 epoch, chấm trên val **100 clip × 16 khung = 1 600 khung** | `scripts/11_sweep.py` với `eval: {split: val, max_clips: 100, max_rows: 16, lpips: true, eval_region_size: 64}` |

Chỉ số của giai đoạn B:
- `l1`, `psnr`, `ssim` trên toàn ảnh.
- `lpips` toàn ảnh, mạng AlexNet như paper.
- `hand_l1`: L1 trong hộp tay.
- `hand_lpips`: LPIPS trên crop tay phóng lên 128².
- `face_l1`: L1 trong hộp mặt.
- `region_l1`: L1 trong 3 hộp 64×64.
- `@large`: chỉ tính trên nhóm khung có cử động lớn.

Một seed. **Không so số sweep15 với số test 100 epoch** (LPIPS 0.0485): đây là vòng lọc nhanh, nên con số tuyệt đối cao hơn.

---

## 5. Kết quả

### 5.1 Giai đoạn A: chất lượng prior (60 clip val × 8 khung)

Prior L1 so với gradient thật, càng thấp càng tốt ✅ (`outputs/d2/re/val_last.json`, bước 30 000):

| | Mặt | Tay trái | Tay phải | Chung |
|---|---|---|---|---|
| RE hồi quy (run5 epoch_100) | **0.0215** | **0.0255** | **0.0262** | **0.0238** |
| RE diffusion, mẫu nhiễu cố định (`diff_l1`) | 0.0290 | 0.0371 | 0.0346 | 0.0325 (+37 %) |
| RE diffusion, trung bình 8 mẫu (`diff_mean_l1`) | – | – | – | 0.0288 (+21 %) |
| Độ lệch giữa 8 mẫu (`diff_std`) | 0.0194 | 0.0280 | 0.0288 | 0.0240 |

Xu hướng theo bước train (`val.csv`, hai dòng cuối) ✅:

| Bước | diff_l1 | diff_mean_l1 | diff_std | reg_l1 |
|---|---|---|---|---|
| 25 000 | 0.0336 | 0.0298 | 0.0253 | 0.0238 |
| 30 000 | 0.0325 | 0.0288 | 0.0240 | 0.0238 |

Các mốc 5 000 đến 20 000 không còn trong hội thoại. File `val.csv` nằm trên Drive (§8).

### 5.2 Giai đoạn B: GReg 15 epoch (sweep15, val 100 clip × 16 khung = 1 600 khung)

Nguồn: `outputs/d2/sweep/sweep_results.csv` và `val_metrics.json` ✅.

| Run | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | hand L1 ↓ | hand LPIPS ↓ | face L1 ↓ | region L1 ↓ | hand L1@large ↓ | LPIPS@large ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| **base** (RE hồi quy) | **0.0210** | **28.17** | **0.9064** | **0.0617** | **0.1382** | **0.2845** | **0.0304** | **0.0507** | **0.1691** | **0.0982** |
| d2_random | 0.0210 (−0.1 %) | không còn lưu | không còn lưu | 0.0643 (+4.1 %) | 0.1422 (+2.9 %) | 0.2911 (+2.3 %) | không còn lưu | 0.0518 (+2.1 %) | 0.1750 (+3.5 %) | không còn lưu |
| d2_fixed | 0.0247 (+17.3 %) | 27.37 (−0.80 dB) | 0.9008 | 0.0745 (+20.7 %) | 0.1524 (+10.3 %) | 0.3194 (+12.2 %) | 0.0312 (+2.6 %) | 0.0573 (+13.0 %) | 0.1878 (+11.0 %) | 0.1221 (+24.3 %) |

Chi phí: cả hai run D2 đều 0.27 s/bước và 3.9 GB VRAM ✅. Thứ hạng trung bình trên 6 chỉ số: base 1.17, d2_random 1.83, d2_fixed 3.00 ✅.

---

## 6. Phân tích nguyên nhân và hiện tượng

### 6.1 Mẫu diffusion kém trung bình có điều kiện theo L1, và đó là điều dự đoán được

- **Lý thuyết:** với cùng điều kiện, L1 kỳ vọng nhỏ nhất đạt được ở **trung vị** của phân phối có điều kiện. Một **mẫu** lấy từ phân phối đó, dù đúng hoàn toàn, có L1 kỳ vọng lớn hơn. Nếu phân phối thật sự đa mode, mẫu đúng có thể chênh khoảng √2 lần về sai số bình phương. Vì vậy so `diff_l1` với `reg_l1` vốn **thiên vị hồi quy**. Giai đoạn A không đủ để kết luận; con số quyết định là giai đoạn B.
- **Bằng chứng ✅:** trung bình 8 mẫu (0.0288) đã gần hồi quy hơn hẳn một mẫu (0.0325). Nhưng nó vẫn kém 0.0238. Nghĩa là ngay cả khi lấy trung bình, RE diffusion vẫn chưa học tốt bằng hồi quy, không chỉ do "mẫu mà so với trung vị".

### 6.2 Độ lệch giữa các mẫu cỡ bằng chính sai số

- `diff_std` = 0.024, gần bằng `reg_l1` = 0.0238 ✅. Một mẫu lệch khỏi trung bình của chính nó trung bình một lượng bằng toàn bộ sai số của RE hồi quy.
- Tay có độ lệch cao hơn mặt (0.028–0.029 so với 0.019) ✅. Điều này đúng với giả thuyết "tay đa mode hơn mặt".
- Nhưng **không tách được** "đa mode thật" với "mô hình chưa hội tụ". Đường val vẫn đang giảm từ 25 000 đến 30 000 bước (`diff_l1` 0.0336 → 0.0325, `diff_std` 0.0253 → 0.0240) ✅. Như vậy một phần độ lệch là nhiễu do chưa train xong ⚠️.

### 6.3 Prior là tín hiệu điều kiện, nên mẫu nhiễu dẫn sai bộ sinh

- Trong GReg, prior không xuất hiện trong ảnh đầu ra. Nó là **3 kênh đầu vào của bộ sinh**. Bộ sinh học tin vào các cạnh trong prior để vẽ ngón tay và đường viền mặt.
- Khi prior lệch 37 % và **lệch một cách nhất quán** (`d2_fixed`: cùng điều kiện luôn cho cùng mẫu sai), bộ sinh vẽ theo cạnh sai. Kết quả là L1 +17 %, hand LPIPS +12 %, LPIPS@large +24 % ✅.
- **Khác biệt lớn nhất nằm ở cử động lớn** (LPIPS@large +24 % so với LPIPS chung +21 %, hand L1@large +11 %). Đó chính là nơi prior được kỳ vọng giúp nhiều nhất, và cũng là nơi RE diffusion lệch nhiều nhất ⚠️ (chưa có `diff_l1` theo nhóm cử động để kiểm chứng).

### 6.4 `fixed` kém hơn `random` nhiều, trái với dự đoán

Dự đoán ban đầu là `fixed` tốt hơn vì tất định và không nhấp nháy. Kết quả ngược lại: `fixed` LPIPS +20.7 %, `random` +4.1 % ✅. Giải thích hợp lý nhất ⚠️:

- **`fixed`:** prior là hàm tất định của điều kiện, nên bộ sinh coi nó là tín hiệu tin cậy và học bám theo. Nhưng tín hiệu này mang sai lệch có hệ thống. Thêm nữa, cùng một tensor nhiễu cho mọi crop có thể để lại một "vân nhiễu" cố định trong mọi prior, và bộ sinh có thể học cả vân đó.
- **`random`:** với cùng một cặp train, prior thay đổi mỗi lần gặp. Bộ sinh học rằng kênh prior **không đáng tin** và giảm phụ thuộc vào nó. Kết quả gần với "GReg không có prior": L1 bằng `base` (0.0210) ✅, LPIPS hơi kém (+4 %), vì mất phần lợi ích mà prior hồi quy tốt mang lại.
- Kiểm chứng đầy đủ cần ablation `no_rpg` (prior = 0) trên cùng giao thức sweep15. Nếu `d2_random` ≈ `no_rpg`, giải thích này đứng vững. Chưa chạy.

### 6.5 Có thể có vấn đề về thang giá trị của mục tiêu

- `G(·)` là độ lớn gradient trên ảnh trong [0,1]. Phần lớn pixel của crop gần 0, chỉ các cạnh có giá trị đáng kể. Sau `x0 = 2·struct − 1`, dữ liệu tập trung sát −1 với phương sai rất nhỏ, trong khi nhiễu ε có độ lệch 1.
- Các mô hình diffusion thường chuẩn hoá dữ liệu về phương sai khoảng 1 (ví dụ hệ số 0.18215 của latent SD). D2 không làm bước này. Với tỉ lệ tín hiệu/nhiễu thấp như vậy, mô hình phải dự đoán vận tốc chính xác tới mức rất nhỏ. Sai số nhỏ của 8 bước Euler đủ để tạo một "nền nhiễu" trên toàn crop, làm L1 tăng đều ⚠️ (giả thuyết, chưa đo histogram mẫu như kế hoạch §4.5 đề xuất).

### 6.6 So sánh với `base` có một yếu tố gây nhiễu

`base` là checkpoint run5 `sweep_clean/ref`, train trên **máy khác, phiên bản thư viện khác**. D2 train trên 4090 này với torch 2.7.1. Ở D4, run `fresh` lệch +12 % so với `base` mà không có lý do rõ ràng ✅ (`d4/sweep`). Điều đó cho thấy chênh lệch máy/seed có thể lên vài phần trăm.

- Với `d2_fixed` (+21 %), sai lệch vượt xa mức nhiễu đó, nên kết luận "kém hơn" vững.
- Với `d2_random` (+4 %), khoảng cách nằm trong vùng không chắc chắn. Chỉ nên kết luận là **không tốt hơn**.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế:**
- Một seed, val 100 clip, 15 epoch.
- `base` không được train lại trên cùng máy.
- RE diffusion chưa hội tụ hẳn (val vẫn giảm ở 30k bước).
- Không có ablation `no_rpg` trên cùng giao thức.
- Chưa thử chuẩn hoá thang giá trị của mục tiêu.
- Ablation 4–5 trong kế hoạch (mục tiêu RGB, fine-tune end-to-end với `detach_prior=false`) chưa làm, vì `d2_fixed` không thắng.

**Nếu muốn cứu hướng này** (theo thứ tự chi phí tăng dần), mỗi bước cần GPU nên phải có quyết định của nhóm:
1. **Không tốn train:** đưa **trung bình K mẫu** (K = 8) làm prior thay vì một mẫu. `diff_mean_l1` cho thấy sai số giảm từ 0.0325 xuống 0.0288. Đây là cách biến diffusion về gần "trung bình có điều kiện", nhưng cũng làm mất đúng ý tưởng "prior sắc".
2. Chuẩn hoá `x0` về phương sai khoảng 1, train thêm (60–100k bước), rồi so lại giai đoạn A.
3. Chạy `no_rpg` và `base` với seed 1 trên cùng máy để có mốc so sánh sạch.
4. **Bài học cho luận văn** (đã có đủ dữ liệu): diffusion có lợi khi **đầu ra của nó được đánh giá trực tiếp** (D1 refiner, tay D6). Nó có hại khi đầu ra trở thành **tín hiệu điều kiện trung gian** cho một mạng hồi quy, vì phương sai của mẫu chuyển thành nhiễu đầu vào.

---

## 8. Trạng thái dữ liệu và sự cố

**Trên Drive** (`GReg_runs/diff_common/outputs/d2/`, remote `gdrive,root_folder_id=1FTgmwzFrRW9qIzLNYAFmt9txxECPgsTr:`):
- D2 xong lúc 15:47 UTC ngày 03/11, trước khi Drive đầy (02:43 UTC ngày 04/11).
- Ngày 05/11 lúc 11:10 UTC, kiểm tra từ máy vast xác nhận `d2/sweep/sweep_results.csv` **có trên Drive** ✅.
- Các file còn lại do vòng sync đẩy lên trước 02:43 UTC ngày 04/11, nhiều khả năng đã có nhưng **chưa kiểm từng file** ⚠️:
  - `re/last.pth`, `re/val.csv`, `re/val_*.jpg` (ảnh lưới: crop normal | thật | mẫu cố định | 3 mẫu ngẫu nhiên | RE hồi quy);
  - các `val_metrics.json`;
  - checkpoint sweep.
- Thư mục `samples` của các run sweep có trên máy; vòng sync chỉ loại `export/**` và thư mục `frames*`.
- Instance 58802139 đã bị xoá ngày 05/11.

**Sự cố khi chạy:**
- **Lỗi lúc 13:10 UTC ngày 03/11.**
  - **Hiện tượng:** sweep truyền `model.rpg.checkpoint` cho cả `base`. Script nạp RE diffusion vào `base` và đóng băng RE + StyleEncoder, nên optimizer có ít nhóm tham số hơn checkpoint. Bước resume báo lỗi `ValueError: loaded state dict contains a parameter group that doesn't match`.
  - **Cách sửa:** commit `c4ea69c` chỉ áp checkpoint khi `rpg.type == diffusion` (`06_train_greg.py:110-118`, nhánh origin).
  - **Thời gian mất:** chạy lại lúc 13:24 UTC, GPU rảnh khoảng 15–20 phút.
  - Commit đã lên GitHub ở `origin/feat/diff-d2-diffusion-re`. Nhánh local `feat/diff-d2-diffusion-re` vẫn đứng ở `546aec8`, cần `git pull` hoặc fast-forward.
