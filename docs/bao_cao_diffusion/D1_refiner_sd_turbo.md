# Báo cáo D1: refiner diffusion một bước (SD-Turbo), có cổng theo hình học

Nhánh code: `feat/diff-d1-refiner`. Thực nghiệm chạy từ 2026-11-03 01:41 đến 11:54 UTC trên vast.ai instance 58802139 (RTX 4090), riêng phần test chạy ngày 2026-11-05. Kế hoạch gốc nằm ở `docs/diffusion_6_huong_ke_hoach.md` §3.

Quy ước: ✅ đã kiểm chứng (bằng code hoặc số đo) · ⚠️ giả thuyết hoặc suy luận, chưa đo riêng.

Có năm tập đánh giá. Mỗi bảng ghi rõ tập của mình, và chỉ so các dòng đo trên cùng tập:

| Mã | Nội dung |
|---|---|
| **val-full** | Toàn bộ export val của run5: 458 clip, 14 198 khung |
| **u16** | Toàn bộ tập test (1 146 clip), 16 khung lấy đều mỗi clip |
| **s200** | 200 clip test × 16 khung, cùng các khung với D5 và D6 |
| **q160** | 40 clip val × 4 khung = 160 khung, dùng cho kiểm tra nhanh lúc train và cho run 0 |
| **test-full** | Toàn bộ tập test, mọi khung (D1 chưa được chấm trên tập này) |

---

## 0. Tóm tắt

- D1 giữ nguyên GReg run5 làm bộ dựng ảnh thô. Sau đó một mạng diffusion **một bước** (SD-Turbo theo công thức pix2pix-turbo) làm sắc lại ảnh.
- Refiner chỉ được sửa trong một **cổng hình học**: tay, mặt và vùng thân mà ảnh nguồn không thấy. Ngoài cổng giữ nguyên pixel của GReg.
- Bản tốt nhất (run3, cổng `freq`) cho kết quả **ổn định trên cả ba tập**:
  - LPIPS giảm 4.6–4.7 %.
  - hand LPIPS giảm 7.0–7.2 %.
  - L1 giữ nguyên.
  - Đổi lại PSNR giảm 0.1–0.3 dB và face L1 tăng 5–7 %.
- Diffusion không huấn luyện (run 0) làm ảnh **tệ hơn** GReg. Bỏ cổng thì L1 tăng 12 % và độ trôi màu tăng 33 %.
- Thêm normal map và mặt nạ vào refiner **không giúp đo được**: run1 và run2 bằng nhau.

**Kết luận:** D1 là cải tiến thật, có ý nghĩa thống kê, nhưng nhỏ. Lợi ích đến chủ yếu từ việc **làm sắc chi tiết tần số cao**, chưa phải từ việc sửa hình học tay. Ghép D1 với tay của D6 cho LPIPS thấp nhất cả chương trình (0.0424 trên s200), nhưng mặt vẫn kém hơn GReg 6 %.

---

## 1. Bối cảnh

**GReg run5 làm gì.** Mô hình nhận ảnh tham chiếu `I_s` và normal map SMPL-X của khung đích `N_t`, rồi:
1. TPSMM (đóng băng) warp ảnh nguồn về tư thế đích.
2. Regional Expert sinh prior gradient cho tay và mặt.
3. Bộ sinh GAN có GRM dựng khung 256².

Trên test-full, run5 đạt L1 0.01735, PSNR 29.20, SSIM 0.9151, LPIPS 0.04837 ✅.

**Lỗi nằm ở đâu.** Phân bổ lỗi trên sweep15, cấu hình run5 ✅ (`docs/diffusion_6_huong_ke_hoach.md:31-37`):

| | toàn ảnh | tay | tay @large | cánh tay @large | LPIPS @small → @large |
|---|---|---|---|---|---|
| L1 / LPIPS | 0.0210 | 0.1382 | 0.1691 | 0.1036 | 0.0236 → 0.0982 |

- L1 ở tay lớn gấp khoảng 6.5 lần L1 toàn ảnh.
- LPIPS tăng khoảng 4 lần khi chuyển động lớn.
- Ở chuyển động lớn, `disocc_frac@large` = 0.39 ✅: 39 % pixel thân **không có trong ảnh nguồn**. Phép warp không lấy được gì ở đó, nên bộ sinh phải tự vẽ.
- Trên val-full, ảnh GReg có hand LPIPS 0.2470, tăng lên 0.2704 ở @large ✅.

**Vì sao chọn hướng này.**
1. Phần lớn ảnh của GReg đã tốt: L1 và SSIM của run5 còn tốt hơn số trong bài báo GReg. Lỗi tập trung ở tay, mặt và vùng bị che. Vậy chỉ cần sửa **cục bộ**, không cần thay cả bộ sinh.
2. Prior của diffusion giỏi tạo chi tiết tần số cao, đúng thứ mà GAN hồi quy hay làm mờ.
3. Diffusion **một bước** rẻ hơn nhiều so với lấy mẫu 25–50 bước, và cho kết quả tất định (không ngẫu nhiên theo seed), nên ít nhấp nháy giữa các khung.

---

## 2. Ý tưởng và điểm mới

### 2.1 Ý tưởng một câu
`Î = Blend(Î_GReg, R(Î_GReg, N_t, masks), G)`. Trong đó R là SD-Turbo một bước được tinh chỉnh bằng LoRA, và G là cổng sinh từ hình học SMPL-X (vùng tay/mặt và độ nhìn thấy V của mesh).

### 2.2 Phần mượn từ công trình trước

| Thành phần | Nguồn |
|---|---|
| SD-Turbo một bước ở t = 999, dùng `x0` dạng đóng; LoRA (UNet r8, VAE r4); `conv_in` train toàn bộ; 4 skip conv 1×1 khởi tạo 0 từ encoder sang decoder của VAE | **pix2pix-turbo** (Parmar et al. 2024). Được viết lại, không chép code (xem 2.4) |
| Mở rộng kênh vào bằng cột trọng số khởi tạo 0 để thêm điều kiện | **DWM** (`_extend_patch_embedding`) |
| Trọng số loss ×10 ở tay và mặt | **SignRefine** Eq. 4 (`w_h = w_f = 10`) |
| Trộn tách tần số: tần số thấp giữ của ảnh gốc | **CraftsMan3D** (K4 trong `docs/bai_hoc_diffusion_mesh_cho_greg.md`) |
| Loss thời gian qua tương ứng bề mặt | **Mesh4D / Generative Rendering** (bai_hoc §3.2) |

### 2.3 Phần là thiết kế riêng của luận văn 🛠
1. **Refiner đặt sau một mô hình hồi quy có hình học.** SignRefine tinh chỉnh một mô hình video nền với điều kiện keypoint 2D. D1 thì giữ GReg và chỉ "vá" ảnh của nó. Diffusion vì vậy không phải tự học danh tính hay nền.
2. **Cổng theo hình học mesh.** Vùng được sửa là `dilate(M_tay ∪ M_mặt) ∪ (thân ∩ (1 − V))`, với V là độ nhìn thấy suy từ tương ứng mesh nguồn–đích. Cổng tính được lúc suy luận vì chỉ dùng mesh đích và mesh nguồn, không dùng ảnh thật (`greg/diffusion/gating.py:21-33`).
3. **Điều kiện hình học đi vào latent.** Normal map đích được mã hoá bằng VAE đóng băng (4 kênh latent). Bốn mặt nạ `[V·C, mặt, tay trái, tay phải]` được average-pool về lưới latent 64². Tất cả nối thêm vào `conv_in` qua cột khởi tạo 0 (`turbo_refiner.py:130-135, 190-201`).
4. **Loss thời gian trên hai khung đích liên tiếp.** Hai khung được nối bằng `correspond()` của mesh, để chi tiết refiner tự vẽ phải di chuyển theo bề mặt (`refiner_losses.py:82-91`).
5. **Cặp train lấy từ GReg epoch_040** thay vì epoch_100. Lý do: epoch_100 đã "nhớ" tập train, nên ảnh thô của nó đẹp hơn lúc test (`configs/diffusion/d1_refiner.yaml:8`).
6. **Hạ mẫu bằng `area`** từ 512 về 256 trước khi tính loss và chấm điểm. Bilinear sẽ gây răng cưa và làm LPIPS đẹp giả (`gating.py:66-72`).

### 2.4 Khác pix2pix-turbo ở mức code
Repo gốc (@`86f5414`) cần `diffusers 0.25.1` vì monkey-patch hàm `forward` của VAE. Ta đã đọc từng dòng và ghi khoảng 20 lỗi ngầm (`docs/code_notes_pix2pix_turbo.md`). D1 viết lại khoảng 200 dòng trên diffusers/peft hiện hành, với các điểm sửa sau (`turbo_refiner.py:14-21`):
- Skip connection gắn bằng `register_forward_pre_hook`.
- Dùng `latent_dist.mode()` thay vì `sample()`, để không thêm nhiễu và không nhấp nháy.
- Đầu vào và đích cùng thang [−1, 1].
- Đặt `lora_alpha = r` tường minh.
- Prompt embedding tính một lần, sau đó bỏ text encoder khỏi bộ nhớ.
- Encode normal map với `capture=False`, để normal map không ghi đè skip activation của ảnh đang được sửa (`turbo_refiner.py:178-185, 195`).

---

## 3. Kiến trúc chi tiết

### 3.1 Sơ đồ (kích thước tensor cho batch B)

```
Đầu vào (cache 256²)                                   work_size = 512 (d1_refiner.yaml:19)
Î_GReg (B,3,256,256) ─resize bicubic─► (B,3,512,512) ─[VAE enc + LoRA r4, có ghi skip]─► z (B,4,64,64)·0.18215
N_t    (B,3,256,256) ─resize bicubic─► (B,3,512,512) ─[VAE enc, no_grad, không ghi skip]─► z_N (B,4,64,64)
masks  (B,4,256,256) = [V·C, mặt, tay T, tay P] ─resize─► 512 ─area-pool─► (B,4,64,64), đổi sang [-1,1]
                                       │
                      concat ──► (B,12,64,64)
                                       │  UNet SD-Turbo, conv_in 4→12 kênh (8 cột mới = 0), LoRA r8
                                       │  t = 999, text = embedding của prompt cố định (B,77,1024)
                                       ▼
                               ε̂ (B,4,64,64)
                x0 = (z − √(1−ᾱ)·ε̂)/√ᾱ ,  ᾱ = ᾱ_999 ≈ 0.0047   (khuếch đại sai số ε̂ khoảng 14.6×)
                                       ▼
          VAE dec + LoRA r4 ; 4 skip 1×1 (khởi tạo 0) cộng vào đầu mỗi up-block:
             enc 64²×512 → dec 64²×512 · enc 128²×256 → dec 128²×512
             enc 256²×128 → dec 256²×512 · enc 512²×128 → dec 512²×256
                                       ▼
                    R (B,3,512,512) ─area─► (B,3,256,256)
                                       ▼
   Cổng G (B,1,256,256) = dilate6(tay ∪ mặt) ∪ dilate(thân·(1−V))  → Blend (soft / hard / freq / none)
                                       ▼
                                Î (B,3,256,256)
```

### 3.2 Từng khối

**a. Đầu vào** (`refiner_data.py:64-75`)
- Ảnh GReg lấy từ export, normal map đích, part map với nhãn 0 nền / 1 thân / 2 mặt / 3 tay trái / 4 tay phải.
- `vis` (V) và `vc` (V·C) là các PNG do `20_export_greg.py` sinh từ tương ứng mesh. C là độ tin cậy của tương ứng.
- Thêm khớp 2D và tâm các vùng.
- Khi bật `pairs=true`, mỗi mẫu gồm hai khung đích liên tiếp `(t−1, t)` dùng cùng khung nguồn.

**b. Encoder VAE.** VAE của SD-Turbo, gắn LoRA r4 vào các conv và attention (`turbo_refiner.py:35, 127-129`). Dùng `mode()` của phân phối hậu nghiệm. Ảnh GReg được encode **có ghi** activation ở đầu 4 down-block để làm skip. Normal map được encode **không ghi**, dưới `no_grad`, và chỉ đi vào UNet.

**c. UNet SD-Turbo.**
- LoRA r8 gắn vào `to_q/k/v/out`, các conv, `proj_in/out` và FFN (`turbo_refiner.py:33-34, 124-126`).
- `conv_in` được thay bằng bản rộng hơn 4 + 4 + 4 = 12 kênh. 4 kênh gốc giữ trọng số SD-Turbo, 8 kênh mới khởi tạo 0, và cả lớp được train (`turbo_refiner.py:39-48, 130-135`). Nhờ khởi tạo 0, ở bước 0 refiner tương đương SD-Turbo gốc, và normal map hay mặt nạ không làm đổi đầu ra. Unit test kiểm điều này (`tests/test_d1_refiner.py`).
- Chỉ một timestep t = 999. Prompt cố định là "a person signing in sign language, sharp hands and fingers, studio video, uniform lighting" (`turbo_refiner.py:36`). Không dùng caption từ ground truth, để tránh rò thông tin.

**d. Decoder VAE và skip** (`turbo_refiner.py:51-100`). Bốn conv 1×1 không bias, khởi tạo 0, đưa activation của encoder (đảo thứ tự) cộng vào đầu vào của từng up-block. Nhờ đó chi tiết pixel của ảnh GReg vượt qua được nút thắt latent 8×. Tính theo kênh, các skip có khoảng 0.49 M tham số.

**e. Cổng và phép trộn** (`gating.py:21-56`). Cổng G là mặt nạ nhị phân:
- Tay và mặt giãn 6 px.
- Cộng thêm phần thân có V = 0, cũng giãn.
- Làm mềm bằng Gaussian với σ = feather/2 = 2 px.

Bốn kiểu trộn:

| Kiểu | Công thức |
|---|---|
| `none` | Î = R |
| `hard` | Î = (1−G)·GReg + G·R |
| `soft` | Như `hard` nhưng dùng G đã làm mềm |
| `freq` | Î = low(GReg) + (1−G)·high(GReg) + G·high(R), với low = Gaussian σ = 8 px và high = x − low |

Với `freq`, **tần số thấp (màu, sáng tối) luôn lấy của GReg ở mọi nơi**. Trong cổng, refiner chỉ thay phần tần số cao.

**f. Tham số train và tham số đóng băng.** Toàn bộ SD-Turbo (VAE và UNet) đóng băng, chỉ train LoRA, `conv_in` và 4 skip. Kế hoạch ghi khoảng 9.6 M tham số train cho cấu hình pix2pix-turbo ✅ (`diffusion_6_huong_ke_hoach.md:148`), cộng thêm 8 cột mới của `conv_in` (khoảng 23 k). Con số đếm chính xác trên checkpoint không còn lưu.

### 3.3 Loss (`refiner_losses.py`, `d1_refiner.yaml:33-39`)

Loss tính trên ảnh refiner **chưa trộn cổng**, sau khi hạ về 256² (`22_train_refiner.py:187-196`):

```
w = 1 + 9·M_tay + 9·M_mặt + 1·thân·(1−V)                        (mặt nạ giãn 3 px)
L = 1.0·L2_w + 5.0·LPIPS-VGG_w(spatial) + 1.0·[L2 + LPIPS]_crop tay + 0.5·L_temp
```

- `L2_w` và `LPIPS_w` là trung bình có trọng số theo `w`. LPIPS dùng VGG ở chế độ `spatial=True`, để bản đồ LPIPS cũng nhân được với `w` (`refiner_losses.py:35-45, 66-80`; `22_train_refiner.py:140`).
- Crop tay: hộp thích ứng theo khớp tay, cạnh 64, phóng lên 128². Chỉ tính tay nhìn thấy và có kích thước ≥ 6 px (`refiner_losses.py:47-64`).
- Loss thời gian: `L_temp = ‖V·C ⊙ (Î_t − warp(Î_{t−1}, F_{t−1→t}))‖₁ / Σ V·C`, với F và V·C lấy từ `correspond()` giữa hai mesh đích (`refiner_losses.py:82-91`).
- Không dùng GAN, không dùng loss CLIP.

### 3.4 Dữ liệu train
- Ảnh vào là export của GReg run5 **epoch_040** trên `train_fit`, tức tập train trừ đi các clip val (`d1_refiner.yaml:8-10`). Ảnh đích là khung thật lấy từ cache.
- Val trong lúc train dùng export epoch_100 trên 40 clip val × 4 khung (q160). Kết quả val được tính sau khi đã trộn cổng `soft` (`22_train_refiner.py:67-88`).
- Kiểm tra "lỗi của epoch_040 trên train có gần lỗi của epoch_100 trên val không" (kế hoạch §3.4) **chưa được chạy**.

### 3.5 Siêu tham số (`d1_refiner.yaml:41-53`) và chi phí đo được

| | run3 (bản đầy đủ) | run1, run2 (ablation) |
|---|---|---|
| Số bước | 5 000 | 3 000 |
| Micro-batch × tích luỹ | 1 cặp (2 khung) × 4 | 1 khung × 4 (`pairs=false`) |
| LR, warmup, weight decay, clip | AdamW 2e-5, 200, 0.01, 1.0 | như run3 |
| Độ chính xác | bf16 autocast, gradient checkpointing | như run3 |
| Tốc độ / VRAM trên 4090 | 2.95 s/bước, 17.2 GB ✅ | 2.51 s/bước, 12.7 GB ✅ (log run2) |
| Thời gian train | khoảng 4.1 h GPU | khoảng 2.1 h GPU mỗi run |

- 2 cặp mỗi micro-batch thì hết VRAM (> 23 GB). Vì vậy dùng 1 cặp và tích luỹ 4 bước (commit `5c12dc7`).
- Cả giai đoạn D1 (run 0, ba lần train, các lần chấm điểm) mất khoảng 10.2 h tính theo đồng hồ, từ 01:41 đến 11:54 UTC, tức khoảng $6.3.
- Thời gian suy luận mỗi khung (`timing.json` của `23_refine_frames.py`): **không còn lưu**.

### 3.6 Suy luận
`23_refine_frames.py` chạy một lần forward cho mỗi khung, tất định. Script ghi `<out>/<blend>/<key>/<row>.png` và chép `index.json` của export, để chấm điểm đúng các khung đã xuất. Không có lấy mẫu ngẫu nhiên và không có CFG.

---

## 4. Giao thức đánh giá

- **Cách chấm:** `scripts/21_eval_frames.py` (bọc `greg/eval/evaluate.py` bằng `FrameDirModel`). Mọi chỉ số tính ở 256², với cùng khung nguồn và cùng `valid_range` như GReg.
- **Các chỉ số:**
  - L1, PSNR, SSIM, LPIPS (AlexNet, giống bài báo).
  - `hand_l1`, `hand_lpips`: crop tay theo hộp thích ứng, phóng lên 128².
  - `face_l1`.
  - `lowfreq_drift`: L1 sau Gaussian σ = 8, tính ngoài vùng tay.
  - `disocc_l1`: L1 trên phần thân có V = 0.
  - `arm_l1`.
  - Các bin chuyển động `@small/@medium/@large` theo độ dịch cổ tay.
- **Mốc so sánh:** GReg run5 epoch_100, chấm lại bằng cùng script trên **cùng các khung**. Cách chấm mới khớp số run5 đã công bố trong phạm vi 0.4 % trên test-full ✅.
- **Thống kê:** bootstrap ghép cặp theo clip, 2 000 lần lặp lại mẫu, khoảng tin cậy 95 %. Đơn vị lặp là clip, vì các khung trong một clip tương quan với nhau. Áp dụng cho u16 và s200.
- **Lưu ý về q160:** con số tuyệt đối trên q160 thấp hơn nhiều so với val-full (GReg 0.0341 so với 0.0484), vì cách lấy 4 khung mỗi clip. Chỉ dùng q160 để so tương đối.

---

## 5. Kết quả

### 5.1 Run 0: diffusion không huấn luyện (q160)
Cấu hình: SD1.5 + ControlNet-Tile img2img, cùng prompt, seed cố định theo (clip, khung), cổng `soft` hoặc không cổng (`24_refine_training_free.py`).

| | L1 | PSNR | LPIPS | hand L1 | hand LPIPS | trôi màu |
|---|---|---|---|---|---|---|
| GReg run5 | **0.0107** | **32.88** | **0.0341** | **0.0835** | **0.2033** | **0.0039** |
| độ mạnh 0.3, cổng soft | 0.0122 | 29.72 | 0.0404 | 0.0903 | 0.2252 | 0.0045 |
| độ mạnh 0.5, cổng soft | 0.0128 | 29.01 | 0.0411 | 0.0944 | 0.2337 | 0.0046 |
| độ mạnh 0.3, không cổng | 0.0232 | 27.59 | 0.0575 | 0.0904 | 0.2333 | 0.0147 |
| độ mạnh 0.5, không cổng | 0.0265 | 26.57 | 0.0642 | 0.0945 | 0.2432 | 0.0169 |

### 5.2 Đường val trong lúc train run3 (q160, cổng soft; GReg: L1 0.0140 / LPIPS 0.0336 / hand LPIPS 0.1993)

| bước | 20 | 1 000 | 2 000 | 3 000 | 4 000 | 5 000 |
|---|---|---|---|---|---|---|
| LPIPS | 0.1324 | 0.0353 | 0.0336 | khoảng 0.0329–0.0332 | 0.0329 | 0.0330 |
| hand LPIPS | 0.4034 | 0.1985 | 0.1936 | khoảng 0.190 | **0.1885** | 0.1894 |
| L1 | 0.0341 | 0.0151 | 0.0145 | 0.0143 | 0.0143 | 0.0143 |

- Ở bước 20 ảnh rất tệ (LPIPS 0.13), vì cột mới và skip mới bắt đầu học. Đến bước 1 000 refiner đã ngang GReg, đến bước 4 000 thì đường val chững lại.
- Ở bước 3 000, log của run1, run2 và run3 nằm lẫn với nhau nên chỉ ghi được khoảng giá trị.
- L1 của refiner trên q160 (0.0143) luôn cao hơn GReg (0.0140).

### 5.3 Run3 với các kiểu cổng, và ablation run1/run2 (val-full, 14 198 khung)

| Cấu hình | L1 | PSNR | SSIM | LPIPS | hand L1 | hand LPIPS | face L1 | disocc L1 | trôi màu | arm L1 | hand L1@large | LPIPS@large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GReg run5** | 0.0173 | **28.94** | 0.9149 | 0.0484 | 0.1192 | 0.2470 | **0.0315** | **0.0547** | 0.0084 | 0.0537 | 0.1455 | 0.0706 |
| **run3 + freq** | **0.0173** (+0.1 %) | 28.83 | 0.9151 | **0.0462 (−4.6 %)** | **0.1169 (−2.0 %)** | **0.2297 (−7.0 %)** | 0.0332 (+5 %) | 0.0553 (+1.2 %) | **0.0084** | **0.0528** | **0.1410** (−3.1 %) | 0.0663 (−6.1 %) |
| run3 + soft | 0.0174 | 28.72 | 0.9153 | 0.0463 (−4.2 %) | 0.1178 | 0.2297 | 0.0340 (+8 %) | 0.0548 | 0.0085 | 0.0532 | 0.1420 | 0.0665 |
| run3 + hard | 0.0174 | 28.69 | 0.9143 | 0.0462 | 0.1178 | 0.2305 | 0.0340 | 0.0548 | 0.0085 | 0.0534 | 0.1420 | 0.0662 |
| run3 + none | 0.0195 (+12.4 %) | 28.21 | **0.9165** | 0.0466 | 0.1178 | **0.2295** | 0.0340 | 0.0548 | 0.0112 (+32.9 %) | 0.0551 | 0.1420 | **0.0654** |
| run1: chỉ ảnh, 3 000 bước, soft | 0.0174 | 28.71 | 0.9149 | 0.0468 (−3.3 %) | 0.1183 | 0.2350 (−4.8 %) | 0.0346 | 0.0548 | 0.0085 | 0.0534 | 0.1433 | 0.0673 |
| run2: + normal + mask, 3 000 bước, soft | 0.0174 | 28.68 | 0.9149 | 0.0468 (−3.2 %) | 0.1178 | 0.2346 (−5.0 %) | – | 0.0546 | 0.0085 | 0.0532 | 0.1421 | 0.0672 |

Run3 = run2 + loss thời gian + 5 000 bước. Ở run3 + hard, hand LPIPS@large là 0.2506 (GReg 0.2704, tức −7.3 %).

### 5.4 Trên tập test, cổng freq (bootstrap ghép cặp theo clip)

**u16** (1 146 clip × 16 khung):

| | L1 | PSNR | SSIM | LPIPS | region L1 | hand L1 | hand LPIPS | face L1 | trôi màu |
|---|---|---|---|---|---|---|---|---|---|
| GReg run5 | 0.01716 | **28.96** | 0.9150 | 0.04868 | **0.0455** | 0.1180 | 0.2519 | **0.0315** | 0.00820 |
| D1 (run3 freq) | 0.01716 (n.s.) | 28.86 | **0.9153** | **0.04640** | 0.0458 | **0.1155** | **0.2337** | 0.0332 | 0.00819 |
| Δ, khoảng tin cậy 95 % | +0.0 %, p = 0.12 | −0.10 dB [−0.12, −0.09] | | **−4.7 %** [−0.00239, −0.00217] | | | **−7.2 %** [−0.0189, −0.0174] | **+5.5 %** [+0.0016, +0.0018] | −0.2 % |

**s200** (cùng các khung với D5 và D6):

| | L1 | PSNR | LPIPS | hand LPIPS | face L1 | trôi màu |
|---|---|---|---|---|---|---|
| GReg run5 | **0.01648** | **29.61** | 0.04758 | 0.2365 | **0.0304** | 0.00774 |
| D1 | 0.01652 (+0.2 %, kém hơn có ý nghĩa) | 29.33 (−0.27 dB) | **0.04535 (−4.7 %)** | **0.2193 (−7.2 %)** | 0.0324 (+6.6 %) | 0.00773 |

### 5.5 D1 trên mô hình khác, và D1 ghép với tay D6

**D1 trên GReg train với D4 aug 100 epoch** (u16; refiner giữ nguyên, **không train lại**):

| | L1 | PSNR | LPIPS | hand L1 | hand LPIPS | face L1 |
|---|---|---|---|---|---|---|
| GReg run5 | **0.01716** | **28.96** | 0.04868 | 0.1180 | 0.2519 | **0.0315** |
| GReg D4aug | 0.01728 | 29.09 | 0.04919 | 0.1145 | 0.2424 | 0.0317 |
| D1 trên D4aug | 0.01733 (+1.0 %) | 28.92 | **0.04676 (−3.9 %)** | **0.1139** | **0.2279 (−9.5 %)** | 0.0334 (+6.3 %) |

Δ so với GReg run5; mọi Δ có ý nghĩa thống kê.

**Ghép tay D6 `ng` vào ảnh D1** (s200). Hộp tay tính từ tư thế đích, viền làm mềm, ảnh cuối = m·D6 + (1−m)·nền:

| | L1 | PSNR | SSIM | LPIPS | region L1 | hand LPIPS | face L1 |
|---|---|---|---|---|---|---|---|
| GReg run5 | 0.01648 | 29.61 | 0.9179 | 0.04758 | 0.0437 | 0.2365 | 0.0304 |
| D1 | 0.01652 | 29.33 | – | 0.04535 | – | 0.2193 | 0.0324 |
| GReg + tay D6 | **0.01564** | **29.95** | **0.9202** | 0.04328 (−9.0 %) | **0.0389** | **0.1841** | **0.0302** (n.s.) |
| **D1 + tay D6** | 0.01576 (−4.3 %) | 29.67 (n.s.) | 0.9199 | **0.04236 (−11.0 %)** | 0.0400 | **0.1841** (−22 %) | 0.0322 (+6.0 %) |

---

## 6. Phân tích nguyên nhân và hiện tượng

**6.1 Vì sao LPIPS và hand LPIPS giảm: refiner chủ yếu "làm sắc".**
- Bằng chứng ✅:
  - (i) run1 **chỉ nhận ảnh GReg**, không có hình học, mà đã lấy được khoảng 70 % lợi ích LPIPS (−3.3 % so với −4.6 %).
  - (ii) Mức giảm ở hand **LPIPS** (−7 %) lớn gấp khoảng 3.5 lần mức giảm ở hand **L1** (−2 %). LPIPS nhạy với texture và cạnh, còn L1 nhạy với việc pixel đặt đúng chỗ.
  - (iii) Lợi ích lớn nhất ở chuyển động lớn (LPIPS@large −6.1 %, hand LPIPS@large −7.3 %), là nơi GReg mờ nhất.
- Giải thích ⚠️: prior của SD-Turbo biết "ngón tay và cạnh sắc trông thế nào". Với đầu vào là ảnh GReg đã đúng bố cục, refiner chỉ cần phục hồi tần số cao. Đây là bài toán gần với khử mờ hơn là sinh mới.

**6.2 Vì sao face L1 tăng 5–8 %.**
- Bằng chứng ✅:
  - face L1 tăng ở mọi kiểu cổng có mở vùng mặt (soft, hard: 0.0340; freq: 0.0332).
  - Khi ghép tay D6 lên **GReg**, face L1 giữ nguyên (0.0302, n.s.). Khi ghép lên **D1**, face L1 tăng 6 %. Vậy lỗi mặt là của D1, không phải của bước ghép.
  - D5 và D6, hai mô hình cũng đi qua VAE của Stable Diffusion, đều kém ở mặt (D6 ng +20 %).
- Hai nguyên nhân cộng lại:
  - (a) **Cổng mở cả mặt** (`include_face: true`, `d1_refiner.yaml:28`), nên mọi thay đổi của refiner trên mặt đều được giữ lại.
  - (b) ⚠️ **VAE của SD nén 8×**. Mặt người ký chỉ khoảng 30–40 px ở 256², tức khoảng 8–10 ô latent ngay cả sau khi phóng lên 512. Mắt, miệng và đường viền khó tái tạo đúng từng pixel, nên refiner "vẽ lại" một khuôn mặt hợp lý nhưng lệch pixel. Trọng số ×10 ở mặt trong loss không bù được giới hạn này.
- Kiểm chứng rẻ nhất là chấm lại với `include_face=false`, không cần train lại. Thí nghiệm này **chưa chạy**.

**6.3 Vì sao PSNR giảm: đánh đổi perception–distortion.**
- Bằng chứng ✅: PSNR giảm 0.10 dB (u16) đến 0.27 dB (s200), trong khi LPIPS giảm. Với cổng `none`, PSNR giảm 0.74 dB, nhưng SSIM lại cao nhất (0.9165) và LPIPS@large tốt nhất.
- Giải thích ⚠️: theo Blau & Michaeli (2018), một bộ ước lượng sinh chi tiết sắc phải trả giá bằng MSE, vì chi tiết đó không trùng pha với ground truth. GReg hồi quy về trung bình có điều kiện nên tối ưu cho MSE.

**6.4 Vì sao cổng `freq` tốt nhất, và vì sao bắt buộc phải có cổng.**
- Bằng chứng ✅:
  - Không cổng: L1 +12.4 %, độ trôi màu +32.9 %, arm L1 +2.7 %, nhưng LPIPS không tốt hơn cổng freq.
  - `freq` có L1 và độ trôi màu bằng GReg (0.0084), PSNR chỉ −0.11 dB so với −0.22 dB của `soft`.
- Giải thích:
  - Refiner làm lệch **màu và độ sáng tần số thấp** trên toàn ảnh, cả ngoài vùng cần sửa ⚠️: VAE và UNet SD có xu hướng dời màu, và loss chỉ có trọng số 1 ở vùng nền.
  - Cổng chặn lỗi ngoài vùng sửa. `freq` còn chặn lỗi tần số thấp **bên trong** vùng sửa, vì chỉ nhận phần chi tiết. Đó đúng là phần refiner làm tốt hơn GReg (6.1).
- `soft` và `hard` gần như bằng nhau, vì cổng đã giãn 6 px nên viền không ảnh hưởng nhiều.

**6.5 Vì sao normal map và mặt nạ không giúp.**
- Bằng chứng ✅: run1 (chỉ ảnh) và run2 (ảnh + normal + 4 mặt nạ) cùng 3 000 bước, cùng cổng: LPIPS 0.0468 so với 0.0468, hand LPIPS 0.2350 so với 0.2346.
- Giải thích ⚠️, chưa tách riêng từng yếu tố:
  - (a) Ảnh GReg **đã mang tư thế**, vì nó được dựng từ chính normal map đó. Normal map không thêm thông tin mà refiner chưa có.
  - (b) Ở lưới latent 64², một bàn tay chỉ chiếm khoảng 4–6 ô, và normal map đi qua VAE (vốn train trên ảnh tự nhiên) mất chi tiết ngón. Điều kiện đến UNet vì vậy quá thô để sửa hình học ngón.
  - (c) Các cột khởi tạo 0, với LR 2e-5 và 3 000 bước, có thể chưa học kịp cách dùng điều kiện mới, trong khi đường "làm sắc ảnh" có sẵn từ prior.
- Hệ quả: D1 hiện là bộ **phục hồi ảnh**, không phải bộ **sửa hình học**. Điều đó khớp với 6.1 và với việc `disocc_l1` không giảm (6.7).

**6.6 Run3 so với run1/run2: không tách được tác dụng của loss thời gian.**
- Run3 khác run2 ở **hai** yếu tố: có loss thời gian, và 5 000 bước thay vì 3 000.
- Trên q160 ở bước 3 000, ba run gần như bằng nhau (LPIPS khoảng 0.0329–0.0332). Đường val của run3 vẫn giảm đến bước 4 000.
- Vậy phần hơn của run3 trên val-full (−4.6 % so với −3.2 %) có thể chỉ do train lâu hơn.
- Độ ổn định thời gian (FVD, `ccm_tc`) **chưa được đo** cho D1, nên chưa có bằng chứng loss thời gian giảm nhấp nháy.

**6.7 Vì sao vùng bị che (disocc) không được sửa.**
- Bằng chứng ✅: `disocc_l1` bằng GReg (0.0547 so với 0.0548 với soft/hard) hoặc kém hơn một chút (0.0553 với freq).
- Giải thích ⚠️:
  - Vùng V = 0 là nơi **không có thông tin** từ ảnh nguồn. Muốn sửa phải tạo nội dung mới đúng với ground truth (ví dụ mặt trong cánh tay, phần áo bị che), mà prior chung không biết.
  - Trọng số loss ở vùng này chỉ ×2 (1 + `w_disocc`=1), so với ×10 ở tay.
  - Với `freq`, phần tần số thấp ở đây giữ của GReg, mà ở vùng disocc lỗi chủ yếu là màu và khối, tức tần số thấp. Đây là giới hạn cố hữu của `freq`.

**6.8 Vì sao diffusion không huấn luyện (run 0) thua.**
- Bằng chứng ✅: ngay cả với cổng, run 0 kém GReg ở L1, LPIPS và hand LPIPS. Độ mạnh lớn hơn (0.5) thì kém hơn nữa. Không cổng thì độ trôi màu tăng 3.8–4.3 lần.
- Giải thích ⚠️: img2img thêm nhiễu rồi khử nhiễu theo prior chung. Prior không biết danh tính người ký, và ControlNet-Tile chỉ giữ bố cục, không giữ chi tiết. Kết quả là "một bàn tay hợp lý" nhưng không phải bàn tay đúng.
- Kết luận cho luận văn: **phải huấn luyện có cặp** (paired). Đây là câu trả lời cho câu hỏi "sao không dùng thẳng diffusion pretrained".

**6.9 Kết quả ổn định giữa các tập, và có tổng quát hoá.**
- ✅ Mức giảm LPIPS gần như giống nhau trên val-full, u16 và s200 (−4.6 / −4.7 / −4.7 %), hand LPIPS cũng vậy (−7.0 / −7.2 / −7.2 %). Khoảng tin cậy hẹp và không chứa 0.
- ✅ Áp nguyên refiner lên một GReg khác (D4aug) mà không train lại vẫn cho LPIPS −3.9 % và hand LPIPS −9.5 %. Lợi ích LPIPS nhỏ hơn trên run5 (−4.7 %).
  - ⚠️ Có thể do lệch phân phối: refiner học trên lỗi của run5 epoch_040.
  - Hand LPIPS tốt nhất (0.228) là tổng của hai cải thiện ở tay (D4aug −3.8 % và D1).

**6.10 Ghép với tay D6 cho thấy D1 còn có ích ngoài vùng tay.**
- ✅ D1 + tay D6 tốt hơn GReg + tay D6 ở LPIPS (0.0424 so với 0.0433), dù hai bản có **cùng** vùng tay.
- Vậy khoảng 2 % LPIPS còn lại đến từ phần D1 sửa **ngoài tay** (mặt, thân bị che, viền). Đổi lại face L1 +6 %. Đây là cùng một đánh đổi như 6.2 và 6.3.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế:**
1. Mới chấm test trên u16 và s200. **Chưa chấm test-full**, chưa có FVD/AKD và chưa có `ccm_tc` (nhấp nháy). Điểm cuối cùng là chưa đo tác dụng của loss thời gian.
2. Mỗi cấu hình chỉ train với một seed.
3. Ablation run3 so với run2 bị lẫn số bước (6.6).
4. Lỗi ở mặt (6.2) chưa có thí nghiệm kiểm chứng.
5. Chưa kiểm phân phối lỗi của epoch_040 trên train so với epoch_100 trên val.
6. Thời gian suy luận mỗi khung không còn lưu. Cần đo lại để điền cột chi phí.

**Việc nên làm tiếp**, xếp theo chi phí tăng dần:
1. **Chấm lại với cổng chỉ ở tay** (`include_face=false`, tuỳ chọn bỏ cả disocc). Chỉ cần suy luận, khoảng 1 h. Checkpoint run3 vẫn còn trên Drive. Kỳ vọng ⚠️: face L1 về mức GReg mà vẫn giữ phần lớn mức giảm LPIPS.
2. Chấm D1 trên **test-full**, thêm FVD và `ccm_tc`.
3. Train run2 thêm đến 5 000 bước, để tách tác dụng của loss thời gian.
4. Nếu muốn D1 sửa hình học chứ không chỉ làm sắc: tăng trọng số disocc, đưa điều kiện hình học ở độ phân giải cao hơn (crop tay 128² riêng), hoặc thử train ở 256 trực tiếp.

---

## 8. Trạng thái dữ liệu (sau khi xoá instance ngày 2026-11-05)

**Còn trên Drive** (`GReg_runs/diff_common`, đồng bộ trước 2026-11-04 02:43 UTC):
- Checkpoint refiner run3.
- Kết quả chấm val của D1 (`results_val.md`, các `summary.json`).
- Log train.
- ⚠️ Không chắc checkpoint run1 và run2 có được đồng bộ hay không.

**Đã mất:**
- Mọi thư mục ảnh của D1 (`frames_val/*`, ảnh test). Các thư mục này bị loại khỏi đồng bộ ngay từ đầu, vì có khoảng 600 k file PNG.
- Điểm test u16 và s200 của D1, D1 trên D4aug, D1 + tay D6 (`per_frame.csv`, `bootstrap.md`). Các kết quả này sinh ra sau khi Drive đã đầy.
- Các con số tổng hợp ở mục 5 còn lại trong lịch sử hội thoại và memory. Bootstrap không thể tính lại nếu không chạy lại.
