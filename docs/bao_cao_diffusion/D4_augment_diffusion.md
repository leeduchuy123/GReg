# Báo cáo D4: Tăng cường dữ liệu bằng diffusion, nhất quán theo clip (kèm phát hiện lỗi "cặp cố định")

Nhánh: `feat/diff-d4-augment` (commit `d70b959`, sửa đánh giá `5d20a58`). Kế hoạch gốc: `docs/diffusion_6_huong_ke_hoach.md` §6.
Ngày chạy: sweep 2026-11-03 20:14–23:28 UTC; bản 100 epoch 2026-11-04 05:27–17:19 UTC. Máy: vast.ai RTX 4090 (instance 58802139, đã xoá).

Quy ước: ✅ đã kiểm chứng (code, log hoặc số đo) · ⚠️ giả thuyết hoặc suy luận chưa đo riêng.
Mọi trích dẫn code dạng `path:dòng` là trên nhánh `feat/diff-d4-augment`, trừ khi ghi khác.

---

## 0. Tóm tắt

- **Ý tưởng:** không đổi kiến trúc GReg. Chỉ đổi **dữ liệu train**:
  - thay nền bằng ảnh do Stable Diffusion 1.5 sinh (offline, 600 ảnh);
  - đổi màu người (quần áo và da).
  - Mọi lựa chọn ngẫu nhiên được rút **một lần cho mỗi (epoch, clip)** và áp giống hệt cho khung nguồn và khung đích, để cặp vẫn "cùng một người, cùng một cảnh".
  - Diffusion không chạy lúc suy luận, nên chi phí suy luận bằng đúng GReg.
- **Phát hiện phụ quan trọng ✅:** `PairDataset.set_epoch()` không tới được các DataLoader worker chạy `persistent_workers`. Vì vậy **mọi epoch dùng lại đúng một tập cặp (nguồn, đích)**, kể cả khi train run5. Đã sửa (tuỳ chọn bật, mặc định tắt) bằng biến epoch trong bộ nhớ chia sẻ.
- **Kết quả:**
  - Vòng lọc 15 epoch: `aug` thắng rõ, LPIPS −6 % so với `base` và −16 % so với `fresh`. Trên nền chưa thấy khi train: LPIPS −7.6 %, L1 −14 %, trôi màu −27 % so với `base`.
  - Train đủ 100 epoch và chấm trên toàn bộ test: **lợi thế mất**. LPIPS 0.0485 → 0.0490 (+0.9 %, kém hơn có ý nghĩa thống kê). Chỉ còn hand LPIPS −3.8 % và PSNR +0.11 dB là tốt hơn có ý nghĩa.
- **Kết luận:** D4 **không** cải thiện tái tạo trong phân phối, nên không đưa vào cấu hình cuối. Nó cho một tín hiệu nhỏ nhưng nhất quán ở tay, và một bằng chứng (chỉ ở 15 epoch) về độ bền với nền lạ. Phát hiện lỗi "cặp cố định" là đóng góp độc lập: cần nêu trong luận văn vì nó áp dụng cho chính baseline.

---

## 1. Bối cảnh

### 1.1 Baseline GReg run5
- Kiến trúc: GReg với TPSMM đóng băng (49.2 M tham số flow), bộ sinh và Regional Expert (34.1 M tham số train), discriminator 2.0 M ✅ (log train 100 epoch).
- Cấu hình: `qk_norm=true`, `regional_perceptual=3`, 100 epoch, batch 4, lr 2e-4, milestone 70/90 ✅ (`run5_config.yaml` lấy từ Drive).
- Số test do chính ta đo lại trên cùng máy ✅: L1 0.01728, PSNR 29.19, SSIM 0.9142, LPIPS 0.04851, hand LPIPS 0.2624, face L1 0.0311, trôi màu 0.0083. Số cũ đã báo là L1 0.01735, LPIPS 0.04837; chênh do PNG 8-bit.

### 1.2 Vấn đề cần giải quyết
- HamNoSys có khoảng 100 người ký, quay trong vài phông studio cho mỗi corpus nguồn. GReg học "người + nền" cố định, nên có thể yếu khi gặp **người mới, nền mới** (giao thức cross-person của luận văn).
- Pose-guided synthesis có một ràng buộc riêng: ảnh đích phải **cùng ngoại hình** với ảnh nguồn, vì mô hình chép ngoại hình từ nguồn. Augmentation thông thường (đổi màu hay nền từng ảnh độc lập) sẽ tạo ra cặp "nguồn áo xanh, đích áo đỏ". Cặp đó dạy mô hình bịa màu thay vì chép màu, tức là phá đúng cơ chế cần học.

### 1.3 Vì sao chọn hướng này trong chương trình diffusion
Trong sáu hướng, D4 là hướng duy nhất dùng diffusion **hoàn toàn offline**:
- SD chỉ sinh ảnh nền một lần, mất khoảng 5 phút GPU.
- Không đụng tới kiến trúc hay chi phí suy luận.
- Chi phí dự kiến thấp nhất, khoảng 3 USD.

Nếu thắng, đây là cải tiến "miễn phí" lúc suy luận.

---

## 2. Ý tưởng và điểm đặc sắc

### 2.1 Ý tưởng
Với mỗi clip, ở mỗi epoch, rút một lần:
1. Có thay nền không (xác suất 0.3); nếu có, dùng ảnh nào trong 500 nền SD dành cho train.
2. Có đổi màu người không (xác suất 0.5); nếu có, bốn hệ số độ sáng, tương phản, độ bão hoà, hue.

Sau đó áp **cùng** các lựa chọn đó cho khung nguồn và khung đích của cặp. Người được tách khỏi nền bằng một mặt nạ mềm (matte), kết hợp phân đoạn ngữ nghĩa và mặt nạ thân SMPL-X.

### 2.2 Điểm mới và đặc sắc
| # | Điểm | Mượn hay tự thiết kế |
|---|---|---|
| 1 | **Ràng buộc nhất quán theo clip** cho pose-guided synthesis: RNG gieo bằng chuỗi `"{seed}/{epoch}/{key}"` (`greg/data/augment.py:82`), nên mọi worker và mọi khung của clip nhận cùng lựa chọn mà không cần đồng bộ trạng thái | 🛠 tự thiết kế |
| 2 | **Diffusion làm nguồn dữ liệu, không phải bộ sinh**: 600 nền SD, prompt chặn người và chữ; chi phí suy luận bằng 0 | Ý tưởng chung "dữ liệu tổng hợp bằng diffusion" là phổ biến; áp vào pose-guided sign video là 🛠 |
| 3 | **Matte lai**: DeepLabV3 (có quần áo, tóc) ∪ mặt nạ thân SMPL-X giãn 2 px (giữ được ngón tay mảnh mà phân đoạn hay bỏ sót) | Hai thành phần có sẵn; cách hợp là 🛠 |
| 4 | **Bài kiểm tra tổng quát hoá có ground truth**: tái tạo clip val sau khi ghép lên 100 nền **giữ lại**, không mô hình nào thấy khi train (script 29). `08_eval_cross_person.py` không có GT nên không chấm được | 🛠 |
| 5 | **Phát hiện lỗi "cặp cố định"** trong code GReg, có unit test chứng minh, sửa có tuỳ chọn và đo riêng bằng run `fresh` | 🛠 (phát hiện trong quá trình làm D4) |

Đổi màu (brightness, contrast, saturation, hue) và thay nền là augmentation kinh điển; bản thân phép biến đổi **không mới**. Cái mới nằm ở ràng buộc theo clip, matte lai và giao thức đo.

### 2.3 Liên hệ giữa augmentation và bản sửa lỗi
Augmentation rút ngẫu nhiên theo `epoch`. Nếu không sửa lỗi, worker luôn thấy `epoch = 0`, nên mỗi clip sẽ nhận **cùng một nền và cùng một màu ở mọi epoch**. Khi đó augmentation chỉ còn là "đổi cố định một lần" thay vì "đa dạng hoá".

Vì vậy run `aug` **bắt buộc** bật cả bản sửa (`configs/sweeps/d4_augment.yaml:18`). Run `fresh` tồn tại để tách tác dụng của bản sửa khỏi tác dụng của augmentation.

---

## 3. Thiết kế chi tiết

### 3.1 Sinh nền bằng SD1.5 (`scripts/27_make_backgrounds.py`)
- **Prompt** (`:19-25`, `:45-47`): `"photo of {màu}{nơi}, empty, no people, {ánh sáng}, background, high quality"`.
  - 7 màu: không màu, light blue, beige, grey, dark, pastel green, warm orange.
  - 20 nơi: tường sơn, văn phòng, giá sách, lớp học, phông studio, gạch, gỗ, hành lang, thư viện, green screen, quán cà phê, …
  - 5 kiểu ánh sáng.
  - Tổng 700 tổ hợp, xáo bằng `Random(0)`; ảnh thứ i dùng tổ hợp thứ i, nên 600 ảnh có 600 prompt khác nhau.
- **Negative prompt:** `person, people, man, woman, hands, face, text, watermark, logo, blurry`. Mục đích là không để nền chứa người hay tay giả, vì chúng sẽ dạy sai mô hình.
- **Sinh ảnh:** SD1.5 fp16, DPM-Solver++ 25 bước, CFG 6.0, seed = chỉ số ảnh (tái lập được), sinh ở 512² rồi thu về 256² (`:38-55`).
- **Chia tập** (`scripts/vast/stages_d4.sh:21-28`): ảnh 0–499 → `backgrounds_train`; ảnh **500–599 giữ lại** cho bài kiểm tra nền lạ.
- **Thời gian thực tế ✅:** 5 phút 16 giây cho 600 ảnh (20:14:38 → 20:19:54).

### 3.2 Matte người (`scripts/28_matte_cache.py`)
```
khung 256² ──bilinear──► 512² ──DeepLabV3-ResNet101 (torchvision, lớp VOC 'person' = 15)──► softmax ──► P_person
                                                                                                   │ thu về 256²
alpha kênh 4 của normal map SMPL-X ──max_pool 5×5 (giãn 2 px)──► M_body ───────────────────────────┤
                                                                                                   ▼
                                   alpha = avg_pool 3×3 ( max(P_person, M_body) ) ──► matte/<key>/<row>.png (8-bit)
```
- Lý do hợp hai nguồn (`:6-9`): mặt nạ SMPL-X **mỏng hơn** quần áo và tóc, nên không cắt được người. Ngược lại, phân đoạn ngữ nghĩa đôi khi **bỏ ngón tay mảnh**. Lấy max giữ được cả hai (`:72`).
- Làm mềm 3×3 (`:73`) để tránh viền răng cưa khi ghép.
- Không dùng RVM như kế hoạch ban đầu: tải RVM cần `torch.hub` chạy mã từ GitHub, và việc đó bị chặn.
- Phạm vi: clip `train_fit` và `val`. Thời gian ✅: 1 h 22 phút (20:19:54 → 21:41:33).
- Lần chạy có cờ `--check` (đo độ phủ của phân đoạn trên thân SMPL-X), nhưng **con số độ phủ không còn lưu**: file nằm trên máy đã xoá và không được báo trong hội thoại.

### 3.3 Phép augmentation (`greg/data/augment.py`)
Lớp `ClipConsistentAugment` gồm hai bước.

**Rút lựa chọn** (`draw`, `:80-89`): `rng = random.Random(f"{seed}/{epoch}/{key}")`.
- Nền: với xác suất `background_p`, chọn một ảnh ngẫu nhiên trong bank.
- Màu: với xác suất `color_p`, rút bốn hệ số:

  | Tham số | Khoảng | Mặc định (`:65-66`) |
  |---|---|---|
  | độ sáng `bf` | 1 ± 0.2 | 0.2 |
  | tương phản `cf` | 1 ± 0.2 | 0.2 |
  | độ bão hoà `sf` | 1 ± 0.3 | 0.3 |
  | hue `hf` | ± 0.08 vòng (≈ ±29°) | 0.08 |

  Hai lựa chọn độc lập, nên theo kỳ vọng: 15 % clip được đổi cả hai, 35 % chỉ đổi màu, 15 % chỉ đổi nền, 35 % giữ nguyên.

**Áp lên một khung** (`apply_frame`, `:91-108`), với `a` là matte:
```
x      = clamp(img · bf)                                   độ sáng
mean   = Σ(x·a) / Σa                                       trung bình màu CHỈ trên người
x      = clamp((x − mean)·cf + mean)                       tương phản quanh màu trung bình của người
HSV(x): H ← (H + hf) mod 1 ;  S ← clamp(S · sf)            hue, độ bão hoà (RGB↔HSV viết tay, test round-trip :32-34)
person = a·x + (1 − a)·img                                 chỉ đổi màu trong matte (quần áo + da cùng lúc)
out    = a·person + (1 − a)·bg                             thay nền (nếu có)
```
- Hàm `__call__` (`:110-118`) dùng **một** lần rút `d` và **một** ảnh nền `bg` cho cả `source_image` và `target_image`. Mỗi khung dùng matte của chính nó (`source_row`, `target_row`).
- Không đổi normal map, khớp hay hộp vùng: hình học giữ nguyên, chỉ ngoại hình đổi.
- Unit test ✅ (`tests/test_d4_augment.py:37-52`): nền bị thay ở cả nguồn và đích, pixel người giữ nguyên, và `draw` tất định.

### 3.4 Sửa lỗi "cặp cố định" (`greg/data/dataset.py`)
**Lỗi ✅:**
- `PairDataset.__getitem__` rút cặp (nguồn, đích) bằng `random.Random((seed·1 000 003 + epoch)·7 919 + index)`.
- Trainer gọi `set_epoch(e)` ở tiến trình chính.
- DataLoader được tạo với `persistent_workers=True` khi `num_workers > 0` (`greg/train_utils.py:38`). Mỗi worker giữ **bản sao** của dataset tạo lúc khởi động, nên `self.epoch` trong worker mãi bằng 0.
- Hệ quả: với `pairs_per_clip = 1`, **mỗi clip chỉ góp đúng một cặp, giống nhau ở cả 100 epoch**. Ở split `train` đó là 4 585 cặp cố định, thay vì mỗi epoch một cặp mới như docstring mô tả.
- Áp dụng cho run5 và mọi baseline.

**Bằng chứng ✅:** `tests/test_d4_augment.py:55-71` dựng DataLoader với 1 worker và `persistent_workers=True`, gọi `set_epoch(0)` rồi `set_epoch(1)`:
- không sửa: batch hai epoch **giống hệt** (`fixed[0] == fixed[1]`);
- có sửa: hai batch khác nhau.

Khi viết test, lần chạy đầu fail đúng ở assertion này, trước khi có bản sửa.

**Cách sửa** (`greg/data/dataset.py:242-257, 278`):
- Thêm `shared_epoch=True` → `self._shared = multiprocessing.Value("i", 0)`. Đây là bộ nhớ chia sẻ, nên worker đọc được giá trị mới.
- `set_epoch` ghi vào đó; `current_epoch()` đọc từ đó.
- `__getitem__` dùng `current_epoch()` cho cả việc rút cặp và cho augmentation (`:283`).
- Bật bằng `data.fresh_pairs_per_epoch: true` (`greg/train_utils.py:28`). **Mặc định tắt**, để baseline vẫn tái lập đúng run5.

### 3.5 Luồng dữ liệu vào GReg
```
                         (một lần, offline)
   SD1.5 + 700 prompt ──► 600 nền 256² ──┬─► 0–499  backgrounds_train
                                         └─► 500–599 backgrounds_heldout (chỉ để đánh giá)
   DeepLabV3 ∪ SMPL-X ──► matte/<key>/<row>.png

                         (mỗi mẫu, trong DataLoader worker)
 index ──► key ──► epoch = shared.value ──► rng_pair(seed, epoch, index) ──► (s, t)
                                   │
                                   └──► rng_aug("seed/epoch/key") ──► {bg?, jitter?}   (một lần cho cả cặp)
 _pair_item(key, s, t) ──► source_image, target_image, normal_t, joints, boxes, ...
                                   │
             ClipConsistentAugment: cùng {bg, jitter} cho source_image và target_image (matte riêng từng khung)
                                   ▼
                    GReg (TPSMM đóng băng → GRM → generator + RE) ── loss run5 không đổi
```
Augmentation chỉ gắn vào loader có split bắt đầu bằng `train` (`greg/train_utils.py:30`). Đánh giá dùng ảnh gốc, trừ script 29 cố ý gắn nền lạ (`greg/eval/evaluate.py` có thêm tham số `item_augment`).

### 3.6 Những gì giữ nguyên so với run5
Kiến trúc, loss và trọng số loss (`perceptual 1`, `regional_perceptual 3`, `struct 10`, `adv 1`), optimizer, lịch lr, TPSMM đóng băng lấy từ checkpoint flow của run5, độ phân giải 256², `pairs_per_clip 1`.

So với run5, chỉ khác ba khoá: `data.fresh_pairs_per_epoch`, `data.augment.background_p = 0.3`, `data.augment.color_p = 0.5`.

### 3.7 Chi phí (đo thật ✅, 4090 với giá $0.62/h)
| Bước | Thời gian | Ghi chú |
|---|---|---|
| Sinh 600 nền | 5 phút | một lần |
| Matte `train_fit` + `val` | 1 h 22 phút | một lần |
| Sweep 15 epoch: `fresh` + `aug` | 1 h 46 phút cho cả hai | 0.20 s/bước, 5.0 GB VRAM, **bằng tốc độ không augment** |
| Chấm nền lạ (3 checkpoint) | 1 phút | |
| Train 100 epoch `aug` | 6 h 28 phút (≈ $4.0) | 05:27:35 → 11:55:23, 2026-11-04 |
| Chấm toàn bộ test | 17.5 phút | |
| **Suy luận** | **+0** | mô hình suy luận giống hệt GReg |

---

## 4. Giao thức đánh giá

| Ký hiệu | Mô tả | Mô hình được so |
|---|---|---|
| **sweep15** | Train 15 epoch trên `train_fit` (milestone 10/13, `configs/sweep_greg.yaml:7-13`), chấm 100 clip val × 16 khung lấy đều, LPIPS AlexNet | `base` = checkpoint `run5 sweep_clean/ref` (train trước đây, **trên máy và phiên bản thư viện khác**, chép từ Drive, `stages_d4.sh:30-37`); `fresh`, `aug` train trên 4090 này |
| **nền lạ** | Script 29: cùng 100 clip val × 16 khung, mỗi clip ghép lên một nền trong 100 nền giữ lại (seed 12345 ≠ seed train), cùng nền cho nguồn và đích, đủ chỉ số có GT | 3 checkpoint của sweep15 |
| **test-full** | `07_eval_reconstruction`, toàn bộ test (1 146 clip, mọi khung), đúng script đã cho số Table 1 của run5 | run5 epoch 100 so với `d4_aug_100` epoch 100 |
| **u16** | Toàn bộ test, 16 khung đều mỗi clip (35 513 khung), qua export PNG + `21_eval_frames --index` | GReg run5, D4 aug, D1 trên mỗi mô hình |
| **bootstrap** | Ghép cặp theo clip, 2 000 lần lấy mẫu lại, trung bình có trọng số theo khung, khoảng tin cậy 95 % của (B − A) | test-full và u16 |

Bootstrap chỉ đo **độ bất định do chọn clip test**. Nó **không** đo độ bất định do huấn luyện (seed, máy). Mọi run ở đây chỉ có một seed.

---

## 5. Kết quả

### 5.1 Vòng lọc 15 epoch (sweep15: val, 100 clip × 16 khung) ✅
| Run | L1 ↓ | LPIPS ↓ | hand L1 ↓ | hand LPIPS ↓ | hand L1 @large ↓ | s/bước | VRAM |
|---|---|---|---|---|---|---|---|
| base (run5 cấu hình, cặp cố định) | 0.0210 | 0.0617 | 0.1382 | 0.2845 | 0.1691 | – | – |
| fresh (chỉ sửa lỗi cặp) | 0.0236 (+12.4 %) | 0.0693 (+12.2 %) | 0.1448 (+4.8 %) | 0.2986 (+4.9 %) | 0.1776 (+5.0 %) | 0.20 | 5.0 GB |
| **aug** (sửa lỗi + nền + màu) | **0.0196 (−7.0 %)** | **0.0580 (−6.0 %)** | **0.1342 (−2.9 %)** | **0.2787 (−2.0 %)** | **0.1653 (−2.3 %)** | 0.20 | 5.0 GB |

`aug` so với `fresh`, đối chứng trực tiếp vì chỉ khác phần augmentation và cùng máy: LPIPS −16.3 %, L1 −17 %, hand LPIPS −6.7 %.

### 5.2 Nền chưa thấy khi train (val, 100 clip × 16 khung = 1 600 khung) ✅
| Run | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | hand LPIPS ↓ | trôi màu ↓ |
|---|---|---|---|---|---|---|
| base | 0.0258 | 26.55 | 0.8945 | 0.0721 | 0.3013 | 0.0132 |
| fresh | 0.0264 | 26.30 | 0.8935 | 0.0763 | 0.3168 | 0.0137 |
| **aug** | **0.0223 (−14 %)** | **27.24 (+0.69 dB)** | **0.9003** | **0.0666 (−7.6 %)** | **0.2837 (−5.8 %)** | **0.0097 (−27 %)** |

Mức suy giảm khi chuyển từ nền thật sang nền lạ, mỗi run so với chính nó (tính từ hai bảng trên):

| Run | LPIPS: thật → lạ | L1: thật → lạ |
|---|---|---|
| base | 0.0617 → 0.0721 (+17 %) | 0.0210 → 0.0258 (+23 %) |
| fresh | 0.0693 → 0.0763 (+10 %) | 0.0236 → 0.0264 (+12 %) |
| aug | 0.0580 → 0.0666 (+15 %) | 0.0196 → 0.0223 (+14 %) |

### 5.3 Train đủ 100 epoch, toàn bộ test (test-full, 07, mọi khung) ✅
| Mô hình | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | hand L1 ↓ | hand LPIPS ↓ | face L1 ↓ | trôi màu ↓ | disocc L1 | arm L1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GReg run5 | **0.0173** | 29.19 | 0.9142 | **0.0485** | 0.1203 | 0.2624 | **0.0311** | **0.0083** | – | – |
| D4 aug 100 ep | 0.0175 | **29.31** | **0.9149** | 0.0490 | **0.1169** | **0.2524** | 0.0313 | 0.0085 | 0.0522 | 0.0516 |

Bootstrap (A = run5, B = D4 aug 100 ep, test-full):
| Chỉ số | A | B | B − A | tương đối | KTC 95 % | Kết luận |
|---|---|---|---|---|---|---|
| LPIPS | 0.04851 | 0.04895 | +0.00044 | +0.9 % | [+0.00035, +0.00054] | **B kém hơn** |
| L1 | 0.01728 | 0.01750 | +0.00022 | +1.3 % | [+0.00020, +0.00025] | B kém hơn |
| PSNR | 29.194 | 29.307 | +0.113 | +0.4 % | [+0.102, +0.124] | **B tốt hơn** |
| hand LPIPS | 0.26236 | 0.25242 | −0.00994 | −3.8 % | [−0.01085, −0.00899] | **B tốt hơn** |
| face L1 | 0.03106 | 0.03133 | +0.00027 | +0.9 % | [+0.00023, +0.00031] | B kém hơn |
| trôi màu | 0.00826 | 0.00852 | +0.00025 | +3.1 % | [+0.00023, +0.00028] | B kém hơn |

Hai cột disocc L1 và arm L1 của run5 trống: lần chấm run5 đó chạy trước khi sửa lỗi metric `5d20a58`, nên không tính hai chỉ số này.

### 5.4 Test 16 khung đều mỗi clip (u16, 35 513 khung), kèm refiner D1 ✅
| Dòng | L1 | PSNR | SSIM | LPIPS | hand L1 | hand LPIPS | face L1 | disocc L1 | trôi màu | arm L1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GReg run5 | 0.0172 | 28.96 | 0.9150 | 0.0487 | 0.1180 | 0.2519 | 0.0315 | 0.0554 | 0.0082 | 0.0537 |
| D4 aug 100 ep | 0.0173 | 29.09 | 0.9148 | 0.0492 | 0.1145 | 0.2424 | 0.0317 | 0.0552 | 0.0084 | 0.0530 |
| D1 trên run5 | 0.0172 | 28.86 | 0.9153 | **0.0464** | 0.1155 | 0.2337 | 0.0332 | 0.0561 | 0.0082 | 0.0528 |
| D1 trên D4 aug | 0.0173 | 28.92 | 0.9148 | 0.0468 | **0.1139** | **0.2279** | 0.0334 | 0.0562 | 0.0084 | **0.0526** |

Bootstrap trên u16 (A = GReg run5):
| So sánh | LPIPS | L1 | PSNR | hand LPIPS | face L1 | trôi màu |
|---|---|---|---|---|---|---|
| D4 aug | +1.1 % (kém) | +0.7 % (kém) | +0.13 dB (tốt) | −3.8 % (tốt) | +0.8 % (kém) | +2.5 % (kém) |
| D1 trên D4 aug | −3.9 % (tốt) | +1.0 % (kém) | −0.05 dB (kém) | **−9.5 %** (tốt) | +6.3 % (kém) | +2.2 % (kém) |
| *(tham chiếu) D1 trên run5* | −4.7 % (tốt) | không khác biệt | −0.10 dB (kém) | −7.2 % (tốt) | +5.5 % (kém) | −0.2 % (tốt) |

Mọi khác biệt trong bảng có p < 0.001, trừ ô ghi "không khác biệt".

---

## 6. Phân tích nguyên nhân và hiện tượng

### 6.1 Vì sao thắng ở 15 epoch nhưng mất ở 100 epoch
1. **Augmentation có ích chủ yếu khi mô hình chưa hội tụ** ⚠️.
   - Ở 15 epoch, `aug` hơn `fresh` 16 % LPIPS dù cả hai cùng thấy cặp mới mỗi epoch.
   - Lời giải thích hợp lý: ghép nền và đổi màu buộc generator dựa vào **ảnh nguồn** (chép màu, chép nền) thay vì "nhớ" ngoại hình từng người ký. Kỹ năng chép này học được sớm, nên giúp nhiều ở lịch ngắn.
   - Ở 100 epoch, baseline cũng đã học đủ việc chép, nên lợi thế co về khoảng 0.
   - Bằng chứng gián tiếp: lợi thế giảm từ −6 % (15 epoch, val) xuống +0.9 % (100 epoch, test). Ta không có mốc 15 epoch trên test, nên chưa tách được hiệu ứng "lịch train" khỏi hiệu ứng "val so với test".
2. **Tập test cùng phân phối nền với train** ✅ (cùng các phông studio HamNoSys).
   - Thay nền nhắm vào **tổng quát hoá**, không vào tái tạo trong phân phối. Trên test-full, nền lạ không bao giờ xuất hiện, nên phần năng lực dành cho nền tổng hợp (30 % clip train) không được "trả công".
   - Bảng nền lạ (§5.2) cho thấy lợi thế rõ hơn khi phân phối lệch, nhưng mới đo ở 15 epoch. **Chưa đo `d4_aug_100` trên nền lạ**, và nay không đo được nữa vì checkpoint đã mất.
3. **Đổi màu có thể tạo ra ngoại hình phi thực tế** ⚠️.
   - Hue ±29° áp lên cả da: da có thể ngả xanh hoặc tím.
   - Mô hình phải học cả các ngoại hình không bao giờ có ở test. Điều này phù hợp với face L1 (+0.9 %) và trôi màu (+3.1 %) kém đi, nhưng chưa có ablation tách `color_p` khỏi `background_p`.
4. **Đối chứng không sạch về seed và máy** ✅ (là hạn chế, không phải nguyên nhân đã chứng minh).
   - run5 được train trước đó trên máy khác (trước khi nâng torch lên 2.7.1). `d4_aug_100` train trên 4090 này. Cả hai chỉ có một seed.
   - Bootstrap cho khoảng tin cậy hẹp (±0.0001 LPIPS), nhưng đó chỉ là độ bất định do chọn clip. Độ dao động giữa hai lần train cùng cấu hình chưa từng được đo.
   - Chương trình chưa từng train lặp cùng một cấu hình, nên không biết nhiễu giữa hai lần train lớn cỡ nào. Vì vậy chưa loại được khả năng khác biệt +0.9 % **nằm trong nhiễu huấn luyện**. Kết luận đúng mức là "D4 không cải thiện LPIPS", không phải "D4 làm hại LPIPS".

### 6.2 Vì sao `fresh` (chỉ sửa lỗi) lại kém `base` ở 15 epoch ⚠️
Kỳ vọng ban đầu: cặp mới mỗi epoch thì đa dạng hơn, nên tốt hơn. Thực tế kém 12 %. Các giả thuyết, xếp theo mức tin:
1. **Nhiễu do máy và thư viện** (mạnh nhất). `base` không được train lại: đó là checkpoint cũ chép từ Drive. `fresh` và `aug` train trên máy mới. Nhận xét quan trọng: chính `aug`, cũng là "cặp mới", **lại tốt hơn `base`**. Vậy "cặp mới" tự nó không thể là thủ phạm gây +12 %, trừ khi augmentation bù lại nhiều hơn 18 %. So sánh nội bộ cùng máy (`aug` so với `fresh`) đáng tin hơn so sánh với `base`.
2. **Nhiệm vụ khó hơn với lịch ngắn.** Với cặp cố định, mỗi clip là một bài toán lặp lại 15 lần, dễ khớp nhanh. Với cặp mới, mỗi epoch là bài mới, nên trong 15 epoch mô hình ở trạng thái "chưa hội tụ" hơn. Theo giả thuyết này, ở 100 epoch `fresh` sẽ đuổi kịp, nhưng chưa train `fresh` 100 epoch.
3. **Tương tác với val:** val dùng khung nguồn là khung đầu clip. Phân phối khoảng cách nguồn–đích của cặp cố định (rút một lần ở epoch 0) và cặp mới là như nhau theo kỳ vọng, nên giả thuyết này yếu.

Cách phân xử: train lại `base`, `fresh`, `aug` với `seed = 1` trên cùng máy (khoảng 3 h). Việc này chưa làm.

### 6.3 Vì sao hand LPIPS vẫn giảm ở mọi giao thức
Đây là tín hiệu **nhất quán nhất** của D4 ✅:
- sweep15: −2.0 %
- nền lạ: −5.8 %
- test-full: −3.8 %, khoảng tin cậy loại trừ 0
- u16: −3.8 %
- hand L1 cũng giảm: −2.8 % (test-full), −3.0 % (u16)

Giải thích ⚠️:
- Thay nền làm thay đổi **độ tương phản tay/nền** từ clip này sang clip khác. Mô hình không thể dựa vào "tay luôn sáng hơn phông xanh" mà phải dựa vào hình học (normal map, prior vùng tay). Điều này đúng ở cả test.
- Đổi màu da buộc mô hình **chép màu da từ nguồn** thay vì nhớ màu trung bình, nên vùng tay (nhỏ, chuyển động nhiều) hưởng lợi nhiều nhất.

Khi đặt D1 lên trên, hand LPIPS cộng dồn: −7.2 % (D1 trên run5) → −9.5 % (D1 trên D4). LPIPS toàn ảnh thì **không** cộng dồn (−4.7 % → −3.9 %). Hai cải tiến cùng nhắm vào tay nhưng qua cơ chế khác nhau: D4 sửa ở bước train, D1 làm sắc sau khi sinh.

### 6.4 PSNR tốt hơn trong khi L1 và LPIPS kém hơn ⚠️
- PSNR (dựa trên MSE, phạt nặng sai số lớn) tăng 0.11 dB. Cùng lúc, L1 (sai số trung bình) tăng 1.3 % và LPIPS tăng 0.9 %.
- Kiểu này thường gặp khi mô hình **ít sai số lớn hơn nhưng nhiều sai số nhỏ hơn**, tức đầu ra "an toàn", mượt hơn: ít khi đặt sai hẳn tay, nhưng mất vân chi tiết. Điều đó khớp với vai trò regularizer của augmentation.
- Chưa kiểm chứng bằng phân bố sai số theo pixel (ví dụ phân vị 99 của |error|). Không còn làm được, vì `per_frame.csv` và khung ảnh đã mất.

### 6.5 Bài kiểm tra nền lạ nói gì, và không nói gì
- **Có:** `aug` bền hơn với nền lạ ở L1 (−14 %) và trôi màu (−27 %). Hai chỉ số tần số thấp này phản ánh trực tiếp việc mô hình "không kéo màu nền cũ vào ảnh".
- **Không:** về LPIPS, mức suy giảm tương đối khi sang nền lạ của `aug` (+15 %) **không nhỏ hơn rõ** so với `base` (+17 %). Lợi thế LPIPS của `aug` trên nền lạ (−7.6 %) gần bằng lợi thế trên nền thật (−6.0 %). Vậy về chất lượng cảm nhận, D4 chưa làm mô hình **bền hơn** mà chỉ **tốt hơn đều** ở 15 epoch, và lợi thế đều đó đã biến mất ở 100 epoch.

---

## 7. Hạn chế và việc nên làm tiếp
1. **Một seed, đối chứng khác máy.** Nên train lại `base`/`fresh`/`aug` (15 epoch, seed 0 và 1) trên cùng máy, khoảng 3–6 h, khoảng $2–4. Đây là điều kiện tối thiểu trước khi viết bất kỳ khẳng định nào về `fresh` hay về khác biệt dưới 2 %.
2. **Chưa đo `d4_aug_100` trên nền lạ.** Đó là bài kiểm tra mà D4 được thiết kế để thắng. Checkpoint đã mất; muốn đo phải train lại 6.5 h (khoảng $4).
3. **Chưa có cross-person thật.** `08_eval_cross_person.py` không có ground truth. Cần FID/FVD trên video cross-person, hoặc đánh giá người xem, để đo đúng mục tiêu tổng quát hoá.
4. **Chưa có ablation riêng nền và riêng màu**, và chưa thử các mức xác suất khác (0.3 và 0.5 là chọn tay).
5. **Hue ±29° trên da** có thể quá mạnh. Nên giới hạn hue trong vùng da, hoặc chỉ đổi màu quần áo bằng mặt nạ phân đoạn quần áo.
6. **Tác động của bản sửa lỗi lên run5 ở 100 epoch chưa đo** (cần `fresh` 100 epoch). Khi viết luận văn, ghi rõ: mọi baseline, kể cả run5, train với cặp cố định.

---

## 8. Trạng thái dữ liệu
| Dữ liệu | Còn hay mất | Ghi chú |
|---|---|---|
| Code, config, unit test | ✅ còn | GitHub, nhánh `feat/diff-d4-augment` |
| Sweep 15 epoch (`base`/`fresh`/`aug`: checkpoint, `val_metrics.json`, `sweep_results.csv`), kết quả nền lạ | ✅ còn trên Drive `GReg_runs/diff_common` | Sync trước khi Drive đầy (2026-11-04 02:43 UTC) |
| 600 ảnh nền, matte | ⚠️ không chắc có trên Drive | Tái tạo được: nền 5 phút (seed cố định, giống hệt), matte khoảng 1.4 h |
| Checkpoint `d4_aug_100`, `eval_test/per_frame.csv`, export u16, khung D1 trên D4 | ❌ **mất** | Sinh ra sau khi Drive đầy; máy đã xoá 2026-11-05 |
| Bảng tổng hợp và bootstrap ở §5.3–5.4 | ✅ còn ở dạng số | Chỉ trong hội thoại và báo cáo này; không tính lại được chỉ số mới |
| Con số độ phủ matte (`--check`) | ❌ không còn lưu | |
