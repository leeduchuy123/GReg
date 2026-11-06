# Báo cáo: GReg + tay D6 (ghép theo vùng, hộp tay lấy từ tư thế đích)

Mã: `docs/bao_cao_diffusion/code/compose_hands.py` (ghép + chấm), `docs/bao_cao_diffusion/code/bootstrap_ci.py` (kiểm định), `docs/bao_cao_diffusion/code/post_d6.sh` (hàng đợi đã chạy).
Ba file được chép nguyên từ scratchpad của phiên; trên máy vast chúng nằm ở `/workspace/`. Số dòng trích dẫn bên dưới là của bản chép này.
Ghép và chấm dùng thư viện của checkout nhánh D1 `feat/diff-d1-refiner` (`/workspace/GReg-d1`, `compose_hands.py:16`).
Ngày chạy: 2026-11-05, 09:04–09:09 UTC (tự động sau khi D6 xong).

Quy ước: ✅ đã kiểm chứng (mã nguồn `path:dòng`, hoặc số đọc từ `summary.json` / `bootstrap.md` trên máy, đã báo trong hội thoại); ⚠️ suy luận hoặc ước lượng, chưa đo.

---

## 0. Tóm tắt

- **Làm gì:** mỗi khung đích ghép từ hai mô hình.
  - Mọi pixel lấy từ **GReg run5**, trừ hai vùng quanh bàn tay.
  - Hai vùng tay lấy từ **D6** (Wan2.1-Fun + LoRA), viền làm mềm.
  - Hộp tay tính **chỉ từ tư thế đích** (khớp SMPL-X), không dùng ảnh thật, nên phương pháp hợp lệ khi suy luận.
- **Kết quả** (200 clip test × 16 khung, bootstrap ghép cặp theo clip, khoảng tin cậy 95 %): GReg + tay D6 `ng` **tốt hơn GReg ở mọi chỉ số đo được**.
  - LPIPS 0.0476 → **0.0433 (−9.0 %)**
  - L1 −5.1 %
  - PSNR +0.35 dB
  - region L1 −11 %
  - hand LPIPS −22 %
  - trôi màu −3 %
  - face L1 không khác biệt có ý nghĩa (−0.5 %, p = 0.061)
- Ghép lên nền **D1** (refiner SD-Turbo) cho LPIPS thấp nhất (**0.0424, −11 %**), nhưng face L1 **+6 %**.
- **Kết luận:**
  - Đây là kết quả dương duy nhất của chương trình D1–D6 không phải đánh đổi chỉ số nào.
  - Thông điệp: diffusion có giá trị khi chỉ được giao **đúng vùng nó giỏi** (bàn tay). GReg giữ danh tính, nền và màu.

---

## 1. Bối cảnh

Trên cùng 200 clip test × 16 khung (s200), hai mô hình có lỗi gần như bổ sung cho nhau theo vùng ảnh ✅ (`docs/bao_cao_diffusion/D6_wan_video_dit.md` §5):

| | LPIPS | hand LPIPS | hand L1 | face L1 | trôi màu |
|---|---|---|---|---|---|
| GReg run5 | **0.0476** | 0.236 | 0.113 | **0.0304** | **0.0077** |
| D6 `ng` | 0.0527 (+11 %) | **0.184 (−22 %)** | **0.091 (−19 %)** | 0.0365 (+20 %) | 0.0104 (+34 %) |

- **GReg** chép ngoại hình từ ảnh nguồn qua warp TPSMM, nên mặt, quần áo và nền rất sát ảnh thật. Bàn tay, nơi hình dạng đổi nhiều và nhiều pixel nguồn không thấy, thì mờ.
- **D6** có prior video nên vẽ tay đẹp. Nhưng nó sinh lại cả khung qua VAE, nên mặt lệch và màu trôi.

Câu hỏi của thí nghiệm: nếu chỉ lấy tay của D6, các chỉ số toàn khung (LPIPS, L1, mặt, màu) có giữ được mức của GReg không, hay đường ghép làm hỏng chúng?

Thí nghiệm này là mục "A" trong hàng đợi sau D6, đã được người dùng đồng ý ngày 2026-11-05 (`post_d6.sh:2-5`).

---

## 2. Ý tưởng và điểm đặc sắc

### 2.1 Ý tưởng

**Phân công theo vùng:** mô hình hồi quy (GReg) lo phần "chép được", mô hình sinh (D6) lo phần "phải vẽ", ở đây là bàn tay.

### 2.2 Điểm mới và vì sao hợp lệ 🛠

1. **Mặt nạ không dùng ảnh thật.**
   - Hộp tay được tính từ khớp tay SMPL-X của **tư thế đích** (`batch["target_joints"]`, `compose_hands.py:35-38`, `:45-47`) và tâm vùng đích (`centers_target`).
   - Đây chính là hình học GReg vẫn nhận làm điều kiện: normal map cũng render từ fit đó.
   - Vì vậy phương pháp chạy được khi sinh video thật, không phải "gian lận lúc đánh giá" (`compose_hands.py:3-5`).
2. **Không train thêm gì.** Ghép là hậu xử lý thuần trên hai thư mục khung đã có. Chi phí ghép không đáng kể; 8 tổ hợp ghép, chấm và bootstrap hết khoảng 5 phút ✅ (09:04:45 → 09:09:52).
3. **Đánh giá hai chiều có kiểm soát:**
   - Ghép **mọi** biến thể D6 (`ng`, `n`, `k`, `ng_ones`) lên **hai nền** (GReg run5, D1), tổng 8 tổ hợp (`post_d6.sh:15-23`).
   - Mọi so sánh đều kiểm định bằng bootstrap **ghép cặp theo clip** (`post_d6.sh:24-41`).
4. **Thiết kế trung thực về chỉ số tay.**
   - Vùng ghép (lề 0.5) bao trọn hộp đo `hand_lpips` (lề 0.25). Theo cấu trúc, chỉ số tay của bản ghép **xấp xỉ của D6** (`compose_hands.py:7-8`).
   - Vì vậy bằng chứng thật nằm ở các chỉ số **toàn khung**, mặt và màu. Báo cáo này đọc kết quả theo đúng thứ tự đó.

### 2.3 Mượn và riêng

| Phần | Nguồn |
|---|---|
| Hộp tay thích ứng từ độ trải của khớp tay | `adaptive_region_boxes` + `region_extents_from_joints`: cải tiến B1 của chính luận văn (`greg/data/regions.py:26-41`, `greg/render/body.py:150-160`, nhánh D1) |
| Ghép có làm mềm viền (alpha blending với mặt nạ mềm) | Kỹ thuật kinh điển |
| Bootstrap theo cụm (cluster bootstrap) | Thống kê kinh điển; ở đây cụm = clip |
| Ý tưởng "diffusion chỉ cho tay, GReg cho phần còn lại" và bộ đánh giá trên tư thế đích | Riêng của luận văn 🛠 |

---

## 3. Kiến trúc chi tiết

### 3.1 Sơ đồ

```
                  ┌──────────────► GReg run5 (epoch 100) ─────► A = Î_GReg     (3, 256, 256)  (PNG đã xuất)
ảnh nguồn I_s ────┤
+ tư thế đích     └──────────────► D6 (Wan2.1-Fun + LoRA) ────► B = Î_D6       (3, 256, 256)  (PNG đã sinh)

khớp 2D SMPL-X đích (B, J, 2) ── region_extents_from_joints ──► độ trải e_lhand, e_rhand (px)
tâm vùng đích centers_t (B, 3, 2) ─┐
                                   ▼
adaptive_region_boxes(hand_margin = 0.5, min_side = 24, max_side = 256)
   side = clamp(e · (1 + 2·0.5), 24, 256) = clamp(2·e, 24, 256);  hộp vuông quanh tâm, kẹp trong khung
region_visibility: tay có tâm ra ngoài khung → bỏ
   ▼
m₀(x, y) = max_tay 1[(x, y) ∈ hộp tay ∧ tay nhìn thấy]               (B, 1, 256, 256) nhị phân
m = AvgPool₉ₓ₉(AvgPool₉ₓ₉(m₀))   (stride 1, pad 4, không đếm pad)   viền mềm khoảng ±8 px
   ▼
Î = m · B + (1 − m) · A                                               (compose_hands.py:61-62)
```

Mã: `compose_hands.py:40-61`.

- **Lớp `ComposeModel`** kế thừa `FrameDirModel`, lớp đọc PNG theo `(clip, hàng)`. Nhờ vậy bản ghép đi qua **đúng** `evaluate_reconstruction` như mọi mô hình khác (`compose_hands.py:29-38`, `:80-82`).
- **Hai lần lọc trung bình 9×9** tương đương một nhân dạng tam giác hai chiều rộng 17 px. Tại mép hộp gốc m ≈ 0.5; m đạt 1 khi vào trong khoảng 8 px và về 0 khi ra ngoài khoảng 8 px ⚠️ (tính từ định nghĩa, `compose_hands.py:57-59`).
- **Quan hệ với hộp đo:**
  - `hand_lpips` đo trên hộp cùng công thức nhưng lề 0.25, tức cạnh 1.5·e (`greg/eval/evaluate.py:82-90`, nhánh D1).
  - Hộp ghép có cạnh 2·e, nên mỗi phía rộng hơn hộp đo 0.25·e. Với e khoảng 30–40 px ⚠️, phần dư là khoảng 7–10 px, vừa đủ để vùng m < 1 gần như nằm ngoài hộp đo.
- **Tham số dòng lệnh:** `--margin 0.5`, `--feather 9` (mặc định, `compose_hands.py:72-73`). Không có tham số nào được tinh chỉnh trên tập test.

### 3.2 Các tổ hợp đã chạy

Theo `post_d6.sh:15-23`:

| Nền A | Tay B | Tên |
|---|---|---|
| GReg run5: `/workspace/outputs/export/run5_epoch_100/test` | D6 `ng`, `n`, `k`, `ng_ones` | `{v}_on_greg` |
| D1 refiner, cổng `freq`: `/workspace/outputs/final/refined_run5/freq` | như trên | `{v}_on_d1` |

### 3.3 Kiểm định: bootstrap ghép cặp theo clip

Mã: `bootstrap_ci.py:35-62`.

1. **Ghép cặp:** chỉ giữ các `(clip, hàng)` mà cả A và B đều có điểm (`:36-38`).
2. **Theo từng chỉ số:**
   - Cộng dồn tổng và số khung của mỗi clip.
   - Lấy mẫu lại **clip** có hoàn lại, 2 000 lần, `rng = default_rng(0)` (`:52-54`, `:71`).
   - Thống kê là **trung bình có trọng số theo khung** của hiệu B − A, đúng thứ `summary.json` báo.
   - Lý do lấy mẫu theo clip: các khung trong một clip tương quan mạnh. Lấy mẫu theo khung sẽ cho khoảng tin cậy hẹp giả tạo (`:3-5`).
3. **Đọc kết quả:**
   - Khoảng tin cậy 95 % theo phân vị 2.5 / 97.5 (`:55`).
   - p hai phía = 2·min(P(d ≤ 0), P(d ≥ 0)) (`:57`).
   - "B tốt hơn / kém hơn" chỉ khi khoảng tin cậy không chứa 0, ngược lại "n.s." (`:58-61`).
4. **Không hiệu chỉnh** cho nhiều phép so sánh ⚠️. Tuy vậy, mọi kết luận chính có p < 0.001.

---

## 4. Giao thức đánh giá

- **Tập s200:** 200 clip test × 16 hàng = **3 200 khung**. Đúng các hàng mà D5 và D6 đã sinh (`index.json` của D5, `post_d6.sh:10`; `compose_hands.py:78-82`).
- **Mốc:**
  - GReg run5 trên đúng các khung đó (`/workspace/outputs/d5/greg_run5_eval_test/per_frame.csv`, `post_d6.sh:24`).
  - D1 lấy từ lần chấm u16 của D1 trên test, ghép cặp trên các hàng chung (`post_d6.sh:31`).
- **Chỉ số** ở 256², cùng bộ chấm như mọi hướng khác: L1, PSNR, SSIM, LPIPS (AlexNet), region L1 (ô 64×64), hand L1 (pixel tay của part map, đã giãn), hand LPIPS, face L1, `lowfreq_drift` (Gaussian σ = 8 px, **ngoài** tay).
- **Kiểm tra trước khi chạy thật:** smoke trên 6 clip (96 khung, 200 lần lấy mẫu) ✅.

---

## 5. Kết quả

### 5.1 Smoke test 6 clip (96 khung), 2026-11-05 01:58 UTC ✅

| | LPIPS | L1 | PSNR | SSIM | hand LPIPS | hand L1 | face L1 |
|---|---|---|---|---|---|---|---|
| GReg run5 | 0.03490 | 0.01102 | 30.75 | 0.9394 | 0.1777 | 0.0811 | 0.02579 |
| D6 `ng` | 0.03957 | – | – | – | – | – | – |
| **GReg + tay D6 `ng`** | **0.03283** (−5.9 %, KTC [−0.00363, −0.00062]) | 0.01069 (n.s.) | 30.95 (n.s.) | 0.9405 (tốt hơn) | **0.1377 (−22.5 %)** | 0.0657 (−18.9 %) | 0.02518 (n.s.) |

### 5.2 Toàn bộ chỉ số trên s200 (3 200 khung) ✅

| Mô hình | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | region L1 ↓ | hand L1 ↓ | hand LPIPS ↓ | face L1 ↓ | drift ↓ |
|---|---|---|---|---|---|---|---|---|---|
| GReg run5 | 0.01648 | 29.61 | 0.91789 | 0.04758 | 0.04369 | 0.1127 | 0.23645 | 0.03040 | 0.00774 |
| D6 `ng` (một mình) | 0.01913 | 28.22 | 0.91282 | 0.05274 | 0.04329 | 0.0909 | 0.18410 | 0.03647 | 0.01041 |
| D1 (một mình) | 0.01652 | 29.33 | không còn lưu | 0.04535 | không còn lưu | không còn lưu | 0.21934 | 0.03240 | 0.00773 |
| **GReg + tay D6 `ng`** | **0.01564** | **29.95** | **0.92018** | 0.04328 | **0.03891** | **0.0900** | **0.18407** | **0.03024** | **0.00749** |
| D1 + tay D6 `ng` | 0.01576 | 29.67 | 0.91988 | **0.04236** | 0.03995 | 0.0910 | **0.18407** | 0.03223 | **0.00749** |

### 5.3 Bootstrap so với GReg run5 (B − A, tương đối, [khoảng tin cậy 95 % của hiệu tuyệt đối], p) ✅

| Bản ghép | LPIPS | L1 | PSNR (dB) | hand LPIPS | face L1 | drift |
|---|---|---|---|---|---|---|
| **tay `ng` trên GReg** | **−9.0 %** [−0.00487, −0.00372] | **−5.1 %** [−0.00098, −0.00071] | **+0.35** [+0.26, +0.43] | −22.2 % | −0.5 % [−0.00035, +0.00001], p = 0.061, **n.s.** | −3.2 % |
| tay `ng_ones` trên GReg | −8.7 % | −4.9 % | +0.33 | −21.5 % | −0.5 %, p = 0.085, n.s. | −3.0 % |
| tay `n` trên GReg | −8.6 % | −4.9 % | +0.27 | −21.5 % | **+0.6 %**, p = 0.035 (kém hơn) | −3.2 % |
| tay `k` trên GReg | −8.2 % | −4.6 % | +0.24 | −20.8 % | **+0.6 %**, p = 0.047 (kém hơn) | −3.1 % |
| tay `ng` trên D1 | **−11.0 %** [−0.00588, −0.00462] | −4.3 % | +0.07, n.s. | −22.2 % | **+6.0 %** (kém hơn) | −3.3 % |
| tay `ng_ones` trên D1 | −10.6 % | −4.1 % | +0.05, n.s. | −21.5 % | +6.0 % (kém hơn) | −3.1 % |
| tay `n` trên D1 | −10.5 % | −4.2 % | −0.01, n.s. | −21.5 % | +7.2 % (kém hơn) | −3.2 % |
| tay `k` trên D1 | −10.2 % | −3.9 % | −0.04, n.s. | −20.8 % | +7.1 % (kém hơn) | −3.2 % |

Các ô không ghi chú khác đều có ý nghĩa với p < 0.001.

Giá trị tuyệt đối LPIPS của 8 bản ghép:

| | `ng` | `ng_ones` | `n` | `k` |
|---|---|---|---|---|
| trên GReg | 0.04328 | 0.04346 | 0.04349 | 0.04366 |
| trên D1 | 0.04236 | 0.04253 | 0.04256 | 0.04273 |

### 5.4 Bootstrap: D1 + tay D6 `ng` so với D1 một mình ✅

| | LPIPS | L1 | PSNR | hand LPIPS | face L1 | drift |
|---|---|---|---|---|---|---|
| D1 → D1 + tay D6 | 0.04535 → 0.04236 (**−6.6 %**) | −4.6 % | +0.34 dB | −16.1 % | −0.5 % (p = 0.042, tốt hơn) | −3.2 % |

---

## 6. Phân tích nguyên nhân và hiện tượng

### 6.1 Vì sao bản ghép tốt hơn **cả hai** thành phần

- L1 của bản ghép (0.01564) thấp hơn cả GReg (0.01648) lẫn D6 (0.01913).
- Lý do: lỗi của GReg **tập trung ở tay**. hand L1 của GReg là 0.113, gấp khoảng 7 lần L1 toàn khung (0.0165) ✅.
- Thay riêng vùng tay bằng D6 (hand L1 0.091) bỏ đi phần lỗi nặng nhất. Ngoài hộp tay, bản ghép **giữ nguyên pixel của GReg**, nên không mang theo lỗi mặt (+20 %) và trôi màu (+34 %) của D6.
- Nói cách khác, bản ghép lấy **giá trị tốt hơn của từng vùng**. Lỗi của nó gần với min(lỗi GReg, lỗi D6) theo từng vùng, không phải trung bình của hai.
- Bằng chứng ✅:
  - region L1 (ô 64×64 quanh mặt và hai tay): 0.0437 → 0.0389 (−11 %).
  - PSNR tăng 0.35 dB.
  - LPIPS toàn khung giảm 9 %. LPIPS nhạy với chi tiết cấu trúc, và tay là nơi GReg mất cấu trúc nhiều nhất (ngón dính, mờ).

### 6.2 Vì sao mặt giữ nguyên (và khi nào không)

- Hộp tay chỉ phủ mặt khi bàn tay ở gần hoặc trước mặt, điều thường gặp trong ngôn ngữ ký hiệu. Khi đó một phần pixel mặt lấy từ D6.
- Mức ảnh hưởng tới mặt phụ thuộc vào **mặt của D6 tệ đến đâu**:

| Biến thể tay | face L1 của D6 một mình | face L1 của bản ghép trên GReg |
|---|---|---|
| `ng`, `ng_ones` | +20 % | −0.5 %, n.s. |
| `n`, `k` | +37 % đến +39 % | +0.6 %, **có ý nghĩa** (p = 0.035 / 0.047) |

- Bằng chứng ✅ này cho thấy đường "rò" qua vùng tay chồng lên mặt là có thật nhưng nhỏ. Nó cũng là thêm một lý do chọn `ng`: warp theo mesh giữ mặt tốt hơn, nên phần mặt bị kéo theo vào hộp tay cũng sạch hơn.
- face L1 của `ng` trên GReg giảm nhẹ (−0.5 %, p = 0.061). Có thể vì ở vùng giáp ranh mặt–tay, khung D6 tốt hơn khung GReg mờ ⚠️.

### 6.3 Vì sao trôi màu còn **giảm** 3 %

- `lowfreq_drift` đo **ngoài** pixel tay, sau Gaussian σ = 8 px.
- Hộp ghép (2·e cộng viền mềm) phủ thêm cổ tay, cẳng tay và nền quanh tay. Ở các vùng đó D6 tốt hơn GReg: `arm_l1` 0.0516 → 0.0487, `disocc_l1` 0.0545 → 0.0503 ✅ (D6 `ng`).
- Thêm vào đó, Gaussian σ = 8 kéo một phần lỗi của tay sang pixel lân cận, nên tay tốt hơn cũng làm vùng quanh tay "đỡ trôi" hơn ⚠️.
- Mức giảm gần như y hệt ở cả 8 bản ghép (−3.0 đến −3.3 %). Nó đến từ vùng hộp tay, không phụ thuộc biến thể.

### 6.4 Chỉ số tay của bản ghép ≈ của D6: do cấu trúc

- hand LPIPS: bản ghép 0.18407, D6 0.18410 ✅. Hộp đo (1.5·e) nằm trong vùng m ≈ 1 của hộp ghép (2·e, viền khoảng 8 px), nên crop tay gần như là crop của D6.
- hand L1 của bản ghép (0.0900) thấp hơn D6 một chút (0.0909), vì một ít pixel tay của part map rơi vào vùng viền 0 < m < 1. Ở đó trộn hai ước lượng sai số gần độc lập làm giảm L1 ⚠️.
- **Không** được trình bày −22 % hand LPIPS như một phát hiện riêng của bản ghép. Đó là thành tích của D6. Đóng góp riêng của bản ghép là các chỉ số toàn khung ở §6.1–6.3.

### 6.5 Ghép lên D1: LPIPS thấp nhất nhưng mặt kém

- D1 + tay D6 có LPIPS 0.0424 (−11 %), thấp hơn bản ghép trên GReg (0.0433).
- **Nguồn lợi thêm:** D1 đã làm sắc mặt và vùng bị che bằng SD-Turbo một bước. Cổng của D1 mở cả mặt, nên LPIPS ngoài tay cũng giảm (D1 một mình: −4.7 %) ✅.
- **Cái giá:** face L1 của D1 kém GReg 6.6 % ✅. Lỗi này nằm ngoài hộp tay, nên bản ghép giữ nguyên nó (+6.0 %), và PSNR không còn hơn GReg (n.s.).
- So với D1 một mình, thêm tay D6 vẫn cải thiện mọi chỉ số (§5.4): LPIPS −6.6 %, hand −16 %, mặt không xấu đi. Như vậy hai cải tiến **cộng dồn được**, vì chúng tác động lên các vùng khác nhau.
- **Chọn bản nào** tuỳ tiêu chí:
  - Nếu ưu tiên độ trung thực, không đánh đổi chỉ số nào: **GReg + tay D6**.
  - Nếu chỉ tối thiểu LPIPS: **D1 + tay D6**.
  - Cải tiến hợp lý tiếp theo ⚠️: cho cổng D1 chỉ mở ngoài mặt (đã đề xuất cho D1), rồi ghép tay D6.

### 6.6 Biến thể D6 nào dùng để ghép gần như không quan trọng

- LPIPS của bốn bản ghép trên GReg nằm trong khoảng 0.0433–0.0437; hand LPIPS trong khoảng 0.184–0.187 ✅.
- Lý do: trong hộp tay, bốn biến thể D6 gần như bằng nhau (`D6_wan_video_dit.md` §6.3). Sự khác biệt giữa các biến thể (mặt, màu) nằm ngoài hộp, và bị loại khi ghép.
- Hệ quả thực tế ⚠️: một biến thể rẻ hơn như `k` (chỉ cần khung nguồn + skeleton, không cần warp mesh) cho bản ghép chỉ kém `ng` khoảng 0.9 % LPIPS. `ng` vẫn tốt nhất ở mọi chỉ số và là bản duy nhất không làm mặt kém đi có ý nghĩa.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế:**
1. **Chỉ 200 trong 1 146 clip test.** Muốn có số so trực tiếp với run5 trên toàn bộ test (0.04851) phải sinh D6 cho cả tập test: khoảng 7.5 GPU-giờ, khoảng $4.7, chưa làm vì hết credit.
2. **Một seed D6, một lần train, 3 000 bước.** Chưa biết lợi ích ghép dao động bao nhiêu giữa các seed.
3. **Chưa có chỉ số thời gian.** Hộp tay di chuyển theo tay từng khung, và hai nguồn có đặc tính thời gian khác nhau (GReg từng khung, D6 theo cửa sổ 49 khung). Có thể có **nhấp nháy ở viền ghép** và ở chỗ nối giữa các cửa sổ D6. FVD và `ccm_tc` chưa đo.
4. **Chỉ số tay bị chi phối bởi cấu trúc** (§6.4). Cần một chỉ số tay độc lập với hộp, ví dụ `hkd` đo keypoint tay bằng RTMPose trên ảnh sinh (kế hoạch §2.3), hoặc đánh giá cảm nhận của người.
5. **Chi phí suy luận** = GReg (khoảng 0.02 s/khung) + D6 (0.186 s/khung, 25 bước, RTX 4090), tức khoảng 10 lần GReg. D6 sinh cả khung chỉ để lấy hai hộp tay.
6. **Tham số ghép chưa tinh chỉnh** (lề 0.5, viền 9×9, chọn trước). Cũng chưa có ablation lề hoặc viền. Tinh chỉnh phải làm trên **val**, không phải test.
7. **Bootstrap không hiệu chỉnh nhiều phép so sánh.** Các kết luận chính có p < 0.001 nên không bị ảnh hưởng, nhưng các ô p ≈ 0.04 (face của `n`/`k`, face của D1 + tay) cần đọc thận trọng.
8. **Vẫn còn xa con số của bài báo GReg** (0.0433 so với 0.0359). Phần lớn khoảng cách có từ bước tái hiện run5, không liên quan tới hướng này.

**Nên làm tiếp:**
1. Train lại D6 `ng` (khoảng 2.5 h) rồi chạy **toàn bộ tập test** cho GReg + tay D6. Đây là con số cần cho bảng chính của luận văn.
2. **D6 chỉ cho vùng tay:** sinh video crop quanh tay ở độ phân giải cao hơn (ví dụ 128–256 px cho mỗi tay). Rẻ hơn nhiều lần so với sinh cả khung, và chi tiết ngón tay có thể tốt hơn.
3. Đo **nhấp nháy ở viền ghép** (ví dụ L1 thời gian trong dải viền) và FVD trên các clip có đủ khung.
4. Ablation lề {0.25, 0.5, 0.75} × viền {5, 9, 17} trên val.
5. Thử D1 với cổng chỉ mở ngoài mặt, rồi ghép tay D6, để có cả LPIPS thấp nhất lẫn mặt không đổi.

---

## 8. Trạng thái dữ liệu

Máy vast 58802139 đã bị xoá ngày 2026-11-05 khoảng 11:20 UTC. Drive đã đầy từ 2026-11-04 02:43 UTC, nên các kết quả dưới đây **chưa từng được đồng bộ**.

| Hạng mục | Trạng thái |
|---|---|
| 8 thư mục `outputs/post/compose/{v}_on_{greg,d1}/` (`per_frame.csv`, `summary.json`) | **mất** |
| Khung ghép | không lưu (ghép tính lại khi chấm, không ghi PNG); khung D6 nguồn **mất**; khung GReg run5 test và D1 test trên máy **mất** |
| `outputs/post/bootstrap.md` (22.6 KB, đủ 10 chỉ số cho mọi cặp) | **mất**. Còn các dòng LPIPS / L1 / PSNR / hand LPIPS / face L1 / drift đã trích trong hội thoại (§5.3–5.4) và bảng `summary.json` của `ng_on_greg` / `ng_on_d1` (§5.2) |
| Smoke 6 clip | số còn trong hội thoại (§5.1); file **mất** |
| Mã `compose_hands.py`, `bootstrap_ci.py`, `post_d6.sh` | **còn**: `docs/bao_cao_diffusion/code/` (chép từ scratchpad ngày 2026-11-06). Chưa có trên nhánh git nào |

Muốn tái lập cần:
1. Checkpoint GReg run5: còn trên Drive (`GReg_runs/run5`).
2. Checkpoint D1 run3: còn trên Drive (`GReg_runs/diff_common/outputs/d1/run3`).
3. Train lại D6 (mất).
4. Chạy `post_d6.sh`. Phụ thuộc: `/workspace/GReg-d1` (nhánh `feat/diff-d1-refiner`, có `--index` trong `21_eval_frames.py`) và `index.json` của 200 clip; file index này cũng **mất**, phải tạo lại bằng cùng cách chọn clip và 16 hàng như D5. ⚠️ Cache test của D6 dùng `pick_keys(test, 200, subset_seed = 0)` (`scripts/32_cache_wan_windows.py:103-104`, `scripts/vast/stages_d6.sh:50-51`). Chưa đối chiếu lại cách D5 chọn 16 hàng trên nhánh `feat/diff-d5-frame-ldm`.
