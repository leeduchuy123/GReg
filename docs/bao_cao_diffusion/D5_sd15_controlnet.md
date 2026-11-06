# D5: LDM theo từng khung (SD1.5 + ControlNet + reference attention), so sánh điều kiện keypoint / normal / mesh

Nhánh `feat/diff-d5-frame-ldm` (commit `3066971`, `d7fffbf`). Code thực sự chạy trên máy là nhánh cục bộ `fix-d5-tmp`: thêm 3 commit sửa lỗi `8b25da6`, `bbc331b`, `7019215`. Nhánh này **chưa được push lên GitHub**. Mọi trích dẫn `path:dòng` dưới đây lấy theo `fix-d5-tmp`.

Quy ước: ✅ đã kiểm chứng (code, log hoặc số đo) · ⚠️ giả thuyết hoặc suy luận, chưa đo.

---

## 0. Tóm tắt

- **Làm gì:** bỏ hẳn bộ sinh GAN của GReg, thay bằng SD1.5 sinh **từng khung**. Một ControlNet nhận điều kiện tư thế. Danh tính lấy từ ảnh nguồn qua *reference attention*, tức self-attention của UNet đọc thêm token của ảnh nguồn.
- **Mục tiêu:** một so sánh có kiểm soát trên cùng backbone, chỉ đổi loại điều kiện: keypoint 2D (`k`), normal map (`n`), normal kèm tín hiệu mesh (`ng`).
- **Đã chạy:** chỉ biến thể `ng`, 12 000 bước trên một RTX 4090, chấm trên **s200** (200 clip test × 16 khung = 3 200 khung). GReg run5 được chấm lại trên đúng các khung đó.
- **Kết quả:** `ng` đạt L1 0.1717 / PSNR 13.27 / SSIM 0.566 / LPIPS 0.383. GReg đạt 0.0165 / 29.61 / 0.918 / 0.0476. D5 kém **khoảng 8–10 lần**, và kém cả việc chép nguyên ảnh nguồn (L1 0.0257). Mô hình bỏ qua tư thế, bịa nền và làm mất danh tính.
- **Kết luận:** ❌ loại D5. `n` và `k` bị bỏ theo quyết định của người dùng. Câu hỏi "keypoint vs normal vs mesh" chuyển sang D6 (Wan2.1) để trả lời.

---

## 1. Bối cảnh

### 1.1 Vị trí trong chương trình
D1–D4 giữ GReg và chỉ chèn diffusion vào một khâu: sau ảnh ra (D1), prior vùng (D2), loss (D3) hoặc dữ liệu (D4). D5 và D6 thì **thay hẳn bộ sinh**: đầu vào vẫn là ảnh nguồn `I_s` cùng hình học SMPL-X của khung đích, nhưng ảnh ra do mô hình diffusion sinh. D5 là bản **theo từng khung, rẻ** (SD1.5, chạy trên 24 GB). D6 là bản **video** (Wan2.1-1.3B). Kế hoạch đặt D5 trước D6 vì chi phí chỉ bằng khoảng 1/10 (`docs/diffusion_6_huong_ke_hoach.md` §7.1).

### 1.2 So với các công trình liên quan
| Công trình | Backbone | Điều kiện tư thế | Giữ danh tính | Ghi chú |
|---|---|---|---|---|
| SignRefine | Wan-Fun-Control | keypoint 2D (RTMPose) render ra ảnh | – | Không công bố code hay dữ liệu. Table 5: thêm MANO và normal mặt **không giúp** tay, nhưng mesh đó fit bằng WiLoR nên nhiễu ✅ (`docs/notes_signrefine_signvip_signs_as_tokens.md`) |
| SignViP | SD1.5 + AnimateDiff + ReferenceNet | đặc trưng tư thế | ReferenceNet | 2.78 B tham số, 1.24 s/khung, DDIM 50 ✅ |
| Animate Anyone / MagicAnimate | SD1.5 + ReferenceNet + temporal | skeleton / DensePose | ReferenceNet là một **bản sao UNet train được** | Train nhiều GPU-ngày |
| **D5 (luận văn)** | SD1.5 + ControlNet | `k` / `n` / `ng` | reference attention dùng chính UNet đóng băng, kèm LoRA | Vừa 24 GB |

### 1.3 Câu hỏi khoa học
Với **cùng backbone, cùng cơ chế giữ danh tính, cùng dữ liệu, cùng số bước và cùng seed**, nếu chỉ đổi biểu diễn tư thế thì kết quả có theo thứ tự `ng > n > k` ở tay không? Nếu có, luận điểm trung tâm "mesh SMPL-X fit tốt giúp tay hơn keypoint 2D" có bằng chứng trực tiếp. Điều này cũng trả lời phản biện từ Table 5 của SignRefine (`docs/diffusion_6_huong_ke_hoach.md` §7.0).

---

## 2. Ý tưởng và điểm đặc sắc

### 2.1 Ba biến thể điều kiện lấy từ **cùng một** fit SMPL-X
Cả ba biến thể lấy từ cùng một fit SMPL-X, nên chỉ có **cách biểu diễn** khác nhau, độ chính xác của tư thế là như nhau:
- **`k`, keypoint 2D (kiểu SignRefine):** khớp SMPL-X chiếu xuống 2D rồi vẽ thành skeleton.
  - Xương thân, 2 × 21 khớp tay (gồm cả đầu ngón), 68 điểm mặt.
  - Mỗi xương một màu riêng để phân biệt trái/phải và từng ngón; điểm mặt là chấm trắng (`greg/diffusion/frame_ldm.py:34-80`).
- **`n`, normal map:** chính điều kiện của GReg.
- **`ng`, normal kèm tín hiệu mesh** (`frame_ldm.py:95`, `scripts/30_train_frame_ldm.py:59-64`). Có 10 kênh:
  - normal map (3 kênh);
  - ảnh nguồn warp sang khung đích theo tương ứng mesh, nhân với `V·C` (3 kênh);
  - `V·C`, mặt nạ "điểm ảnh này có thật trong ảnh nguồn và tin được" (1 kênh);
  - **CCM**, toạ độ T-pose của từng điểm ảnh đích (3 kênh).

  Đây là những thông tin **chỉ mesh mới cho được**: điểm nào nhìn thấy, tương ứng nguồn ↔ đích, và vị trí chuẩn hoá trên cơ thể.

### 2.2 Cái gì mới, cái gì mượn
| Thành phần | Nguồn |
|---|---|
| ControlNet sao từ encoder UNet, zero-conv | Mượn (Zhang et al. 2023), dùng `ControlNetModel.from_unet` của diffusers ✅ |
| Reference attention: self-attention đọc thêm token của ảnh nguồn | Ý tưởng mượn từ "reference-only" / Animate Anyone. **Hiện thực riêng** 🛠: không có ReferenceNet; dùng chính UNet đóng băng cùng LoRA, chạy một lần ở t = 0 |
| Điều kiện `ng` (warp mesh + V·C + CCM đưa vào ControlNet) | **Riêng của luận văn** 🛠. CCM theo bai_hoc §K2 |
| Loss trọng số vùng ×10 ở tay và mặt | Mượn từ SignRefine Eq. 4 (`w_h = w_f = 10`) ✅ |
| Thiết kế thí nghiệm "cùng backbone, chỉ đổi điều kiện" | **Riêng** 🛠: SignRefine và SignViP không có đối chứng này |
| Nhiễu khởi tạo chung cho mọi khung của một clip | 🛠, để giảm nhấp nháy khi sinh từng khung độc lập |

Điểm đặc sắc nằm ở **thiết kế thí nghiệm**: mọi thứ trừ điều kiện đều được giữ cố định, kể cả cặp train (cùng seed, `30_train_frame_ldm.py:7-8`). Kiến trúc tự nó không mới.

---

## 3. Kiến trúc chi tiết

### 3.1 Sơ đồ (biến thể `ng`, làm việc ở 512²)

```
Cache 256²  ──bicubic ×2──►  512²  (30_train_frame_ldm.py:37-38: KHÔNG render lại cache 512 như kế hoạch)

Ảnh nguồn I_s (3,512,512) ─VAE enc (đóng băng, mode())─► z_ref (4,64,64)
        │  UNet SD1.5 (đóng băng + LoRA r16), t = 0, no_grad, bank.mode = "write"
        │  → mỗi self-attention (attn1) lưu hidden_states của nó: (B, H·W, C), C ∈ {320,640,1280}
        ▼
   Reference bank  (16 lớp attn1: 64²·320, 32²·640, 16²·1280, 8²·1280 token …)

Điều kiện c (10,512,512) = [N_t (3) | warp(I_s)·V·C (3) | V·C (1) | CCM (3)]
        │  ControlNet (bản sao encoder + mid block của UNet, train toàn bộ;
        │  conditioning embedding: conv 10 → … → 320 kênh ở 64², zero-conv)
        ▼
   12 residual down + 1 residual mid  ──────────────────────────────┐
                                                                     ▼
z_t (4,64,64) ─► UNet SD1.5 (đóng băng + LoRA r16 trên to_q/k/v/out)
                  attn1:  Q = x_t,  K,V = [x_t ; bank]   (bank.mode = "read")
                  attn2:  cross-attention với prompt cố định (CLIP-L, 77×768)
                ─► ε̂ (4,64,64)

Train:  loss = Σ w·‖ε̂ − ε‖² / Σ w,   w = 1 + 9·maxpool(mask mặt ∪ tay trái ∪ tay phải, 64²)
Infer:  DDIM 25 bước, không CFG, z_T = randn(seed = chỉ số clip) dùng chung cho mọi khung của clip
        → VAE dec → 512² → hạ mẫu area về 256²
```

### 3.2 Từng khối

**Backbone.** SD1.5 (`stable-diffusion-v1-5`): VAE, UNet và text encoder đều **đóng băng** (`frame_ldm.py:162-163`). Prompt cố định, được mã hoá một lần, text encoder sau đó bị xoá (`frame_ldm.py:32`, `:180-186`):

> "a person signing in sign language, sharp hands and fingers, studio video, uniform lighting"

**ControlNet** (`frame_ldm.py:178`):
- Tạo bằng `ControlNetModel.from_unet(unet, conditioning_channels=C)`, khởi tạo từ trọng số encoder của UNet; các zero-conv ở đầu ra bằng 0.
- Số kênh điều kiện là `{"k": 3, "n": 3, "ng": 10}` (`frame_ldm.py:83`).
- Điều kiện được đưa về 512². Normal, warp và V·C dùng nội suy bilinear; CCM dùng nearest để không trộn toạ độ của hai bộ phận khác nhau (`frame_ldm.py:86-96`).
- Với `ng`, điều kiện tính **trên GPU cho từng cặp** bằng `MeshCorrespondence` (`edge_px=4`) và `canonical_coordinate_map`. Ảnh warp dùng hàm `warp` (grid sample) của GReg (`30_train_frame_ldm.py:41-64`).

**Reference attention** (`frame_ldm.py:100-152`, `:222-227`):
- Mọi processor `attn1` của UNet được thay bằng `RefSelfAttnProcessor`; `attn2` (cross-attention) giữ `AttnProcessor2_0` (`:150`).
- **Pha ghi** `write_reference`: chạy UNet trên latent **sạch** của ảnh nguồn với `t = 0`, dưới `no_grad`. Mỗi `attn1` lưu `hidden_states.detach()` (`:121`, `:224`).
- **Pha đọc**: mỗi `attn1` tính `K, V` trên `[token hiện tại ; token đã lưu]` (`:126`). Query vẫn chỉ là token hiện tại, nên ảnh sinh "tra cứu" được ngoại hình của ảnh nguồn.
- Vì `t = 0`, bank chỉ cần tính **một lần mỗi khung** rồi dùng lại cho cả 25 bước lấy mẫu.
- Hệ quả thiết kế: pha ghi không có gradient, nên **cách UNet mã hoá ảnh nguồn không được học**. LoRA chỉ học được cách *đọc* bank, qua `to_q/to_k/to_v` của pha đọc ⚠️ (suy ra từ code, xem §6).

**LoRA** (`frame_ldm.py:164-167`): r = 16, α = 16, khởi tạo gaussian, gắn trên `to_q, to_k, to_v, to_out.0` của mọi attention trong UNet (cả `attn1` và `attn2`). ControlNet được tạo **trước** khi gắn LoRA, nên không có LoRA.

**Số tham số train** (log in ra số tham số không còn lưu):
- ControlNet khoảng 361 M ⚠️ (kích thước chuẩn của ControlNet-SD1.5).
- LoRA khoảng 3.2 M ⚠️, tính tay: 16 transformer block, mỗi block 14·r·C + 1536·r tham số.

**Loss** (`frame_ldm.py:237-254`, `30_train_frame_ldm.py:166-172`):
- Dự đoán ε theo lịch DDPM của SD1.5; `t` lấy đều trong [0, 1000).
- MSE trung bình theo kênh latent, nhân trọng số `w = 1 + 9 · adaptive_max_pool(mask, 64²)`.
- `mask` = mặt nạ part 2/3/4 (mặt, tay trái, tay phải), giãn 3 px, cứng.
- Tương đương **×10 ở tay và mặt**.

**Lấy mẫu** (`frame_ldm.py:256-270`, `scripts/31_sample_frame_ldm.py:74-91`):
- `DDIMScheduler` 25 bước, guidance 1 (không CFG, nên chỉ một lượt UNet mỗi bước).
- Nhiễu đầu `randn(seed = chỉ số clip)`, giống nhau cho mọi khung của clip.
- Mỗi khung sinh **độc lập**: không có attention theo thời gian.

**Dữ liệu train** (`configs/diffusion/d5_frame_ldm.yaml:9-11`, `30_train_frame_ldm.py:96-103`):
- Cặp (nguồn, đích) của GReg trên tập `train_fit` (train trừ val), 8 cặp mỗi clip.
- 50 % ảnh nguồn là khung hợp lệ đầu tiên của clip (tư thế nghỉ, giống giao thức test).
- Chỉ dùng clip có mesh buffer.

### 3.3 Siêu tham số (`configs/diffusion/d5_frame_ldm.yaml`)
| | Giá trị |
|---|---|
| Độ phân giải làm việc | 512² (từ cache 256² phóng bicubic) |
| Bước train | 12 000 |
| Batch | 2 × tích luỹ 2 = 4 cặp/bước, tổng cộng 48 000 cặp |
| lr | ControlNet 1e-5, LoRA 1e-4, AdamW, weight decay 0.01, warmup 500, clip grad 1.0 |
| Độ chính xác | bf16 autocast (`cache_enabled=False`), gradient checkpointing cho UNet và ControlNet |
| Lấy mẫu | DDIM 25 bước, không CFG |

### 3.4 Khác kế hoạch §7.2
- **Không render lại cache 512²** (kế hoạch ước tính khoảng 4 h); dùng ảnh 256² phóng lên.
- **Không có CFG** và không dropout điều kiện (kế hoạch: CFG 2, dropout p = 0.1).
- **Không có attention thời gian** (K3, Generative Rendering) lúc infer, và không có `L_temp`.
- **Không có IP-Adapter.**
- Không chạy ablation 4–6.

Các lựa chọn này đều để vừa ngân sách: cùng thời gian và tiền cho cả ba biến thể ✅ (đọc từ code). Ảnh hưởng của chúng tới kết quả được bàn ở §6.

### 3.5 Chi phí đo được
| | Giá trị |
|---|---|
| Tốc độ train | 0.93–0.95 s/bước ✅ (log bước 200, 250, 850, 8 900–9 000) |
| Thời gian train | 12 000 bước ≈ 3.1 h. Bắt đầu 01:39:36 UTC 04/11; đến 04:02 UTC đạt bước 9 000 sau 142 phút ✅ |
| VRAM | `max_memory_allocated` 10.5 GB ở bước 200–850, 17.2 GB ở bước 8 900 (gồm cả preview). `nvidia-smi` báo 19 220 MiB ✅ |
| Lấy mẫu + chấm 3 200 khung | Khoảng 30 phút ⚠️ (suy từ log: train xong khoảng 04:50, bước chấm GReg bắt đầu 05:19). `timing.json` (s/khung) không còn lưu |
| Tiền | Khoảng 3.7 h × $0.62 ≈ $2.3 ⚠️, chưa tính khoảng 2 h GPU bỏ không (xem §5.3) |

---

## 4. Giao thức đánh giá

- **Tập s200** (`scripts/31_sample_frame_ldm.py:61-70`):
  - 200 clip test, chọn bằng `pick_keys(…, 200, seed=0)`.
  - Ảnh nguồn là khung hợp lệ đầu tiên (`source_row=0`).
  - 16 khung đích lấy đều trong clip.
  - Tổng 3 200 khung, danh sách ghi vào `index.json`.
- **Đầu ra:** sinh ở 512², hạ mẫu bằng `area` về 256² (bằng độ phân giải cache) rồi lưu PNG (`31_sample_frame_ldm.py:87-91`).
- **Cách chấm:** `scripts/21_eval_frames.py`, cùng bộ chỉ số với mọi nhánh: L1, PSNR, SSIM, LPIPS-AlexNet, region L1, hand L1/LPIPS, face L1, arm L1, disocc L1, lowfreq drift, và các chỉ số "@large" ở cử động lớn.
- **Đối chứng:** GReg run5 (epoch 100) chấm lại trên **đúng 3 200 khung** đó bằng `--index` ✅. Dòng chép ảnh nguồn (`source_copy_l1`) và dòng ảnh warp TPSMM (`warped_l1`) được tính trong cùng lần chấm.
- Tập s200 cũng là tập đo D6, nên D5 và D6 so trực tiếp được.

---

## 5. Kết quả

### 5.1 Đường train (biến thể `ng`, 8 clip val, preview DDIM 25 bước) ✅
| Bước | 10 (smoke) | 2 000 | 4 000 | 6 000 | 8 000 |
|---|---|---|---|---|---|
| L1 preview | 0.2957 | 0.2028 | 0.1903 | 0.2003 | 0.1900 |

- Loss train dao động 0.017–0.144 (ví dụ bước 8 900 / 8 950 / 9 000: 0.0417 / 0.0172 / 0.1411). Đây là bình thường với loss ε khi `t` lấy ngẫu nhiên, nên không dùng để đọc hội tụ.
- L1 preview giảm từ 0.30 xuống 0.20 trong 2 000 bước đầu, rồi **đi ngang** từ 2k đến 8k.

**Ảnh preview ở bước 8 000** (4 cột: nguồn | điều kiện | D5 | đích) ✅ (quan sát trực tiếp lúc chạy, ảnh không còn lưu):
- **Không theo tư thế:** bản đồ tư thế để tay buông, nhưng ảnh sinh lại giơ tay như đang ký.
- **Bịa nền:** nền xanh trơn của dữ liệu biến thành nền "vẽ".
- **Lỗi bố cục:** có hàng sinh ra **hai người**, có hàng mất hẳn danh tính người ký.

### 5.2 Kết quả trên s200 (3 200 khung) ✅
| Chỉ số | GReg run5 | D5 `ng` | D5 so với GReg |
|---|---|---|---|
| L1 ↓ | **0.0165** | 0.1717 | +942 % |
| PSNR ↑ | **29.61** | 13.27 | −16.3 dB |
| SSIM ↑ | **0.9179** | 0.5661 | −38 % |
| LPIPS ↓ | **0.0476** | 0.3829 | +705 % |
| region L1 ↓ | **0.0437** | 0.1728 | |
| hand L1 ↓ | **0.1127** | 0.2215 | +97 % |
| hand LPIPS ↓ | **0.2365** | 0.5288 | +124 % |
| face L1 ↓ | **0.0304** | 0.1438 | +373 % |
| arm L1 ↓ | **0.0516** | 0.2212 | +329 % |
| disocc L1 ↓ | **0.0545** | 0.1865 | +242 % |
| lowfreq drift ↓ | **0.0077** | 0.1559 | **+1 913 %** |
| hand L1 @large ↓ | **0.1425** | 0.2294 | +61 % |
| LPIPS @large ↓ | **0.0708** | 0.4038 | +470 % |

Hai mốc tham chiếu trên cùng các khung: **chép nguyên ảnh nguồn** L1 0.0257; **ảnh warp TPSMM** của run5 L1 0.0208. Tỉ lệ vùng phải vẽ mới (`disocc_frac`) = 0.218.

Mốc ngoài: Wan2.1-Fun **chưa train** (D6 zero-shot) trên cùng s200 đạt LPIPS 0.144 ✅. Tức là một mô hình video không train vẫn tốt hơn D5 đã train 12 000 bước.

### 5.3 Biến thể `n` và `k`: không chạy
Lúc 04:06 UTC ngày 04/11, sau khi xem preview bước 8 000, người dùng đồng ý **bỏ `n` và `k`**. File trạng thái `d5_{train,eval}_{n,k}.done` được ghi "skipped" ✅. Lý do:
1. Khoảng cách tới GReg là khoảng 10 lần, và nằm ở **toàn ảnh** (nền, màu, danh tính), không phải ở tay. `n` và `k` chỉ khác `ng` ở điều kiện, và còn **ít** thông tin hơn `ng` (không có ảnh warp). Vì vậy không có lý do để kỳ vọng chúng thu hẹp được khoảng cách đó ⚠️.
2. Bỏ hai biến thể tiết kiệm khoảng 6–7 h GPU (≈ $4), đủ để chạy D4 100 epoch, D1 trên test và D6 với credit còn lại.
3. So sánh `k`/`n`/`ng` chỉ có nghĩa khi backbone đã sinh được ảnh hợp lý. Câu hỏi này được trả lời ở D6. Trên s200, D6 cho LPIPS `ng` 0.0527 < `n` 0.0544 < `k` 0.0553 và face L1 0.0365 < 0.0417 < 0.0422 ✅.

### 5.4 Sự cố khi chạy ✅
| Lúc (UTC) | Lỗi | Sửa |
|---|---|---|
| 03/11 23:29 | `ModuleNotFoundError: greg.diffusion.turbo_refiner` (D5 import prompt từ module chỉ có ở nhánh D1) | `8b25da6`: định nghĩa `DEFAULT_PROMPT` ngay trong `frame_ldm.py` |
| 04/11 01:35 | `CheckpointError`: gradient checkpointing chạy lại attention trong backward, nhưng lúc đó bank đã chuyển sang `off`, nên K/V mất token tham chiếu và lệch kích thước | `bbc331b`: bank giữ chế độ `read` qua cả backward (`frame_ldm.py:247-248`) |
| 04/11 01:36 | `CheckpointError` lần hai. Autocast lưu bản bf16 của trọng số LoRA từ pha ghi (chạy `no_grad`), rồi pha có grad dùng lại bản đó, nên **LoRA không nhận gradient** và lượt tính lại thấy tensor khác | `7019215`: `torch.autocast(..., cache_enabled=False)` (`30_train_frame_ldm.py:167-170`) |
| 04/11 05:19 | Bước chấm GReg trên cùng khung lỗi vì `21_eval_frames.py` của nhánh D5 không có `--index` | Chạy từ checkout D1 (nhánh có `--index`) |

Lỗi thứ ba đáng ghi lại: nếu chỉ tắt checkpointing thay vì sửa tận gốc, mô hình vẫn train "bình thường" nhưng LoRA **đứng yên** mà không ai biết. Lỗi đầu khiến GPU bỏ không khoảng 2 h (23:29 → 01:33) vì lúc đó watcher chưa bật.

---

## 6. Phân tích nguyên nhân

### 6.1 Thất bại nằm ở toàn ảnh, không chỉ ở tư thế ✅
Phân rã sai số cho thấy chỗ hỏng nặng nhất **không phải ở tay**:

| | GReg | D5 | Tỉ lệ |
|---|---|---|---|
| lowfreq drift (màu và nền, ngoài tay, sau Gaussian σ = 8) | 0.0077 | 0.1559 | **×20** |
| L1 toàn ảnh | 0.0165 | 0.1717 | ×10 |
| face L1 | 0.0304 | 0.1438 | ×4.7 |
| hand L1 | 0.1127 | 0.2215 | ×2 |

Sai số tần số thấp gấp 20 lần nghĩa là **màu nền, màu áo và vị trí người đều sai**, không chỉ chi tiết ngón tay. Điều này khớp với preview: nền bị bịa, có khi hai người, mất danh tính.

**Vì sao PSNR chỉ 13 dB:** PSNR 13.27 dB tương ứng MSE ≈ 10^(−1.327) ≈ 0.047, tức RMSE ≈ 0.22 trên thang [0, 1] ✅ (tính từ số đo). Sai trung bình khoảng 22 % cường độ ở **mọi** điểm ảnh chỉ xảy ra khi phần lớn khung hình (nền chiếm đa số diện tích) bị sai. GReg chép nền gần như nguyên vẹn qua warp, nên giữ được khoảng 29.6 dB.

### 6.2 Mô hình không dùng thông tin có sẵn trong điều kiện ✅
Kênh `warp(I_s)·V·C` của `ng` đã chứa sẵn ảnh nguồn được warp sang khung đích. Chỉ cần chép kênh đó ra là đạt L1 cỡ 0.021 (ảnh warp TPSMM trên cùng khung đạt 0.0208; ảnh nguồn nguyên vẹn đạt 0.0257). D5 cho L1 0.17, **kém gấp 8 lần việc chép ảnh nguồn**.

Vậy sau 12 000 bước, đường ControlNet gần như chưa truyền được nội dung của điều kiện vào ảnh ra. Ảnh ra chủ yếu đến từ prior của SD1.5.

### 6.3 Các nguyên nhân có khả năng (xếp theo mức tin cậy)

1. **ControlNet chưa "bắt" được điều kiện. Ngân sách train quá nhỏ.** ⚠️ (mạnh)
   - 12 000 bước × 4 cặp = 48 000 mẫu, lr ControlNet 1e-5 ✅ (cấu hình).
   - ControlNet khởi đầu bằng zero-conv nên thoạt đầu UNet **không thấy** điều kiện. Bài ControlNet mô tả hiện tượng "hội tụ đột ngột": mô hình bỗng theo điều kiện sau một số bước nhất định, và họ train với batch lớn trên hàng trăm nghìn tới hàng triệu ảnh.
   - L1 preview đi ngang 0.20 → 0.19 từ 2k đến 8k ✅, cho thấy chưa đạt tới điểm đó.
   - Điều kiện `ng` 10 kênh lại **khác hẳn** ảnh tự nhiên: CCM, mặt nạ, normal. Lớp conditioning embedding phải học từ đầu.

2. **Reference attention quá yếu để giữ danh tính và nền.** ⚠️ (mạnh)
   - Bank được ghi ở t = 0 bằng UNet đóng băng, và `detach` ✅ (`frame_ldm.py:121`). Cách mã hoá ảnh nguồn **không được học**; chỉ LoRA r16 (khoảng 3 M tham số) học cách đọc nó.
   - Animate Anyone và SignViP dùng ReferenceNet là **cả một UNet train được** cho đúng việc này.
   - Ở bước nhiễu lớn, query của ảnh nhiễu và key của ảnh sạch t = 0 nằm ở hai phân bố khác nhau. Attention vì vậy có thể bỏ qua bank ⚠️.
   - Hệ quả quan sát được: nền và danh tính bị thay bằng prior của SD ✅ (preview, drift ×20).

3. **Prior của SD1.5 cùng prompt cố định kéo về ảnh "đẹp" chung chung.** ⚠️
   - Prompt có "studio video, uniform lighting" ✅ (`frame_ldm.py:32`).
   - Khi điều kiện và tham chiếu yếu, mô hình sinh "một người trong studio" theo prior. Điều này khớp với nền vẽ, ảnh hai người, tay giơ lên kiểu "đang ký" bất kể điều kiện. Ảnh nhiều người là lỗi bố cục quen thuộc của SD1.5 ở 512² khi điều khiển yếu.
   - Không có CFG ✅, nên không có cơ chế khuếch đại ảnh hưởng của điều kiện lúc lấy mẫu.

4. **Lệch miền: ảnh đích 256² phóng bicubic lên 512².** ⚠️
   - Ảnh đích bị mờ so với ảnh 512² SD1.5 đã học ✅ (`30_train_frame_ldm.py:37-38`, `:171`). Mô hình phải học "sinh ảnh mờ", ngược với prior.
   - Ảnh ra cuối cùng còn qua VAE của SD, vốn mất chi tiết ở mặt và tay nhỏ ở 256² (cũng thấy ở D1 và D6: face L1 kém hơn GReg).
   - Yếu tố này giải thích được sai số ở chi tiết, **không** giải thích được drift ×20.

5. **Dữ liệu ít và đơn điệu.** ⚠️ Khoảng 4 000 clip train, ít người ký, nền đơn giản. Đủ cho GAN của GReg (vì GReg chủ yếu *chép* từ ảnh nguồn), nhưng ít để fine-tune ControlNet khoảng 361 M tham số tới mức theo tư thế từng ngón.

6. **Sinh từng khung độc lập.** ✅ (thiết kế) Không có attention thời gian, nên mỗi khung có thể "chọn" một người hoặc nền khác. Nhiễu chung theo clip giảm được phần này nhưng không loại bỏ. Yếu tố này ảnh hưởng tới độ nhất quán và FVD hơn là L1 từng khung.

### 6.4 Đối chiếu với D6 (cùng s200)
D6 (Wan2.1-Fun-1.3B-InP + LoRA, 3 000 bước) đạt LPIPS 0.0527 với cùng loại điều kiện `ng`. Khác biệt chính so với D5:
- **Inpainting đúng nghĩa:** điểm ảnh tin được (V·C) được đặt vào khe "masked video", nên mô hình *giữ* chúng thay vì phải học cách chép.
- **Danh tính** qua CLIP ảnh và khung đầu thật.
- **Prior video.**

Đối chiếu này ủng hộ nguyên nhân 1 và 2: nút thắt của D5 là **cơ chế đưa ảnh nguồn và điều kiện vào mô hình**, không phải loại điều kiện ⚠️. Muốn kiểm chứng phải chạy D5 với ReferenceNet hoặc với ảnh warp đưa thẳng vào latent.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế của kết quả:**
- Chỉ một biến thể (`ng`), một seed, 12 000 bước. **Chưa trả lời** được câu hỏi `k`/`n`/`ng` trên SD1.5.
- Lấy mẫu chỉ ở 25 bước DDIM, không CFG; chưa thử CFG hay số bước khác.
- s200 chỉ là 200 trong 1 146 clip test, dù với GReg tập này đại diện tốt (LPIPS 0.0476 so với 0.0485 trên toàn tập test).

**Nếu muốn cứu D5** (theo thứ tự rẻ đến đắt, đều ⚠️ chưa thử):
1. Đưa `warp(I_s)·V·C` thẳng vào **latent** (concat vào `conv_in` kiểu inpainting, như D6 và DWM) thay vì chỉ qua ControlNet. Có thể dùng sẵn SD1.5-inpainting làm backbone.
2. Train lâu hơn (≥ 50k bước) với batch lớn hơn và lr ControlNet 1e-4; thêm dropout điều kiện cùng CFG.
3. Thay reference attention bằng ReferenceNet train được (cần GPU lớn hơn) hoặc IP-Adapter.
4. Render cache 512² thật (khoảng 4 h) thay cho phóng bicubic.
5. Chỉ khi ảnh đã hợp lý (L1 ≲ chép ảnh nguồn) mới chạy `n`/`k` để làm so sánh có kiểm soát.

**Khuyến nghị cho luận văn:** ghi D5 là **kết quả âm có giải thích**: "LDM từng khung với ControlNet và reference attention nhẹ, ở ngân sách 1 GPU, không học được việc chép danh tính và nền". Phần so sánh điều kiện dùng D6. Không nên đầu tư thêm vào D5 nếu ngân sách vẫn như hiện tại.

---

## 8. Trạng thái dữ liệu

| Thứ | Ở đâu | Còn không |
|---|---|---|
| Code D5 (bản gốc) | GitHub `feat/diff-d5-frame-ldm` | ✅ |
| 3 commit sửa lỗi (`8b25da6`, `bbc331b`, `7019215`) | nhánh cục bộ `fix-d5-tmp` | ✅ cục bộ, **chưa push** |
| Checkpoint `ng` (`final.pt`, `last.pt`) | `/workspace/outputs/d5/ng` trên vast | ❌ mất (máy đã xoá 05/11) |
| Ảnh preview `samples/step_*.jpg` | vast | ❌ mất |
| 3 200 khung test, `index.json`, `timing.json` | vast | ❌ mất |
| `per_frame.csv`, `summary.json`, `results_test.md` | vast | ❌ mất |
| Log loss / s/bước / VRAM (`logs/loss.csv`) | vast | ❌ mất; chỉ còn các dòng trích ở §3.5 và §5.1 |
| Số liệu tổng hợp ở §5 | hội thoại và memory | ✅ |

Drive đầy từ 04/11 02:43 UTC (403 `storageQuotaExceeded`). D5 bắt đầu train lúc 01:39 và mọi output sinh sau mốc đó, nên **không có gì của D5 lên được Drive**. `index.json` của s200 cũng mất. Tuy vậy tập này tái tạo được tất định từ `pick_keys(test, 200, seed=0)`, `source_row=0`, 16 khung lấy đều.
