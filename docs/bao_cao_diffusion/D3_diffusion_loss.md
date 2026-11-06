# Báo cáo D3: Diffusion làm hàm loss khi huấn luyện GReg (E-LatentLPIPS)

Nhánh `feat/diff-d3-diffusion-loss`.
- Code chính ở `origin/feat/diff-d3-diffusion-loss` @`e4fd080`.
- Commit sửa lỗi `0211c3a` nằm trên nhánh local `fix-d3-tmp` và **chưa được push** lên `feat/diff-d3-diffusion-loss`. Xem §8.
- Chạy trên vast.ai instance 58802139 (RTX 4090) ngày 2026-11-03.

Ký hiệu: ✅ đã kiểm chứng (code, log hoặc số đo) · ⚠️ giả thuyết hoặc ước lượng, chưa đo riêng.

---

## 0. Tóm tắt

- **Làm gì:** thêm vào loss huấn luyện của GReg một khoảng cách cảm nhận tính trong **không gian latent của Stable Diffusion 1.5**, gọi là E-LatentLPIPS. Mô hình diffusion (VAE encoder + VGG trên latent) bị **đóng băng** và chỉ đóng vai "người chấm" lúc train. Khi suy luận, GReg chạy y như cũ, **không có diffusion và không tốn thêm chi phí**.
- **Ba biến thể, so với `base` (sweep15, val 100 clip × 16 khung)** ✅:

  | Biến thể | Thay đổi | LPIPS | hand LPIPS | L1 |
  |---|---|---|---|---|
  | `elp1` | thêm E-LatentLPIPS toàn ảnh | −0.8 % | +1.7 % | +1.5 % |
  | `elp1_hand` | thêm cả trên crop tay | +2.0 % | +2.6 % | +3.6 % |
  | `elp1_noadv` | bỏ 3 PatchGAN, chỉ giữ critic diffusion | **+13.7 %** | +7.1 % | +12.7 % |

- **Kết luận:**
  - D3 bị loại.
  - Latent của SD không thêm thông tin đáng kể so với VGG perceptual mà GReg đã dùng.
  - Một critic diffusion **đóng băng** không thay được discriminator **được huấn luyện**.
  - Thêm loss làm thời gian train tăng khoảng 45–100 % mà không đổi lại được gì.

---

## 1. Bối cảnh

### 1.1 GReg run5 làm gì (phần liên quan tới loss)

GReg sinh khung đích `Î` từ ảnh nguồn `I_s` và normal map SMPL-X `N_t`:
1. TPSMM đóng băng warp `I_s`.
2. Regional Expert tạo prior cho mặt và tay.
3. GRM điều biến bộ sinh.
4. Ảnh cuối `Î = O·I_st + (1−O)·I_gen`.

Số tham số train là 34.1 M, discriminator 2.0 M ✅.

Loss gốc (`greg/losses.py:1-3, 126-162`):

```
L_G = λ_per · L_VGG(Î, I_t)                                      VGG-19 pyramid 4 scale (1, ½, ¼, ⅛), 5 lớp × trọng số 10
    + λ_rper · (1/R) Σ_r w_r · L_VGG(crop_r(Î)↑128, crop_r(I_t)↑128)   crop mặt / tay T / tay P, phóng lên 128²
    + λ_struct · L_struct                                         L1 prior RE với gradient crop đích
    + λ_adv · L_adv                                               3 PatchGAN theo vùng, đầu vào (crop RGB, gradient) 4 kênh
λ_per = 1, λ_rper = 3 (run5), λ_struct = 10, λ_adv = 1, GAN dạng bce non-saturating
```

Cấu hình lấy từ `configs/hamnosys_greg.yaml` và phần `common` của sweep ✅.

### 1.2 D3 sửa phần nào

D3 chỉ **cộng thêm số hạng** vào `GRegLoss.generator_loss` (`losses.py:164-171`, nhánh D3). Biến thể `noadv` thì bỏ số hạng GAN và không tạo discriminator (`06_train_greg.py:110-112`). Kiến trúc GReg, dữ liệu và lịch train giữ nguyên.

### 1.3 Vì sao chọn hướng này

- **Rẻ lúc suy luận:** đây là hướng duy nhất trong sáu hướng đưa tri thức diffusion vào **mà tốc độ suy luận vẫn bằng GReg**. Diffusion chỉ xuất hiện lúc train.
- **Giả thuyết:** không gian latent của SD được học trên hàng tỉ ảnh, nên có thể nhạy hơn VGG-ImageNet với kết cấu tự nhiên như da, ngón tay, vải. Dùng nó làm thước đo cảm nhận thì GReg có thể sinh chi tiết sắc hơn.
- **Câu hỏi nghiên cứu:** "critic diffusion đóng băng" có **thay thế hoặc bổ sung** được GAN không? GAN trong GReg vốn khó ổn định.

---

## 2. Ý tưởng và điểm mới

### 2.1 Chọn công thức: vì sao E-LatentLPIPS, không phải DMD hay DIFT

Nhóm đã đọc code của ba họ phương pháp (`docs/code_notes_diffusion_as_loss.md` §0, §1.5, §2, §7) ✅:

| Công thức | Thứ đóng băng | Vấn đề với GReg trên 24 GB | Quyết định |
|---|---|---|---|
| **DMD / DMD2** (distribution matching) | UNet "real score" | Cần thêm **UNet "fake score" phải train** (0.86 B tham số; fp32 + AdamW khoảng 14 GB riêng cho phần này), cập nhật 5–10 lần cho mỗi bước G ⇒ vượt 24 GB, chậm gấp 5–10 lần. Critic không còn "đóng băng", nên câu hỏi nghiên cứu mất ý nghĩa. Real score chỉ thấy text, không thấy tư thế hay danh tính, nên chỉ khớp phân phối biên ("ảnh trông thật"), không ràng buộc tái tạo | **Loại** |
| **DIFT** (đặc trưng UNet SD ở t ≈ 261, `up_blocks[1]`) | VAE + UNet SD | Đắt gấp 3–6 lần VGG pyramid, cần thêm 4–7 GB. Đặc trưng ở 1/16 độ phân giải (ảnh 256 cho lưới 16², không thấy ngón tay). Code gốc bọc `no_grad` + `detach`, phải sửa | **Để sau**, chỉ làm nếu E-LatentLPIPS không cải thiện tay. Thực tế chưa viết code |
| **E-LatentLPIPS** (Kang et al. 2024, Diffusion2GAN) | VAE encoder SD + VGG16-bn train trên latent | Loss có cặp, gói pip sẵn, khoảng 12 ms/+0.6 GB mỗi lần gọi trên A100 (số của bài báo, chưa gồm VAE encode) | **Làm trước** |

### 2.2 Điểm đặc sắc

1. **Đưa không gian latent của mô hình diffusion vào làm loss cho bài toán sinh ảnh ký hiệu điều khiển bằng tư thế.** Tri thức từ dữ liệu khổng lồ của SD vào mà GReg không phải đổi kiến trúc hay tốn thêm lúc suy luận ⚠️ (theo những gì nhóm đã đọc, chưa thấy công trình sign-language nào làm điều này).
2. **Biến thể theo vùng tay (`elp1_hand`):** áp cùng thước đo latent lên crop tay phóng to, thay vì chỉ toàn ảnh. Đây là thiết kế của nhóm, vì E-LatentLPIPS gốc chỉ dùng cho toàn ảnh.
3. **Thí nghiệm "critic diffusion thay GAN" (`elp1_noadv`):** kiểm tra trực tiếp liệu một thước đo cảm nhận đóng băng có đủ để bỏ discriminator. Diffusion2GAN vẫn giữ GAN (trọng số GAN 0.25 so với E-LatentLPIPS 1.0, theo code notes §4).
4. **Đo độ trôi màu (`lowfreq_drift`):** đây là rủi ro đã biết của loss trong latent SD, vì latent mã hoá màu toàn cục yếu (code notes §4).

### 2.3 Mượn từ đâu, phần nào là của nhóm

| Thành phần | Nguồn |
|---|---|
| E-LatentLPIPS: VGG16-bn bỏ 3 max-pool, đầu vào 4 kênh, train ImageNet trong latent SD, hiệu chỉnh tuyến tính trên BAPPS; ensemble augmentation khả vi | Kang et al. 2024 (arXiv 2405.05967), gói `elatentlpips` (**giấy phép CC BY-NC**: chỉ dùng cho nghiên cứu, cần ghi rõ trong luận văn) ✅ |
| VAE encoder SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` (diffusers) |
| **Của nhóm:** cắm vào `GRegLoss`; truyền gradient qua VAE encoder về ảnh RGB của GReg (Diffusion2GAN sinh thẳng latent nên không cần bước này); biến thể crop tay; thí nghiệm bỏ GAN; ép các op StyleGAN2 chạy đường PyTorch tham chiếu | 🛠 |

---

## 3. Kiến trúc chi tiết

### 3.1 Module `ELatentLPIPSLoss`

`greg/diffusion/latent_lpips.py`, các dòng trích dẫn theo bản `fix-d3-tmp`.

```
Î (B,3,256,256) ∈[0,1] ──×2−1──► SD1.5 VAE encoder [ĐÓNG BĂNG, gradient ĐI QUA] ─ latent_dist.mode() ─► z0 (B,4,32,32)
I_t (B,3,256,256)      ──×2−1──► SD1.5 VAE encoder [no_grad]                   ─ latent_dist.mode() ─► z1 (B,4,32,32)
                                                                                          │
              ELatentLPIPS(encoder="sd15", augment="bg")  [ĐÓNG BĂNG, BN luôn ở eval]     ▼
                 1. normalize=True: nhân hệ số latent SD (0.18215) bên trong gói
                 2. augment "bg" (blit + geometric), CÙNG tham số cho z0 và z1 (ensemble augmentation)
                 3. latent-VGG16-bn → đặc trưng nhiều tầng → chuẩn hoá kênh → hiệu số² → trọng số tuyến tính (kiểu LPIPS)
                 4. add_l1_loss=True: + |z0 − z1|₁ trên latent
                                                                                          ▼
                                                                         L_elp (vô hướng, mean theo batch)
```

- Chỉ nạp encoder, `vae.decoder = None` (`latent_lpips.py:37-40`).
- Lấy `mode()` của phân phối latent thay vì `sample()`, nên loss tất định (`:53-56`).
- `train()` bị ghi đè để module **luôn ở eval**, giữ đúng thống kê BatchNorm của latent-VGG (`:50-51`).
- Mọi tham số có `requires_grad=False` (`:45-47`).
- Gradient chỉ chảy qua `z0`, tức qua VAE encoder về ảnh `Î` (`:58-63`).
- `scale=1.0`: ảnh 256² cho latent 32². SD được train ở 512² (latent 64²). Tuỳ chọn `scale=2` (phóng ảnh trước khi encode, tốn khoảng 4 lần) có sẵn nhưng **chưa thử**.
- **Sửa lỗi môi trường** (`fix-d3-tmp`, `latent_lpips.py:19-30, 43`): augmentation của gói dùng op StyleGAN2-ADA (`upfirdn2d`, `bias_act`, `filtered_lrelu`), mặc định JIT-compile CUDA plugin và cần `nvcc`/`CUDA_HOME`. Image vast không có `nvcc`, nên lỗi `OSError` xuất hiện ngay ở lần gọi loss đầu tiên. Bản sửa ép các op dùng đường PyTorch tham chiếu. Trên latent 4×32² chi phí này không đáng kể.

**Số tham số đóng băng** không được log. Ước lượng ⚠️: VAE encoder SD1.5 khoảng 34 M, latent-VGG16 khoảng 15 M. Không có tham số nào được train thêm.

### 3.2 Biến thể crop tay (`elpips_hand`)

`losses.py:166-171`:

```
crop_regions(Î, boxes, 64)[:, 1:]    (B,2,3,64,64)    hai hộp tay 64×64 cố định quanh tâm khớp (regions.mode = fixed)
   → resize_patches(·, 256)          (B,2,3,256,256)  phóng 4×
   → flatten                         (2B,3,256,256)
   → chỉ giữ tay hiển thị (w_r > 0)
   → ELatentLPIPSLoss  ⇒  latent 32² cho mỗi crop tay
```

Làm tương tự với `I_t`. Không có tay nào hiển thị thì loss = 0 (vẫn giữ đồ thị tính toán).

### 3.3 Loss tổng của D3

```
L_G = λ_per·L_VGG + λ_rper·L_rVGG + λ_struct·L_struct + λ_adv·L_adv
    + λ_elp · L_elp(Î, I_t)
    + λ_elph · L_elp(crop_tay(Î)↑256, crop_tay(I_t)↑256)
```

| Run | λ_elp | λ_elph | λ_adv | Discriminator |
|---|---|---|---|---|
| `base` | 0 | 0 | 1 | 3 PatchGAN |
| `elp1` | 1 | 0 | 1 | 3 PatchGAN |
| `elp1_hand` | 1 | 1 | 1 | 3 PatchGAN |
| `elp1_noadv` | 1 | 0 | **0** | **không tạo** (`06_train_greg.py:110-112`) |

Trọng số lấy theo khuyến nghị đọc code: bắt đầu ở 1.0, song song với `perceptual: 1.0` (code notes §4). Kế hoạch có quét λ ∈ {0.5, 1, 2} nhưng sweep thực tế chỉ chạy λ = 1.

Lưu ý về độ lớn tương đối: trong log GReg cùng cấu hình, `perceptual` dao động khoảng 50–75 và `regional_perceptual` khoảng 20–150 ✅ (log D2 ngày 03/11, cùng `GRegLoss`). Giá trị `elpips` lúc train **không còn lưu**. Nếu nó cùng bậc với LPIPS thường (≤ 1) cộng L1 latent, thì phần đóng góp của nó trong tổng loss rất nhỏ ⚠️. Xem §6.1.

### 3.4 Huấn luyện

- **Cấu hình** (`configs/sweeps/d3_diffusion_loss.yaml`):
  - kế thừa `configs/sweep_greg.yaml`: 15 epoch, milestone [10, 13], batch 4, lr G và D 2e-4, split `train_fit`, `pairs_per_clip: 1`;
  - cộng `model.gr.qk_norm=true` và `loss.weights.regional_perceptual=3`, đúng cấu hình run5;
  - TPSMM đóng băng (checkpoint flow của run5).
- **`base`:** checkpoint `run5/outputs/sweep_clean/ref`, **không train lại**. Bản này đã được chấm ở D2 và được chép sang D3 (`stages_d3.sh:14-22`).
- **Chi phí** ✅ (`sweep_results.csv`, cột `sec_per_step_median` và `vram_gb_max`):

| Run | s/bước | VRAM tối đa | So với GReg thường (0.20 s, 5.0 GB, lấy từ run `fresh` của D4 cùng máy) |
|---|---|---|---|
| elp1 | 0.29 | 8.8 GB | +45 % thời gian, +3.8 GB |
| elp1_hand | 0.40 | 15.5 GB | **×2 thời gian**, ×3 VRAM |
| elp1_noadv | 0.27 | 8.7 GB | +35 % (dù không có discriminator) |

- **Thời gian:** cả sweep chạy từ 16:06 đến 20:14 UTC ngày 03/11, khoảng 4 h 08 phút cho 3 run ✅. Tức khoảng 2.6 USD ở 0.62 USD/h ⚠️ (tính từ mốc thời gian).
- **Lúc suy luận:** chi phí thêm bằng **0**. Loss không có mặt khi suy luận, GReg đầu ra có cùng kiến trúc với run5.

---

## 4. Giao thức đánh giá

- **sweep15:** GReg train 15 epoch, chấm trên **val 100 clip × 16 khung = 1 600 khung** bằng `scripts/11_sweep.py` (`eval: split val, max_clips 100, max_rows 16, lpips true, eval_region_size 64`). Một seed.
- **Chỉ số:**
  - `l1`, `lpips` (AlexNet) trên toàn ảnh;
  - `hand_l1`: L1 trong hộp tay;
  - `hand_lpips`: LPIPS trên crop tay phóng lên 128²;
  - `hand_l1@large`: chỉ trên khung cử động lớn;
  - `lowfreq_drift`: L1 sau lọc Gauss σ = 8 px, ngoài vùng tay, đo trôi màu tần số thấp.
- `rank` = thứ hạng trung bình trên 6 chỉ số `rank_metrics` của sweep.
- **Không so với số test 100 epoch** (LPIPS 0.0485). Số sweep15 chỉ dùng để so giữa các run trong cùng sweep.

---

## 5. Kết quả

Sweep15, val 100 clip × 16 khung. Nguồn: `outputs/d3/sweep/sweep_results.csv` ✅.

| Run | rank | L1 ↓ | LPIPS ↓ | hand L1 ↓ | hand LPIPS ↓ | hand L1@large ↓ | lowfreq drift ↓ | s/bước | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **base** | **1.33** | **0.0210** | 0.0617 | 0.1382 | **0.2845** | **0.1691** | **0.0114** | – | – |
| elp1 | 1.67 | 0.0213 (+1.5 %) | **0.0612 (−0.8 %)** | **0.1377 (−0.4 %)** | 0.2893 (+1.7 %) | 0.1695 (+0.2 %) | 0.0119 (+4.2 %) | 0.29 | 8.8 GB |
| elp1_hand | 3.00 | 0.0218 (+3.6 %) | 0.0630 (+2.0 %) | 0.1395 (+0.9 %) | 0.2919 (+2.6 %) | 0.1725 (+2.0 %) | 0.0121 (+6.2 %) | 0.40 | 15.5 GB |
| elp1_noadv | 4.00 | 0.0237 (+12.7 %) | 0.0702 (+13.7 %) | 0.1479 (+7.0 %) | 0.3047 (+7.1 %) | 0.1797 (+6.2 %) | 0.0142 (+24.3 %) | 0.27 | 8.7 GB |

PSNR, SSIM và face L1 của các run D3 nằm trong các `val_metrics.json` trên Drive nhưng **không có trong hội thoại**, nên không được ghi ở đây.

Để đặt cạnh các hướng khác trong cùng giao thức sweep15 (LPIPS): D2 `d2_random` 0.0643, D4 `aug` 0.0580 (−6.0 %), D4 `fresh` 0.0693 ✅.

---

## 6. Phân tích nguyên nhân và hiện tượng

### 6.1 `elp1` ≈ `base`: latent SD gần như không thêm thông tin

**Hiện tượng ✅:** mọi chênh lệch nằm trong ±2 % (LPIPS −0.8 %, hand LPIPS +1.7 %, L1 +1.5 %). Thứ hạng 1.67 so với 1.33 của `base`.

**Nguyên nhân:**
1. **Tín hiệu trùng lặp ⚠️.** E-LatentLPIPS được thiết kế để **xấp xỉ LPIPS-VGG** trong không gian latent: nó hiệu chỉnh trên cùng bộ phán đoán cảm nhận BAPPS. GReg đã tối ưu trực tiếp VGG-19 ở 4 scale × 5 lớp (trọng số 10 mỗi lớp), cộng VGG trên crop vùng 128² (×3). Thêm một thước đo cùng loại, trên biểu diễn thô hơn, không đưa thêm hướng gradient mới đáng kể.
2. **Độ phân giải latent quá thô cho chi tiết tay ✅.** Ảnh 256² cho latent 32², mỗi ô latent ứng với 8×8 pixel. Ngón tay rộng vài pixel, nhỏ hơn một ô latent, nên VAE encoder đã nén mất phần lớn chi tiết này trước khi VGG nhìn thấy. Đúng như cảnh báo trong code notes §4: "latent 32² ⇒ mất chi tiết < 8 px (ngón tay)". Đó cũng là lý do hand LPIPS không giảm (+1.7 %).
3. **Ngoài phân phối ⚠️.** SD 1.5 được train ở 512² (latent 64²). Ảnh 256² là ngoài phân phối của VAE, nên đặc trưng latent kém đại diện hơn. Tuỳ chọn `scale=2` để sửa điều này chưa được thử.
4. **Trọng số tương đối nhỏ ⚠️.** Với λ = 1, nếu L_elp cùng bậc với LPIPS (≤ 1) thì nó nhỏ hơn số hạng VGG (khoảng 50–75) hai bậc độ lớn. Không có log `elpips` để khẳng định. Quét λ ∈ {0.5, 1, 2} như kế hoạch chưa làm.

### 6.2 `elp1_hand` kém hơn đều trên mọi chỉ số

**Hiện tượng ✅:** kém `base` ở cả 6 chỉ số (thứ hạng 3.0), độ trôi màu +6.2 %, thời gian ×2, VRAM 15.5 GB.

**Nguyên nhân ⚠️:**
- Crop tay 64² được **phóng 4 lần bằng nội suy** lên 256² trước khi vào VAE. Encoder nhận một ảnh mờ, không giống ảnh tự nhiên 256². Loss khi đó so sánh "biểu diễn của ảnh mờ", nên không thưởng cho chi tiết thật.
- Crop dùng hộp **cố định** 64×64, nên có nhiều nền quanh tay. Nó cũng trùng hoàn toàn với crop của regional VGG (×3) và PatchGAN tay. Ba loss cùng kéo một vùng theo các hướng hơi khác nhau.
- Mức chênh +2–3 % nằm gần vùng nhiễu (§6.4), nhưng việc kém **đồng loạt** trên 6/6 chỉ số và chi phí gấp đôi đủ để loại biến thể này.

### 6.3 `elp1_noadv`: critic diffusion đóng băng không thay được GAN

**Hiện tượng ✅:** bỏ 3 PatchGAN làm mọi chỉ số kém hẳn: LPIPS +13.7 %, L1 +12.7 %, hand LPIPS +7.1 %, độ trôi màu +24.3 %.

**Nguyên nhân:**
1. **Hai loại tín hiệu khác bản chất ✅.**
   - E-LatentLPIPS là loss **có cặp**, tức một khoảng cách cố định tới ảnh đích. Mọi loss có cặp (VGG, L1, latent-LPIPS) khi không chắc chắn đều kéo đầu ra về **trung bình có điều kiện**, tức ảnh mờ.
   - Discriminator là thước đo **được huấn luyện liên tục** để phân biệt thật và giả. Nó phạt chính kiểu mờ đó và đẩy bộ sinh tạo cạnh sắc.
   - Một critic đóng băng không thích nghi theo lỗi hiện tại của bộ sinh, nên không thay được vai trò này. Đó là lý do DMD phải train "fake score" (§2.1) và Diffusion2GAN vẫn giữ GAN.
2. **GAN trong GReg còn giúp cả L1 ⚠️.** Thông thường GAN làm L1 kém đi một chút. Ở đây bỏ GAN làm L1 tăng 12.7 %. Các PatchGAN của GReg nhận cả crop RGB **và bản đồ gradient** của vùng (4 kênh). Vì vậy chúng ép cạnh của tay và mặt đúng vị trí. Đây là ràng buộc cấu trúc chứ không chỉ là "độ thật".
3. **Trôi màu +24 % ⚠️.** Không có discriminator nhìn màu thật trong crop vùng, và latent SD mã hoá màu toàn cục yếu (code notes §4). Vì vậy tần số thấp ngoài vùng tay trôi nhiều hơn. Bản `elp1` (vẫn có GAN) cũng có độ trôi màu +4 %, gợi ý rằng số hạng latent tự nó đã đẩy nhẹ màu theo hướng sai.
4. Đây là **câu trả lời âm rõ ràng** cho câu hỏi "critic diffusion thay GAN?" và vượt xa vùng nhiễu.

### 6.4 Độ tin cậy của các chênh lệch nhỏ

`base` là checkpoint run5 train trên **máy khác, phiên bản thư viện khác**. Các run D3 train trên 4090 này với torch 2.7.1. Trong D4 cùng giao thức, run `fresh` lệch +12 % so với `base` ✅. Điều đó cho thấy chênh lệch máy, phiên bản và seed có thể lên vài phần trăm.

- Các chênh lệch ±2 % của `elp1` và `elp1_hand` **không có ý nghĩa thống kê**. Kết luận đúng là "không cải thiện", không phải "làm hại".
- Chênh lệch của `elp1_noadv` (+13.7 %) là đáng tin.

---

## 7. Hạn chế và việc nên làm tiếp

**Hạn chế:**
- Một seed, val 100 clip, 15 epoch.
- `base` không được train lại trên cùng máy.
- Chỉ thử λ = 1.
- Chưa thử `scale=2` (ảnh 512 vào VAE).
- Không còn log giá trị `elpips` để biết độ lớn tương đối.
- DIFT (bước 2 của kế hoạch) chưa được viết code. Chuỗi tự động chuyển sang D4 sau khi sweep D3 xong.

**Nếu muốn đào sâu** (đều cần GPU, nên phải có quyết định của nhóm):
1. Quét λ_elp ∈ {2, 5, 10} để kiểm tra giả thuyết "trọng số quá nhỏ" (§6.1, ý 4).
2. Thử `scale=2` để VAE làm việc ở độ phân giải gốc của SD.
3. DIFT trên `up_blocks[2]` (1/8 độ phân giải) với t ≈ 100–260 trên crop tay. Dự trù +50–150 % thời gian train và +4–7 GB (code notes §7).
4. Train lại `base` với seed 1 trên cùng máy để có mốc so sánh sạch.

**Bài học cho luận văn** (đã có đủ dữ liệu):
- Tri thức diffusion đưa vào **qua loss có cặp đóng băng** không vượt được VGG perceptual mà GReg đã có. Lý do là thông tin trùng lặp và latent quá thô so với chi tiết tay.
- GAN theo vùng là thành phần **không thể thay** trong GReg: bỏ đi thì LPIPS +14 %.
- Muốn diffusion giúp tay, phải để diffusion **vẽ** (D1, D6), không chỉ **chấm điểm**.

---

## 8. Trạng thái dữ liệu và sự cố

**Trên Drive** (`GReg_runs/diff_common/outputs/d3/`, remote `gdrive,root_folder_id=1FTgmwzFrRW9qIzLNYAFmt9txxECPgsTr:`):
- D3 xong lúc 20:14 UTC ngày 03/11, trước khi Drive đầy (02:43 UTC ngày 04/11).
- Ngày 05/11 lúc 11:10 UTC, kiểm tra từ máy vast xác nhận `d3/sweep/sweep_results.csv` **có trên Drive** ✅.
- Checkpoint, `val_metrics.json` và log của các run sweep được sync trước khi Drive đầy, nhiều khả năng đã có nhưng **chưa kiểm từng file** ⚠️.
- Instance 58802139 đã bị xoá ngày 05/11.

**Sự cố khi chạy:**
- **Lỗi lúc 15:51 UTC ngày 03/11.**
  - **Hiện tượng:** run `elp1` lỗi ngay ở bước loss đầu tiên: `OSError: CUDA_HOME environment variable is not set`. Op augmentation StyleGAN2 trong gói `elatentlpips` cố JIT-compile CUDA.
  - **Cách sửa:** commit `0211c3a` ("E-LatentLPIPS augmentation ops on their PyTorch reference path") áp bằng patch trên máy vast.
  - **Thời gian mất:** chạy lại lúc 16:06 UTC, GPU rảnh khoảng 15 phút.
- **Việc còn treo:** `0211c3a` chỉ có trên nhánh local `fix-d3-tmp`. `origin/feat/diff-d3-diffusion-loss` vẫn ở `e4fd080`, tức **chưa có bản sửa**. Cần push `fix-d3-tmp:feat/diff-d3-diffusion-loss`. Nếu không, chạy lại D3 trên máy không có `nvcc` sẽ lỗi lần nữa.
