# Hai hướng cải tiến dựa trên diffusion cho GReg

Ngày lập: 2026-10-29. Tài liệu đi kèm: `docs/huong_cai_tien_keo_tay.md` (danh sách đầy đủ các hướng, gồm A1 và B1).

- **Hướng 1 — "Warp rồi Diffuse":** video diffusion DiT đóng băng, điều kiện là ảnh nguồn đã warp bằng mesh SMPL-X cùng mặt nạ nhìn thấy, chỉ huấn luyện adapter/LoRA. Trần chất lượng cao nhất, có tính mới.
- **Hướng 2 — Bộ tinh chỉnh diffusion một bước:** đặt sau GReg, kiểu pix2pix-turbo / OSEDiff. Là hướng cải thiện nhiều nhất trên mỗi đơn vị compute.

Quy ước trong file:
- ✅ = đã kiểm chứng bằng nguồn (link ở mục 6).
- ⚠️ = suy luận / thiết kế đề xuất / chưa có số liệu, **phải đo thử trước khi viết vào luận văn**.

---

## 0. Bối cảnh và điều kiện tiên quyết

### 0.1 Vì sao cần diffusion

Kết quả `run3` cho thấy pipeline GAN + warp của GReg có ba giới hạn cấu trúc:

1. **Vùng không nhìn thấy trong ảnh nguồn** (lòng bàn tay, mặt trong cẳng tay, phần thân bị tay che ở khung nguồn) phải do một UNet nhỏ tự vẽ. Dữ liệu chỉ khoảng 4.6k clip, loss là L1 + perceptual + GAN vùng, nên kết quả mờ.
2. **Generator gần như không đóng góp:** L1 val của GReg là 0.0330, chỉ warp là 0.0335, copy nguyên nguồn là 0.0387.
3. **Mỗi khung được sinh độc lập**, không có prior về chuyển động hay tính nhất quán theo thời gian.

A1 (flow + mặt nạ nhìn thấy từ mesh) và B1 (mặt nạ tay/mặt từ mesh) sửa được phần warp và loss vùng, nhưng **không** sửa được giới hạn 1 và 3. Diffusion mang vào prior học từ hàng triệu video người thật, đúng chỗ còn thiếu.

### 0.2 Điều kiện tiên quyết chung cho cả hai hướng

| Thành phần | Vai trò trong Hướng 1 | Vai trò trong Hướng 2 |
|---|---|---|
| A1: flow mesh `F_geo`, mặt nạ nhìn thấy `V_geo`, độ tin cậy `C` | Tín hiệu điều kiện chính (video warped + mask) | Cải thiện đầu vào thô của refiner |
| B1: mặt nạ tay/mặt từ `parts/*.png` | Trọng số loss vùng, mask cho cross-attention của adapter | Trọng số loss vùng |
| Chỉ số đo theo vùng và theo bin chuyển động (mục C2 của file kia) | Bắt buộc để so sánh | Bắt buộc để so sánh |
| Bộ lọc fade-in (`02b_mark_valid_rows.py`) | Bắt buộc | Bắt buộc |
| GReg (đã có A1 + B1) huấn luyện đủ 100 epoch | Baseline so sánh | **Đầu vào** của refiner |

→ **Không nên bắt đầu diffusion trước khi A1 + B1 chạy xong.** Cả hai hướng đều dùng lại chúng; nếu thiếu thì không biết phần cải thiện đến từ đâu.

### 0.3 Dữ liệu sẵn có cần lưu ý

- Cache hiện tại ở 256×256. Script tiền xử lý đã có `--size` (`scripts/02_preprocess_hamnosys.py:52`), và video gốc độ phân giải cao vẫn còn trên Drive. Vì vậy có thể tạo cache 480 hoặc 512 mà không phải sửa code. ✅ (đã đọc code)
- fps: video trên 30 fps được resample về 25 (`greg/data/frames.py:24`). Wan sinh mặc định ở 16 fps. ⚠️ Cần quyết định resample 25 → 16 hay giữ 25 (xem 1.6).
- Có khoảng 5.7k clip HamNoSys (4585 train / 1146 test). Bộ dữ liệu này nhỏ so với dữ liệu huấn luyện diffusion, nhưng đủ cho LoRA/adapter.

---

## 1. Hướng 1 — "Warp rồi Diffuse": video DiT đóng băng + adapter, điều kiện hình học từ mesh

### 1.1 Ý tưởng một câu

Cho một video diffusion DiT đã huấn luyện sẵn xem **chính ảnh nguồn đã được warp đúng vị trí bằng mesh SMPL-X**, kèm mặt nạ "pixel nào nhìn thấy được thật". Mô hình chỉ cần:
- giữ nguyên texture ở vùng nhìn thấy,
- **sinh** vùng bị che bằng prior của nó,
- làm mượt theo thời gian.

Mô hình không phải tự đoán chuyển động.

### 1.2 Vì sao tin đây là hướng mạnh nhất

1. **Chia việc đúng chỗ.** Phần hình học (ai đi đâu, cái gì che cái gì) được giải chính xác bằng mesh: SMPL-X khớp ảnh thật với sai số 2.5 px, kể cả ngón tay. Phần ngoại hình (texture vùng chưa thấy, nếp vải, độ sắc) giao cho prior diffusion. Đây đúng là hai thứ mà GReg đang làm yếu.
2. **Tiền lệ trong ngôn ngữ ký hiệu.** SignRefine (ECCV 2026) dùng chính backbone Wan-Fun-Control-1.3B, đóng băng, thêm adapter vùng tay/mặt. Họ báo cáo cải thiện tới 30% độ chính xác tư thế tay so với baseline mạnh nhất, và hơn 80% lựa chọn ưa thích của người dùng ngôn ngữ ký hiệu. ✅ Nhưng SignRefine **chỉ điều kiện bằng keypoint 2D**.
3. **Tiền lệ "render hình học thô → diffusion làm renderer nâng cao".** ViewCrafter (2024) render point cloud thô (có lỗ, méo), VAE-encode rồi đưa vào video diffusion. Diffusion đóng vai "enhanced renderer", lấp vùng che khuất và sửa hình học sai. Họ cũng cho thấy điều kiện bằng render tốt hơn điều kiện bằng tọa độ Plücker. ✅ StreetCrafter dùng render LiDAR làm điều kiện ở mức pixel. ✅
4. **Khoảng trống chưa ai lấp.** Theo các nguồn đã khảo sát, chưa nhóm nào kết hợp **tương ứng hình học từ mesh SMPL-X (warp + visibility)** với renderer video diffusion cho ngôn ngữ ký hiệu. Các nhóm diffusion ký hiệu (SignViP, SignRefine, SignMimic) đều dùng keypoint 2D ± render tay HaMeR. ✅ Đây có thể là đóng góp chính của luận văn.
5. **Tính nhất quán thời gian có sẵn**, vì đây là mô hình video, không sinh từng khung.

### 1.3 Phản biện cần đưa vào luận văn: GenWarp

GenWarp (NeurIPS 2024, Sony) cố ý **không** dùng ảnh warped làm điều kiện. Lý do của họ: ảnh warp từ depth ước lượng đơn ảnh bị nhiễu, làm méo chi tiết (ví dụ biển hiệu neon bị méo). Thay vào đó họ đưa vào **embedding tọa độ đã warp**, để diffusion tự học "warp chỗ nào, sinh chỗ nào" bằng cross-view attention + self-attention. ✅

**Lập luận của mình:**
- Hình học của mình không đến từ depth đơn ảnh nhiễu, mà từ mesh SMPL-X đã fit tối ưu theo thời gian.
- Vùng mesh không mô hình được (tay áo rộng, tóc) được đánh dấu bằng mặt nạ `C`.
- Vì vậy ảnh warped đáng tin hơn nhiều so với trường hợp của GenWarp.

**Nhưng nên làm ablation**, vì đây chính là câu hỏi một reviewer sẽ hỏi: (a) chỉ ảnh warped + mask; (b) chỉ bản đồ tọa độ tương ứng (mỗi pixel đích mang tọa độ (u, v) của điểm nguồn tương ứng, kiểu GenWarp); (c) cả hai.

### 1.4 Kiến trúc đề xuất ⚠️ (thiết kế, chưa thử)

```
              ┌────────────── Điều kiện mỗi khung đích t ──────────────┐
I_s ──┐       │ I_st  = warp(I_s, F_geo,t)       (RGB, 3 kênh)          │
      │       │ V_t   = V_geo,t · C_t             (1 kênh, 0..1)        │
mesh ─┼──A1──►│ N_t   = normal map đích           (3 kênh)              │
      │       │ M_t   = mặt nạ tay trái/phải/mặt  (3 kênh, từ parts)     │
      │       │ (tùy chọn) UV_t = tọa độ nguồn tương ứng (2 kênh)       │
      │       └──────────────────────────────────────────────────────────┘
      │                          │
      │          VAE-encode video I_st và N_t → latent (nén 4× thời gian, 8× không gian)
      │          Mask V_t, M_t: downsample về lưới latent
      │                          │
      │       ┌──────────────────▼──────────────────────────┐
      └──────►│ Wan-Fun-Control-1.3B / Wan-VACE-1.3B (ĐÓNG BĂNG) │◄── ảnh tham chiếu I_s
              │  + LoRA trên attention/FFN (rank 16–32)          │    (đặc trưng danh tính)
              │  + kênh vào mới cho [V, M, UV], khởi tạo 0        │
              │  + adapter vùng tay/mặt (kiểu SignRefine):        │
              │    crop N_t vùng tay 128–256² → encoder nhỏ →     │
              │    masked cross-attention vào vài block DiT       │
              └──────────────────┬──────────────────────────────┘
                                 ▼
                        video đích (81 khung)
                                 │
          (tùy chọn) dán lại pixel warped nơi V≈1 và ngoài vùng tay: Î = V·I_st + (1−V)·Î_diff
```

**Chọn backbone:**

| Lựa chọn | Ưu | Nhược | Trạng thái |
|---|---|---|---|
| Wan2.1-Fun-1.3B-Control | Đã có kênh `control_video`; SignRefine dùng chính nó; DiffSynth-Studio có script train LoRA + validate | Control gốc nhận một video điều kiện (skeleton/edge/depth); cần mở rộng để nhận thêm mask | ✅ có trong DiffSynth-Studio |
| Wan2.1-VACE-1.3B | Thiết kế cho đa điều kiện: video điều khiển + mask + ảnh tham chiếu (video inpainting) — khớp rất tự nhiên với "giữ vùng nhìn thấy, sinh vùng che" | Chưa kiểm chứng chi tiết API ở 1.3B; SignRefine dùng nó làm baseline (kém hơn SignRefine) | ⚠️ cần đọc model card |
| Wan2.1-Fun-V1.1-1.3B-InP / I2V | Có khung đầu làm ảnh tham chiếu | Không có kênh điều kiện hình học | ✅ có script LoRA |

→ Khuyến nghị: **thử VACE-1.3B trước** (khung "mask + video tham chiếu" trùng với bài toán). Nếu API không phù hợp thì dùng Fun-1.3B-Control và mở rộng kênh vào. ⚠️

**Cách đưa điều kiện vào:**
1. **Ghép kênh (concat) ở latent**, giống cách Fun-Control/ViewCrafter đưa render vào. Phần trọng số cho kênh mới khởi tạo bằng 0, để lúc bắt đầu mô hình hành xử y như mô hình gốc (ý tưởng zero-conv của ControlNet; SignRefine cũng khởi tạo gần 0 cho output projection của adapter ✅).
2. **Adapter vùng tay/mặt:** SignRefine ghi nhận ở độ phân giải latent, tay + mặt chỉ chiếm khoảng 15% khung hình. Họ dùng encoder vùng riêng trên crop 256×256, đưa vào DiT qua **masked cross-attention**, kèm "sink token" để attention có chỗ "không chú ý" ngoài vùng. ✅ Mình làm tương tự, nhưng crop **normal map SMPL-X của tay** thay vì keypoint, nên có thông tin 3D của ngón tay.
3. **Ảnh tham chiếu `I_s`:** qua nhánh tham chiếu sẵn có của backbone (CLIP image / reference latent) để giữ danh tính, màu áo, nền.

### 1.5 Hàm mất mát

- **Loss gốc của backbone:** flow matching (Wan là rectified-flow DiT). ⚠️ Kiểm tra lại dạng loss chính xác trong DiffSynth-Studio.
- **Trọng số theo vùng:** nhân trọng số latent thuộc tay/mặt. SignRefine đặt **tay và mặt gấp 10 lần nền**. ✅ (theo bản review Pith; nên đọc lại bài gốc để xác nhận). Mặt nạ lấy từ B1, downsample về lưới latent.
- **(Tùy chọn) trọng số theo vùng che khuất:** tăng nhẹ trọng số ở nơi `V=0` (vùng phải sinh), vì đó là chỗ GReg đang yếu nhất. ⚠️
- Không cần GAN loss, vì diffusion đã tự cho độ sắc.

### 1.6 Chi tiết dữ liệu và độ phân giải ⚠️

- **Độ phân giải:** Wan làm việc tốt ở 480p (ví dụ 480×832). Video ngôn ngữ ký hiệu đã crop vuông, nên đề xuất crop vuông **480×480** (bội số của 16) từ video gốc, dùng `--size 480` khi tiền xử lý lại. Khi so với GReg, **resize output về 256** rồi mới tính metric, để bảng kết quả công bằng.
- **Số khung:** Wan yêu cầu dạng 4n+1 (81, 49, 33…). Clip HamNoSys thường ngắn (một ký hiệu), nên đề xuất cắt theo cửa sổ 49 hoặc 81 khung, có chồng lấn; clip ngắn hơn thì pad bằng khung cuối và che loss phần pad.
- **fps:** hoặc resample về 16 fps cho khớp prior của Wan (mất thông tin tay khi chuyển động nhanh), hoặc giữ 25 fps và để LoRA tự thích nghi. Nên thử nhanh cả hai trên một tập nhỏ.
- **Khung nguồn:** dùng khung đầu hợp lệ (tay buông) làm ảnh tham chiếu, đúng giao thức reconstruction của GReg.
- **Tiền tính sẵn:** latent VAE của video đích và video warped (tiết kiệm VRAM khi train; musubi-tuner/DiffSynth đều hỗ trợ cache latent ✅).

### 1.7 Compute

| Hạng mục | Số liệu | Nguồn |
|---|---|---|
| Inference Wan2.1-T2V-1.3B | **8.19 GB VRAM** (công bố chính thức), khoảng **4 phút** cho 5 s video 480p trên RTX 4090, cần `--offload_model True --t5_cpu`, không lượng tử hóa | ✅ model card / README |
| Đo độc lập | 4090: 281 s; A100: 170 s; H100: 85 s (480p). SaladCloud phản bác "8 GB là đủ" trong thực tế | ✅ InstaSD, SaladCloud |
| Train LoRA 1.3B trên 24 GB | Cộng đồng báo cáo 1.3B train LoRA trên 1 GPU 24 GB là đủ (diffusion-pipe, có người nói dùng chưa tới một nửa VRAM; nhưng nhiều ví dụ train ở 512px với ảnh/clip ngắn). Ngưỡng tối thiểu cho LoRA video khoảng 24 GB | ✅ Civitai notes, hướng dẫn cộng đồng |
| Train ở 81 khung × 480p | **Chưa tìm được con số đo chính xác** | ⚠️ phải đo |
| SignRefine (để tham chiếu) | 4× GH200 | ✅ |

**Ước tính cho luận văn ⚠️:**
- **Inference toàn bộ test:** 1146 clip × khoảng 3–5 phút/clip trên 4090 ≈ **60–95 giờ GPU**. → Báo cáo trên một tập con cố định (ví dụ 200 clip, cố định seed), hoặc giảm số bước lấy mẫu. Ghi rõ trong luận văn.
- **Train:** bắt đầu ở 256–320 px, 33–49 khung trên 24 GB để kiểm tra pipeline. Bản chính thức nên **thuê 1× A100/H100 80 GB trên vast.ai**, vài ngày.

**Công cụ:**
- DiffSynth-Studio: hỗ trợ `PAI/Wan2.1-Fun-1.3B-Control`, có script inference, low-VRAM inference, full training, LoRA training và validate LoRA. Ví dụ LoRA cho Fun-InP: `--height 480 --width 832 --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 --learning_rate 1e-4`. Bản validate Control dùng video điều khiển 81 khung 480×832. ✅
- musubi-tuner (kohya-ss): hỗ trợ `t2v-1.3B`; có fp8 + block swap để giảm VRAM, và cache latent. ✅
- diffusion-pipe: cũng train được Wan 1.3B trên 24 GB. ✅

### 1.8 Đánh giá

- **Giữ toàn bộ chỉ số của GReg** (L1, PSNR, SSIM, LPIPS, FVD, AKD) ở 256², để đặt cạnh Table 1.
- **Thêm:**
  - L1/LPIPS vùng cánh tay + bàn tay (mặt nạ B1);
  - MPJPE tay bằng RTMPose-X / DWPose (chuẩn 2026 theo SignRefine);
  - phân tầng theo bin dịch chuyển cổ tay.
- **Lưu ý về nghịch lý perception–distortion:** diffusion thường **tăng** chất lượng cảm nhận (LPIPS, FVD) nhưng có thể **giảm** PSNR/SSIM, vì nó sinh chi tiết hợp lý chứ không phải chi tiết đúng từng pixel. Phải nói rõ điều này khi viết, nếu không bảng kết quả sẽ khó hiểu.
- **Lưu ý về metric vòng tròn:** một bản review tự động (Pith) cho rằng con số 30% MPJPE của SignRefine bị thổi phồng, vì điều kiện và metric dùng cùng loại keypoint. ✅ (lời phê bình, chưa phải kết luận). Mình **điều kiện bằng SMPL-X** nhưng **đo bằng RTMPose**, nên ít bị vòng tròn hơn. Đây là một điểm có thể nêu.
- **Đánh giá người dùng** (nếu có thể): hỏi người biết ngôn ngữ ký hiệu xem ký hiệu có đọc được không. Ở đây hình dạng tay sai đồng nghĩa với sai nghĩa.

### 1.9 Ablation đề xuất

| # | Biến thể | Trả lời câu hỏi |
|---|---|---|
| 1 | Backbone zero-shot, chỉ điều kiện normal map (không LoRA) | Prior thô làm được đến đâu |
| 2 | + LoRA, điều kiện = normal map | Fine-tune giúp bao nhiêu |
| 3 | + video warped I_st (không mask) | Warp từ mesh giúp bao nhiêu |
| 4 | + mặt nạ V·C | Mặt nạ nhìn thấy giúp bao nhiêu |
| 5 | + adapter tay/mặt + loss vùng ×10 | Phần dành riêng cho tay |
| 6 | Thay I_st bằng bản đồ tọa độ UV (kiểu GenWarp) | Trả lời phản biện GenWarp |
| 7 | Warp bằng TPSMM thay vì mesh | Tách đóng góp của A1 bên trong diffusion |
| 8 | Có / không dán lại pixel warped ở vùng V≈1 | Giữ chi tiết so với tính liền mạch |

### 1.10 Rủi ro và cách giảm

| Rủi ro | Hệ quả | Cách giảm |
|---|---|---|
| Diffusion "bịa" hình dạng tay | **Sai nghĩa ký hiệu**, nặng hơn cả mờ | Adapter tay + normal map tay độ phân giải cao; loss vùng tay; đo MPJPE tay |
| Trôi danh tính (mặt, màu áo đổi dần) | Giảm ID similarity | Ảnh tham chiếu; dán lại pixel warped ở vùng nhìn thấy (ablation 8) |
| Diffusion bỏ qua điều kiện warp (vì prior quá mạnh) | Mất lợi thế của A1 | Khởi tạo 0 cho kênh mới nhưng tăng dần lr; kiểm tra bằng ablation 3 so với 2 |
| SignMimic: animator diffusion dùng nguyên bản thất bại nặng trên video ký hiệu (chỉ riêng làm sạch pose đã giảm FVD 1309 → 471 trên ASL 50K) | Zero-shot rất tệ | Không dùng zero-shot làm phương pháp chính; LoRA + điều kiện chuyên biệt là bắt buộc ✅ |
| Inference chậm | Không đánh giá hết test set | Tập con cố định; giảm số bước; nêu rõ thời gian/khung so với GReg |
| Dữ liệu nhỏ (4.6k clip) | Overfit LoRA | Rank nhỏ (16–32); dừng sớm theo val; augmentation màu |
| VRAM thực tế cao hơn công bố | OOM trên 24 GB | Cache latent; fp8/block swap (musubi); gradient checkpointing; thuê 80 GB |

### 1.11 Giấy phép ⚠️
- Wan2.1: Apache-2.0 (cần xác nhận lại trên repo trước khi công bố code).
- SignAvatars / video gốc: chỉ dùng cho nghiên cứu; không phát hành lại video sinh từ danh tính người ký thật khi chưa kiểm tra điều khoản.

### 1.12 Kế hoạch triển khai (theo mốc)

1. **M0 (sau A1 + B1):** xuất video warped, `V·C`, mặt nạ tay/mặt cho ~50 clip val ở 480². Chạy zero-shot Fun-Control và VACE với normal map để xem prior thô (ablation 1).
2. **M1:** dựng data loader cho DiffSynth-Studio (cửa sổ 4n+1 khung, cache latent). Train LoRA nhỏ ở 256–320 px trên 24 GB, chỉ điều kiện normal map (ablation 2). Mục đích: kiểm tra pipeline end-to-end.
3. **M2:** thêm kênh warp + mask (ablation 3–4). Đo VRAM thật ở 480p, rồi quyết định có thuê A100/H100 không.
4. **M3:** thêm adapter tay/mặt + loss vùng (ablation 5). Train bản chính.
5. **M4:** đánh giá trên tập con test cố định, phân tầng theo bin, cộng thêm ablation 6–8.

---

## 2. Hướng 2 — Bộ tinh chỉnh diffusion một bước sau GReg

### 2.1 Ý tưởng một câu

Giữ nguyên GReg (đã có A1 + B1) làm "bộ dựng thô". Sau đó cho một mô hình diffusion **một bước** (nền SD-Turbo + LoRA) nhận ảnh thô, normal map và mặt nạ, rồi trả ra ảnh sắc nét. Đây là bài toán dịch ảnh sang ảnh có cặp (paired image-to-image).

### 2.2 Vì sao đáng làm

1. **Nhắm thẳng vào lỗi "mờ".** Generator GReg học bằng L1 + perceptual nên kết quả trung bình hóa. Prior của mô hình ảnh lớn giỏi khôi phục nếp vải, ngón tay, kết cấu tóc.
2. **Rẻ và nhanh:**
   - pix2pix-turbo chạy **0.11 s/ảnh trên A100, 0.29 s trên A6000** ở 512×512 ✅, nhanh hơn Hướng 1 hàng nghìn lần.
   - Chỉ huấn luyện LoRA + vài lớp nhỏ. OSEDiff chỉ có **8.5M tham số huấn luyện** ✅.
3. **Không đập bỏ pipeline.** GReg, các bảng kết quả và ablation hiện có vẫn giữ nguyên; refiner là một tầng cộng thêm, dễ báo cáo dạng "GReg → GReg + A1 + B1 → + refiner".
4. **Có tiền lệ mạnh trong bài toán gần giống:** khôi phục ảnh thật (real-world super-resolution) bằng SD một bước. OSEDiff (NeurIPS 2024) khởi đầu diffusion **trực tiếp từ ảnh chất lượng thấp** thay vì từ nhiễu ngẫu nhiên, nên bỏ được tính ngẫu nhiên; nhanh hơn StableSR khoảng 28 lần. ✅ "Ảnh GReg mờ" về bản chất là một dạng ảnh suy giảm.

### 2.3 Chọn nền tảng

| Lựa chọn | Cơ chế | Ưu | Nhược | Code |
|---|---|---|---|---|
| **pix2pix-turbo** (Parmar et al. 2024) | SD-Turbo một bước; LoRA ở mọi module; skip connection + zero-conv từ encoder sang decoder VAE (giữ chi tiết đầu vào); retrain lớp đầu UNet; loss tái tạo + LPIPS + GAN (vision-aided, D dùng CLIP) | Dành sẵn cho dịch ảnh có cặp; ngang ControlNet ở Sketch2Photo/Edge2Image mà chỉ một bước | Không có số VRAM train chính thức; có báo cáo sụp đổ (output giống hệt nhau) khi train trên ảnh y tế 256 px (issue #93); thiếu package `vision_aided_loss` (issue #38) | ✅ GaParmar/img2img-turbo |
| **OSEDiff** (Wu et al., NeurIPS 2024) | LoRA ở VAE encoder + UNet, decoder đóng băng; loss dữ liệu + VSD (variational score distillation) trong latent, với SD đóng băng + một bản SD LoRA làm regularizer | Mạnh cho "phục hồi chi tiết"; LPIPS/FID tốt nhất trên RealSR tại thời điểm công bố | Train 4× A100 khoảng 1 ngày, batch 16; VSD tốn thêm 2 mạng; TSD-SR chỉ ra VSD kém tin cậy khi đầu vào quá tệ | ✅ (paper; repo cần kiểm tra) |

→ **Khuyến nghị: pix2pix-turbo trước** (paired, loss đơn giản, code rõ). Chỉ chuyển sang kiểu OSEDiff nếu pix2pix-turbo cho chi tiết chưa đủ.

### 2.4 Kiến trúc đề xuất ⚠️

```
GReg(+A1+B1) ──► Î_coarse (256², upsample → 512²)
                     │
                     ▼  VAE encoder (LoRA)  ──skip + zero-conv──┐
     [N_t, M_t, V_t] ─► conv nhỏ (khởi tạo 0) ─► cộng vào input  │
                     ▼                                          │
              UNet SD-Turbo một bước (LoRA, lớp đầu retrain)      │
                     ▼                                          │
              VAE decoder (LoRA) ◄──────────────────────────────┘
                     ▼
              Î_refined (512²) ──► (tùy chọn) Î = M_keep·Î_coarse + (1−M_keep)·Î_refined
```

- **Đầu vào thêm:** normal map `N_t` (để refiner biết hình dạng tay đúng là gì, không chỉ làm nét thứ đang có), mặt nạ tay/mặt `M_t`, mặt nạ nhìn thấy `V_t`. Đưa vào qua conv khởi tạo 0, cộng vào lớp đầu, để lúc bắt đầu refiner hành xử như pix2pix-turbo gốc.
- **Prompt văn bản:** cố định (ví dụ "a person signing, sharp hands") hoặc rỗng. Không cần text.
- **Độ phân giải:** SD-Turbo tối ưu ở 512. Train ở 512 với đích lấy từ video gốc tiền xử lý lại với `--size 512`. Như vậy refiner còn đóng vai **siêu phân giải ×2**. Khi so với GReg, resize về 256.
- **(Tùy chọn) dán lại:** giữ pixel của Î_coarse ở vùng chắc chắn tốt (V cao, không phải tay) để tránh refiner làm trôi danh tính. Đưa vào ablation.

### 2.5 Dữ liệu huấn luyện: điểm dễ sai nhất ⚠️

Refiner học từ cặp (đầu ra GReg, ảnh thật). **Nếu sinh đầu ra GReg trên chính tập train của GReg, ảnh thô sẽ "đẹp hơn thực tế"** (GReg đã nhớ tập train), nên refiner học sửa lỗi nhẹ hơn lỗi gặp ở test. Ba cách tránh:

1. **K-fold:** chia train thành 2–3 phần; train GReg trên phần này, sinh ảnh thô cho phần kia. Tốn thêm 2–3 lần train GReg.
2. **Dùng checkpoint GReg sớm** (ví dụ epoch 20–40) để sinh ảnh thô cho tập train. Rẻ hơn, nhưng phân phối lỗi khác.
3. **Tổng hợp suy giảm (degradation):** lấy ảnh thật, áp blur cục bộ ở tay, tạo "tay ma" bằng cách trộn với ảnh warped, làm giãn tay áo, rồi dùng làm đầu vào. Tương tự cách SR thật tổng hợp suy giảm. Có thể trộn với cách 1 hoặc 2.

→ Khuyến nghị: **2 + 3**, rồi kiểm tra trên val xem phân phối lỗi có khớp không. Dùng `val.txt` (10% train) đã có sẵn trong `04_make_splits.py`.

### 2.6 Hàm mất mát

- Theo pix2pix-turbo: L2/L1 tái tạo + LPIPS + GAN (vision-aided discriminator). ✅
- **Trọng số theo vùng (B1):** tăng trọng số pixel tay/mặt (MimicMotion tăng trọng số vùng tay giúp giảm ngón tay méo ✅; SignRefine dùng ×10 ✅).
- **Loss nhất quán thời gian ⚠️:** refiner xử lý từng khung nên dễ nhấp nháy. Dùng flow mesh giữa hai khung đích liên tiếp (tính từ A1, miễn phí) để warp `Î_refined(t−1)` sang t, rồi phạt chênh lệch ở vùng nhìn thấy. Cần lấy mẫu cặp khung liên tiếp khi train.
- **(Tùy chọn) loss keypoint tay:** chạy RTMPose/DWPose đóng băng trên ảnh tinh chỉnh, phạt lệch keypoint tay. Giúp tránh "bịa" hình dạng tay.

### 2.7 Compute

| Hạng mục | Số liệu | Nguồn |
|---|---|---|
| Inference pix2pix-turbo | 0.11 s (A100), 0.29 s (A6000) ở 512² | ✅ |
| Train pix2pix-turbo | **Không có con số VRAM chính thức.** Lệnh mẫu: 512 px, batch 2, xformers. Script có `--gradient_checkpointing`, mixed precision, chỉnh LoRA rank | ✅ (repo, issue #38) |
| Tiền lệ 1 GPU 24 GB | MariData (2026) train mô hình một bước cùng loại trên **1× RTX 4090** nhờ fp16 + xformers + gradient checkpointing | ✅ |
| OSEDiff | 4× A100, khoảng 1 ngày, batch 16, LoRA rank 4 | ✅ |

**Ước tính ⚠️:**
- Cấu hình mặc định (512, batch 2, không mixed precision) có lẽ **không vừa** 24 GB. Phải dùng batch 1 + gradient accumulation, fp16/bf16, gradient checkpointing, xformers.
- **Inference toàn test set:** khoảng 100k khung × 0.3 s ≈ **8 giờ** trên A6000. Khả thi, đánh giá được toàn bộ test set (khác với Hướng 1).

### 2.8 Đánh giá
- Như mục 1.8 (giữ chỉ số GReg ở 256², thêm vùng tay, MPJPE tay, bin chuyển động).
- **Bắt buộc thêm chỉ số thời gian**, vì xử lý từng khung: FVD, và sai số warp giữa khung liên tiếp (warping error dùng flow mesh).
- Báo cáo cả **trước và sau refiner** trên cùng ảnh thô, để tách đóng góp.

### 2.9 Ablation đề xuất

| # | Biến thể | Câu hỏi |
|---|---|---|
| 1 | Refiner chỉ nhận Î_coarse | Prior SD tự làm được đến đâu |
| 2 | + N_t | Điều kiện hình học có giúp tay đúng hình không |
| 3 | + M_t, V_t + loss vùng | Phần dành riêng cho tay |
| 4 | + loss thời gian | Đổi nhấp nháy lấy gì |
| 5 | Dữ liệu: checkpoint sớm vs suy giảm tổng hợp vs cả hai | Cách tạo cặp huấn luyện |
| 6 | Có / không dán lại vùng an toàn | Danh tính so với độ liền mạch |
| 7 | pix2pix-turbo vs kiểu OSEDiff | Lựa chọn nền |
| 8 | Refiner trên GReg gốc (TPSMM) vs GReg + A1 + B1 | Refiner có "che" được lỗi warp không, hay bắt buộc phải sửa A1 trước |

Ablation 8 rất đáng làm: nó cho thấy **refiner không thay được A1**. Một refiner không thể dựng lại cánh tay mà warp đã kéo giãn thành cao su; nó chỉ làm nét thứ đã gần đúng.

### 2.10 Rủi ro và cách giảm

| Rủi ro | Cách giảm |
|---|---|
| Nhấp nháy theo thời gian | Loss thời gian bằng flow mesh; mô hình một bước là tất định với cùng đầu vào (không lấy mẫu nhiễu ngẫu nhiên), nên vốn ít nhấp nháy hơn diffusion nhiều bước; sau cùng có thể chuyển lên Hướng 1 |
| "Bịa" hình dạng tay, sai nghĩa | Điều kiện N_t; loss keypoint tay; đo MPJPE tay trước/sau |
| Sụp đổ (output giống nhau), như issue #93 ở 256 px | Train ở 512; giữ skip connection; theo dõi đa dạng output trên val từ sớm |
| Lệch phân phối train/test của ảnh thô | Mục 2.5 |
| PSNR/SSIM giảm dù ảnh đẹp hơn | Giải thích perception–distortion; báo LPIPS/FVD/MPJPE tay song song |
| Giấy phép SD-Turbo | ⚠️ SD-Turbo theo giấy phép phi thương mại của Stability AI (cần xác nhận). Đủ cho nghiên cứu, ghi rõ trong luận văn |

### 2.11 Kế hoạch triển khai

1. **M0:** cài img2img-turbo, sửa thiếu `vision_aided_loss`, chạy lại ví dụ gốc để chắc môi trường đúng.
2. **M1:** tiền xử lý lại một tập con ở 512²; sinh ảnh thô từ checkpoint GReg sớm + suy giảm tổng hợp; train refiner chỉ nhận Î_coarse (ablation 1). Đo VRAM thật.
3. **M2:** thêm N_t, M_t, V_t, loss vùng (ablation 2–3).
4. **M3:** thêm loss thời gian (ablation 4); đánh giá toàn bộ test set.
5. **M4:** ablation 5–8.

---

## 3. So sánh hai hướng

| Tiêu chí | Hướng 1: Warp rồi Diffuse | Hướng 2: Refiner một bước |
|---|---|---|
| Sửa vùng bị che (lòng bàn tay…) | **Mạnh** (sinh có điều kiện + prior video) | Vừa (chỉ làm nét thứ GReg đã vẽ) |
| Sửa mờ | Mạnh | **Mạnh** |
| Nhất quán thời gian | **Có sẵn** | Phải thêm loss |
| Tính mới cho luận văn | **Cao** (chưa ai kết hợp mesh SMPL-X + video DiT cho ký hiệu) | Trung bình (ứng dụng kỹ thuật SR vào SLVG) |
| Compute train | 24 GB để thử; nên thuê 80 GB cho bản chính | 1× 24 GB, cần tối ưu bộ nhớ |
| Inference | Phút/clip; đánh giá trên tập con | 0.1–0.3 s/khung; đánh giá toàn test |
| Rủi ro kỹ thuật | Cao (pipeline lớn, nhiều ẩn số VRAM/API) | Thấp–vừa |
| Phụ thuộc A1/B1 | Dùng chúng làm điều kiện | Dùng chúng để có ảnh thô tốt |

**Lộ trình kết hợp đề xuất:** A1 + B1 → **Hướng 2** (kết quả chắc chắn, rẻ) → **Hướng 1** (đóng góp chính, nếu có GPU). Hai hướng không loại trừ nhau. Video sinh từ Hướng 1 vẫn có thể đi qua refiner ở Hướng 2 nếu cần siêu phân giải.

---

## 4. Những điều chưa chắc chắn cần kiểm chứng trước khi viết luận văn

1. VRAM thật khi train LoRA Wan-1.3B ở 480p, 49–81 khung, có/không fp8 + block swap.
2. API điều kiện đa kênh (mask + reference + control) của **Wan2.1-VACE-1.3B** và **Wan2.1-Fun-V1.1-1.3B-Control** (có nhận ảnh tham chiếu không?).
3. Dạng loss chính xác (flow matching) trong DiffSynth-Studio cho Wan, và cách thêm trọng số theo vùng.
4. VRAM thật khi train pix2pix-turbo ở 512 với batch 1 + gradient checkpointing.
5. Con số "tay/mặt ×10" của SignRefine (đang lấy từ bản review Pith, cần đọc bài gốc).
6. Giấy phép Wan2.1 (Apache-2.0?) và SD-Turbo (phi thương mại?).
7. Code chính thức của OSEDiff và SignRefine (đã có hay chưa).

---

## 5. Tóm tắt các con số đã kiểm chứng

| Con số | Ý nghĩa | Nguồn |
|---|---|---|
| 8.19 GB, ~4 phút / 5 s 480p trên 4090 | Inference Wan2.1-T2V-1.3B (có offload) | Wan2.1 model card |
| 281 s / 170 s / 85 s | 480p trên 4090 / A100 / H100 | InstaSD |
| ~15% | Diện tích tay + mặt ở lưới latent | SignRefine |
| tới 30%, >80% | Cải thiện độ chính xác tư thế tay; tỉ lệ người dùng ký hiệu ưa thích | SignRefine (tự báo cáo; Pith phê bình metric vòng tròn) |
| 1309 → 471 | FVD MimicMotion trên ASL 50K chỉ nhờ làm sạch pose | SignMimic |
| 0.11 s / 0.29 s | pix2pix-turbo 512² trên A100 / A6000 | Parmar et al. 2024 |
| 8.5M; 4× A100 × 1 ngày | Tham số huấn luyện và compute của OSEDiff | OSEDiff |
| ~28× | OSEDiff nhanh hơn StableSR | OSEDiff |

---

## 6. Tài liệu tham khảo

**Diffusion cho ngôn ngữ ký hiệu**
- SignRefine (ECCV 2026), Pelykh, Fish, Mercanoglu Sincan, Bowden: https://arxiv.org/abs/2609.08496 · HTML: https://arxiv.org/html/2609.08496 · Project: https://cogvis-cvssp.github.io/papers/signrefine/ · Review Pith (phê bình metric): https://pith.science/paper/2609.08496
- SignMimic (2026): https://arxiv.org/abs/2609.14122
- SignViP (NeurIPS 2025): https://arxiv.org/abs/2506.15980 · code: https://github.com/umnooob/signvip
- GReg (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_GReg_Geometry-Aware_Region_Refinement_for_Sign_Language_Video_Generation_ICCV_2025_paper.pdf

**Render hình học → diffusion ("warp rồi diffuse")**
- ViewCrafter (2024): https://arxiv.org/abs/2409.02048 · https://drexubery.github.io/ViewCrafter/
- StreetCrafter (2024): https://arxiv.org/abs/2412.13188
- GenWarp (NeurIPS 2024), phản biện: https://arxiv.org/abs/2405.17251 · weights: https://huggingface.co/Sony/genwarp
- Splatting-guided diffusion (về hallucination): https://arxiv.org/abs/2502.12752

**Animation người bằng diffusion có hình học 3D**
- Champ (ECCV 2024): https://arxiv.org/abs/2403.14781
- RealisDance: https://arxiv.org/abs/2409.06202
- VividPose: https://arxiv.org/abs/2405.18156
- MimicMotion (tăng trọng số vùng tay): https://arxiv.org/abs/2406.19680
- Wan-Animate: https://arxiv.org/abs/2509.14055

**Backbone và công cụ Wan**
- Wan2.1 repo: https://github.com/Wan-Video/Wan2.1 · model card 1.3B: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Benchmark độc lập: https://www.instasd.com/post/wan2-1-performance-testing-across-gpus · https://blog.salad.com/benchmarking-wan2-1/
- DiffSynth-Studio, tài liệu Wan: https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Wan.md
  - script LoRA Fun-InP: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh
  - validate LoRA Fun-Control: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py
- musubi-tuner (Wan): https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md
- Ghi chú train LoRA Wan 1.3B trên 24 GB: https://civitai.com/articles/12980/wan21-lora-training-note · https://www.stablediffusiontutorials.com/2025/03/wan-lora-train.html

**Diffusion một bước / refiner**
- pix2pix-turbo / CycleGAN-Turbo (Parmar et al. 2024): https://arxiv.org/abs/2403.12036 · code: https://github.com/GaParmar/img2img-turbo
  - issue #38 (thiếu `vision_aided_loss`): https://github.com/GaParmar/img2img-turbo/issues/38
  - issue #93 (khó tái lập, output sụp đổ): https://github.com/GaParmar/img2img-turbo/issues/93
- MariData (train một bước trên 1× RTX 4090): https://arxiv.org/abs/2606.03246
- OSEDiff (NeurIPS 2024): https://arxiv.org/abs/2406.08177
- TADSR (phê bình/cải tiến VSD): https://arxiv.org/abs/2508.16557
- InstantDrag (báo cáo loss phân kỳ khi train kiểu pix2pix-turbo): https://arxiv.org/abs/2409.08857
