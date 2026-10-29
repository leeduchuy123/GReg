# Các hướng cải tiến GReg cho lỗi "kéo tay bị mờ / biến dạng"

Ngày lập: 2026-10-29. Dựa trên (1) kết quả `run3` trên Drive (flow stage 1 đủ 100 epoch, stage 2 mới chạy sweep 15 epoch), (2) đọc code hiện tại, (3) khảo sát tài liệu 2019–2026 với 22 nguồn, 25 claim được kiểm chứng chéo (23 đứng vững, 2 bị bác bỏ, ghi ở cuối).

---

## 0. Tóm tắt một trang

**Lỗi gốc, nhìn thấy trực tiếp trong ảnh mẫu `run3`:**

| # | Quan sát | Bằng chứng trong run3 |
|---|---|---|
| L1 | Flow TPSMM không di chuyển được cẳng tay/bàn tay, chỉ kéo giãn ống tay áo như cao su | cột `warped I_s->t` của clip 11775, 15063 |
| L2 | Mặt nạ che khuất O tính ở 64×64, mềm, nên tay sinh ra bị trong suốt (tay ma) | cột `occlusion O`, output clip 13141 |
| L3 | Generator phải tự vẽ cả cánh tay không có texture, kết quả mờ | L1 toàn ảnh: copy nguồn 0.0387 → chỉ warp 0.0335 → GReg đầy đủ 0.0330 |
| L4 | Loss vùng bàn tay bị nền pha loãng: ô 64×64 nhưng bàn tay chỉ ~25×25 px (~15 % diện tích); prior của hand expert gần như trống | cột `regional priors R` |
| L5 | Train lấy cặp khung ngẫu nhiên, test luôn lấy khung đầu (tay buông) làm nguồn → test toàn chuyển động lớn | `dataset.py` vs `SequenceDataset` |
| L6 | Chỉ số đo toàn ảnh không nhạy với tay; kết quả sweep chỉ là nhiễu | chênh lệch ±5 %, 1 seed, L1 của `base` tăng theo epoch |

**Tin tốt:** mesh SMPL-X khớp ảnh thật rất sát, kể cả ngón tay (sai số khớp 2.5 px, `overlay_grid.png`). Mọi hướng xếp hạng cao bên dưới đều tận dụng điều này.

**Xếp hạng (tác động kỳ vọng × khả thi trên 1 GPU 16–24 GB):**

| Hạng | Hướng | Tác động lên lỗi kéo tay | Chi phí | Bằng chứng |
|---|---|---|---|---|
| 1 | Flow + mặt nạ nhìn thấy tính giải tích từ mesh SMPL-X, thay TPSMM | Rất lớn (L1, L2, L3) | Thấp, bỏ luôn stage 1 | Mạnh |
| 2 | Mặt nạ bàn tay/mặt từ mesh thay ô vuông; loss có trọng số theo vùng tay | Lớn cho bàn tay (L4) | Rất thấp | Trung bình (gián tiếp) |
| 3 | Sửa giao thức train/test + chỉ số đo theo vùng và theo biên độ chuyển động | Gián tiếp nhưng bắt buộc (L5, L6) | ~0 GPU | Trung bình |
| 4 | Warp trong không gian đặc trưng nhiều tầng, giữ nhánh warp RGB, gate bằng visibility | Vừa–lớn (L3) | Thấp | Mạnh (ablation LWG, Intrinsic Flow) |
| 5 | Đa khung tham chiếu, hợp nhất có trọng số | Lớn cho vùng bị che (lòng bàn tay, mặt trong cẳng tay) | Vừa; đổi giao thức | Mạnh về cơ chế |
| 6 | Bộ tinh chỉnh flow phần dư (quần áo, tóc, lỗi fit) | Vừa | Thấp | Trung bình |
| 7 | Kênh điều kiện riêng cho bàn tay + loss keypoint tay | Vừa cho bàn tay | Thấp–vừa | Trung bình |
| 8 | Ổn định theo thời gian (temporal) nhẹ | Giảm rung, không sửa mờ | Thấp–vừa | Trung bình |
| 9 | Neural texture trong không gian UV (atlas) + inpainting | Lớn về lý thuyết cho vùng chưa thấy | Cao (kiến trúc mới) | Trung bình |
| 10 | Renderer diffusion (fine-tune adapter/LoRA trên mô hình đóng băng) | Trần chất lượng cao nhất | Rất cao | Mạnh nhưng compute 4–8 GPU lớn |
| 11 | Tinh chỉnh bàn tay hậu kỳ bằng diffusion inpainting | Vừa, có nguy cơ rung | Vừa | Trung bình |
| — | Không khuyến nghị: NODE-Flow làm hướng chính, chuyển tư thế từng bước kiểu PATN, SAM làm mặt nạ chính, train diffusion từ đầu | xem mục 6 | | |

**Lộ trình đề xuất:** làm 3 → 1 → 2 → 4 trước (tất cả rẻ và nhắm thẳng vào L1–L6), chạy stage 2 đủ 100 epoch, rồi mới cân nhắc 5–8. Mục 9–11 là hướng "phần mở rộng" của luận văn nếu còn thời gian và GPU.

---

## 1. Nhóm A: Flow và che khuất (sửa gốc rễ)

### A1. Flow + mặt nạ nhìn thấy tính từ mesh SMPL-X (hạng 1)

**Thay đổi gì.** Bỏ `FlowEstimator` (TPSMM: ResNet-18, 50 keypoint, 10 TPS, dense motion 64×64). Thay bằng phép tính giải tích:

```
Với mỗi pixel đích p:
  f = face_idx_t(p), b = bary_t(p)                      (rasterizer đã có: greg/render/rasterize.py)
  q = Σ_k b_k · Π_s(V_s[f_k])                           vị trí của chính điểm bề mặt đó trên ảnh nguồn
  z_q = Σ_k b_k · depth_s(V_s[f_k])
  V_geo(p) = [ |depth_buffer_s(q) − z_q| < ε ]          nhìn thấy ở nguồn hay không (z-buffer)
  C(p)     = f(|det ∇F|, n·v)                           độ tin cậy: mức kéo giãn, góc nhìn
```

Mặt nạ O của GReg trở thành `O = V_geo · C` (có thể cho một mạng nhỏ học phần dư trên nền này), ở độ phân giải 256² với biên sắc nét.

**Tại sao nó sửa được lỗi.** Trường warp 2D mượt (TPS, hay bất kỳ trường vận tốc tích phân nào) là một đồng phôi: không thể biểu diễn tay gập đè lên thân (ánh xạ nhiều-một, gián đoạn ở biên che khuất). Chiếu từ 3D có z-buffer thì sự gián đoạn tự xuất hiện. Khi tay buông, mặt trước ống tay áo vẫn nhìn thấy trong ảnh nguồn, nên texture cánh tay được lấy từ nguồn thay vì bắt generator vẽ.

**Chi phí.** Tiền xử lý thêm `face_idx` (uint16 PNG) + `bary` mỗi khung, và tọa độ chiếu + depth của 10 475 đỉnh mỗi khung (float16, ~63 KB/khung). Hoặc tính trực tiếp trên GPU trong vòng lặp train. Không cần huấn luyện stage 1 nữa (tiết kiệm 100 epoch).

**Code tham khảo (public).** Liquid Warping GAN / iPERCore (SMPL → correspondence map → flow, Neural Mesh Renderer), Dense Intrinsic Appearance Flow (`ly015/intrinsic_flow`, có visibility map 3 mức: thấy / không thấy / nền).

**Bằng chứng.** LWG: flow từ mesh "chính xác hơn và ít lệch hơn affine từ keypoint". Intrinsic Flow: cơ chế face_idx + barycentric + visibility giống hệt. Cả hai là GAN 2019, chạy được trên 1 GPU.

**Rủi ro.** SMPL-X không có quần áo và tóc: ống tay áo rộng, tóc dài không có tương ứng thật → cần A3 (refiner) và C(p). Lỗi fit (2.5 px) lan thẳng vào flow; ở bàn tay có thể lớn hơn. Nhòe chuyển động trong video thật làm ground truth tay cũng mờ.

### A2. Warp trong không gian đặc trưng nhiều tầng + giữ nhánh RGB (hạng 4)

**Thay đổi gì.** Encoder mã hóa ảnh nguồn một lần; ở mỗi tầng UNet, warp feature bằng flow A1 (thu nhỏ tương ứng) rồi nhân với `V_geo` ở tầng đó; vẫn giữ nhánh warp RGB toàn độ phân giải để blend cuối. Repo đã có mẫu sẵn: `InpaintingNetwork` của stage 1 làm đúng việc này nhưng bị bỏ ở stage 2.

**Bằng chứng.** Ablation của Liquid Warping GAN trên iPER: chỉ warp RGB SSIM 0.822 / LPIPS 0.113; chỉ warp feature 0.830 / 0.103; cả hai (LWB) 0.840 / 0.087. Intrinsic Flow: chỉ warp feature mất chi tiết tần số cao, thêm nhánh pixel + trọng số mềm `x̂ = z·x_warp + (1−z)·x_gen` tăng FashionIS 4.800 → 4.898; gating theo visibility tăng FashionIS 4.491 → 4.898 trong khi SSIM toàn ảnh **không đổi** (0.778) — chính là hiện tượng L6.

**Lưu ý.** Intrinsic Flow báo cáo failure case khi nhánh pixel trộn vùng không nhất quán do lệch tư thế lớn → trọng số z phải bị chặn bởi `V_geo · C`.

### A3. Bộ tinh chỉnh flow phần dư ΔF (hạng 6)

**Thay đổi gì.** Mạng nhỏ nhận (feature nguồn đã warp, N_t, V_geo) → ΔF, khởi tạo 0, phạt độ lớn + độ mượt. Sửa phần mesh không mô hình được: tay áo, tóc, lỗi fit ngón tay. Đây là chỗ hợp lý để thử NODE-Flow làm **ablation** (so 1 lớp conv / GRU 3 bước / NODE midpoint). Vì phần dư nhỏ, kỳ vọng NODE không thắng đáng kể, nhưng câu hỏi được trả lời bằng số.

**Code tham khảo.** "Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction" (Lu et al. 2022): residual flow từng tầng + attention map + occlusion map, cỡ GAN.

### A4. Đa khung tham chiếu (hạng 5)

**Thay đổi gì.** Ngoài khung đầu (tay buông), cho thêm 1–2 khung nguồn có tay giơ (lộ lòng bàn tay, mặt trong cẳng tay). Mỗi nguồn có flow + V_geo riêng; hợp nhất theo trọng số `C` hoặc attention, **không cộng thẳng**.

**Bằng chứng.** LWG hợp nhất `X̂_t = BS(X_s1,T1) + BS(X_s2,T2) + X_t`; bản TPAMI 2022 (AttLWB) chỉ ra cộng thẳng làm phình biên độ feature ở vùng chồng lấn gây artifact, phải thay bằng attention + SPADE.

**Lưu ý giao thức.** Thay đổi protocol đánh giá reconstruction của GReg (1 khung nguồn) → phải báo cáo thành thiết lập riêng ("k-shot"), không so trực tiếp với Table 1 của bài báo.

---

## 2. Nhóm B: Bàn tay

### B1. Mặt nạ bàn tay/mặt từ mesh thay ô vuông cố định (hạng 2) — đề xuất của bạn, đã kiểm chứng trên run3

**Hiện trạng.** `region_boxes` tạo ô 64×64 cố định quanh tâm khớp. Với bàn tay, ~85 % ô là nền/áo. Ba loss vùng (VGG vùng, L_struct, PatchGAN vùng) và hand expert đều bị pha loãng; prior tay học được gần bằng 0.

**Thay đổi gì (không cần SAM).** Cache đã có `parts/*.png` với nhãn 3 = tay trái, 4 = tay phải, lấy từ skinning weight SMPL-X (`greg/render/body.py:32`), khớp pixel với `N_t`.
1. Ô crop = bounding box của mặt nạ tay + lề 15 %, resize về 64 (hoặc 96 để "phóng to" tay); paste ngược bằng resize nghịch đảo.
2. Nhân mặt nạ tay (giãn 3–5 px để phủ cổ tay/tay áo) vào L1/VGG vùng và vào đầu vào discriminator vùng.
3. Trọng số loss toàn ảnh: tăng trọng số pixel thuộc tay (kiểu "hand-region enhancement" của MimicMotion) để L3 không bị nền chi phối.

**Khi nào mới cần SAM / SAM 2.** Chỉ khi mesh lệch tay áo rộng hoặc tay nhòe chuyển động; chạy offline một lần trên khung đích của tập train (lúc inference không cần mask vì RE chỉ nhận crop normal). HaMeR/WiLoR cho MANO tốt nhưng hay rớt tay khi che khuất nặng, nên SMPL-X đã fit sẵn vẫn là nguồn mask tin cậy hơn; HaMeR/WiLoR hợp hơn cho **metric/loss keypoint tay** (B2).

**Bằng chứng (gián tiếp, chưa có ablation "mesh-mask vs ô vuông" trong tài liệu).** SignRefine (ECCV 2026): trọng số MSE theo vùng mặt/tay + masked cross-attention theo bbox; bỏ masked attention làm MPJPE tay trái 6.638 → 7.309, mặt +10 %; tác giả quy chất lượng tay cho việc **phóng to vùng tay lên 256×256**. MimicMotion: tăng trọng số loss vùng tay giảm ngón tay méo (ablation định tính). "Giving a Hand to Diffusion Models" (Pelykh/Bowden, FG 2024, code public): sinh tay kèm mặt nạ tay riêng rồi outpaint thân, chính xác hơn điều kiện toàn thân.

→ Vì chưa có ai đo trực tiếp, đây là **ablation có giá trị đóng góp** cho luận văn.

### B2. Loss và chỉ số keypoint tay (hạng 7, phần loss)

**Thay đổi gì.** Chạy RTMPose-X / DWPose (hoặc WiLoR) đóng băng trên ảnh sinh ra; loss MSE keypoint tay (kiểu HANDI) hoặc Region-Aware Cycle Loss (RACL, 2409.09149) tách lỗi tay khỏi lỗi toàn thân. Cùng bộ đó dùng làm metric MPJPE tay (chuẩn 2026 trong SignRefine) và Hand SSIM / Hand PE trên patch 60×60 quanh khớp giữa (FPDM 2412.07333, có giá trị tham chiếu trên PHOENIX).

**Chi phí.** Thêm một mạng pose đóng băng trong vòng lặp train (vài trăm MB VRAM, chậm ~20–30 %). Có thể chỉ bật ở nửa sau huấn luyện.

### B3. Kênh điều kiện riêng cho bàn tay + gating theo pixel (hạng 7, phần kiến trúc)

**Thay đổi gì.** Ngoài `N_t` toàn ảnh, render thêm tay ở độ phân giải cao (crop 128² từ mesh tay, có depth), đưa vào generator qua nhánh riêng; trộn các nguồn điều kiện bằng gating theo pixel.

**Bằng chứng.** Champ (ECCV 2024): SMPL một mình không đủ cho tay/mặt, bỏ skeleton phụ làm PSNR 29.84 → 29.05, FVD 170 → 184 (nhưng Champ dùng SMPL không có ngón tay, nên bài học chuyển sang SMPL-X yếu hơn). RealisDance: tay thật nhờ render tay 3D HaMeR + adaptive pose gating (chỉ so sánh định tính, không có ablation số). Bản DiT của DreamActor-M1 (2512.21905) thêm HaMeR nhiều khung + "sharpness guidance" để tay rõ cả khi video nguồn nhòe — đúng chế độ tay chuyển động nhanh của ngôn ngữ ký hiệu.

---

## 3. Nhóm C: Giao thức huấn luyện và đánh giá (hạng 3, ~0 GPU)

### C1. Lấy mẫu cặp theo biên độ chuyển động
- 50 % cặp: nguồn = khung đầu hợp lệ (khớp với test), đích ngẫu nhiên.
- 50 % cặp: ngẫu nhiên, nhưng ưu tiên khoảng cách tư thế lớn (khoảng cách cổ tay 2D hoặc khoảng cách tham số SMPL-X), theo curriculum tăng dần và **giữ lại cặp chuyển động nhỏ** để không quên (tiền lệ: curriculum theo stride khung trong 2610.33566).

### C2. Chỉ số đo
- L1 / LPIPS tính riêng trên mặt nạ **cánh tay + bàn tay** (từ `parts`), trên mặt.
- MPJPE tay (RTMPose-X / DWPose), Hand SSIM; AKD giữ nguyên để so với bài báo.
- **Phân tầng theo biên độ chuyển động** (bin khoảng cách cổ tay nguồn–đích: nhỏ / vừa / lớn). Không có bin, cải thiện ở tay bị trung bình toàn ảnh nuốt mất (minh họa: Intrinsic Flow SSIM không đổi 0.778 trong khi FashionIS +0.4).
- Baseline bắt buộc trong mọi bảng: "copy nguồn" và "chỉ warp".

### C3. Vệ sinh run
- Bật bộ lọc fade-in (`02b_mark_valid_rows.py`, đã có trong code, run3 chưa có).
- Chạy stage 2 đủ 100 epoch trước khi tune siêu tham số; sweep 15 epoch hiện tại không có ý nghĩa thống kê (1 seed, ±5 %).

---

## 4. Nhóm D: Ổn định theo thời gian (hạng 8)

GReg sinh từng khung độc lập. Với flow giải tích (A1), tính nhất quán đã tốt hơn vì không còn keypoint học nhiễu. Nếu vẫn rung:
- Loss nhất quán: warp output khung t−1 sang t bằng flow mesh giữa hai khung đích (có sẵn miễn phí), phạt sai khác ở vùng `V` thấy được.
- Temporal attention nhẹ trên backbone đóng băng: ablation 2603.17398 cho thấy bỏ nó FVD 86 → 211; nhưng loss hiệu latent giữa khung **không** giúp trong cùng bài → ưu tiên attention hơn loss nếu phải chọn.

Không sửa lỗi mờ, chỉ giảm nhấp nháy; xếp sau A–C.

---

## 5. Nhóm E: Hướng nặng, trần chất lượng cao hơn

### E1. Neural texture trong không gian UV của SMPL-X (hạng 9)
Texture/feature gắn trên bề mặt cơ thể, bất biến tư thế; tích lũy từ nhiều khung; vùng chưa thấy được inpaint trong không gian UV (DINAR, ICCV 2023, code `SamsungLabs/DINAR`, dùng latent diffusion inpainting; cần 2 khung cùng người khác tư thế mỗi bước — SignAvatars có sẵn). Bản lai flow + atlas UV (2412.06174, IEEE TMM 2025, train 1× RTX 3090 Ti, 600k iter) tự nêu thất bại ở "arms, hands, feet" khi DensePose thiếu tương ứng và không có code. Điểm yếu chung: tay áo rộng, texture không đổi theo tư thế.

### E2. Renderer diffusion (hạng 10)
Tình trạng 2025–2026 của SLVG đang dịch chuyển sang đây: SignViP (NeurIPS 2025, SD1.5 + ReferenceNet + AnimateDiff, 4× A6000, code public, tay MPJPE tăng gấp đôi ngoài PHOENIX), SignRefine (ECCV 2026, Wan-Fun-Control-1.3B đóng băng + adapter vùng, 4× GH200, −30 % MPJPE tay trên NVSign), SignMimic (MimicMotion đóng băng + chỉ huấn luyện module làm sạch pose; chỉ làm sạch pose đã giảm FVD 1309 → 471 trên ASL 50K). Champ (SMPL depth/normal/semantic + DWPose, 8× A100, inference ~20 GB), RealisDance (DWPose + SMPL + HaMeR, code + weights, SD1.5), MIMO, VividPose (SMPL-X dense maps, không có code), Wan-Animate (14B, weights public, chỉ dùng zero-shot).

**Khả thi trên 1 GPU 16–24 GB:** chỉ ở mức (a) inference zero-shot làm baseline/upper bound, hoặc (b) fine-tune adapter / LoRA trên renderer đóng băng (SD1.5-ReferenceNet kiểu RealisDance, hoặc Wan-1.3B kiểu SignRefine) với điều kiện **normal map + visibility SMPL-X** của luận văn — điểm chưa nhóm nào khai thác với renderer diffusion. Bài học từ SignMimic: diffusion off-the-shelf thất bại trên video ký hiệu nếu không có xử lý pose/điều kiện chuyên biệt. Text2Sign (2607.13164) chứng minh diffusion 1 GPU L4 chỉ đạt 64×64, không có metric tay.

### E3. Tinh chỉnh bàn tay hậu kỳ (hạng 11)
HandRefiner / RHanDS / 2506.12680: inpaint vùng tay bằng SD1.5 ControlNet (8–12 GB) với điều khiển mesh/depth tay; mesh tay SMPL-X có sẵn làm điều khiển. Nhược điểm: xử lý từng khung → rung theo thời gian; dễ "sửa" nhầm hình dạng tay làm sai nghĩa ký hiệu. Chỉ dùng nếu B1–B3 chưa đủ.

---

## 6. Những hướng KHÔNG khuyến nghị (và lý do)

| Hướng | Lý do |
|---|---|
| NODE-Flow (Mirvakhabova, CVPRW 2025) làm hướng chính | Biến `t` của nó là thời gian **tinh chỉnh ước lượng**, không phải thời gian chuyển động; đầu ra vẫn là một trường 2D mượt và vẫn warp một lần → không sửa được L1. Tham số 4.7M → 9.2M, không nhanh hơn. Chỉ giữ làm ablation trong A3. |
| Chuyển tư thế từng bước qua tư thế trung gian (PATN, CoP) | Khi đã có mesh, tương ứng đỉnh-đỉnh không phụ thuộc đường đi → bước trung gian không thêm thông tin. Tài liệu ghi nhận PATN mất chi tiết, mờ, lặp mẫu giả (DRN, PoNA). |
| Tích phân trường vận tốc 2D | Đồng phôi theo cấu trúc, không biểu diễn được che khuất. |
| SAM / SAM 2 làm mặt nạ chính | `parts/*.png` từ mesh đã khớp pixel với `N_t` và miễn phí; SAM thêm phụ thuộc nặng và một nguồn lệch thứ hai. Chỉ dùng phụ (B1). |
| Train diffusion từ đầu | 4–8 GPU A100/A6000/GH200 theo mọi nguồn tự báo cáo. |
| Avatar 3D Gaussian / NeRF từ một ảnh | Không có claim nào sống sót qua kiểm chứng trong vòng khảo sát này; chi phí cao, chưa thấy áp dụng cho ngôn ngữ ký hiệu. |

---

## 7. Lộ trình thí nghiệm

| Bước | Nội dung | Mục nào | GPU |
|---|---|---|---|
| E0 | Chẩn đoán trên ~50 clip val: L1 vùng cánh tay/bàn tay của warp TPSMM vs warp mesh, chỉ ở nơi `V_geo = 1`, chia theo bin cổ tay; đo tỉ lệ pixel tay đích không thấy ở nguồn | A1, C2 | CPU được |
| E1 | Thay TPSMM bằng A1 (`model.flow.type: mesh`), O = V_geo·C + phần dư học; bật fade filter; sửa sampling C1; huấn luyện stage 2 100 epoch | A1, C1, C3 | 1 GPU |
| E2 | B1: mặt nạ tay từ mesh cho loss/discriminator + crop sát; ablation mesh-mask vs ô vuông | B1 | 1 GPU |
| E3 | A2: warp feature nhiều tầng vs RGB vs cả hai | A2 | 1 GPU |
| E4 | A3: ΔF conv / GRU / NODE | A3 | 1 GPU |
| E5 | A4 đa khung (báo cáo thành thiết lập riêng), B2 loss keypoint tay, B3 kênh tay | A4, B2, B3 | 1 GPU |
| E6 | (tùy chọn) E2: LoRA trên renderer diffusion với điều kiện SMPL-X; zero-shot Wan-Animate / MimicMotion làm upper bound | E2 | ≥ 24 GB |

Mọi bảng kết quả: có baseline "copy nguồn" và "chỉ warp", có bin biên độ chuyển động, có MPJPE tay.

---

## 8. Nguồn chính

Ngôn ngữ ký hiệu
- GReg (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_GReg_Geometry-Aware_Region_Refinement_for_Sign_Language_Video_Generation_ICCV_2025_paper.pdf
- SignRefine (ECCV 2026): https://arxiv.org/abs/2609.08496
- SignMimic (2026): https://arxiv.org/abs/2609.14122
- SignViP (NeurIPS 2025): https://arxiv.org/abs/2506.15980
- Text2Sign 1-GPU (2026): https://arxiv.org/abs/2607.13164
- FPDM, Hand SSIM / Hand PE trên PHOENIX: https://arxiv.org/pdf/2412.07333
- Danh mục SLP: https://github.com/VIPL-SLP/Awesome-Sign-Language-Processing

Flow từ mesh, warp feature, visibility, đa nguồn
- Liquid Warping GAN (ICCV 2019): https://arxiv.org/pdf/1909.12224 ; AttLWB (TPAMI 2022): https://arxiv.org/abs/2011.09055
- Dense Intrinsic Appearance Flow (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Dense_Intrinsic_Appearance_Flow_for_Human_Pose_Transfer_CVPR_2019_paper.pdf
- Residual flow, multi-source (2022): https://arxiv.org/html/2202.00843v1
- One-shot flow + neural texturing (TMM 2025): https://arxiv.org/abs/2412.06174
- DINAR (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Svitov_DINAR_Diffusion_Inpainting_of_Neural_Textures_for_One-Shot_Human_Avatars_ICCV_2023_paper.html
- PATN mở rộng và nhược điểm: https://arxiv.org/pdf/2103.11622

Bàn tay
- MimicMotion (hand-region loss weighting): https://arxiv.org/pdf/2406.19680
- WiLoR (CVPR 2025): https://arxiv.org/html/2409.12259v2
- Giving a Hand to Diffusion Models (FG 2024): https://arxiv.org/abs/2403.10731
- HandRefiner: https://arxiv.org/html/2311.17957v2
- Region-Aware Cycle Loss: https://arxiv.org/pdf/2409.09149

Diffusion có dẫn hướng 3D
- Champ (ECCV 2024): https://arxiv.org/abs/2403.14781
- RealisDance: https://arxiv.org/abs/2409.06202
- VividPose: https://arxiv.org/abs/2405.18156
- DreamActor-M1: https://arxiv.org/abs/2504.01724
- Wan-Animate: https://arxiv.org/abs/2509.14055
- MIMO (CVPR 2025): https://arxiv.org/pdf/2409.16160
- StableAnimator: https://github.com/Francis-Rings/StableAnimator

Huấn luyện / temporal
- Curriculum theo biên độ chuyển động (2026): https://arxiv.org/html/2610.33566
- Temporal attention nhẹ (2026): https://arxiv.org/html/2603.17398v1

NODE-Flow (để đối chiếu): https://openaccess.thecvf.com/content/CVPR2025W/IMW/papers/Mirvakhabova_Learning_Optical_Flow_Field_via_Neural_Ordinary_Differential_Equation_CVPRW_2025_paper.pdf

---

## 9. Giới hạn của khảo sát này

- Hai claim bị bác bỏ khi kiểm chứng chéo, **không trích dẫn**: (a) ablation "bỏ 3D hands → Hand SSIM 0.614 → 0.477, FVD 275 → 383" của SignViP; (b) ablation feature-warping / occlusion-map của 2412.06174. Vì vậy luận điểm "điều kiện tay 3D cải thiện tay đo được" chỉ còn dựa trên lý giải của RealisDance (không có ablation số) và ablation skeleton toàn ảnh của Champ.
- Phần lớn số liệu định lượng về warp mesh / feature / visibility là GAN 2019 trên iPER và DeepFashion (một lần chạy, không phương sai, không phải video ký hiệu). Chuyển sang tay nhỏ và chuyển động lớn trong SignAvatars là suy luận; E0–E3 là để kiểm chứng.
- Chưa có nguồn nào đo trực tiếp "mặt nạ tay từ mesh/SAM vs ô vuông" → B1 là ablation mới của luận văn.
- GReg không có code chính thức, không nêu Hr×Wr; phương trình L_struct có thể lỗi đánh máy (grayscale vs gradient).
- SignRefine và SignMimic rất mới (tự báo cáo, chưa tái lập độc lập); tình trạng code SignRefine chưa xác nhận; 2412.06174 không có code.
- Số liệu compute (Champ 8× A100, ~20 GB inference; SignRefine 4× GH200; SignViP 4× A6000; StableAnimator train 40–70 GB) là tự báo cáo.
