# Tổng hợp 8 bài báo diffusion / mesh 3D: note đọc kỹ paper + code

Ngày lập: 2026-10-29. PDF của cả 8 bài nằm trong `ref papers/`. File đi kèm:
- `docs/bai_hoc_diffusion_mesh_cho_greg.md`: những điểm hay nhất áp dụng được cho GReg, kèm đề xuất chỉnh sửa.
- `docs/diffusion_improvement.md`: hai hướng diffusion H1 (warp-then-diffuse) và H2 (refiner một bước).

Quy ước (giữ như các file docs khác):
- ✅ = đã kiểm chứng trong paper (mục/bảng/phương trình) hoặc code (`file:dòng`).
- ⚠️ = suy luận, thiết kế đề xuất hoặc chưa có số liệu, **phải kiểm tra trước khi viết vào luận văn**.

**Về trích dẫn `file:dòng`:** đường dẫn tính từ gốc repo, tại đúng commit trong bảng dưới. Nếu repo cập nhật, số dòng có thể lệch; khi đó checkout đúng commit để đối chiếu.

---

## A. Danh sách bài và trạng thái code

| # | Bài | Venue | Code chính thức | Commit đã đọc | License | Đã đọc | Liên quan GReg |
|---|---|---|---|---|---|---|---|
| 1 | PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion | CVPR 2026 | **Không có** (không repo, không project page) | – | – | Paper + phụ lục arXiv v4. Code mở của khối nền BD3-LM (`block_diff_mask`), **không phải code chính thức** | ★★ (ý tưởng attention global + part) |
| 2 | CraftsMan3D: High-fidelity Mesh Generation with 3D Native Diffusion and Interactive Geometry Refiner | CVPR 2025 | **Có một phần**: [HKUST-SAIL/CraftsMan3D](https://github.com/HKUST-SAIL/CraftsMan3D) (trước là `wyysf-98/CraftsMan`). Chỉ có bản single-image; **refiner và bản multi-view trong paper không phát hành** | `f87d6a5` (2025-06-26) | Mâu thuẫn: README ghi MIT, `gradio_app.py` ghi AGPL-3.0, không có file LICENSE; `volume_decoders.py` mang header phi thương mại của Tencent | Paper + toàn bộ ~12.5k dòng | ★★ (Poisson blend, ControlNet-Tile refiner) |
| 3 | Single Mesh Diffusion Models with Field Latents for Texture Generation (arXiv 2312.09250) | CVPR 2024 | **Có**: [google-research/mesh_diffusion](https://github.com/google-research/google-research/tree/master/mesh_diffusion) (JAX/Flax). Tiền xử lý mesh nằm ở repo riêng `twmitchel/field_latent_preprocess` (chưa đọc) | `1ae1fde` (2026-10-28) | Apache-2.0 | Paper + supplementary + toàn bộ 15 file (~4.8k dòng) | ★★ (texture trên bề mặt, inpainting kiểu thay latent) |
| 4 | Dexterous World Models (DWM) | CVPR 2026 | **Có**: [snuvclab/dwm](https://github.com/snuvclab/dwm) + checkpoint HF (CogVideoX-Fun-5B, Wan2.1-Fun-14B) | `19de79b` (2026-10-26) | Apache-2.0 | Paper + code của tác giả (train/pipeline/model/data/render tay) | ★★★★★ (gần như đúng công thức H1) |
| 5 | Generative Rendering: Controllable 4D-Guided Video Generation with 2D Diffusion Models | CVPR 2024 | **Không có**. Phương pháp được bảo hộ bằng patent Mỹ US 12,664,751 (Adobe). Không tìm được bản tái hiện | – | – | Paper + phụ lục A–E (arXiv). Code mở gần nhất về cơ chế: [SyncMVD](https://github.com/LIU-Yuxin/SyncMVD) @`faa9562`, **không phải code của bài** | ★★★★ (tương ứng mesh → nhất quán thời gian) |
| 6 | Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models | ICLR 2026 | **Không có**. Project page ghi "Coming soon"; link ẩn danh trong phụ lục đã hết hạn (`repository_expired`) | – | – | Paper + toàn bộ phụ lục | ★★★ (metric, làm mượt SMPL-X, neo đỉnh) |
| 7 | Towards 3D-Aware Video Diffusion Models: Render-Free Human Motion Control with Mesh Tokenization ("MeshToken", arXiv 2606.02000) | 2026 (arXiv) | **Chưa có**: [jingyunliang/MeshToken](https://github.com/jingyunliang/MeshToken) chỉ có README 1 dòng. Đọc thay bằng code bài trước của cùng tác giả: [alibaba-damo-academy/Uni3C](https://github.com/alibaba-damo-academy/Uni3C) (SIGGRAPH Asia 2025, chỉ có inference) | Uni3C `75ed6e2` (2025-09-20) | MeshToken: chưa có; Uni3C: xem repo | Paper + code của tác giả Uni3C | ★★★★ (Uni3C ≈ code "warp-then-diffuse") |
| 8 | Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video | CVPR 2026 | **Có**: [jzr99/Mesh4D](https://github.com/jzr99/Mesh4D), chỉ có inference (không có script train, nhưng còn LightningModule train) | `30269e3` (2026-01-09) | Tencent Hunyuan3D 2.1 Community License (phi thương mại, không áp dụng ở EU/UK/Hàn Quốc) | Paper + code của tác giả | ★★★★ (tọa độ chính tắc, tiền lệ tương ứng barycentric) |

**Tóm lại:** 4/8 bài có code chính thức (CraftsMan3D chỉ một phần, Mesh4D chỉ inference). 1 bài có repo rỗng, đã đọc thay bằng Uni3C. 3 bài không có code.

---

## B. Mỗi bài một câu: điều đáng học nhất

| # | Bài | Điều đáng học nhất cho GReg |
|---|---|---|
| 1 | PartDiffuser | Điều kiện **global + part** tốt hơn hẳn chỉ part (CD×10³ 17.8 so với 54.7) hoặc chỉ global (29.1) ✅. Adapter tay của H1 nên cho key/value = [token toàn cục + token crop tay], không chỉ crop tay. |
| 2 | CraftsMan3D | Diffusion "làm nét" âm thầm làm **lệch tần số thấp** (màu, tông da) ✅ (Fig. 7/10). Dùng blend Poisson hoặc tách tần số thay cho blend cứng `V·I_warp + (1−V)·Î`. |
| 3 | Field Latents | Thay latent vùng đã biết bằng latent gốc re-noise ở mỗi bước (Eq. 14) để **giữ vùng nhìn thấy, chỉ sinh vùng che**. Ngoài ra có ý tưởng "bộ nhớ texture trên bề mặt SMPL-X". |
| 4 | DWM | Đưa điều kiện "tĩnh" vào **khe masked-video của mô hình inpainting** (mô hình bắt đầu như hàm đồng nhất), thêm render tay bằng **kênh concat zero-init + LoRA r64**. Thay "video tĩnh" bằng video warped của A1 và mask bằng `V·C`, ta có ngay H1. |
| 5 | Generative Rendering | Tương ứng mesh tiêm vào **trước và sau self-attention** cho nhất quán dài hạn tốt hơn cả mô hình T2V ✅. Chép đặc trưng thẳng qua tương ứng, không qua attention, thì hỏng ✅ (Fig. 6). |
| 6 | Animating the Uncaptured | Điều kiện chỉ bằng khung đầu + text gây morphing và bịa vật thể ✅. Cần điều kiện hình học dày theo từng khung. Làm mượt tham số theo thời gian giảm lỗi gia tốc 2.65 → 1.50 ✅. |
| 7 | MeshToken (+Uni3C) | Bằng chứng cho "render-free" yếu. Điều kiện căn pixel thắng về fidelity (PSNR 22.57 so với 16.78) ✅. Uni3C cho mẫu code controller: point cloud render lại + mask lỗ + zero-init cộng vào DiT. |
| 8 | Mesh4D | Positional embedding từ **vị trí chính tắc 3D**, dùng chung giữa các khung. Ở GReg tương ứng với **bản đồ tọa độ T-pose SMPL-X từng pixel**, có sẵn từ A1 và có giá trị cả ở vùng bị che. |

---

## C. So sánh chéo

### C.1 Hình học đi vào mô hình sinh bằng cách nào

| Bài | Biểu diễn hình học | Cách đưa vào | Căn pixel? | Khởi tạo khối mới |
|---|---|---|---|---|
| DWM | Render mesh tay (L xanh / R đỏ, Phong đèn tại camera) + video cảnh tĩnh | VAE-encode → **concat kênh** trước patch-embed; cảnh vào khe masked-video của InP | Có | Cột trọng số mới = 0, patch-embed train full |
| Uni3C (đại diện MeshToken) | Point cloud render lại + mặt nạ lỗ + Plücker | Controller 20 tầng (dim 1024) nhìn cả `x_t`, **cộng residual** vào 20/40 block đầu | Có | `zero_module` cho output |
| MeshToken | 54 token VQ-HPS/khung (SMPL, không ngón) + vector quỹ đạo | **Cross-attention theo từng latent frame** sau text-CA | Không | `to_out` zero-init |
| Generative Rendering | UV + depth render | ControlNet-depth + tiêm đặc trưng attention qua UV (pre/post-attn) + nhiễu UV | Có (qua UV) | Training-free |
| Mesh4D | Điểm mẫu trên mesh khung 1 + skinning | Sin-cos PE của vị trí chính tắc; mask attention theo xương | 3D | Zero-init mọi output projection; mẹo "nhân đôi /2" khi nạp trọng số VAE |
| PartDiffuser | Point cloud global + theo part (PartField) | Cross-attention: block i chỉ thấy `[C_global, C_part_i]` | 3D | – |
| CraftsMan3D | Ảnh đa góc / normal render | Token CLIP + DINO cộng theo vị trí patch → cross-attn; refiner: ControlNet-Tile trên normal | – | ModLN zero-init |
| Field Latents | Trường vector tiếp tuyến trên mesh | Field convolution; inpainting bằng thay latent mỗi bước | Bề mặt | – |

**Kết luận chéo ⚠️:** với bài toán tái tạo có camera cố định như GReg, ba bài có code (DWM, Uni3C, Mesh4D) cùng chỉ về một công thức: **điều kiện căn pixel + kênh mới zero-init + backbone đóng băng / LoRA**. Token 3D (MeshToken) chỉ đáng làm **bổ sung** cho vùng bị che (ngón tay chồng nhau).

### C.2 Công thức huấn luyện thực tế đọc được từ code

| | DWM (CogVideoX-5B) | DWM (Wan-14B) | Mesh4D | Field Latents | CraftsMan3D |
|---|---|---|---|---|---|
| Loss | v-pred viết dưới dạng x0 có trọng số `1/(1−ᾱ)` | Flow matching, MSE, bỏ phần tử \|diff\|>50 | Flow matching (SiT) | **L1** (paper ghi L2) | ε / Min-SNR γ=5 (paper không nhắc) |
| Timestep | Đều {0..999} | Đều theo chỉ số lịch, shift 5, phân tầng theo rank | – | Cosine (paper ghi linear) | – |
| Fine-tune | LoRA r64 (q,k,v,o) + patch-embed full | LoRA r64 mọi Linear + patch-embed full | Train full DiT mới | Từ đầu, một mesh | Từ đầu |
| lr / opt | 1e-4, AdamW (0.9, 0.95), wd 0.01, cosine, warmup 100 | như bên trái | LR hằng (quirk `steps=1e9`) | 5e-4 (paper ghi 1e-3) | – |
| Dữ liệu | 17 010 clip × 49 khung, 480×720, 8 fps | như bên trái | Cửa sổ T=6 khung | 1 mesh | Objaverse |
| Dropout điều kiện | 0 (có code drop tay → target tĩnh nhưng tắt) | như bên trái | 10%/khung, CFG 5 | – | Theo batch (lỗi; nên theo mẫu) |
| Compute | GPU lớp 80 GB | – | 68 GB khi overfit (comment config) | 8×V100 × ~2 ngày / mesh | 32×A800 × 7 ngày |

### C.3 Những điểm code khác paper đáng ghi nhớ
- **DWM:** loss là v-pred/flow matching, không phải ε như Eq. 7. Prompt sinh từ video GT nên rò thông tin. Mesh tay tổng hợp (Character Creator) khác mesh tay thật (MANO/HaMeR). Toàn bộ 18 cảnh test TRUMANS đều có trong train.
- **Mesh4D:** paper ghi DINO-Giant nhưng code dùng DINOv2-Large. Mỗi cửa sổ chỉ T=6 khung, trong khi paper nói "toàn chuỗi". Bản phát hành bỏ bước căn chỉnh CPD.
- **CraftsMan3D:** ngưỡng TSDF 1/128 (paper 1/256), winding number 0.5 (paper 0.75). Nhiều lỗi khiến không tái lập được số trong paper.
- **Field Latents:** loss L1, lịch cosine, lr 5e-4/200K bước (paper: L2, linear, 1e-3/300K). Attention dùng trọng số thực dù paper ghi phức.
- **Animating the Uncaptured:** không công bố trọng số loss; "feature mapper" không được mô tả.
- **PartDiffuser:** thân bài và phụ lục mâu thuẫn ở vài chỗ; thiếu `w(t)`, β/ω và cách tính `C_part_i`.

---

## D. Hướng dẫn đọc

Mỗi note có cùng cấu trúc: 0. Thông tin nhanh · 1. TL;DR · 2. Bài toán · 3. Phương pháp · 4. Kết quả & ablation · 5. Đọc code (hoặc pseudo-code nếu không có code) · 6. Hạn chế · 7. Liên hệ với GReg.

Thứ tự đọc khuyến nghị nếu ít thời gian: **Bài 4 (DWM) → Bài 7 (MeshToken/Uni3C) → Bài 8 (Mesh4D) → Bài 5 (Generative Rendering)**. Bốn bài còn lại chủ yếu cho ý tưởng phụ.

Mục lục:
1. [PartDiffuser (CVPR 2026)](#bài-1--partdiffuser-part-wise-3d-mesh-generation-via-discrete-diffusion-cvpr-2026)
2. [CraftsMan3D (CVPR 2025)](#bài-2--craftsman3d-high-fidelity-mesh-generation-with-3d-native-diffusion-and-interactive-geometry-refiner-cvpr-2025)
3. [Single Mesh Diffusion with Field Latents (CVPR 2024)](#bài-3--single-mesh-diffusion-models-with-field-latents-for-texture-generation-cvpr-2024)
4. [Dexterous World Models (CVPR 2026)](#bài-4--dexterous-world-models-cvpr-2026)
5. [Generative Rendering (CVPR 2024)](#bài-5--generative-rendering-controllable-4d-guided-video-generation-with-2d-diffusion-models-cvpr-2024)
6. [Animating the Uncaptured (ICLR 2026)](#bài-6--animating-the-uncaptured-humanoid-mesh-animation-with-video-diffusion-models-iclr-2026)
7. [MeshToken / Render-Free Motion Control (2026)](#bài-7--towards-3d-aware-video-diffusion-models-render-free-human-motion-control-with-mesh-tokenization-meshtoken-arxiv-260602000-062026)
8. [Mesh4D (CVPR 2026)](#bài-8--mesh4d-4d-mesh-reconstruction-and-tracking-from-monocular-video-cvpr-2026)


---

## Bài 1 — PartDiffuser: Part-wise 3D Mesh Generation via Discrete Diffusion (CVPR 2026)

Quy ước: ✅ = kiểm chứng trực tiếp trong bài (ghi rõ mục/bảng/phương trình). ⚠️ = suy luận, diễn giải hoặc thông tin lấy từ code/bài của khối nền, **không** phải của PartDiffuser.

Nguồn đã đọc: bản CVPR open-access (PDF trong `ref papers/`), bản arXiv HTML v4 (có **Supplementary Material**: Phụ lục A, B, C — bản CVPR không có phần này), code mở của BD3-LM (`kuleshov-group/bd3lms`, `models/dit.py`).

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Yichen Yang*, Hong Li*, Haodong Zhu, Linin Yang, Guojun Lei, Sheng Xu, Baochang Zhang (Beihang; CUC; ZJU) ✅ |
| Venue | CVPR 2026 (open access CVF), arXiv 2511.18801 (v4 có phụ lục) ✅ |
| Bài toán | Point cloud → artist mesh (mesh "kiểu nghệ sĩ", ít mặt, topo sạch) ✅ |
| Link | https://arxiv.org/abs/2511.18801 · https://arxiv.org/html/2511.18801v4 · CVF: `openaccess.thecvf.com/content/CVPR2026/papers/Yang_PartDiffuser_..._CVPR_2026_paper.pdf` |
| Code | **Không có code chính thức.** Đã kiểm tra lại bằng WebSearch (29/10/2026): không có repo GitHub, không có project page; bài không ghi link code. Trang ResearchGate có dòng "Code is available at github.com/SAITPublic/DOCTR" — đó là **nhầm lẫn của ResearchGate** (thuộc bài DOCTR khác), không phải PartDiffuser. |
| Mô hình | DiT ~**0.3B** tham số, discrete (masked) diffusion kiểu BD3-LM, max seq **4096** token, block **1024** token (§4.1) ✅. Số layer / hidden dim / số head: **không công bố** ✅ |
| Compute train | Pre-train **8× H100 80GB × 3 ngày** + fine-tune **4× H100 80GB × 2 tuần** (§4.1) ✅ → ≈ 576 + 1344 ≈ **1.9k H100-giờ** ⚠️ (tự nhân) |
| Inference | **63.8 s/mesh** với T = 1024 bước/block; 17.1 s với T = 256 (Bảng 3) ✅. GPU đo inference không ghi rõ. |
| Dữ liệu | ~**81K** mẫu từ Objaverse + 3D-Front, đã lọc (§4.1, Phụ lục A) ✅ |

---

### 1. TL;DR

- Mesh được cắt thành **N part ngữ nghĩa** (PartField), mỗi part được token hóa bằng **BPT** thành một **block 1024 token** (pad). Các part được xếp theo **BFS trên đồ thị kề part** ✅ (§3.2).
- Sinh **semi-autoregressive**: **tự hồi quy giữa các part** (part i chỉ thấy part < i), **diffusion rời rạc song song bên trong part** (masked diffusion, lấy mẫu kiểu LLaDA: dự đoán tất cả, giữ token tự tin nhất, remask phần còn lại) ✅ (§3.3.4). Đây đúng là **Block Diffusion (BD3-LM)** với "block = part ngữ nghĩa" thay cho block độ dài cố định.
- Điều kiện hình học: point cloud → encoder Michelangelo → **1 đặc trưng global + N đặc trưng part**. Cross-attention trong mỗi DiT block chỉ nhận **C_dyn,i = [C_global, C_part_i]** khi xử lý block i ("part-aware cross-attention") ✅ (§3.3.1–3.3.2).
- Ablation quan trọng nhất cho mình (Bảng 2): **global + part (CD 17.8) > chỉ global (29.1) >> chỉ part (54.7)** ✅. Tức là đặc trưng cục bộ giúp chi tiết, nhưng **bỏ ngữ cảnh global thì sụp mạnh nhất**.
- Bước lấy mẫu là nút cổ chai: mặc định **T = 1024 bước cho block 1024 token = 1 token/bước** (số lần forward bằng AR!). Giảm còn 256 bước thì nhanh 3.7× nhưng CD 17.8 → 44.7 (Bảng 3) ✅.
- Với GReg: **không** lấy phần sinh mesh token. Lấy **ý tưởng điều kiện phân cấp global + vùng** và **cách dựng attention mask theo vùng** cho adapter tay/mặt của H1; ý tưởng "block-AR + diffusion trong block" chỉ đáng cân nhắc theo **trục thời gian** (chunk khung), không theo bộ phận cơ thể.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Đầu vào:** point cloud của vật thể (lấy mẫu từ mesh GT; cùng phân đoạn part với mesh) ✅ (§3.2.1).
**Đầu ra:** chuỗi token BPT → giải mã thành tam giác (mesh kiểu nghệ sĩ) ✅.

**Vì sao khó (theo lập luận của tác giả, §1, §2.1):**
1. AR từng token (MeshGPT, MeshAnything, BPT, TreeMeshGPT…) bị **tích lũy lỗi**: sai sớm làm hỏng phần sau.
2. **Xung đột global vs local:** để giữ topo toàn cục đúng, mô hình AR có xu hướng làm mượt/đơn giản hóa chi tiết tần số cao.
3. Diffusion rời rạc song song hoàn toàn (như TSSR) thì khó giữ tính đúng topo → tác giả chọn đường lai: block-AR (BD3-LM) với block = part.

⚠️ Nhận xét: lập luận "tích lũy lỗi" chỉ được giảm một phần — mỗi part vẫn dài tới 1024 token và mặc định vẫn sinh 1 token/bước, chỉ là **thứ tự theo độ tự tin** thay vì trái→phải. Phần lợi thật có lẽ đến từ (a) attention hai chiều trong part và (b) điều kiện cục bộ C_part_i.

---

### 3. Phương pháp chi tiết

#### 3.1 Phân đoạn part và thứ tự (§3.2.1, Phụ lục A) ✅

- Dùng **PartField** trên mesh; số cụm được chọn ngẫu nhiên trong khoảng [K_min, K_max] (main text). Phụ lục A cho công thức theo số mặt F:
  - `K_min = min(⌊F·0.5/500⌋, 4)`, `K_max = min(⌊F·2.0/500⌋, 4)` ✅ (Phụ lục A, đúng như LaTeX trong HTML).
  - ⚠️ Hệ quả: **số part tối đa = 4**. Khớp với max seq 4096 = 4 × 1024. Với F ≥ 1000 mặt thì K_max = 4; với F ≥ 4000 thì K_min = 4 — nhưng dữ liệu đã lọc F < 4000. Vậy "part" ở đây rất **thô** (2–4 khối lớn), không phải hàng chục bộ phận nhỏ.
- Point cloud dùng cùng phân đoạn với mesh (để có C_part_i) ✅.
- **Thứ tự part:** dựng **ma trận kề giữa các part**, BFS từ một điểm khởi đầu ngẫu nhiên; BFS giữ tính cục bộ (part gần nhau trên đồ thị → gần nhau trong chuỗi) ✅ (§3.2.1).
  - Mâu thuẫn nhỏ: main text nói "chọn ngẫu nhiên một **vertex**"; Phụ lục A nói "chọn ngẫu nhiên **hai part** khác nhau làm gốc BFS" để tăng cường dữ liệu (mỗi mesh → 2 thứ tự) ✅. ⚠️ Có lẽ phụ lục là bản đúng với thực tế cài đặt.
- **Robustness với phân đoạn:** thay PartField bằng SAMPart3D trên Objaverse: CD 19.11 so với 17.81 (×10⁻³) ✅ (Phụ lục C.1). Tác giả nhận xét part thô hơn dễ chạm trần ngân sách token → đơn giản hóa cục bộ.

#### 3.2 Token hóa mesh (§3.2.2, Phụ lục A) ✅

- Dùng **BPT** (Blocked and Patchified Tokenization, Weng et al. CVPR 2025): gom các mặt kề nhau thành **patch** quanh một **đỉnh tâm** + các đỉnh láng giềng; dùng **từ vựng kép** (token đỉnh tâm có loại riêng) nên không cần token ngắt patch ✅.
- Mỗi part được token hóa **riêng**, pad tới đúng 1024, rồi ghép các block lại ✅.
- **Lọc block:** độ dài mã của mỗi part phải nằm trong **[128, 1024]** token; mẫu có part ngoài khoảng bị loại (Phụ lục A). Main text chỉ nhắc ngưỡng trên 1024 ✅.
- Lọc mesh F < 4000 mặt; chia train/val/test **9:1:1** ✅ (Phụ lục A). ⚠️ Tỷ lệ "9:1:1" như ghi trong bài cộng lại 11 phần — có thể là lỗi đánh máy, không rõ.
- ⚠️ BPT gốc (tencent/bpt, code mở) lượng tử hóa tọa độ thành block + offset; bài không nói độ phân giải lượng tử, kích thước từ vựng. Không đoán số.

#### 3.3 Điều kiện phân cấp (§3.3.1) ✅

- Encoder point cloud **Michelangelo** (pre-trained, đóng băng hay không: **không nói**) → `C_pc = [C_global, C_part_1, …, C_part_N]`.
  - `C_global`: "a single global feature vector" — hình dạng tổng thể.
  - `C_part_i`: đặc trưng hình học cục bộ của part i.
- **Không nói** C_part_i được tính thế nào (encode riêng tập điểm của part? pool đặc trưng điểm theo nhãn part?), số token mỗi đặc trưng, chiều. ✅ (đã tìm trong cả main + phụ lục: không có).
  - ⚠️ Hai cách hợp lý: (a) chạy Michelangelo riêng trên điểm của part i → lấy shape token/latent tokens; (b) chạy một lần trên toàn cloud, rồi pool theo nhãn part. Nếu mỗi "feature vector" thật sự chỉ là **1 vector**, thì K/V của cross-attention chỉ dài **2 token** — rất nghèo; khả năng cao mỗi đặc trưng là một chuỗi token (Michelangelo encoder kiểu Perceiver cho ra 1 shape token + nhiều latent token). Không thể xác nhận.

#### 3.4 Part-Aware Diffusion Block (§3.3.2) ✅

DiT block chuẩn + chèn cross-attention giữa self-attention và FFN:

```
Z  = SelfAttn(LN(Z)) + Z                     # có composite mask
Ẑ  = CrossAttn(Q = LN(Z), K = C_dyn, V = C_dyn) + Z
Z_out = FFN(Ẑ) + Ẑ                           # "standard FFN with residual"
```

- **C_dyn** = concat(C_global, C_part_i) với i là part đang xử lý. Khi train song song, **mỗi token block** dùng đặc trưng riêng [C_global, C_part_i] của nó ✅ (§3.3.4).
- ⚠️ Cài đặt tương đương khi train cả chuỗi một lượt: đặt K/V = toàn bộ C_pc (1 + N nhóm token), và dùng **mask cross-attention khối**: query thuộc block i chỉ được nhìn nhóm global + nhóm part i. Đây chính là cái mình có thể mượn (mục 5, 7).
- Điều kiện timestep: bài nói "DiT-based" nên ⚠️ có lẽ adaLN theo t (BD3-LM gốc dùng adaLN từ σ — đã thấy trong `models/dit.py`), nhưng bài không mô tả.

#### 3.5 Self-attention: composite mask (§3.3.2, Phụ lục B.1, Hình 7) ✅

- Chuỗi train dài **2N block**: N block nhiễu `X_t` + N block sạch `X_0` (đúng kỹ thuật BD3-LM: nối x_t và x_0 để học mọi block trong 1 forward) ✅.
- Mask = **giao (AND)** của:
  1. **Block Diffusion Mask** (từ BD3-LM): hai chiều trong block, nhân quả theo block giữa các block.
  2. **Block-Aware Padding Mask**: trong block hiện tại thì nhìn hết **kể cả pad**; giữa các block thì **cấm nhìn pad của block khác** ✅.
- Phương trình tường minh: **không có**, chỉ có hình minh họa N = 3 ✅.
- Block Diffusion Mask trong code BD3-LM (✅ đọc từ `bd3lms/models/dit.py`, **không phải code PartDiffuser**), với n = độ dài chuỗi, vị trí ≥ n là x_0:
  - `M_BD` (block diagonal): `block_q == block_kv` và cùng loại (cùng nhiễu hoặc cùng sạch).
  - `M_OBC` (offset block-causal): query nhiễu ở block b nhìn **block sạch < b**.
  - `M_BC` (block-causal): query sạch ở block b nhìn **block sạch ≤ b**.
  - Mask = M_BD | M_OBC | M_BC. Truyền vào flex-attention (`create_block_mask`) hoặc SDPA (`attn_mask` bool).
- ⚠️ Vì block nhiễu i được nhìn block nhiễu i (hai chiều) và block sạch < i, nên mỗi block nhiễu học đúng `p(X_i | X_i^t, X_<i)` — khớp với phương trình nhân tử hóa (§3.3.3).

#### 3.6 Mục tiêu (§3.3.3) ✅

- Nhân tử hóa: `p(X | C_pc) = ∏_{i=1..N} p(X_i | X_<i, C_pc)`.
- Mỗi part: masked diffusion, dự đoán token sạch:
  `L_i = E_{t, X_i^t ~ q(·|X_i^0)} [ w(t) · (−log p_θ(X_i^0 | X_i^t, X_<i, C_dyn,i)) ]`
  với `w(t)` là trọng số "derived from the noise schedule". Dạng cụ thể của w(t): **không ghi**.
- ⚠️ Theo MDLM/BD3-LM: forward q thay token bằng [MASK] với xác suất 1 − α_t; NELBO có trọng số `α'_t / (1 − α_t)`; với lịch tuyến tính α_t = 1 − t thì trọng số ∝ **1/t**, CE chỉ tính trên vị trí bị mask. Có thể PartDiffuser dùng đúng vậy vì "similar to BD-LM", nhưng không xác nhận được.

#### 3.7 Lịch nhiễu / curriculum (Phụ lục B.1) ✅

- **Pre-train:** lịch nhiễu **bị kẹp (clipped)** `t ~ U[β, ω]` (theo BD3-LM) để "học nhanh đặc trưng hình học cốt lõi"; giá trị β, ω: **không công bố**.
- **Fine-tune:** lịch tuyến tính đầy đủ `U[0, 1]` để khớp toàn bộ quỹ đạo khử nhiễu lúc inference.
- ⚠️ Ý nghĩa: BD3-LM đề xuất kẹp t để giảm phương sai gradient (tránh các t quá nhỏ — gần như không có mask — và quá lớn). Code BD3-LM có tham số tìm độ rộng kẹp `algo.clip_search_widths=[0.5,…,0.9]` (README) — tức khoảng kẹp được tìm theo dữ liệu.

#### 3.8 Lấy mẫu (§3.3.4, §4.5, Phụ lục B.2 Algorithm 1) ✅

```
X_accum ← ∅
for i in 1..N:
    Z ← SamplePrior(B, L_block)            # block toàn [MASK]
    X_ctx ← concat(X_accum, Z)
    C_dyn ← concat(C_global, C_part_i)
    for t = T .. 1:
        P(X0|Xt) ← Model(X_ctx, t, C_pc, block_idx=i)   # cross-attn dùng C_dyn theo block_idx
        Z ← SampleStep(P, X_ctx, t, t-1)                 # LLaDA: dự đoán, giữ token tự tin, remask phần còn lại
        X_ctx ← update_context(X_accum, Z)
    X_accum ← concat(X_accum, Z)
```

- Số part N do điều kiện C_pc quyết định (đã biết trước từ phân đoạn point cloud) ✅ — tức **không** phải sinh số part.
- Trong block: **chiến lược LLaDA**: "iteratively predicting and selectively remasking tokens" ✅. ⚠️ LLaDA mặc định là *low-confidence remasking*: mỗi bước dự đoán mọi vị trí đang mask, chuyển ⌈L/T⌉ vị trí có xác suất argmax cao nhất thành token cố định, phần còn lại giữ [MASK].
- **Mặc định T = 1024 = độ dài block** → ⚠️ đúng 1 token/bước → **4096 forward cho 4 part**, nhiều như AR (nhưng mỗi forward phải xử lý cả block hai chiều). Bài thừa nhận đây là chế độ "high-fidelity".
- Quan sát (Phụ lục C.2, Hình 9) ✅: trong một part, mô hình dựng **"khung xương" thưa** trước, mặt xuất hiện đồng thời ở nhiều vị trí; ở giai đoạn nhiễu, mặt mới có xu hướng **bám theo biên các part đã sinh** → dùng ngữ cảnh part trước để giữ kết nối topo.
- Không đề cập CFG, nhiệt độ, top-p, KV-cache. ⚠️ BD3-LM có KV-cache cho block trước; Algorithm 1 "pass full context" nên có thể không cache.

#### 3.9 Dữ liệu & train (§4.1, Phụ lục A) ✅

- Objaverse + 3D-Front → PartField → BFS (2 thứ tự/mesh) → BPT → lọc block [128, 1024] → ~81K mẫu.
- 2 giai đoạn: pre-train 3 ngày 8×H100, fine-tune 2 tuần 4×H100. LR, batch, số bước, optimizer: **không công bố**.

---

### 4. Kết quả & ablation quan trọng

#### 4.1 So sánh chính (Bảng 1) ✅ — test 300 mẫu ngẫu nhiên mỗi bộ; CD ×10³

| Dataset | Method | CD↓ | HD↓ | EMD↓ | F1↑ |
|---|---|---|---|---|---|
| Objaverse | MeshAnythingV2 | 24.402 | 0.253 | 0.136 | 0.285 |
| | BPT | 86.837 | 0.493 | 0.256 | 0.138 |
| | TreeMeshGPT | 36.938 | 0.306 | 0.157 | 0.279 |
| | **Ours** | **17.813** | **0.238** | **0.115** | **0.343** |
| HSSD | MeshAnythingV2 | 9.197 | 0.170 | 0.077 | 0.383 |
| | BPT | 14.029 | 0.186 | 0.093 | 0.369 |
| | TreeMeshGPT | **4.522** | **0.135** | 0.061 | 0.441 |
| | **Ours** | 5.364 | 0.138 | **0.059** | **0.471** |
| 3DFront | MeshAnythingV2 | 10.406 | 0.169 | 0.076 | 0.420 |
| | BPT | 15.652 | 0.203 | 0.096 | 0.421 |
| | TreeMeshGPT | 7.793 | 0.161 | 0.073 | **0.462** |
| | **Ours** | **6.461** | **0.147** | **0.068** | 0.453 |

- Objaverse: CD tốt hơn MeshAnythingV2 27%, F1 cao hơn ~20% ✅ (§4.3.2). HSSD: TreeMeshGPT thắng CD, HD ✅.
- ⚠️ Lưu ý công bằng: (1) baseline dùng checkpoint công bố, train trên dữ liệu khác; test Objaverse/3DFront có thể được lấy từ chính phân chia 9:1:1 đã lọc (< 4000 mặt, mỗi part 128–1024 token) — tức thuận lợi phân phối cho PartDiffuser; bài không nói rõ test được lọc theo tiêu chí nào. (2) HSSD không có trong train → có thể xem là kiểm tra tổng quát hóa, ở đó lợi thế không rõ. (3) Không có số cho từng seed/độ lệch chuẩn, không có số mặt/tính hợp lệ topo (manifold, tỷ lệ thất bại), không có user study.

#### 4.2 Ablation điều kiện phân cấp (Bảng 2, Objaverse) ✅

| Biến thể | CD↓ | HD↓ | EMD↓ | F1↑ |
|---|---|---|---|---|
| Full (global + part) | **17.813** | **0.238** | **0.115** | **0.343** |
| Chỉ global | 29.125 | 0.297 | 0.144 | 0.281 |
| Chỉ part | 54.728 | 0.356 | 0.205 | 0.239 |

- Cách làm: **không** train lại từ đầu; khởi tạo từ trọng số full, sửa cross-attention nhận điều kiện bị cắt, train tiếp **30%** số bước gốc ✅ (§4.4).
- Diễn giải của tác giả: global quyết định cấu trúc/lắp ráp các part; part quyết định chi tiết. Hình 4: "chỉ part" giữ chi tiết cục bộ nhưng lắp ghép sai ✅.
- ⚠️ Hạn chế: biến thể ablation chỉ được thích nghi 30% bước → có thể thiệt thòi so với train đầy đủ; nhưng khoảng cách (17.8 vs 54.7) đủ lớn để kết luận "global là bắt buộc" khá chắc.

#### 4.3 Số bước lấy mẫu (Bảng 3, Objaverse) ✅

| k | T/block | Thời gian (s/mesh) | CD↓ | HD↓ |
|---|---|---|---|---|
| 1 (mặc định) | 1024 | 63.8 | 17.813 | 0.238 |
| 2 | 512 | 33.7 | 27.657 | 0.299 |
| 4 | 256 | 17.1 | 44.720 | 0.321 |

- Ít bước → phải mở nhiều token độ tin thấp cùng lúc → bề mặt vỡ, patch bất thường; cấu trúc toàn cục vẫn ổn (Hình 5) ✅.
- ⚠️ Tức là lợi thế "song song" của discrete diffusion **chưa** được hiện thực hóa: ở chế độ nhanh, chất lượng còn kém hơn cả MeshAnythingV2 (24.4).

#### 4.4 Thiếu gì trong ablation ⚠️
- Không ablate **block = part** so với **block độ dài cố định** (BD3-LM thuần) — đây là câu hỏi cốt lõi để chứng minh "part" có ích, nhưng không có.
- Không ablate thứ tự part (BFS vs ngẫu nhiên), số part, bỏ composite padding mask, bỏ curriculum clipped→full.
- Không so với AR cùng kiến trúc/cùng dữ liệu (để tách lợi ích của diffusion khỏi lợi ích dữ liệu/điều kiện).

---

### 5. Code

**Không có code chính thức.** Các đoạn dưới đây là ⚠️ **pseudo-code tự viết** theo mô tả của bài + code mở của BD3-LM (`kuleshov-group/bd3lms`, hàm `block_diff_mask` trong `models/dit.py` — đã đọc) và chiến lược remask của LLaDA. **Không phải code PartDiffuser.**

#### 5.1 Composite self-attention mask (BD3-LM mask ∧ padding mask)

```python
import torch

def composite_self_mask(part_lens, L_blk):
    """part_lens: list[int] độ dài thật (không pad) của N part; mỗi block pad tới L_blk.
    Chuỗi train = [x_t (N*L_blk) | x_0 (N*L_blk)]. Trả về bool [2n, 2n], True = được nhìn."""
    N = len(part_lens); n = N * L_blk
    idx = torch.arange(2 * n)
    is_x0 = idx >= n
    pos = torch.where(is_x0, idx - n, idx)
    blk = pos // L_blk
    is_pad = (pos % L_blk) >= torch.tensor(part_lens)[blk]

    q_x0, k_x0 = is_x0[:, None], is_x0[None, :]
    q_b,  k_b  = blk[:, None],   blk[None, :]
    # --- Block Diffusion Mask (đúng logic của bd3lms.block_diff_mask) ---
    m_bd  = (q_b == k_b) & (q_x0 == k_x0)                 # 2 chiều trong block (cùng nhiễu/cùng sạch)
    m_obc = (q_b > k_b) & k_x0 & ~q_x0                    # block nhiễu i  -> block sạch < i
    m_bc  = (q_b >= k_b) & k_x0 & q_x0                    # block sạch i   -> block sạch <= i
    m_block = m_bd | m_obc | m_bc
    # --- Block-Aware Padding Mask (theo mô tả Phụ lục B.1) ---
    same_blk = m_bd                                       # "trong block hiện tại": nhìn cả pad
    m_pad = same_blk | ~is_pad[None, :]                   # khác block: cấm nhìn pad
    return m_block & m_pad
```

#### 5.2 Part-aware cross-attention (chọn [C_global, C_part_i] theo block)

```python
def part_cross_mask(N, L_blk, n_glob, n_part, two_halves=True):
    """K/V = concat(C_global [n_glob tok], C_part_1..N [n_part tok mỗi part]).
    Query thuộc block i chỉ được nhìn global + part i."""
    q_blk = torch.arange(N * L_blk) // L_blk
    if two_halves: q_blk = torch.cat([q_blk, q_blk])     # block nhiễu i và block sạch i cùng dùng part i
    kv_owner = torch.cat([torch.full((n_glob,), -1),
                          torch.arange(N).repeat_interleave(n_part)])
    return (kv_owner[None, :] == -1) | (kv_owner[None, :] == q_blk[:, None])

class PartAwareBlock(nn.Module):
    def forward(self, z, c_pc, self_mask, cross_mask, t_emb):
        z = z + self.sa(modulate(self.ln1(z), t_emb), attn_mask=self_mask)
        z = z + self.ca(self.ln2(z), c_pc, c_pc, attn_mask=cross_mask)   # Q = token mesh, K/V = C_pc
        return z + self.ffn(modulate(self.ln3(z), t_emb))
```

Lúc inference chỉ có block i đang khử nhiễu nên chỉ cần truyền thẳng `C_dyn = cat(C_global, C_part_i)` cho các query của block đó (Algorithm 1 dòng 6, 10).

#### 5.3 Vòng lấy mẫu semi-AR + LLaDA low-confidence remasking

```python
@torch.no_grad()
def sample(model, c_glob, c_parts, L_blk=1024, T=1024, MASK=MASK_ID):
    ctx = torch.empty(1, 0, dtype=torch.long)
    for i, c_part in enumerate(c_parts):                    # N part theo thứ tự BFS
        z = torch.full((1, L_blk), MASK)
        c_dyn = torch.cat([c_glob, c_part], dim=1)
        n_per_step = split_evenly(L_blk, T)                 # T=L_blk -> 1 token/bước
        for s in range(T):
            logits = model(torch.cat([ctx, z], 1), t=1 - s / T, c_dyn=c_dyn)[:, -L_blk:]
            prob, pred = logits.softmax(-1).max(-1)
            masked = z == MASK
            conf = torch.where(masked, prob, torch.full_like(prob, -1.0))
            k = n_per_step[s]
            sel = conf.topk(k, dim=-1).indices                 # giữ k token tự tin nhất
            z.scatter_(1, sel, pred.gather(1, sel))            # phần còn lại vẫn [MASK] (remask)
        ctx = torch.cat([ctx, z], 1)                         # part i thành ngữ cảnh sạch cho part i+1
    return bpt_decode(strip_pad(ctx))
```

#### 5.4 Code mở của các khối nền (không phải code chính thức)
- **BD3-LM** (`github.com/kuleshov-group/bd3lms`): mask khối (`block_diff_mask`, flex-attention hoặc SDPA), train 2n token (x_t ‖ x_0), cắt output về n token đầu; sampling có `kv_cache`, `nucleus_p`; tham số tìm khoảng kẹp nhiễu `clip_search_widths`. ✅ (đọc `models/dit.py` + README)
- **LLaDA** (`ML-GSAI/LLaDA`): sampling semi-AR theo block với low-confidence remasking. ⚠️ (theo hiểu biết chung, không đọc lại trong phiên này)
- **BPT** (`whaohan/bpt` / Tencent), **PartField** (NVIDIA), **Michelangelo** (`NeuralCarver/Michelangelo`): đều có code mở. ⚠️ chưa đọc chi tiết; không cần cho GReg.

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

**Tác giả tự nêu (§5.1)** ✅: dung lượng mô hình & cửa sổ ngữ cảnh giới hạn quy mô/độ phức tạp tài sản; cần nghiên cứu thêm lấy mẫu song song và quan hệ số bước–chất lượng; phụ thuộc phân đoạn thượng nguồn, hướng tới train end-to-end và đầu vào đa phương thức.

**Mình thấy thêm** ⚠️:
1. **Quy mô rất nhỏ:** tối đa 4 part × 1024 token, mesh < 4000 mặt. "Part" thực chất là 2–4 khối lớn.
2. **Không thật sự nhanh:** mặc định 1 token/bước; chế độ nhanh thua baseline AR.
3. **Thiếu ablation cốt lõi** (block-theo-part vs block cố định; thứ tự BFS; padding mask). Khó biết "part" đóng góp bao nhiêu so với "chỉ là BD3-LM + cross-attn Michelangelo".
4. **Mô tả mơ hồ:** cách lấy C_part_i, số token điều kiện, w(t), β/ω, siêu tham số train đều không có → **khó tái lập**, không có code.
5. **Mâu thuẫn nhỏ main text vs phụ lục** (BFS từ vertex vs từ 2 part; ngưỡng lọc dưới 128; tỷ lệ split 9:1:1).
6. **Cần phân đoạn point cloud lúc inference** (C_part_i lấy từ point cloud đã phân đoạn) → mô hình nhận sẵn "bản đồ part" — một dạng điều kiện mạnh mà baseline không có.
7. Đánh giá chỉ bằng CD/HD/EMD/F1 — không đo chất lượng topo kiểu nghệ sĩ (số mặt, manifold, tam giác suy biến), vốn là mục tiêu chính của dòng bài này.

**Câu hỏi mở:** block-theo-part có tốt hơn block cố định không? Lợi ích đến từ cross-attn cục bộ hay từ thứ tự? Có cần AR giữa part không, hay chỉ cần cross-attn theo part + sinh song song toàn bộ?

---

### 7. Liên hệ với GReg

**Nói thẳng:** sinh mesh token **không phải bài toán của mình**. SMPL-X có topo cố định; mesh đích đã có sẵn từ fitting. Mình sinh **ảnh/video**, điều kiện bằng mesh. Giá trị của bài nằm ở **cách tổ chức điều kiện theo vùng** và **cách viết mask**, không ở mô hình.

#### 7.1 Nên lấy (theo thứ tự ưu tiên)

| # | Ý tưởng từ PartDiffuser | Áp vào GReg (H1/H2) | Độ khó |
|---|---|---|---|
| 1 | **Điều kiện phân cấp global + part trong cross-attn**, có ablation "global only / part only" ✅ Bảng 2 | Adapter tay/mặt của H1: K/V của token thuộc vùng tay = **[token global + token crop tay]**, không chỉ crop tay. Global có thể là: token ảnh tham chiếu I_s (hoặc normal map toàn khung, downsample) ⚠️. Bài cho bằng chứng định lượng rằng **chỉ có part** thì hỏng nặng nhất → biện minh trực tiếp cho thiết kế này và cho một **ablation đối xứng** trong luận văn: {global, hand-only, global+hand}. | Thấp–trung bình |
| 2 | **Mask cross-attention theo chủ sở hữu** (query của block i chỉ thấy nhóm K/V global + nhóm part i) | Khác biệt quan trọng: ở PartDiffuser mỗi block = một part nên mask theo **block**; ở mình token latent nằm trên lưới không gian → mask theo **từng token**, dựng từ mặt nạ B1 (`parts/*.png`) downsample về lưới latent (8× không gian, 4× thời gian của Wan) ⚠️. Gợi ý: nhãn token = {body, tay trái, tay phải, mặt} theo đa số pixel; **dilate** mặt nạ tay 1–2 token để query sát biên vẫn thấy crop tay; query "body" chỉ thấy nhóm global (hoặc thêm "sink token" kiểu SignRefine để không bị ép attend). Code: dùng lại logic `part_cross_mask` (mục 5.2), đổi `q_blk` từ chỉ số block sang nhãn vùng của từng token. | Trung bình |
| 3 | **Block-Aware Padding Mask** (pad trong block thì nhìn được, pad ở block khác thì cấm) | Khi crop tay có số token biến thiên (tay khuất/ra ngoài khung → không có crop), pad các nhóm K/V về cùng kích thước và **cấm attend vào pad** của nhóm không tồn tại. Đơn giản nhưng hay quên. | Thấp |
| 4 | **Block diffusion theo thời gian** (AR giữa các block, diffusion song song trong block; train một lượt bằng x_t ‖ x_0 + mask BD3-LM) | ⚠️ Nếu cần video dài hơn cửa sổ Wan (81 khung), có thể sinh **theo chunk khung**: chunk k điều kiện vào chunk sạch < k. Đây là cách áp dụng "semi-AR" hợp lý hơn cho mình (trục thời gian), giải đúng điểm yếu "không có prior thời gian" mà không cần cả clip trong một cửa sổ. Nhưng với Wan đóng băng (bidirectional, flow matching) thì phải đổi mask attention + train lại nhiều → **rủi ro cao**, chỉ nên ghi là hướng mở. Clip HamNoSys ngắn nên có lẽ không cần. | Cao |
| 5 | **Curriculum lịch nhiễu: kẹp t trước, đầy đủ sau** | ⚠️ Với H2 (refiner một bước) hoặc H1 kiểu SDEdit: đầu vào warped đã gần đúng, nên có thể train/khởi động với t trong khoảng giữa (không cần t ≈ 1). Chỉ là gợi ý; bài không cho số β, ω. | Thấp |
| 6 | **Bằng chứng "ít bước → hỏng chi tiết, cấu trúc vẫn giữ"** (Bảng 3, Hình 5) | Cảnh báo cho H2/distillation: số bước thấp thường giữ được bố cục thân người nhưng hỏng **ngón tay**. Phải đo metric **vùng tay** (B1 mask, MPJPE tay) khi giảm bước, không chỉ metric toàn khung. | Thấp (chỉ là giao thức đánh giá) |
| 7 | Quan sát "sinh khung xương tự tin trước, bám biên vùng đã có" (Phụ lục C.2) | ⚠️ Song song khái niệm: pixel nhìn thấy (V=1) ≈ token đã unmask/tự tin; vùng khuất (V=0) ≈ token [MASK] cần sinh, nên bám biên vùng đã biết. Củng cố cách đặt bài toán H1 như **inpainting có điều kiện** (VACE mask) thay vì sinh lại toàn khung. Chỉ dùng như lập luận trực giác trong luận văn. | — |

#### 7.2 Không nên lấy (và lý do)

- **Token hóa mesh (BPT), discrete diffusion trên token, Michelangelo, PartField:** không liên quan — mình không sinh mesh; phân đoạn của mình đã **chính xác theo định nghĩa** (SMPL-X part segmentation), không cần mô hình phân đoạn học.
- **AR giữa các bộ phận cơ thể (thân → tay → mặt):** ⚠️ không hợp với ảnh 2D: vùng tay che thân, ranh giới pixel phụ thuộc che khuất, sinh tuần tự theo vùng làm tăng số lần chạy và dễ tạo đường nối. Nếu muốn "tay sinh sau, điều kiện vào thân", phương án thực tế hơn là **một pass tinh chỉnh tay** (crop tay phóng to → diffusion inpaint → dán lại), tức 2 "block" thô, chứ không phải khung semi-AR kiểu PartDiffuser.
- **Thứ tự BFS / chọn số part ngẫu nhiên:** không có ý nghĩa với số vùng cố định (body/2 tay/mặt).
- **Số liệu của bài:** không dùng để biện minh cho hiệu quả trong video/ảnh; chỉ trích như một bằng chứng (ở miền 3D) rằng điều kiện global + cục bộ tốt hơn từng phần riêng lẻ.

#### 7.3 Cách trích trong luận văn ⚠️
- Phần related work về "part-aware conditioning": PartDiffuser (CVPR 2026) + SignRefine (masked cross-attn vùng tay/mặt) là hai tiền lệ; PartDiffuser cho ablation rõ ràng nhất về vai trò của ngữ cảnh global trong điều kiện cục bộ (Bảng 2).
- Ablation đề xuất bổ sung cho bảng 1.9 của `docs/diffusion_improvement.md`: biến thể 5a "adapter chỉ crop tay" vs 5b "adapter global + crop tay" (mask theo token từ B1), đo L1/LPIPS vùng tay + MPJPE tay.

---

## Bài 2 — CraftsMan3D: High-fidelity Mesh Generation with 3D Native Diffusion and Interactive Geometry Refiner (CVPR 2025)

Quy ước: ✅ = đã kiểm chứng trong paper (ghi mục/bảng/phương trình) hoặc trong code (ghi `file:dòng`, đường dẫn tương đối từ gốc repo `CraftsMan3D/`). ⚠️ = suy luận/đánh giá của mình, chưa kiểm chứng.

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Weiyu Li\*, Jiarui Liu\*, Hongyu Yan\*, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, Xiaoxiao Long (HKUST, LightIllusions, Adobe Research) ✅ |
| Venue | CVPR 2025, trang 5307–5317 (bản Open Access) ✅. Bản arXiv đầu tiên: 2405.14979 (5/2024, tên "CraftsMan") ✅ README |
| Project / demo | craftsman3d.github.io, HF Space `wyysf/CraftsMan` ✅ README |
| Code | github.com/HKUST-SAIL/CraftsMan3D (trước là wyysf-98/CraftsMan). Bản clone chỉ có 1 commit `f87d6a5` (2025-06-26) ✅ |
| Trọng số | `craftsman3d/craftsman` (v1-5, VAE kiểu Michelangelo + DiT, điều kiện 1 ảnh) và `craftsman3d/craftsman-doravae` (DoraVAE + rectified flow, README ghi "not the official version") ✅ README |
| **Phần ĐÃ phát hành** | VAE + DiT điều kiện **một ảnh**, code train (có lỗi, xem 5.4), pipeline suy luận, script watertight + sharp-edge sampling ✅ |
| **Phần CHƯA phát hành** | (1) Mô hình điều kiện **đa góc nhìn (MV)** – chính là mô hình trong paper (ToDo chưa tick, config `clip-mvrgb-modln-...yaml` mà README nhắc tới **không tồn tại**); (2) **toàn bộ giai đoạn 2 – bộ refiner dựa trên normal** (ControlNet-Tile, cross-view attention, continuous remeshing, Poisson blending, relative Laplacian, "magic brush"): grep toàn repo không có dòng nào ✅ |
| `hy3dshape`? | **Không có** thư mục này. Thứ duy nhất lấy từ Hunyuan3D-2 là `craftsman/models/autoencoders/volume_decoders.py` (giải mã volume phân cấp kiểu FlashVDM), mang header *Tencent Hunyuan Non-Commercial License* ✅ `volume_decoders.py:1-13`. README có ToDo "support training Hunyuan3D-2 / TripoSG" (chưa làm) ✅ |
| License | **Mâu thuẫn**: README ghi MIT; `gradio_app.py` ghi AGPL-3.0; `volume_decoders.py` mang license Hunyuan phi thương mại; repo **không có** file LICENSE ✅. ⚠️ Không nên chép nguyên code vào GReg; chỉ nên tái hiện ý tưởng. |
| Compute train | 32×A800, batch 32/GPU, 7 ngày (README) ✅; paper chỉ ghi 32 A800 (§4.1) ✅. Config phát hành dùng `batch_size: 40` ✅ |
| Compute infer | Coarse ~5 s, refine ~20 s (Hình 1, Bảng 1, Bảng 3) ✅; refine chạy trên "GTX 3080" (README) ✅ |
| Kích thước | VAE 185M (paper §4.1) / 184M (header config); DiT 500M (paper) / 547M (header config); bộ mã hóa CLIP-L + DINOv2-B đóng băng 514M; trainable 548M ✅ `configs/image-to-shape-diffusion/clip-dinov2-pixart-diffusion-dit32.yaml:1-12` |
| Supplementary | Paper liên tục dẫn "please refer to the supplementary" (chi tiết dữ liệu, setting train, kết quả CLAY) nhưng **PDF không kèm** supplementary ✅ |

---

### 1. TL;DR

- Sinh mesh 3D từ một ảnh/text theo hai bước "thợ thủ công": (1) DiT 3D-native khử nhiễu trên **latent set** (3DShape2VecSet/Michelangelo) cho ra **mesh thô** trong ~5 s; (2) **refiner dựa trên normal map**: render normal của mesh thô → ControlNet-Tile (đã fine-tune trên normal) "vẽ thêm chi tiết" → dùng normal đã tăng cường làm giám sát để tối ưu đỉnh mesh (~20 s), tự động hoặc tương tác bằng cọ vẽ mặt nạ.
- Mẹo then chốt của bước 2: **Poisson Normal Blending** loại bỏ thành phần tần số thấp mà diffusion lén đưa vào (gây méo toàn cục), và **Relative Laplacian** phạt thay đổi tọa độ Laplace so với mesh ban đầu (thay vì ép về 0, gây co mesh).
- DiT điều kiện bằng **ảnh đa góc nhìn** (sinh bởi MV-diffusion khi suy luận), đặc trưng CLIP + DINOv2, camera đưa vào qua **ModLN** trong bộ mã hóa ảnh; ablation cho thấy MV + camera giúp CD 0.0317 → 0.0188 (Bảng 4).
- Code phát hành chỉ là bước 1 bản một ảnh (+ bản DoraVAE rectified flow); refiner và bản MV **không có**. Liên hệ với GReg chủ yếu ở **ý tưởng**: refine bằng Tile-diffusion có ràng buộc tần số thấp, chia sẻ K/V giữa các khung không cần train, zero-init cho nhánh mới, winding number cho kiểm tra trong/ngoài.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

- **Đầu vào:** một ảnh tham chiếu hoặc prompt text. Khi suy luận, ảnh/text được đưa qua một MV-diffusion có sẵn (MVDream cho text; Era3D/Wonder3D/CRM cho ảnh) để có ảnh đa góc nhìn ✅ §3.2.
- **Đầu ra:** mesh tam giác (không texture), topology đều (do marching cubes trên trường TSDF), sau đó có thể refine chi tiết bề mặt ✅ §3, Hình 4.
- **Vì sao khó** (theo paper §1, §2):
  1. SDS chậm (hàng chục phút đến giờ), không ổn định, lỗi đa mặt (Janus) ✅.
  2. MV + tái tạo (LRM) nhanh nhưng phụ thuộc chất lượng ảnh 2D; tự che khuất, ánh sáng phức tạp, bất nhất giữa các view → artefact ✅ (Hình 2: InstantMesh hỏng ở vùng bị che).
  3. 3D-native bị giới hạn bởi dữ liệu 3D: Objaverse chủ yếu là mesh không kín (open, double-sided, non-manifold, point cloud quét) → <1% dùng trực tiếp được để lấy SDF ✅ §3.1.
  4. Các mô hình 3D-native trước (Michelangelo, CLAY, Direct3D) bề mặt quá mượt, thiếu chi tiết, train bất ổn ✅ §2.2.
  5. Không hỗ trợ người dùng chỉnh sửa tương tác ✅ §1.

---

### 3. Phương pháp chi tiết

#### 3.1 Tiền xử lý dữ liệu: watertight bằng visibility + winding number (§3.1)

- **Paper:** đặt 50 camera trên mặt cầu, lưới 256 hoặc 512. Ô lưới "không nhìn thấy bởi tất cả camera" thì kiểm tra thêm **winding number**; nếu > 0.75 thì coi là bên trong ✅ §3.1. Tỉ lệ chuyển watertight thành công trên Objaverse tăng **60% → 80%** ✅. Thời gian (Hình 3): Mesh2SDF 178 s, visibility check 3 s, visibility + winding number 6 s ✅.
- **Code thực tế** (`watertight_and_sampling.py:234-341`) ✅:
  1. Camera trực giao, phân bố Fibonacci trên mặt cầu bán kính 4 (`:15-29`, `:270`), render depth 1024² bằng nvdiffrast (`:303-307`).
  2. `visibility_check` (`:191-231`): điểm lưới **nhìn thấy từ ít nhất một view** nếu depth của nó < depth render (không có epsilon); đồng thời lưu `dist` = min |Δdepth| dọc tia.
  3. Điểm **đang được coi là nhìn thấy** mà winding number > ngưỡng → lật thành "bên trong" (`:311-317`). Ngưỡng mặc định **0.5**, không phải 0.75 như paper.
  4. Gần bề mặt (`dist < 1`): thay bằng khoảng cách chính xác điểm–mesh qua `igl.point_mesh_squared_distance` (`:320-326`); điểm không nhìn thấy → dấu âm (`:329`).
  5. Marching cubes ở **mức offset 1 voxel** (`sample_size/grid_resolution` ≈ 2.1/256), không phải mức 0 (`:333-334`) → mặt mỏng/hở trở thành vỏ kín hơi dày.
- Sau watertight: lấy 100k điểm bề mặt + normal, 2×100k điểm gần bề mặt (nhiễu Gauss σ = 0.001 và 0.01), 100k điểm thể tích đều trong [−1.05, 1.05]³; SDF bằng `pysdf` (`:571-587`). Có tùy chọn **sharp-edge sampling** của Dora (Blender `edges_select_sharp`, ngưỡng góc 15° ở CLI) (`:344-527`, `:540`).

#### 3.2 Shape VAE: latent set Perceiver (§3.2, Hình 5a)

- Theo Michelangelo [64] bỏ lớp contrastive; mã hóa point cloud + normal thành **tập latent** Z; giải mã thành trường **TSDF** (thay occupancy) với ngưỡng 1/256 "để tối ưu ổn định" ✅ §3.2.
- **Kiến trúc trong code** (`craftsman/models/autoencoders/michelangelo_autoencoder.py`) ✅:
  - Embed điểm: Fourier 8 tần số log-space (không nhân π) → 3×17 = 51 chiều, nối với normal 3 chiều → `Linear` lên width 768 (`:22-67`, `:369-373`, `:412-417`).
  - **Query của encoder = chính các điểm bề mặt được FPS** (`use_downsample: true`): FPS từ N điểm còn `num_latents` điểm, lấy embedding của chúng làm query (`:437-451`). Một cross-attention (query ← toàn bộ điểm) rồi 8 lớp self-attention, LayerNorm cuối (`:456-460`).
  - Bottleneck KL: `pre_kl` Linear 768 → 2×64 cho từng token; `post_kl` 64 → 768 (`:582-586`). Latent cho diffusion: **768 token × 64 chiều** (config diffusion).
  - Decoder: 16 lớp self-attention trên latent (`:590-599`, `:671-682`), rồi **một** cross-attention từ điểm truy vấn (Fourier → Linear) vào latent → LN → Linear ra 1 giá trị (`:478-518`).
- **Loss VAE** (`craftsman/systems/shape_autoencoder.py:30-78`, `:92-96`) ✅: MSE trên TSDF (cắt ±τ rồi chia τ, nên giá trị nằm trong [−1, 1]; `craftsman/data/base.py:158-159`) + λ_KL·KL với λ_KL = 0.001. KL lấy **trung bình** theo (token, chiều) (`michelangelo_autoencoder.py:311-318`) chứ không phải tổng → regularization rất yếu. ⚠️ Gần như một AE tất định có chút nhiễu.

#### 3.3 DiT khử nhiễu điều kiện đa góc nhìn (§3.2, Hình 5b)

- Điều kiện (Eq. 1) ✅:
  `c = φ_clip(ŷ, ModLN(π)) + φ_mlp(φ_dino-v2(ŷ, ModLN(π)))`
  với ŷ là ảnh MV, π là tham số camera, ModLN là adaptive LayerNorm (FiLM) điều biến bộ mã hóa ảnh theo camera (theo Instant3D [20]); φ_mlp "một MLP nhỏ" để căn đặc trưng DINO với CLIP.
- Loss (Eq. 2) ✅: ε-prediction chuẩn `E‖ε − ε_θ(S_t, t, c)‖²`.
- Backbone: DiT [41] với **adaLN-single** của PixArt-α [4] để giảm tham số ✅ §3.2.
- **Train bằng ảnh MV ground-truth render**, suy luận bằng ảnh MV do MV-diffusion sinh ra ✅ §4.1, §4.2. ⚠️ Có khoảng cách train/test (ảnh MV sinh ra không nhất quán bằng ảnh render) – paper không phân tích.
- Chi tiết kiến trúc trong code ở 5.2.

#### 3.4 Refiner hình học dựa trên normal (§3.3, Hình 6) – **chỉ có trong paper**

1. **Sinh normal hướng dẫn trung gian:** render normal của mesh thô từ nhiều view → **ControlNet-Tile** [62] đã fine-tune trên dữ liệu normal (Objaverse + HumanNorm [15]) để "tăng cường" normal với chi tiết, có prompt text ("a dragon with vivid details, normal map") và **mặt nạ tùy chọn** ✅ §3.3, Hình 6a.
2. **Cross-view attention không cần huấn luyện** (Eq. 3) ✅: trong self-attention của mọi view, thay K, V bằng
   `K = W^K[z^0, …, z^K], V = W^V[z^0, …, z^K]`
   tức K/V được nối từ tất cả các view và chia sẻ chung. Tác giả giải thích nó chạy được không cần train là nhờ (a) ràng buộc mạnh từ normal thô và (b) Tile-ControlNet chỉ "ảo giác" thêm chi tiết mà không thay đổi nhiều đầu vào ✅.
3. **Tối ưu mesh trực tiếp** bằng continuous remeshing [39] (Palfinger), loss (Eq. 4) ✅:
   `L_remeshing = Σ_i ‖n̂_i − R_n(V, F, π_i)‖₁`
4. **Poisson Normal Blending** (Eq. 5) ✅: `n_fused = Γ(n̂, R_n(V,F,π), m)`. Lý do (Hình 7): normal do SD sinh chứa **thay đổi tần số thấp** so với normal gốc → mesh bị kéo giãn/méo toàn cục. Poisson blending chỉ giữ gradient (chi tiết cục bộ) của normal tăng cường, còn điều kiện biên/tần số thấp lấy từ normal render; m là mặt nạ render từ mesh thô.
5. **Relative Laplacian Smoothing** (Eq. 6) ✅: tính tọa độ Laplace ban đầu `V^W_init = W_init·V_init`; mỗi bước cập nhật `x ← x + λ·v(W·V − V^W_init)`. Không ép tọa độ Laplace về 0 (gây co mesh) mà phạt *thay đổi* so với mesh thô.
6. **Tương tác ("magic brush"):** người dùng tô vùng cần sửa → mặt nạ nhị phân → chỉ vùng đó được cập nhật ✅ §3.3.

#### 3.5 Lịch train, suy luận, dữ liệu

- Dữ liệu: Objaverse (+ Objaverse-XL) đã lọc (bỏ point cloud, cấu trúc mảnh, mesh thủng, mesh không texture) rồi watertight ✅ §3.1. README phát hành danh sách 190k và 320k UUID, cùng mẫu 100k cho diffusion ✅.
- Train diffusion: 32 A800, ảnh MV ground-truth ✅ §4.1. Không có số bước/epoch trong paper (đẩy sang supplementary).
- Suy luận: MV-diffusion → DiT (DDIM) → TSDF → marching cubes → (tùy chọn) refine 20 s ✅.

---

### 4. Kết quả & ablation quan trọng

**Bảng 1 – GSO, 30 shape ngẫu nhiên, căn ICP** ✅

| Nhóm | Phương pháp | CD ↓ | IoU ↑ | Thời gian |
|---|---|---|---|---|
| Tái tạo | One-2-3-45 | 0.0629 | 0.4086 | ~45 s |
| Tái tạo | Zero123 | 0.0339 | 0.5035 | ~10 phút |
| Tái tạo | **InstantMesh** | **0.0187** | **0.6353** | ~10 s |
| SDS | RealFusion | 0.0819 | 0.2741 | ~90 phút |
| SDS | Magic123 | 0.0516 | 0.4528 | ~60 phút |
| 3D sinh | Point-E | 0.0426 | 0.2875 | ~40 s |
| 3D sinh | Shap-E | 0.0436 | 0.3584 | ~10 s |
| 3D sinh | Michelangelo | 0.0404 | 0.4002 | ~3 s |
| 3D sinh | One-2-3-45++ | 0.0437 | 0.3386 | ~20 s |
| 3D sinh | **CraftsMan3D** | 0.0291 | 0.5347 | ~5 s |

⚠️ Nhận xét thẳng: trên GSO, CraftsMan3D **thua rõ** InstantMesh (CD 0.0291 vs 0.0187, IoU 0.5347 vs 0.6353). Paper gọi là "comparable" và lập luận GSO quá đơn điệu.

**Bảng 2 – tập con Objaverse có tự che khuất** ✅: InstantMesh CD 0.04909 / IoU 0.50151; CraftsMan3D **0.03943 / 0.53215**. ⚠️ Không nói rõ kích thước tập con và cách chọn.

**Bảng 3 – refine mesh, 20 object Objaverse, mesh vào bị giảm còn 500 mặt** ✅: CLIP similarity Fantasia3D 0.2567 (~15 phút), Latent-NeRF 0.2725 (~1 giờ), **Ours 0.2821 (~20 s)**. ⚠️ CLIP similarity là thước đo yếu cho chất lượng hình học.

**Bảng 4 – ablation DiT điều kiện MV** (tập con tự chọn) ✅

| Biến thể | CD ↓ | IoU ↑ |
|---|---|---|
| Không điều kiện MV (một ảnh) | 0.0317 | 0.5892 |
| Không inject camera | 0.0249 | 0.6561 |
| Cost volume (kiểu CLAY) | 0.0223 | 0.6583 |
| **Đầy đủ (MV + ModLN camera)** | **0.0188** | **0.7059** |

→ Đa góc nhìn đóng góp nhiều nhất; camera qua ModLN tốt hơn cost volume (cost volume cần pose chính xác để back-project) ✅ §4.4.

**Hình 10 – ablation refiner** (chỉ định tính) ✅: không PF + không Re-Laplace → "đầu nhọn kỳ quặc" do bias toàn cục của normal diffusion; thêm PF → hết bias nhưng mesh co lại; thêm Re-Laplace → hình hợp lý.

⚠️ Thiếu: không có ablation định lượng cho Poisson blending/Relative Laplacian, cross-view attention training-free, CLIP+DINO so với chỉ CLIP, TSDF so với occupancy, hay ngưỡng winding number.

---

### 5. Đọc code

#### 5.1 Bản đồ repo (~12.5k dòng Python; phần lớn là bản sao HF CLIP/DINOv2)

```
CraftsMan3D/
├── train.py                 # Lightning trainer (theo cấu trúc threestudio)
├── inference.py, gradio_app.py
├── train_autoencoder.sh     # gọi launch.py + config KHÔNG tồn tại (hỏng)
├── train_diffusion.sh
├── watertight_and_sampling.py   # watertight (visibility + winding number) + sharp-edge sampling (Dora)
├── configs/
│   ├── shape-autoencoder/michelangelo-l768-e64-ne8-nd16.yaml
│   └── image-to-shape-diffusion/{clip-dinov2-pixart-diffusion-dit32[-finetune|-mv].yaml,
│                                 DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32.yaml}
└── craftsman/
    ├── pipeline.py          # CraftsManPipeline: rembg, căn ảnh, sample, giải mã mesh
    ├── data/{base.py, Objaverse.py}
    ├── systems/{shape_autoencoder.py, shape_diffusion.py (DDPM/DDIM), shape_rectified_flow.py, utils.py}
    ├── models/autoencoders/{michelangelo_autoencoder.py, dora_autoencoder.py, volume_decoders.py (Hunyuan/FlashVDM)}
    ├── models/denoisers/{pixart_denoiser.py, utils.py}      # DiT adaLN-single
    ├── models/conditional_encoders/{base.py, clip_dinov2_encoder.py, dinov2_encoder.py,
    │                                clip/, dinov2/, dinov2_with_registers/}  # bản HF + ModLN
    ├── models/transformers/{attention.py, perceiver_1d.py, utils.py}
    └── utils/               # config, scheduler, checkpoint, saving (threestudio)
```

#### 5.2 Đi qua các file cốt lõi

**a) Hệ diffusion DDPM – `craftsman/systems/shape_diffusion.py`** ✅
- VAE đóng băng (`:69-71`); latent = `kl_embed * z_scale_factor`, mặc định **1.0**, tức không chuẩn hóa latent (`:83-88`).
- Timestep đều trong [0, 1000) (`:111-117`), `add_noise` của diffusers, ε-pred hoặc v-pred tùy scheduler (`:126-131`).
- **Min-SNR-γ weighting, γ = 5 mặc định** (`:42`, `:139-159`) – paper không nhắc. Config không ghi đè `snr_gamma`, nên mọi lần train đều dùng Min-SNR.
- Sampling: DDIM 50 bước, CFG 7.5, nhánh uncond là embedding rỗng (`:219-284`, `systems/utils.py:144-219`).

**b) Hệ rectified flow – `craftsman/systems/shape_rectified_flow.py`** (dùng cho bản DoraVAE) ✅
- Lấy mẫu t theo **logit-normal** (mean 0, std 1) như SD3 (`:47-50`, `:129-137`, `systems/utils.py:21-40`); scheduler `FlowMatchEulerDiscreteScheduler` với **shift = 3.0** (config Dora).
- Nội suy tuyến tính `x_t = (1−σ)x_0 + σε` (`:140-141`).
- **`precondition_outputs=True`**: output mạng được hiểu là vận tốc v, đổi thành x̂_0 = x_t − σ·v̂ và loss MSE với x_0 (`:147-163`). ⚠️ Tương đương `σ²‖v̂ − (ε − x_0)‖²`: loss velocity có trọng số σ², giảm trọng số ở mức nhiễu thấp.
- Suy luận: Euler `x ← x − Δt·v̂`, 30 bước, CFG 7.5 (`systems/utils.py:221-295`).

**c) Denoiser – `craftsman/models/denoisers/pixart_denoiser.py` + `utils.py`** ✅
- Đầu vào: Linear 64 → 1024; 32 block DiT, width 1024, 16 head (config). **Không có positional embedding** cho token latent (`:151`), đúng với bản chất "tập" của latent set.
- adaLN-single: một `t_block` dùng chung sinh 6×width tham số (`:83-86`, `:153`), mỗi block cộng thêm `scale_shift_table` riêng đã học (`denoisers/utils.py:50,55`).
- Block: self-attn (có modulate + gate) → **cross-attn không norm, không gate** (`denoisers/utils.py:57`) → MLP (có modulate + gate).
- **Identity init**: zero `c_proj` của self-attn, cross-attn, MLP ở mọi block (`pixart_denoiser.py:104-111`) → lúc khởi tạo, mỗi block là phép đồng nhất.
- Hợp nhất điều kiện CLIP + DINO: xem mục 5.3 (3).

**d) Bộ mã hóa điều kiện – `craftsman/models/conditional_encoders/`** ✅
- CLIP ViT-L/14 @224 và DINOv2-base @224 → mỗi bên 257 token; DINO 768 chiều được chiếu lên 1024 bằng `Linear(768, 1024, bias=False)` **khởi tạo 0** (`clip_dinov2_encoder.py:161-163`, `:281-286`). Đây chính là "φ_mlp" trong Eq. 1, thực tế chỉ là một Linear.
- CLIP và DINO đóng băng hoàn toàn; chỉ tham số `mod_norm` (ModLN camera) được train khi bật (`clip_dinov2_encoder.py:144-159`). Config phát hành `encode_camera: false`.
- **ModLN** zero-init (`dinov2/modeling_conditional_dinov2.py:36-55`) chèn sau `norm1`/`norm2` của mỗi layer DINO/CLIP (`:69-70`, `:80`, `:93`). Camera embedding = sin/cos của ma trận c2w 4×4 làm phẳng (32 chiều) (`base.py:72-78`). Có sẵn 4 camera cố định trước/phải/sau/trái (`base.py:42-67`).
- **CFG dropout 10%** (`base.py:32`, `:100-104`), embedding rỗng là **tensor 0** (`zero_uncond_embeds`, `clip_dinov2_encoder.py:129-132`).

**e) Dữ liệu – `craftsman/data/base.py`** ✅
- Mỗi mẫu chọn ngẫu nhiên **một trong 20 view** render (`idx: [0..19]`, `:190`), nền xám 0.5, crop theo bbox alpha rồi đặt vào canvas 512 với lề 30 px (`:18-43`, `:166-201`).
- Supervision TSDF: cắt ±τ rồi chia τ (`:158-159`). Lỗi `except` → lấy mẫu ngẫu nhiên khác (`:222-227`).

**f) Pipeline suy luận – `craftsman/pipeline.py`** ✅
- `rembg` nếu ảnh không có alpha, căn đối tượng chiếm 95% khung, nền trắng (`:96-166`).
- Khi dùng bf16: **VAE ép về fp16** ("shape vae only support fp16"), CLIP/DINO và DiT để bf16 (`:70-74`).
- Giải mã: lưới `2^mc_depth` (mặc định 8 → 256³) qua `HierarchicalVolumeDecoding`, rồi `skimage` marching cubes, đảo thứ tự đỉnh của mặt (`michelangelo_autoencoder.py:220-279`).

**g) Giải mã volume phân cấp – `craftsman/models/autoencoders/volume_decoders.py`** (từ Hunyuan3D-2/FlashVDM) ✅
- Độ phân giải [64, 128, 256] cho octree 256 (`:199-205`). Ở mỗi cấp chỉ truy vấn các ô có **đổi dấu với lân cận** hoặc **|logit| < 0.95** (`:245-246`), giãn (dilate) bằng conv 3×3×3 toàn số 1, phần còn lại gán −10000 (`:221-275`). Chỉ hỗ trợ batch = 1 (`squeeze(0)`, `grid_logits[0]`).

#### 5.3 Những điểm hay nhất trong code

**(1) Encoder dùng chính điểm bề mặt (FPS) làm query → số token latent linh hoạt**
`craftsman/models/autoencoders/michelangelo_autoencoder.py:437-456`
```python
idx = fps(pos, batch, ratio=ratio)          # ratio = num_latents / N
query = data.view(bs*N, -1)[idx].view(bs, -1, data.shape[-1])
latents = self.cross_attn(query, data)
```
Không có query học được, nên số latent chỉ là một tham số lúc chạy. Config VAE ghi `num_latents: 1024`, nhưng config diffusion dùng **768** với cùng VAE ✅ (so sánh hai file config). Query lại mang thông tin vị trí của điểm bề mặt, nên có "khung xương" hình học sẵn. Dora/TripoSG/Hunyuan3D-2 tận dụng đúng tính chất này để tăng số token khi fine-tune.

**(2) Dora: chia đôi ngân sách token cho điểm cạnh sắc**
`craftsman/models/autoencoders/dora_autoencoder.py:392-395, 467-502`
```python
self.downsmaple_num_latents = num_latents // 2
...
query = torch.cat([query, sharp_query], dim=1)
latents = self.cross_attn(query, data)
latents = latents + self.cross_attn_sharp(query, sharp_data)
```
Một nửa query lấy FPS từ điểm đều, nửa còn lại từ điểm trên cạnh sắc. Hai cross-attention riêng (dữ liệu đều / dữ liệu sắc) được cộng lại. FPS chạy trên tọa độ fp16 cho nhanh (`:475`, `:487`). Đây là **lấy mẫu theo tầm quan trọng ở mức token**: dồn dung lượng biểu diễn vào vùng tần số cao.

**(3) Hợp nhất CLIP + DINO theo từng token, căn theo vị trí patch, DINO khởi tạo 0**
`clip_dinov2_encoder.py:161-163, 284-285` và `pixart_denoiser.py:135-140`
```python
self.linear_proj = nn.Linear(768, 1024, bias=False); nn.init.constant_(self.linear_proj.weight, 0)
visual_embeds = torch.cat([clip_embeds, dino_embeds], dim=1)        # [B, 257+257, 1024]
clip_feat, dino_feat = context.chunk(2, dim=2)
visual_cond = clip_weight * self.clip_embed(clip_feat) + dino_weight * self.dino_embed(dino_feat)
```
Cả hai backbone đều patch 14 ở 224 → 16×16 + CLS = 257 token, nên token thứ i của CLIP và của DINO **cùng vị trí ảnh** và có thể cộng trực tiếp: số token điều kiện không tăng gấp đôi. Mỗi nguồn có LN + Linear riêng. DINO bắt đầu từ 0, nên mô hình khởi đầu như chỉ dùng CLIP rồi dần học thêm đặc trưng cục bộ của DINO. ⚠️ Paper không ablate lựa chọn này.

**(4) Identity-init cho DiT + zero-init cho ModLN**
`pixart_denoiser.py:104-111`, `dinov2/modeling_conditional_dinov2.py:44-47`
```python
nn.init.constant_(block.attn.c_proj.weight, 0); nn.init.constant_(block.cross_attn.c_proj.weight, 0)
nn.init.constant_(block.mlp.c_proj.weight, 0)
...
nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)   # ModLN: x*(1+scale)+shift bắt đầu = x
```
Mọi đường mới (khối DiT, camera ModLN chèn vào bộ mã hóa ảnh **đóng băng**, chiếu DINO) đều bắt đầu là phép đồng nhất hoặc bằng 0. Đây là cùng một nguyên lý với zero-conv của ControlNet. Chèn ModLN vào ViT pretrained mà không phá đặc trưng ban đầu là một cách rẻ để đưa thêm tín hiệu toàn cục (camera) vào backbone đóng băng.

**(5) TSDF chuẩn hóa về [−1, 1] → vừa ổn định khi train vừa cho phép giải mã phân cấp**
`data/base.py:158-159`, `volume_decoders.py:245-246`
```python
ret["sdf"] = sdfs.clip(-tau, tau) / tau
...
curr_points += grid_logits.squeeze(0).abs() < 0.95
```
Xa bề mặt, đầu ra bão hòa ở ±1, nên "|logit| < 0.95" là một dải gần bề mặt rất rõ. Bộ giải mã phân cấp chỉ cần tinh chỉnh trong dải này. Thiết kế loss (truncation) và thiết kế suy luận (coarse-to-fine theo vùng quan tâm) khớp với nhau.

**(6) Watertight: visibility nhanh + winding number sửa lỗi "nhìn xuyên lỗ thủng"**
`watertight_and_sampling.py:310-317`
```python
visibility, dist = visibility_check(grid_points, rendererd_imgs['vert_depth'], ...)
winding_numbers = igl.fast_winding_number_for_meshes(V, F, grid_points)
visibility[visibility & (winding_numbers > winding_number_thres)] = False
```
Visibility check (kiểu CLAY) rất nhanh nhưng sai khi mesh có lỗ: camera "nhìn thấy" điểm bên trong qua lỗ. Generalized winding number gần như bền vững với lỗ thủng và mesh không kín, nên dùng để lật lại những điểm này. Mặt nạ rẻ (render) kết hợp với phép thử trong/ngoài toàn cục (winding) chỉ ở chỗ nghi ngờ.

**(7) Marching cubes ở mức offset 1 voxel thay vì mức 0**
`watertight_and_sampling.py:333-334` – với mặt mỏng/hở (lá, giấy) mà UDF ≈ 0, mức 0 cho bề mặt suy biến. Lấy mức ε = 1 voxel cho ra vỏ kín có độ dày, đổi lại mesh hơi "béo" thêm 1 voxel.

**(8) Min-SNR-γ cho ε-pred và preconditioning kiểu SD3 cho flow**
`shape_diffusion.py:139-159`, `shape_rectified_flow.py:147-163`
```python
mse_loss_weights = torch.stack([snr, gamma*torch.ones_like(t)], 1).min(1)[0] / snr   # eps-pred
...
output = output * (-sigmas) + noisy_z ; target = latents                              # flow, loss trên x0
```
Hai mẹo ổn định train có sẵn từ diffusers, không ghi trong paper nhưng là mặc định thực tế.

**(9) Attention luôn được checkpoint ở mức op, kể cả khi tắt `use_checkpoint`**
`craftsman/models/transformers/attention.py:57, 62, 213, 218` – `checkpoint(self.attention.forward_qkv_fuse, (x,), (), True)`: chỉ tính lại phần softmax-attention khi backward, giữ activation của Linear. ⚠️ Mặt trái: tham số **bên trong** hàm được checkpoint không nhận gradient, vì `params=()` và `CheckpointFunction` chỉ tính grad cho inputs + params truyền vào (`craftsman/utils/checkpoint.py:31-32, 59-64`). Nếu bật `qk_norm`, LayerNorm của Q/K (`attention.py:74-75, 107-108`) sẽ **không được học**. (Đã grep GReg: không dùng helper checkpoint tự viết kiểu này, nên không bị ảnh hưởng.)

**(10) Scale √√d nhân vào cả q và k trước einsum ("ổn định hơn với fp16")**
`attention.py:80, 91-93` – chia tỉ lệ trước tích vô hướng để tránh tràn số fp16, softmax ép lên fp32. Đây là mẹo từ code shap-e/point-e. Thực tế đường `use_flash` (SDPA) được dùng nên mẹo này ít khi chạy.

#### 5.4 Khác biệt giữa paper và code (và lỗi trong code phát hành)

| Chủ đề | Paper | Code | Mức |
|---|---|---|---|
| Điều kiện | Ảnh MV + camera ModLN | Config phát hành: **1 ảnh**, `n_views: 1`, `encode_camera: false`; bản MV chưa phát hành | ✅ |
| Refiner normal | Mục 3.3, Eq. 3–6 | **Không có** | ✅ |
| Ngưỡng TSDF | 1/256 | `tsdf_threshold: 0.0078125` = **1/128** (config VAE) | ✅ |
| Ngưỡng winding number | 0.75 | Mặc định **0.5** (`watertight_and_sampling.py:241`) | ✅ |
| Logic winding | "Ô không nhìn thấy bởi mọi camera → kiểm tra winding" | Kiểm tra winding cho điểm **đã nhìn thấy** để lật thành bên trong (`:317`) | ✅ (cách đọc paper hơi mơ hồ) |
| φ_mlp (DINO → CLIP) | "small MLP" | 1 `Linear` không bias, init 0 | ✅ |
| Loss diffusion | ε-MSE (Eq. 2) | ε-MSE + **Min-SNR γ = 5**; bản Dora: rectified flow, logit-normal, shift 3 | ✅ |
| Batch | README 32/GPU | config 40 | ✅ |
| Lịch β | Không nêu | Config chính: DDPM mặc định (linear 1e-4 → 0.02); config finetune: `scaled_linear` 0.00085 → 0.012 kiểu SD | ✅ (không biết checkpoint v1-5 dùng cái nào ⚠️) |
| Không shuffle | – | `DataLoader(...)` **không có `shuffle=True`** (`craftsman/data/Objaverse.py:46-48`) → mỗi epoch đi cùng thứ tự | ✅ code; ⚠️ ảnh hưởng thực tế chưa rõ |
| Đăng ký module | – | Config `-mv` và Dora dùng `system_type: "pixart-diffusion-system"` và `condition_model_type: "cond-embedder"` – **không được đăng ký** ở đâu (chỉ có `shape-diffusion-system`, `shape-rectified-flow-system`, `clip-dinov2-embedder`, `dinov2-embedder`) → `craftsman.find` báo lỗi | ✅ |
| RF training | – | Gọi `self.condition.forward_visual_embeds` và `encode_text` – **không được định nghĩa** ở đâu; nhánh ảnh 4D dùng `bs`, `n_images` chưa gán (`shape_rectified_flow.py:118, 131`) | ✅ |
| Dora encoder | – | Encoder bắt buộc có `sharp_pc` khi `with_sharp_data=True` (`dora_autoencoder.py:443-449`), nhưng dataloader không bao giờ trả `sharp_surface` | ✅ |
| VAE training | – | `train_autoencoder.sh` gọi `launch.py` + config không tồn tại; dataset dùng `self.cfg.data_type` (không phải field của config, `base.py:130`); config VAE dùng `root_dir` thay vì `local_dir`; định dạng `samples.npz` của script watertight (`near_surface_points`, `volume_rand_points`) **không khớp** khóa mà loader đọc (`rand_points`, `sdfs`) | ✅ |
| `extract_geometry` (Michelangelo) | – | Gọi `time.time()` nhưng file **không import `time`** (`michelangelo_autoencoder.py:229, 251`) → ⚠️ có thể NameError với checkpoint v1-5; bản Dora có import `time` | ✅ thiếu import |
| `qkv_bias` của DiT | – | `qkv_bias=self.cfg.drop_path` (= 0.) → **Q/K/V không có bias** (`pixart_denoiser.py:74`), có vẻ là lỗi đánh máy | ✅ |
| Grad checkpoint DiT | `use_checkpoint: true` | `auto_grad_checkpoint` kiểm tra thuộc tính `grad_checkpointing` mà không nơi nào đặt → checkpoint block **không bao giờ bật** (`denoisers/utils.py:65-71`) | ✅ |
| CFG dropout | – | `random.random()` gọi **một lần cho cả batch** (`base.py:100`) → 10% số batch hoàn toàn uncond, không phải 10% số mẫu | ✅ |
| Sharp-edge sampling | – | Đọc lại file mesh **gốc** (chưa chuẩn hóa tâm/tỉ lệ ở `:551-552`) qua Blender, trộn đỉnh từ bpy với đỉnh từ trimesh theo chỉ số (`:443-447`, `:456-458`) | ✅ code; ⚠️ có thể lệch hệ tọa độ/chỉ số |
| Nền ngẫu nhiên | – | `background_color == [-1,-1,-1]` → `randint(0,256)` trong khi ảnh ở [0,1] (`base.py:183-184`) | ✅ (không dùng trong config) |

⚠️ Kết luận: code phát hành dùng được để **suy luận** và **fine-tune bản DDPM một ảnh**. Train VAE và train rectified-flow cần sửa nhiều chỗ. Không thể dùng để tái hiện số liệu của paper.

#### 5.5 Hyperparameter thực tế (từ config)

| Nhóm | Giá trị |
|---|---|
| VAE | width 768, 12 head, 8 lớp encoder, 16 lớp decoder, `embed_dim` 64, Fourier 8 tần số, `n_samples` 16384 điểm bề mặt (+ normal), `n_supervision` 16384, TSDF τ = 1/128, λ_KL 1e-3, AdamW lr 1e-4 β(0.9, 0.99), warmup tuyến tính 5k bước rồi Cosine T_max 5k (⚠️ CosineAnnealingLR tuần hoàn, với 100k epoch thì LR dao động), bf16-mixed, batch 8 |
| Latent | 768 token × 64 (Michelangelo/v1-5); 2048 token × 64 (DoraVAE); `z_scale_factor` 1.0 |
| DiT | width 1024, 32 lớp, 16 head, `context_dim` 1024, adaLN-single, không PE, drop_path 0 |
| Điều kiện | CLIP ViT-L/14 @224 + DINOv2-base @224, CFG dropout 0.1, uncond = 0. Tên config Dora gợi ý `dinov2reglarge518` và code `dinov2_encoder.py:76-81` hỗ trợ `dinov2-with-registers-large` (@518 → 37² + 1 = 1370 token sau khi bỏ register, `:110-111`, `:185-188`), nhưng **nội dung** yaml Dora trong repo vẫn ghi CLIP-L + DINOv2-base @224 → ⚠️ config thật của checkpoint DoraVAE nằm trên HF, không có trong repo |
| Diffusion | DDPM 1000 bước, ε-pred, Min-SNR γ 5; suy luận DDIM 50 bước, CFG 7.5, eta 0 |
| Flow (Dora) | FlowMatchEuler shift 3.0, logit-normal(0, 1), preconditioning x0; suy luận Euler 30 bước, CFG 7.5 |
| Optim diffusion | AdamW lr 1e-4 (bản chính) / 2e-4 + Cosine T_max 5000 eta_min 1e-6 (bản mv/Dora), fp16-mixed, batch 40, DDP |
| Data | 20 view/object, chọn ngẫu nhiên 1 view/mẫu, nền xám 0.5, canvas 512 lề 30 px |
| Giải mã | Bounds ±1.05 (pipeline) / ±1.1 (gradio), octree 2^8 = 256 (pipeline) / 2^7 (gradio, validation), FlashVDM [64 → 128 → 256] |
| Watertight | 50 view, lưới 256, depth 1024², winding ngưỡng 0.5, MC offset 1 voxel |

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Kết quả chính không tái hiện được bằng code công khai:** mô hình MV và refiner không phát hành; nhiều config trỏ tới module không tồn tại ✅.
2. **So sánh định lượng mỏng:** GSO chỉ 30 shape và thua InstantMesh; Bảng 2/4 dùng "tập con tự chọn" không mô tả; refiner chỉ đo bằng CLIP similarity trên 20 object ✅/⚠️.
3. **Thiếu ablation cho các mẹo quan trọng nhất của bước 2** (PF, Re-Laplace chỉ có hình định tính; cross-view attention không có ablation) ✅.
4. **Khoảng cách train/test của điều kiện MV:** train bằng ảnh render GT, test bằng ảnh MV sinh ra; paper không đo độ nhạy với lỗi bất nhất giữa các view ⚠️.
5. **Refiner tối ưu từng mesh (~20 s)**, phụ thuộc chất lượng ControlNet-Tile normal và số view; vùng không có view nào phủ thì không được refine ⚠️.
6. **Không có texture**, và tác giả thừa nhận khả năng điều khiển của latent set diffusion còn hạn chế ✅ §5.
7. **License mâu thuẫn** (MIT / AGPL / Hunyuan phi thương mại) ✅.
8. **Câu hỏi mở:** cross-view attention training-free có còn tác dụng khi khác biệt giữa các view lớn không (paper tự thừa nhận nó chạy được "một phần" nhờ ràng buộc mạnh)? Poisson blending trong miền normal có xử lý được thay đổi topology cục bộ (thêm lỗ, gờ lớn) không? ⚠️

---

### 7. Liên hệ với GReg

**Nói thẳng:** đây là bài sinh **hình dạng 3D tĩnh**, không có thời gian, không có ảnh người thật, không có warp. Phần lớn đóng góp (latent set VAE, watertight, marching cubes, DiT 3D) **không chuyển** sang được video 2D. Giá trị cho GReg nằm ở **bước 2 (refine bằng diffusion có kiểm soát)** và một vài **mẹo kỹ thuật**. Cấu trúc "coarse → refine" của CraftsMan trùng gần như hoàn toàn với Hướng 2 (GReg cho ảnh thô → refiner diffusion), nên các bài học về chống "trôi tần số thấp" áp dụng trực tiếp.

#### 7.1 Kỹ thuật chuyển được

| # | Ý tưởng từ CraftsMan3D | Áp dụng cho GReg | Hướng | Độ khó | Ghi chú |
|---|---|---|---|---|---|
| T1 | **Poisson Normal Blending** – diffusion "tăng chi tiết" nhưng lén đổi tần số thấp, nên chỉ lấy gradient/chi tiết từ diffusion còn biên/tần số thấp giữ từ nguồn tin cậy (Eq. 5, Hình 7) | Sau refiner: (a) **hợp nhất trong miền gradient**: vùng `V=1` lấy pixel warp/GReg làm điều kiện biên Dirichlet, vùng che khuất lấy gradient từ ảnh diffusion → giải Poisson (`cv2.seamlessClone` hoặc FFT/CG solver); thay cho phép trộn cứng `V·I_warp + (1−V)·Î_diff` trong `diffusion_improvement.md` §1.4; hoặc (b) **tách tần số**: `Î = LowPass(I_GReg) + HighPass(Î_diff)` trong vùng tay. Thêm loss/metric "drift tần số thấp" `‖LP(Î) − LP(I_GReg)‖₁` | H2 (cũng dùng được cho H1 ở bước dán lại) | Thấp | ⚠️ Diffusion refiner 2D nổi tiếng đổi màu áo/tông da; đây là chỗ mẹo này có giá trị nhất. Cần đo: màu (ΔE) ở vùng nhìn thấy trước/sau |
| T2 | **ControlNet-Tile trên ảnh thô** ("ảo giác chi tiết mà không đổi nhiều đầu vào", §3.3) | **Baseline training-free cho H2**: SD1.5 + ControlNet-Tile (điều kiện = output GReg) + tùy chọn ControlNet-normal (điều kiện = normal SMPL-X, cần đổi quy ước màu/trục sang kiểu NormalBAE), img2img strength ≈ 0.3–0.5, chỉ trong mặt nạ tay + vùng `V=0` (B1). Có số liệu trước khi train pix2pix-turbo/OSEDiff | H2 | Thấp (1–2 ngày) | ⚠️ Chậm (nhiều bước), nhưng là mốc so sánh rẻ và dễ giải thích trong luận văn |
| T3 | **Cross-view attention không train** (Eq. 3: K/V nối từ mọi view, chia sẻ chung) | Coi "view" = **khung hình**: trong self-attention của refiner per-frame, nối K/V của khung nguồn `I_s` (giữ danh tính) và khung t−1, t+1 (mượt thời gian). Không cần train; gắn bằng attention processor của diffusers | H2 (bổ sung prior thời gian mà H2 thiếu) | Trung bình | Điều kiện CraftsMan nêu cho việc "training-free chạy được" là ràng buộc mạnh từ đầu vào thô + Tile ít thay đổi đầu vào; H2 thỏa mãn điều này (GReg output ràng buộc mạnh). ⚠️ Paper không ablate nên phải tự đo (tLPIPS/flicker) |
| T4 | **Generalized winding number** bền với mesh hở (`watertight_and_sampling.py:311-317`) | **A1 – tiếp xúc tay–thân/mặt**: trong ngôn ngữ ký hiệu tay thường chạm mặt/ngực, fit SMPL-X dễ xuyên vào nhau; z-buffer khi đó cho visibility sai. Tính `igl.fast_winding_number` của mesh thân (bỏ tay) tại các đỉnh tay: WN > 0.5 → đỉnh đang xuyên → hạ độ tin cậy `C` hoặc đánh dấu che khuất. SMPL-X có lỗ (mắt/miệng), đúng trường hợp WN được thiết kế cho | A1 | Thấp–trung bình | ⚠️ Đầu tiên chạy chẩn đoán xem bao nhiêu % khung có xuyên (kiểu E0) rồi mới làm. Lưu ý thêm: visibility check của CraftsMan **không có epsilon độ sâu** (`:225`); A1 cần epsilon |
| T5 | **Zero/identity-init mọi đường mới** (DINO proj = 0, DiT block = I, ModLN = 0; mục 5.3 (4)) | H1: kênh vào mới `[V, M, UV]`, adapter tay, LoRA đều khởi tạo để mô hình gốc không đổi ở bước 0 (đã có trong kế hoạch). Mới: **ModLN zero-init chèn vào encoder đóng băng** (DINOv2/CLIP) để đưa tín hiệu toàn cục, ví dụ tham số pose tay MANO/SMPL-X, vào encoder crop tay của adapter mà không fine-tune encoder | H1 | Thấp | ✅ khớp với ControlNet/SignRefine; ⚠️ phần ModLN là đề xuất |
| T6 | **Hợp nhất CLIP + DINO theo token cùng vị trí patch**, chiếu DINO zero-init (mục 5.3 (3)) | Encoder ảnh tham chiếu `I_s` / crop tay cho adapter H1: cộng token DINOv2 (tương ứng cục bộ tốt) vào token CLIP mà Wan-I2V đang dùng, không tăng số token cross-attn | H1 | Trung bình | ⚠️ Paper không ablate. Chỉ đáng thử nếu chi tiết tay/danh tính còn yếu sau khi làm xong phần chính |
| T7 | **Mẹo train**: Min-SNR-γ = 5 (ε-pred), logit-normal + shift 3 (flow), preconditioning x0, CFG dropout 10% với uncond = 0 | H1 (Wan là flow matching): giữ lịch t mặc định của Wan khi LoRA, chỉ cân nhắc dịch t về nhiễu thấp nếu mục tiêu là chi tiết. **CFG dropout nên theo từng mẫu và từng điều kiện** (warp, normal, mask bỏ độc lập) để làm multi-condition CFG; CraftsMan bỏ cả batch (`base.py:100`) là cách yếu hơn | H1 | Thấp | ✅ code; ⚠️ khuyến nghị |
| T8 | **Refine cục bộ theo mặt nạ** ("magic brush") + **tính toán phân cấp chỉ ở dải gần bề mặt** (FlashVDM, `|logit|<0.95`) | Chỉ chạy refiner/diffusion ở vùng tay + vùng `V=0`; crop tay độ phân giải cao (256–512) rồi dán lại (giống SignRefine). Tiết kiệm compute, giảm nguy cơ làm hỏng vùng đã tốt | H2, H1 | Thấp–trung bình | ⚠️ |
| T9 | **Giám sát bằng normal** (Eq. 4: L1 giữa normal render và normal mục tiêu) | Metric/loss "nhất quán hình học tay": chạy bộ ước lượng normal có sẵn (DSINE/Sapiens-normal ⚠️ cần kiểm chứng license/chất lượng) trên khung sinh ra, so L1 với normal SMPL-X **chỉ trong mặt nạ tay** | Đánh giá; có thể là loss phụ cho H2 | Cao | ⚠️ Normal SMPL-X là normal thân trơn, không có quần áo; chỉ đáng tin ở tay/mặt. Ưu tiên thấp |

#### 7.2 Ưu tiên đề xuất ⚠️

1. **T1 (Poisson/tách tần số)**: rẻ nhất, áp dụng cho mọi refiner; đưa vào ngay khi có output H2/H1 đầu tiên, kèm metric drift màu ở vùng nhìn thấy.
2. **T2 (baseline Tile-ControlNet training-free)**: cho số liệu H2 trong vài ngày, trước khi train.
3. **T3 (chia sẻ K/V giữa khung)**: bổ sung tính nhất quán thời gian cho H2 mà không cần train.
4. **T4 (winding number cho xuyên mesh)**: sau khi chẩn đoán thấy lỗi visibility ở các khung tay chạm mặt.

#### 7.3 KHÔNG nên lấy (và lý do)

- **Latent set VAE / 3DShape2VecSet / TSDF / marching cubes / FlashVDM:** dành cho hình 3D; GReg làm việc trên ảnh/video và (ở H1) đã có video VAE của Wan. Không có bài toán 3D tương ứng.
- **Train DiT từ đầu** (500M tham số, 32×A800 × 7 ngày): không khả thi và không cần, dữ liệu chỉ ~4.6k clip; H1 đã chọn hướng backbone đóng băng + LoRA.
- **Tối ưu mesh bằng continuous remeshing + relative Laplacian:** topology SMPL-X cố định và đã được fit tối ưu; thêm chi tiết vào mesh không đổi được chất lượng ảnh (quần áo, tóc không có trong SMPL-X). Tối ưu ~20 s mỗi khung cũng quá chậm cho video. Ý tưởng "phạt *thay đổi* so với trạng thái ban đầu thay vì ép về 0" đã nằm trong T1 (giữ tần số thấp).
- **Điều kiện MV + camera ModLN:** GReg có một camera cố định; "đa view" tương ứng là đa khung, đã có T3.
- **Chép code của repo:** nhiều lỗi (mục 5.4), license mâu thuẫn và có phần Hunyuan phi thương mại. Nên tự viết lại các ý tưởng (phần nào cũng chỉ vài chục dòng).

#### 7.4 Câu cần viết trong luận văn (nếu dùng) ⚠️

- Khi giải thích vì sao cần ràng buộc tần số thấp cho refiner: dẫn CraftsMan3D Hình 7/10 và Eq. 5 làm tiền lệ ("diffusion enhancement introduces low-frequency changes … Poisson blending eliminates global distortions").
- Khi dùng chia sẻ K/V giữa các khung không cần train: dẫn Eq. 3 cùng các nguồn video-editing (TokenFlow, Text2Video-Zero; chưa đọc trong ghi chú này). Nói rõ CraftsMan3D chỉ chứng minh định tính.

---

## Bài 3 — Single Mesh Diffusion Models with Field Latents for Texture Generation (CVPR 2024)

Quy ước: ✅ = kiểm chứng trực tiếp trong bài (ghi rõ mục/bảng/phương trình) hoặc trong code (ghi `đường_dẫn.py:DÒNG`, đường dẫn tính từ thư mục `mesh_diffusion/`). ⚠️ = suy luận, tự tính toán hoặc phỏng đoán của mình.

Nguồn đã đọc: PDF arXiv 2312.09250v3 (bản chính + **Supplementary Material** §8–§10, Bảng 3–4), toàn bộ code chính thức `google-research/mesh_diffusion` (15 file, ~4.9k dòng, đọc hết từng dòng, **không chạy**). Code tiền xử lý mesh nằm ở repo khác (`twmitchel/field_latent_preprocess`) và **chưa đọc**. Mọi nhận định phụ thuộc vào dữ liệu tiền xử lý (log map, góc vận chuyển song song) đều được đánh dấu ⚠️.

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Thomas W. Mitchel (Google Research → PlayStation), Carlos Esteves, Ameesh Makadia (Google Research) ✅ |
| Venue | CVPR 2024; arXiv 2312.09250 (v3, 28/05/2024) ✅ |
| Link | Paper: https://arxiv.org/abs/2312.09250 · Project: https://single-mesh-diffusion.github.io/ · Code: https://github.com/google-research/google-research/tree/master/mesh_diffusion ✅ (README) |
| Trạng thái code | Có code **train + sample FLDM** và **kiến trúc FL-VAE** (encoder/decoder). **Không có** script train FL-VAE, **không có** code tiền xử lý mesh trong repo này (README trỏ sang repo `field_latent_preprocess`) ✅. Có checkpoint FL-VAE và checkpoint FLDM cho một mesh (con sư tử biển Objaverse, 200K bước) + dữ liệu đã tiền xử lý **65 GB** cho 500 bản remesh ✅ (README). |
| License | Apache 2.0 (header mọi file) ✅. Mesh sư tử biển: CC-BY ✅ (README). |
| Framework | JAX 0.4.25 + Flax 0.8.1 + Optax 0.2.1; dữ liệu qua TFRecord/TensorFlow 2.15; `clu` cho checkpoint/metrics ✅ (`requirements.txt`). Toàn bộ mạng dùng `complex64`. |
| Compute | FL-VAE: 1.5M iter, batch 16, ~5 giờ trên 1×A100 (§9.1) ✅. FLDM: 300K iter, batch 16, **~2 ngày trên 8×V100 16 GB**, phải rematerialize (§9.2) ✅. Sampling: **~10 phút/mẫu** (DDPM 1000 bước) ✅. |
| Dữ liệu | FL-VAE: 1K mesh phẳng 10K đỉnh phủ lên ảnh 512² của OpenImagesV4 ✅ (§6). FLDM: **một** mesh có texture, nhân 500 bản remesh ~30K đỉnh ✅ (§6). |

---

### 1. TL;DR

- Biểu diễn texture trên mesh bằng **Field Latents (FL)**: mỗi đỉnh mang **d = 8 vector tiếp tuyến** (viết dưới dạng số phức trong khung tiếp tuyến cục bộ), mã hóa texture của one-ring quanh đỉnh. Một FL-VAE **huấn luyện trên ảnh phẳng** dùng được zero-shot trên mọi mesh 3D, vì nó chỉ "nhìn" one-ring qua **log map** (làm phẳng cục bộ) ✅ (§4, §6).
- Decoder là neural field: màu tại điểm q được dự đoán từ latent z_p của đỉnh p cộng với **tọa độ log_p q chiếu lên chính các vector latent** (`log_p q · z̄_p`). Cách này cho chi tiết **dưới mức tam giác** tốt hơn hẳn nội suy barycentric (Bảng 1: PSNR 22.38 so với 21.33 và INF 18.86) ✅.
- **FLDM**: DDPM trên trường vector latent, mạng khử nhiễu là U-Net 2 tầng nông bằng **field convolution** (tích chập trên mặt cong cho đặc trưng vector tiếp tuyến). Timestep và nhãn được tiêm **vào đối số của filter** để giữ tính đẳng biến ✅ (§5, §9.2).
- Toàn bộ pipeline **đẳng biến với phép đẳng cự (isometry)**. Nhờ vậy model không phụ thuộc lựa chọn khung tiếp tuyến tùy ý, tự lặp lại chi tiết ở các vùng giống nhau cục bộ, và có thể **"chuyển texture sinh tạo"** sang mesh khác gần đẳng cự chỉ bằng cách sample trên mesh mới ✅ (§3, §6.5).
- Với GReg: **không** nên lấy bộ máy số phức/field conv. SMPL-X có topology cố định nên không có vấn đề "khung tiếp tuyến mơ hồ" mà bài này giải. Đáng lấy là **ý tưởng "texture/latent sống trên bề mặt + inpaint trên bề mặt"**, **decoder theo tọa độ cục bộ**, và **inpainting kiểu thay thế latent đã nhiễu** (Eq. 14). Hai bài được trích dẫn là DINAR [51] và Grigorev 2019 [20] gần bài toán của GReg hơn nhiều ⚠️.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Bài toán ✅ (§1, §3):** cho **một** mesh tam giác có texture (UV atlas). Học phân phối "patch texture" của chính nó để sinh các **biến thể texture mới trên cùng hình học** (không đổi geometry). Có thêm các chế độ có điều khiển: theo nhãn người dùng tô (label-guided), inpainting, chuyển sang mesh mới, chỉnh tỉ lệ chi tiết.

- **Đầu vào train:** 1 mesh + texture atlas; 500 bản remesh khác triangulation (dùng chung UV atlas) ✅ (§6); tùy chọn một file PLY gán nhãn theo mặt ✅ (`latents/io_utils.py:25-46`, trường `quality` của face).
- **Đầu ra:** latent tại mọi đỉnh → giải mã ra **texture atlas** (mặc định 1024²) bằng cách decode từng texel ✅ (`sin_mesh_ddm/train.py:531-578`, `sin_mesh_ddm/input_pipeline.py:58-61`).

**Vì sao khó ✅ (§1–§3):**
1. **Không có lưới chuẩn.** Ảnh có hướng "lên/phải" chuẩn, còn mặt cong thì không: khung tọa độ ở mỗi điểm chỉ xác định **sai khác một phép quay tùy ý** (gauge ambiguity). Không thể bê nguyên VAE/UNet ảnh sang.
2. **Các cách 2D hóa đều mất mát:** render nhiều view + SDS/projection thì không nhất quán giữa các view và mang theo ánh sáng giả. Rasterize vào lưới 3D/triplane (Sin3DM) thì bị aliasing chi tiết. UV atlas (Point-UV diffusion) thì hỏng khi atlas bị cắt vụn.
3. **Dữ liệu 3D có texture đẹp rất ít**, nên bài chọn mô hình **chỉ học từ một mẫu** (kiểu SinGAN/SinDiffusion). Để không học thuộc mẫu, mạng phải có **receptive field nhỏ**.
4. **Bộ nhớ:** tích chập trên one-ring không đều (8 láng giềng/đỉnh, 30K đỉnh, kênh phức) rất tốn bộ nhớ, dẫn đến phải rematerialize ✅ (§9.2).

---

### 3. Phương pháp chi tiết

#### 3.1 Ký hiệu và biểu diễn (✅ §4, §9 "Tangent bases")

- Tại đỉnh p: pháp tuyến n_p; chọn **tùy ý** một vector đơn vị e₁ ⊥ n_p, đặt e₂ = n_p × e₁ ✅ (§9). Vector tiếp tuyến v ∈ T_pM được viết thành số phức v ≡ r·e^{iθ} ∈ ℂ_p.
- Đổi khung (gauge) một góc β thì **mọi** vector tại p nhân với e^{-iβ}. Một isometry γ: M→N có vi phân dγ|_p tác động như phần tử của **U(1)**, tức là nhân với một số phức đơn vị ✅ (§8.1).
- **Nhiễu Gauss trên tangent bundle** 𝒯𝒩_M(0, I_d): tại mỗi điểm ε(p) = ε₁ + iε₂, với ε₁, ε₂ ~ 𝒩(0, I_d) (Eq. 1). Phân phối này đẳng hướng nên bất biến khi quay (Eq. 2) ✅. Đây là chìa khóa để cả forward process lẫn reverse process đều đẳng biến.
- **Field Latent:** z_p ∈ ℂ^d với **d = 8** (mọi thí nghiệm), tức 8 vector tiếp tuyến/đỉnh = 16 số thực/đỉnh ✅ (§6).

#### 3.2 FL-VAE Encoder (✅ Eq. 3, 21–25; code `latents/ring_vae.py:52-175`)

Encoder chạy **từng đỉnh độc lập** trên one-ring B_p:
1. Lấy mẫu **64 điểm** q_i đều trong one-ring (điểm đầu là chính p, log = 0) ✅ (§9.1; `config.attn_neigh = 64`, `sin_mesh_ddm/configs/default.py:52`). Với mỗi q_i có màu ψ(q_i) ∈ ℝ³ và log_p q_i ∈ ℂ.
2. **Nâng màu vô hướng thành vector tiếp tuyến:** h_i = MLP(ψ(q_i)) · log_p q_i ∈ ℂ^C, với C = 128 (Eq. 21). Đổi khung tại p thì log quay theo, nên h_i quay theo: **đẳng biến**.
3. **Token** ξ_p = mean_i [ MLP(ψ(p), ψ(q_i), |log_p q_i|) · log_p q_i ] (Eq. 22; code dùng thêm ψ(p) của tâm, `ring_vae.py:60-71`).
4. Chuỗi {h_i} ∪ {ξ_p} đi qua **8 lớp VN-Transformer** (attention đa đầu trên chiều mẫu + 2 "tangent vector neuron", có residual) ✅ (§9.1; `ring_vae.py:89-129, 161-162`).
5. Lấy token ra để dự đoán: **μ_p = Linear_ℂ(ξ_p) ∈ ℂ^d** (đẳng biến). **ln σ_p** được dự đoán từ một đặc trưng **bất biến** Re(ξ ⊙ conj(Lξ)) (Eq. 24–25; `ring_vae.py:132-175`).

**Điều kiện đẳng biến (Eq. 7):** dγ|_p · μ_p = μ_{γ(p)} và σ_p = σ_{γ(p)} (σ vô hướng). Khi đó, nhờ Eq. 2, z = μ + σ⊙ε cũng quay đúng như một vector ✅ (§8.1, Eq. 15).

#### 3.3 FL-VAE Decoder: điểm hay nhất của bài (✅ Eq. 4–5, 18; `ring_vae.py:209-240`)

Muốn biết màu tại điểm q trong một tam giác kề p, decoder là neural field F với hai loại đặc trưng **bất biến gauge**:
- **Bất biến nội tại:** phần tam giác trên của ma trận Hermit z_p z_p^*: |z_i|² (d số thực) và Re/Im của z_i z̄_j với i<j. Code ghép thành d + 2·d(d−1)/2 = **d² = 64 số thực** ✅ (`ring_vae.py:217-229`).
- **Tọa độ định vị:** c_pq = log_p q · z̄_p ∈ ℂ^d (Eq. 5). Với từng kênh i: Re(c) = ⟨log_p q, z_i⟩ (tích vô hướng), Im(c) = det(z_i, log_p q) ✅ (bài viết "inner product and determinant"). Đổi khung thì log và z quay cùng góc, nên tích với liên hợp **triệt tiêu góc quay**.
- F là **MLP 5 lớp, 512 kênh, SiLU** (đầu vào 64 + 16 = 80 số thực, đầu ra RGB) ✅ (§9.1; `latents/layers.py:138-153`).
- Mỗi điểm q trong tam giác (p₁,p₂,p₃) nhận **ba dự đoán** ψ_{p_k}(q). Ba dự đoán này chỉ được **trộn barycentric lúc inference** ✅ (§4). Lúc train FL-VAE, mỗi dự đoán được giám sát riêng bằng L1 trên one-ring (Eq. 26).

**Trực giác ⚠️:** mỗi latent z_i đóng vai một "khung cục bộ có hướng + độ lớn". Decoder biết q nằm ở đâu **so với hướng của từng latent**, nên có thể vẽ cạnh, sọc, gradient có hướng **bên trong** tam giác. Nội suy barycentric các đặc trưng bất biến (ablation "FL-VAE Bary.", `ring_vae.py:243-267`) chỉ cho được tín hiệu tuyến tính từng mảnh của các bất biến nên mờ hơn.

**Loss FL-VAE ✅ (Eq. 26–28):** L = ‖ψ(q) − ψ_p(q)‖₁ + λ·KL, với **λ = 0.01**. KL của Gauss phức: −½ Σ(1 + ln σ² − |μ|² − σ²). Mesh train được **cố ý thô hơn** mesh dùng lúc inference (10K đỉnh trên ảnh 512², so với 30K đỉnh lúc dùng), để mỗi one-ring chứa nhiều tần số cao hơn. Bài cho rằng việc này giúp bền với mật độ lấy mẫu ✅ (§6). Mesh phẳng sinh bằng farthest point sampling + Delaunay ✅ (§9.1). Khuyến nghị kèm theo: **remesh đều** mesh đích vì VAE không thấy one-ring méo lúc train ✅ (§9.1).

#### 3.4 Field Latent Diffusion Model (✅ Eq. 8–13, 29–39)

- **Forward:** Z_t = √ᾱ_t Z + √(1−ᾱ_t) ε, với ε ~ 𝒯𝒩_M(0, I_d) (Eq. 8); T = 1000 ✅ (§6).
- **Mục tiêu:** dự đoán nhiễu, ‖ε − ε_θ(Z_t, t, ρ)‖² (Eq. 10). ρ ∈ L²(M, ℝ^m) là embedding vô hướng theo đỉnh (nhãn, timestep).
- **Reverse:** DDPM chuẩn, các hệ số C₁, C₂, C₃ ở Eq. 29–31 ✅.
- **Đẳng biến (Eq. 12–13):** chỉ cần mạng thỏa γ·ε_θ(Z,t,ρ) = ε_θ(γZ, t, γρ). Chứng minh bằng quy nạp ở §8.2 ✅.
- **Scale latent (Eq. 38–39):** với X ~ 𝒯𝒩(0, I₁) thì E|X(p)| = √(π/2) (phân phối chi 2 bậc tự do), nên bài scale **từng kênh** Z_i ← √π / (√2·E|Z_i|) · Z_i để latent cùng thang với nhiễu ✅ (§9.2). Code làm khác, xem 5.4.

#### 3.5 Field convolution có embedding (✅ Eq. 32–37, §9.2)

Ký hiệu cực: X_c(q) = ϱ e^{iϕ_c(q)}, log ≡ r_pq e^{iθ_pq}, φ_pq là góc **vận chuyển song song** từ T_qM sang T_pM. Field conv:

  [X ∗ f]_{c'}(p) = Σ_c ∫_{B_p} ϱ_c(q) e^{i(ϕ_c(q) + φ_pq)} · f_{c'c}( r_pq e^{i(θ_pq − ϕ_c(q))}, π(q) ) dq   (Eq. 34–35)

Diễn giải:
- **Giá trị** được vận chuyển song song về khung của p (e^{iφ_pq}), nên cộng các vector ở các khung khác nhau là hợp lệ.
- **Đối số của filter** là vị trí tương đối **đo trong khung do chính đặc trưng X_c(q) định nghĩa** (θ − ϕ_c). Vì vậy nó bất biến gauge. Đây là ý tưởng "spatially adaptive alignment": đặc trưng tự chọn hướng cho filter.
- Phần mở rộng của bài: filter nhận thêm **embedding vô hướng π** (timestep, nhãn). Bài nói đã thử cộng embedding (phá đẳng biến) và nhân embedding (huấn luyện không ổn định), còn đưa embedding vào đối số filter thì vừa giữ đẳng biến vừa ổn định ✅ (§5).
- So với Field Conv gốc: support chỉ gồm các đỉnh kề trong one-ring; filter là **MLP 3 lớp, 8 kênh ẩn** (PointConv trick) thay vì cơ sở Fourier ✅ (§9.2).

⚠️ **Lưu ý ký hiệu:** Eq. 32 viết "log_p q ≡ r_pq e^{iθ_pq}". Nếu θ_pq đo trong khung của p còn ϕ_c(q) đo trong khung của q, thì hiệu θ − ϕ **không** bất biến khi đổi khung tại p. Để đẳng biến thật, đại lượng này phải là log_q p biểu diễn trong khung của q (như công thức "scatter" của bài Field Convolutions gốc), hoặc phải cộng thêm φ_pq. Code (`layers.py:180-192`) nhân `logs` với conj(khung của x tại láng giềng) mà **không** cộng góc vận chuyển, nên `logs_` trong dữ liệu tiền xử lý hẳn đã ở khung của láng giềng. Mình chưa đọc repo tiền xử lý để xác nhận.

#### 3.6 Kiến trúc mạng khử nhiễu (✅ §9.2; `sin_im_ddm/net.py:348-471`)

- **FCResNet block** (`latents/layers.py:366-411`): residual = Linear_ℂ(x). Nhánh chính: [VN-MLP có norm → FieldConv] → MLP(emb) → [VN-MLP có norm → FieldConv có emb]. Block cuối có thêm norm + activation.
- **U-Net 2 tầng, 1 lần downsample:** 2 block ở tầng mịn (đi xuống), 4 block ở tầng thô, 4 block ở tầng mịn (đi lên), **tổng 10** ✅. 96 kênh ở tầng mịn, 192 ở tầng thô ✅. Chỉ có **một** skip connection (x0) ✅ (`net.py:425-451`).
- **Receptive field ≈ 28-ring ≈ 6% diện tích** mesh 30K đỉnh ✅ (§9.2). ⚠️ Tự đếm: 2 block × 2 conv = 4 ring ở tầng mịn khi đi xuống, 4 block × 2 conv = 8 ring ở tầng thô (≈ 16 ring mịn nếu tầng thô có cạnh dài gấp 2), 4 block × 2 conv = 8 ring khi đi lên: 4 + 16 + 8 = 28. Khớp với con số của bài.
- **Pool/unpool:** mean-pool có trọng số diện tích và **vận chuyển song song**. Unpool là nearest neighbor kèm vận chuyển ✅ (`layers.py:590-620`).
- **Phi tuyến (Eq. 19):** kiểu Vector Neuron. X' = Q + ReLU(−Re(Q K̄))/(|K|²+δ) · K, tức là "nếu Q ngược hướng K thì bỏ thành phần dọc K". Code thay ReLU bằng SiLU ✅ (`layers.py:60-82`).
- **Chuẩn hóa (Eq. 20):** chỉ chia **độ lớn** cho căn của trung bình |X|² **có trọng số diện tích trên toàn mesh**, rồi nhân hệ số phức học được ✅ (`layers.py:29-57`). Không có bias ở bất kỳ đâu (bias phá đẳng biến) ✅.

#### 3.7 Train, inference, điều khiển

- **Train FLDM ✅ (§6, §9.2):** 500 bản remesh ~30K đỉnh (để mạng **không học thuộc connectivity**), encode mỗi bản một lần. Adam, LR 1e-3 → 1e-5 (cosine), batch 16, 300K iter, linear schedule. Code khác (5.4).
- **Inference:** DDPM 1000 bước trên một bản remesh **test** (khác triangulation lúc train) → decode ✅ (`train.py:1162-1248`).
- **Label-guided (§6.3):** nhãn theo mặt → nhãn theo đỉnh → embedding vô hướng π ✅.
- **Inpainting (§6.4, Eq. 14):** sau mỗi bước reverse, ở vùng mask giữ lại thì thay Z_{t−1} bằng latent gốc **được noise lại đến mức t−1**: Z_{t−1} ← m·Z^{gốc}_{t−1} + (1−m)·Z̃_{t−1}. Đây là RePaint **không** có bước resample. Biên được khớp nhờ tích chập lan thông tin ✅.
- **Chuyển texture (§6.5):** train trên M, sample trên N gần đẳng cự cục bộ. Chạy được cả giữa topology khác nhau (sọ genus 0 sang genus 3) ✅. Hình 11 cho thấy Complex Functional Maps hỏng khi topology đổi ✅.
- **Chỉnh tỉ lệ (§6.6):** khoảng cách giữa các đỉnh được chuẩn hóa bởi r = √(A/(π|V|)). Vì vậy remesh thô hơn thì họa tiết to hơn ✅ (Hình 7).

---

### 4. Kết quả & ablation quan trọng

**Bảng 1: nén/tái tạo texture**, 16 vật thể Google Scanned Objects, metric tính trong atlas; DSSIM và LPIPS nhân 10 ✅:

| Phương pháp | PSNR ↑ 30K (5K) | DSSIM ↓ | LPIPS ↓ |
|---|---|---|---|
| **FL-VAE** | **22.38** (20.59) | **0.51** (0.83) | **1.02** (1.81) |
| FL-VAE (Bary.), ablation bỏ log-coordinate | 21.33 (19.61) | 0.66 (1.01) | 1.31 (2.03) |
| INF [27] (một neural field cho mỗi mesh) | 18.86 (16.45) | 1.16 (1.64) | 2.15 (2.73) |

→ Log-coordinate thắng barycentric khoảng **+1 dB PSNR**, LPIPS giảm ~22% ✅. Lưu ý: FL-VAE là **một model pre-train trên ảnh phẳng dùng zero-shot**, còn INF được fit riêng cho từng mesh, nhưng vẫn thua xa ✅.

**Bảng 2: sinh không điều kiện**, 10 asset (9 Objaverse + 1 GSO, Bảng 4), 64 mẫu/asset, render 24 view ✅:

| | SIFID ↓ | LPIPS ↑ (đa dạng) |
|---|---|---|
| **FLDM** | **3.27** | 1.15 |
| Sin3DM | 6.58 | 2.20 |

Caption ghi SIFID và LPIPS "được nhân hệ số 10⁵ và 10" (bản trích xuất văn bản chỉ còn "105", mình hiểu là 10⁵) ⚠️. FLDM thua về LPIPS-đa-dạng, nhưng bài lập luận Sin3DM sinh cả geometry nên LPIPS bị thổi phồng. Ví dụ: mẫu Sin3DM bên phải có LPIPS 2.91 nhưng texture kém (SIFID 2.42 so với 0.71 của mẫu FLDM) ✅. ⚠️ Đây là lập luận định tính. Bài không có user study, không có đo đa dạng chỉ riêng texture.

**Ablation và quan sát khác:**
- **DiffusionNet thay cho field conv** (§9.2): train được, nhưng mẫu **ít đa dạng** và "răng cưa" (ragged). Bài giải thích: DiffusionNet có support toàn cục nên dễ học thuộc mẫu duy nhất, và chiếu lên cơ sở riêng tần thấp làm hỏng latent vốn không liên tục ✅. **Chỉ có kết quả định tính.**
- **Mesh train >10K đỉnh** cho mẫu đa dạng và đẹp hơn mesh thô ✅ (§9.2, định tính).
- Inpainting, label-guided, chuyển texture, chỉnh tỉ lệ: **chỉ có hình** (Hình 4–7, 9–11), không có số ✅.
- **Không có** ablation định lượng cho: số chiều latent d, receptive field, việc bỏ đẳng biến, số bản remesh.

---

### 5. Đọc code

#### 5.1 Bản đồ repo

```
mesh_diffusion/
├── latents/                     # Biểu diễn + các lớp dùng chung
│   ├── ring_vae.py      (267)   # FL-VAE: ring_encoder, Sampling, ring_decoder (log-coord), scalar_decoder (ablation Bary.)
│   ├── complex_attn.py  (466)   # MultiHeadAttention của Flax sửa cho số phức: softmax(Re(q·k̄))
│   ├── layers.py        (756)   # norm, VN-neuron, FieldConv(+Emb), FCEResBlock*, pool/unpool có transport, VQ (bỏ dở)
│   ├── dataset.py       (186)   # TFRecord encode/parse
│   └── io_utils.py       (70)   # đọc PLY có nhãn (face 'quality'), npz, png
├── sin_im_ddm/                  # Tên gây hiểu nhầm: thực ra là "lõi diffusion + mạng", KHÔNG phải baseline ảnh
│   ├── diffusion.py     (316)   # lịch beta (linear/cos/exp), DDPM/DDIM, noise tangent, inpaint
│   └── net.py           (896)   # fc_unet_2 (không nhãn), fc_unet_2_lab (có nhãn), _signal (HKS), _quantize, fc_next, fc_unet_3
└── sin_mesh_ddm/
    ├── main.py           (70)   # absl entry
    ├── train.py        (1248)   # encode latent trước → vòng train pmap → sample + decode atlas
    ├── input_pipeline.py(301)   # lấy dữ liệu conv/pool theo tầng, map nhãn qua remesh bằng kd-tree, HKS
    └── configs/default*.py      # mặc định + cấu hình train/sample sư tử biển có nhãn
```

⚠️ Lưu ý: `sin_im_ddm/` **không** chứa baseline single-image diffusion. Nó chứa module diffusion và mạng U-Net field-conv mà `sin_mesh_ddm/train.py` import (`train.py:45-46`).

#### 5.2 Đi qua từng file

**`latents/ring_vae.py`**
- `square_mag_norm_encoder` (33-49): chia độ lớn, thống kê theo **axis (0,1)**, tức **qua mọi đỉnh trong batch và mọi mẫu**. Đây là thống kê theo batch (xem 5.4).
- `init_conv` (52-71): tạo token = trung bình của MLP(màu tâm, màu mẫu, |log|) · log (Eq. 22).
- `complex_mhattn`/`quad_mhattn` (74-129): pre-norm → attention → residual → norm → 2×`dense_neuron` → residual. Lặp 4 lần trong mỗi `quad_mhattn`, gọi 2 lần nên tổng **8 lớp** ✅.
- `ring_encoder` (141-175): nâng màu (MLP 3 lớp × log), bỏ mẫu 0 (tâm, log = 0), nối token vào cuối, chạy 8 lớp, lấy token ra. `mean` = Dense phức không bias. `ln_var` = tanh(đặc trưng bất biến) → Dense(2d) → clip ≤ 3 → trả về **số phức (log-var riêng cho phần thực và phần ảo)**.
- `Sampling` (183-201): reparameterization trên (Re, Im) riêng rẽ.
- `ring_decoder` (209-240): bất biến Hermit + `pix_logs * conj(vlatents[pix_tri])` → `mlp5`.
- `scalar_decoder` (243-267): ablation "Bary.": nội suy barycentric các bất biến rồi đưa vào MLP.

**`latents/complex_attn.py`**: bản sao `flax.linen.MultiHeadDotProductAttention`. Chỉ đổi phần trọng số: `jnp.real(einsum(q, conj(k)))` (92-99). Giá trị V vẫn là số phức, nhân với trọng số softmax thực. Phần còn lại (decode cache, mask) không dùng.

**`latents/layers.py`**: xem 5.3. Có thêm `INF` (156-172, MLP có skip + sigmoid, có lẽ dùng cho baseline INF), các biến thể block đã thử (`FCEResBlock2`: cộng embedding chiếu lên `bases`; `FCEResBlock_mult`: **nhân** embedding phức, đúng biến thể bài nói "không ổn định"; `FCENextBlock`), và VQ `quantized_projection`/`quantized_embedding` (645-756). **Pipeline chính không dùng** các phần này ⚠️ (grep: chỉ `fc_unet_2_quantize` dùng `quantized_embedding`).

**`sin_im_ddm/diffusion.py`**: `linear_beta` (57-60, β từ 1e-4 đến 2e-2 theo căn), `cosine_beta` (63-72, s = 0.008, clip 0.999), `exp_beta`. `predict_prev` = DDPM posterior chuẩn (168-182). `encode_batched_tangent_latent_samples` (271-294): mỗi mesh trong batch lấy **một** t. `encode_inpaint_mask` (297-316).

**`sin_im_ddm/net.py`**: `fc_unet_2_lab` (348-471) là mạng của bài. `fc_unet_2_signal` (474-596) điều kiện theo tín hiệu phổ (HKS/eigen); bài không nhắc đến biến thể này ⚠️.

**`sin_mesh_ddm/input_pipeline.py`**: `get_conv_data` (73-120) lấy theo tầng `hei_mass_l` (diện tích đỉnh), `hei_neigh_l` (chỉ số láng giềng), `logs_l` (log tới láng giềng), `xport_l` (góc vận chuyển), `unpool_adj_l_l+1`/`unpool_vals_l_l+1` (ánh xạ tầng + góc), cắt còn `k_conv = 8` láng giềng. `map_mask` (187-202): mỗi mặt được lấy 15 điểm đều (`regular_samples(4)`), sau đó tra kd-tree NN từ đỉnh của từng bản remesh để chuyển nhãn. `compute_hks` (250-258): heat kernel signature.

**`sin_mesh_ddm/train.py`**: `create_train_state` (72-178) nạp **trọng số encoder/decoder đã train sẵn** (khóa `Checkpointring_encoder_0`; tiền tố "Checkpoint" cho thấy lúc train FL-VAE họ bọc module bằng `nn.checkpoint` ⚠️). `train_step` (198-269): L1 loss + `conj` gradient. `sample_latents` (379-451). `reconstruct` (531-578). `train_and_evaluate` (662-1248): encode 500 mesh **một lần** → tính sigma → vòng train → sample định kỳ.

#### 5.3 Những điểm hay nhất trong code

**(1) Khung tiếp tuyến do chính đặc trưng định nghĩa, có xử lý vector 0**: `latents/layers.py:180-192`
```python
mask = 1.0 * jnp.greater(jnp.abs(x), 1.0e-4)
angle = jnp.angle(mask * x + (1.0 - mask))           # |x|≈0 → góc 0
frame = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))
return complex(logs[...,0], logs[...,1])[..., None] * jnp.conjugate(frame)[neigh, :]
```
Mỗi **kênh** c tại mỗi láng giềng có khung riêng, quay theo hướng của X_c(q). Vị trí tương đối được viết trong khung đó nên bất biến gauge. Vector gần 0 không có hướng, nên code gán góc 0 để tránh gradient NaN của `angle(0)`. Đây là chi tiết dễ quên khi tự viết lại.

**(2) Vận chuyển song song gộp vào "trọng số" phức**: `sin_im_ddm/net.py:376-389` và `latents/layers.py:207, 220`
```python
mass[0][neigh[0]] * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0]))   # w_pq = A_q · e^{iφ_pq}
weights = weights / jnp.sum(jnp.abs(weights), axis=1, keepdims=True)
feat = jnp.sum(x[neigh, :, None] * weights[..., None, None] * rel, axis=1)
```
Tích phân trên B_p trong Eq. 34 thành tổng có trọng số diện tích. Phép quay vận chuyển nằm sẵn trong pha của trọng số, nên chỉ cần **một phép nhân phức**. Chuẩn hóa bằng |w| nên pha không ảnh hưởng tổng trọng số.

**(3) Filter = MLP thực dùng chung + trộn kênh bằng Dense phức (PointConv trick)**: `latents/layers.py:210-226`
MLP 2 lớp (`hdim` kênh) biến vị trí tương đối (Re, Im) thành `hdim` hàm cơ sở **thực**. Các hàm cơ sở này **dùng chung cho mọi kênh** (Dense chỉ tác động trục cuối). Sau đó `Dense(features, use_bias=False, param_dtype=complex64)` trộn các cặp (kênh, cơ sở). Kết quả tương đương f_{c'c}(u) = Σ_h W_{c',(c,h)} φ_h(u_c) với W phức: filter phức đầy đủ C'×C nhưng chỉ tốn một MLP nhỏ.

**(4) Tiêm embedding vào đối số filter, theo từng kênh**: `latents/layers.py:271-280`
```python
jnp.concatenate((jnp.real(rel)[..., None], jnp.imag(rel)[..., None],
                 jnp.tile(emb[:, None, :, None], (1, rel.shape[1], 1, 1))), axis=-1)
```
Đầu vào của MLP filter cho kênh c là bộ ba (Re u_c, Im u_c, π_c). Embedding đã được chiếu về đúng `features` kênh (`layers.py:401-402`), mỗi kênh nhận **một số vô hướng riêng**. Cách này rẻ (filter chỉ có 3 đầu vào) và giữ đẳng biến vì π bất biến. Bài nhấn mạnh đây là cách duy nhất vừa đẳng biến vừa ổn định (§5). Biến thể nhân embedding còn nằm trong `FCEResBlock_mult` (`layers.py:469-520`).

**(5) Nhãn được nhúng bằng chính hàm sinusoidal của timestep, theo từng đỉnh, ở từng tầng**: `sin_im_ddm/net.py:401-415`
```python
lab_0 = batched_timestep_embedding(labels[0] + 1, emb_features)
emb_0 = jnp.concatenate((jnp.tile(emb[None, :], (labels[0].shape[0], 1)), lab_0), axis=-1)
```
Nhãn nguyên được coi như một "timestep" (cộng 1 để nhãn 0 vẫn khác vector 0). Kết quả nối với embedding t rồi qua MLP, cho mỗi đỉnh một embedding riêng. Nhãn ở tầng thô lấy qua chỉ số `hei_1` (`input_pipeline.py:112-118`).

**(6) Vector Neuron "mềm" cho số phức**: `latents/layers.py:69-82`
```python
dot = q.real * k.real + q.imag * k.imag          # Re(q·k̄) = <q,k> trong mặt phẳng tiếp tuyến
k2  = k.real * k.real + k.imag * k.imag
k   = (jax.nn.silu(-1.0 * dot) / (k2 + EPS)) * k
return q + k
```
Đây là phi tuyến **đẳng biến U(1)** duy nhất cần dùng. Khi ⟨q,k⟩ < 0, hàm này gần như chiếu bỏ thành phần dọc k. SiLU thay ReLU cho mượt.

**(7) Norm theo diện tích, hệ số phức**: `latents/layers.py:45-57`. `mass / sum(mass)` biến trung bình thành **tích phân trên bề mặt**, nên mesh dày hay thưa ở một vùng không làm lệch thống kê (bất biến remesh). `scale` là số phức (quay toàn cục mọi vector cùng góc), vẫn đẳng biến vì U(1) giao hoán.

**(8) Attention đẳng biến chỉ bằng một dòng**: `latents/complex_attn.py:92-99`. `softmax(Re(q·k̄)/√d)` bất biến khi mọi token quay cùng góc (mọi mẫu của one-ring cùng nằm trong khung của p). V phức nhân trọng số thực nên đẳng biến. Attention **trên tập mẫu** cũng bất biến hoán vị ✅ (§9.1).

**(9) Decode từng đỉnh rồi mới trộn barycentric (chỉ lúc inference)**: `sin_mesh_ddm/train.py:543-567`
```python
pix_tris = jnp.reshape(pix_tris, (pix_tris.shape[0], -1))          # (B, P*3): 3 đỉnh/texel
pix_logs = jnp.reshape(pix_logs, (pix_logs.shape[0], -1, 2))       # log_{p_k} texel
...
pix_recon = np.reshape(pix_recon, (pix_recon.shape[0], -1, 3, 3))
pix_recon = np.sum(pix_recon * pix_bary[..., None], axis=2)
```
Mỗi texel được decode **3 lần** (một lần từ mỗi đỉnh), sau đó trộn theo barycentric. Decode chia lô 100K điểm (`train.py:551-564`).

**(10) Encode latent một lần, dùng giá trị trung bình, không chạy VAE trong vòng train**: `sin_mesh_ddm/train.py:843-857`. Latent của 500 mesh được tính trước (dùng `l_mean`, **không sample**) và cache vào `tf.data`. Encoder chạy theo 4 lô đỉnh để vừa bộ nhớ (803-804). Vòng train chỉ còn U-Net.

**(11) Batch = nhiều mesh, mỗi GPU 1 mesh + tích lũy gradient**: `sin_mesh_ddm/train.py:681-684, 161-162`. `num_multi_steps = batch_size // số_GPU`, sau đó `optax.MultiSteps`. Mỗi "mẫu" là **cả một mesh 30K đỉnh** với một timestep. ⚠️ Vì vậy `num_steps` đếm micro-step: với 8 GPU và batch 16 thì 200K micro-step chỉ là ~100K lần cập nhật.

**(12) Remat cho field conv**: `latents/layers.py:318, 348`: `nn.checkpoint(FieldConv)`. Đây chính là chi tiết "rematerialize tensors" làm train chậm trong §9.2 ✅.

**(13) Gradient phức phải lấy liên hợp**: `sin_mesh_ddm/train.py:241`: `grads = jax.tree.map(jnp.conj, grads)`. JAX trả về liên hợp của hướng giảm dốc nhất cho tham số phức. Nếu quên dòng này, Adam sẽ đi sai hướng. Đây là bẫy kinh điển khi train mạng phức trong JAX.

**(14) Pool/unpool có vận chuyển**: `latents/layers.py:590-620`. Downsample: trọng số = diện tích × e^{iφ} khi scatter vào đỉnh thô. Tổng trọng số được tính **trước** khi gắn pha (`610-616`). Unpool: lấy đỉnh thô gần nhất rồi quay bằng `xport[:, 0, 1]`.

**(15) Chuyển nhãn giữa các triangulation bằng kd-tree trên điểm lấy mẫu đều trong mặt**: `sin_mesh_ddm/input_pipeline.py:153-202`. Cách này đơn giản và bền: không cần correspondence giữa 500 bản remesh.

#### 5.4 Khác biệt paper và code

| # | Paper | Code | Ghi chú |
|---|---|---|---|
| 1 | Loss L2 (Eq. 10) | **L1** trên ε (`train.py:229-233`) | ✅ |
| 2 | Linear schedule (§6) | `ddm_schedule = 'cos'` trong mọi config (`configs/default.py:77`, `default_train_labeled.py:34`) | ✅ |
| 3 | LR 1e-3 → 1e-5, 300K iter | LR **5e-4**, warmup 1000, **200K** bước (ghi chú "SEAL") (`default.py:88-100`) | ✅ |
| 4 | Filter MLP 8 kênh ẩn; VN-MLP 3 lớp | `hdim = 16`, `mlp_layers = 2` trong config train/sample (`default_train_labeled.py:35-36`) | ✅ |
| 5 | Scale latent **theo kênh** cho E\|Z_i\| = √(π/2) (Eq. 39) | **Scale nhiễu** σ = √2·mean\|z\| **toàn cục** (`train.py:864-865`, `diffusion.py:284`) | ⚠️ Tự tính: nhiễu có E\|ε\| = σ√(π/2) = √π·mean\|z\| ≈ 1.77·mean\|z\|, còn paper cho tỉ lệ 1. Tức là SNR theo phương sai của code thấp hơn ~π lần. Có thể đây là lựa chọn có chủ ý (dịch lịch về phía nhiều nhiễu). |
| 6 | Mean: MLP VN 2 lớp; ln σ: MLP 2 lớp, **σ vô hướng** | Mean: 1 Dense phức; ln σ: tanh → 1 Dense → **log-var riêng cho Re và Im** (`ring_vae.py:168-175`) | ⚠️ σ khác nhau giữa Re và Im thì Gauss không đẳng hướng, nên Eq. 7 không còn đúng tuyệt đối. Ảnh hưởng nhỏ vì FLDM chỉ dùng μ. |
| 7 | "Linear layer trong attention dùng trọng số phức" | `ComplexMultiHeadDotProductAttention` để `param_dtype = float32` mặc định (`complex_attn.py:231`), và `nn.Dense` trong `complex_mhattn`/`directional_features` cũng thực | ✅ Vẫn đẳng biến (ma trận thực giao hoán với e^{iβ}), chỉ kém biểu diễn hơn. |
| 8 | Norm "độc lập batch" (Eq. 20) | Encoder chuẩn hóa qua **mọi đỉnh trong batch** (`ring_vae.py:43-45`) | ⚠️ Latent của một đỉnh phụ thuộc nhẹ vào các đỉnh cùng lô (encode chia 4 lô, `train.py:803`). |
| 9 | Phi tuyến ReLU (Eq. 19) | SiLU | ✅ |
| 10 | "Sampling from the distributions" (§5) | Dùng **μ** (`train.py:851`) | ✅ |
| 11 | — | `use_ema=True` sẽ lỗi: `state.ema_state` không có trong `TrainState` (`train.py:50-58, 1048`) | ✅ (mặc định `use_ema=False`) |
| 12 | — | `state.sigma` khởi tạo 1.0 nên nhánh `compute_sigma` không bao giờ chạy (`train.py:176, 709-712`). Sigma luôn được tính lại từ latent ở mỗi lần chạy. | ✅ Không phải bug với cùng dữ liệu, nhưng sigma không lưu trong checkpoint. |
| 13 | — | `quantized_projection`: `theta = arctan2(β, α) + π·[α<0]` (`layers.py:666-672`). arctan2 đã cho đủ góc, cộng thêm π sẽ chọn **góc xa nhất**. | ⚠️ Có vẻ là bug, nhưng nằm trong code chết. |

#### 5.5 Hyperparameter thực tế (theo config train sư tử biển có nhãn)

| Nhóm | Giá trị | Nguồn |
|---|---|---|
| Latent | d = 8 phức/đỉnh; ~30K đỉnh/mesh; 500 mesh train, 16 mesh test | `default.py:27, 35, 114-115` |
| FL-VAE | encoder C = 128, 8 head, 64 mẫu/one-ring; decoder MLP 5 lớp × 512 | `train.py:87-90`; §9.1 |
| U-Net | 96 / 192 kênh; `k_conv` = 8 láng giềng; `hdim` = 16; `mlp_layers` = 2; emb dim = 2·96 = 192 | `default_train_labeled.py:35-37`; `net.py:398` |
| Diffusion | T = 1000, cosine, ε-pred, L1 | `default.py:107`; `train.py:229` |
| Tối ưu | Adam(0.9, 0.999, 1e-8), LR 0 → 5e-4 (warmup 1000) → 1e-5 cosine, batch 16, 200K bước, không EMA | `default.py:84-112` |
| Sampling | DDPM 1000 bước, 64 mẫu | `default.py:32` |
| Texture | atlas 1024² mặc định | `input_pipeline.py:61` |

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Mỗi mesh một model** (single-example). Train ~2 ngày trên 8×V100 cho **một** texture ✅ (§9.2). Không có prior chung giữa nhiều vật thể. Bài tự nêu "thời gian train dài" là hạn chế ✅ (§7).
2. **Không điều khiển được hướng** của họa tiết sinh ra. Đây là hệ quả trực tiếp của tính đẳng biến ✅ (§7). Bài đề xuất điều kiện bằng trường vector người dùng (chưa làm).
3. **Chỉ texture, không geometry, không ánh sáng.** Texture học từ scan chứa **shading bị nướng sẵn**, và FLDM tái tạo luôn shading đó ⚠️.
4. **Đánh giá mỏng:** Bảng 2 chỉ 10 asset, so với đúng một baseline (Sin3DM). Không có số cho inpainting/transfer/label. Không có ablation định lượng cho đẳng biến, d hay receptive field ✅.
5. **Chi phí hạ tầng:** tiền xử lý tạo **65 GB** cho một mesh (500 remesh kèm cấu trúc láng giềng/log/transport của 2 tầng) ✅ (README). Pipeline JAX + TFRecord, field conv không có kernel tối ưu.
6. **Sampling chậm:** DDPM 1000 bước, ~10 phút ✅. Code có hàm DDIM (`diffusion.py:202-249`) nhưng không dùng.
7. **Phụ thuộc chất lượng mesh:** cần remesh đều. One-ring méo nằm ngoài phân phối train của VAE ✅ (§9.1).
8. **Câu hỏi mở ⚠️:** (a) Có thật cần đẳng biến khi mesh có tham số hóa chuẩn (template cố định)? (b) Tỉ lệ nhiễu/latent của code khác paper, ảnh hưởng thế nào? (c) Log-coordinate decoder còn hơn barycentric bao nhiêu khi mật độ đỉnh xấp xỉ mật độ pixel?

---

### 7. Liên hệ với GReg

#### 7.1 Khác biệt nền tảng cần nói thẳng ⚠️

- Bài này giải bài toán **"bề mặt không có khung chuẩn, triangulation tùy ý"**. SMPL-X thì ngược lại: **topology cố định (10,475 đỉnh), có UV chuẩn, có tương ứng đỉnh-đỉnh giữa mọi tư thế và mọi người**. A1 của GReg đã khai thác đúng điều này (barycentric cùng mặt giữa mesh nguồn và mesh đích, `greg/geometry/mesh_flow.py:128-160`). Với template cố định, ta có thể chọn khung tiếp tuyến **cố định theo đỉnh của template** (ví dụ từ cạnh hoặc từ UV) rồi để nó biến dạng theo mesh. Khi đó vấn đề gauge biến mất, toàn bộ bộ máy U(1)/số phức/field conv **không cần thiết**.
- Đẳng biến isometry của bài tương ứng với một tính chất có thật: khớp nối người **gần đẳng cự** trừ vùng quanh khớp. Nhưng GReg đã có tương ứng **chính xác** qua topology chung, nên không cần "học" bất biến tư thế.
- Bài là **single-example**. GReg cần prior **xuyên người** (nhiều signer) và **trong vài trăm ms/khung**, không phải 10 phút.

#### 7.2 Những ý tưởng đáng lấy (xếp theo mức khả thi)

**(A) "Bộ nhớ texture trên bề mặt" để chở ngoại hình từ nguồn sang đích và giữ nhất quán thời gian.** ⚠️ Khả thi cao, rẻ.
- Chiếu các pixel nhìn thấy của I_s lên SMPL-X (qua `face_idx`/`bary` mà GReg đã rasterize được, `greg/render/rasterize.py:25, 139-147`) thành **texture theo đỉnh/UV** kèm mặt nạ "đã quan sát".
- Render texture đó vào khung đích. Về hình học, kết quả **tương đương** warp A1 khi chỉ có một nguồn. Lợi ích thật xuất hiện khi **tích lũy nhiều khung**: trong H1 sinh theo cửa sổ 49/81 khung, vùng lòng bàn tay **đã được sinh** ở cửa sổ trước có thể được ghi vào bộ nhớ texture và render làm điều kiện cho cửa sổ sau. Như vậy lòng bàn tay trông giống nhau giữa các cửa sổ, nhất quán theo thời gian, theo kiểu "3D-consistent".
- Dùng như **kênh điều kiện bổ sung** cho H1 (bên cạnh I_st, V_t): một ảnh "render từ bộ nhớ" + mặt nạ "đã từng quan sát".
- ⚠️ Mật độ: tự ước lượng ở 256², mỗi đỉnh nhìn thấy của tay ứng với khoảng 1–3 pixel, nên texture theo đỉnh **đã gần đủ** độ phân giải. Ở 480² thì cần UV atlas hoặc latent theo đỉnh kèm decoder dưới mức tam giác, xem (C).

**(B) Prior inpainting trên bề mặt cho vùng chưa từng thấy (lòng bàn tay, mặt trong cẳng tay).** ⚠️ Khả thi trung bình. Nên làm **trong không gian UV/đỉnh của SMPL-X**, không theo kiểu FLDM.
- Cơ chế inpainting Eq. 14 (thay latent đã nhiễu ở vùng quan sát) áp dụng được cho bất kỳ diffusion nào trên miền bề mặt. Nhưng prior phải học **từ nhiều signer**. 4.6k clip, mỗi clip chiếu được một phần texture, có thể tạo cặp (texture một phần, texture gộp từ cả clip) để train mô hình **hoàn thiện texture tay/da**. Mục tiêu train là texture gộp từ mọi khung trong clip, nơi lòng bàn tay thường lộ ra ở các khung khác.
- Hai bài được trích dẫn sát hơn nhiều so với FLDM: **DINAR** [51] (ICCV 2023, diffusion inpainting neural texture cho avatar một ảnh, trên UV SMPL-X) và **Grigorev et al.** [20] (CVPR 2019, *coordinate-based texture inpainting for pose-guided human image generation*, gần như đúng bài toán của GReg). → **Nên đọc hai bài này trước khi đầu tư hướng (B).**
- Bài học từ FLDM có thể chuyển sang: UV atlas có **đường nối (seam)** làm hỏng U-Net 2D (Point-UV diffusion, §2 ✅), mà bàn tay SMPL-X là các đảo UV nhỏ cắt quanh cổ tay và ngón ⚠️. Diffusion **trên đỉnh** (graph/mesh conv trên topology cố định, receptive field cục bộ) tránh được seam. Nếu dùng UV thì cần padding/xử lý seam.
- Rủi ro lớn nhất ⚠️: **shading bị nướng vào texture**. Lòng bàn tay quay về camera có ánh sáng khác mu bàn tay. Texture chép từ nguồn sang tư thế mới sẽ sai sáng tối. Vì vậy texture bề mặt chỉ nên là **điều kiện**, còn renderer cuối (H1 video DiT hoặc generator GReg cùng normal map) phải tự chỉnh shading. Không dán thẳng texture lên ảnh output.
- **Baseline rẻ nên thử trước ⚠️:** điền lòng bàn tay bằng thống kê màu da của phần tay nhìn thấy (mean/cov theo signer) cộng một "palm texture trung bình" trên UV, rồi render làm điều kiện. Nếu baseline này đã giảm mờ đáng kể thì không cần diffusion bề mặt.

**(C) Decoder theo tọa độ cục bộ thay vì nội suy barycentric.** ⚠️ Khả thi cao nhưng lợi ích có thể nhỏ ở 256².
- Khi render đặc trưng theo đỉnh (neural texture) vào ảnh, GReg hiện chỉ có `interpolate_vertex_attribute` (barycentric). Bảng 1 cho thấy giải mã **từng đỉnh với vị trí tương đối của pixel** rồi mới trộn thắng barycentric ~1 dB ✅. Với template cố định, "vị trí tương đối" chỉ cần là (u,v) cục bộ trong khung cố định của đỉnh, **không cần số phức**.
- Đáng thử nếu chuyển lên 480/512, khi một tam giác tay phủ nhiều pixel.

**(D) Chi tiết kỹ thuật nhỏ lấy được:**
- Tiêm điều kiện **vào đối số filter/attention** thay vì cộng vào đặc trưng, khi cần giữ một bất biến nào đó ✅ (§5). Với GReg thì ít cần.
- Inpainting kiểu "thay vùng đã biết bằng bản noise lại" (Eq. 14) là cách **không cần train** để ép video DiT của H1 giữ đúng vùng nhìn thấy (V_t ≈ 1) trong latent. Nó bổ sung cho việc dán pixel sau cùng ở §1.4 của `docs/diffusion_improvement.md` ⚠️. Lưu ý: latent Wan nén 8× không gian, nên mask phải downsample và sẽ bị nhòe biên.
- Chuẩn hóa thống kê **theo diện tích bề mặt** (`layers.py:45-57`) khi tính loss/metric trên mesh, để vùng mesh dày (tay, mặt SMPL-X rất dày) không áp đảo ⚠️.
- Receptive field nhỏ chống học thuộc trên dữ liệu ít ✅ (§5). Đáng nhớ nếu train một prior texture tay nhỏ trên ~4.6k clip.

#### 7.3 Những thứ KHÔNG nên lấy ⚠️

- Mạng số phức, field convolution, VN-Transformer, tiền xử lý log map/vận chuyển song song: chỉ cần khi **không** có khung chuẩn. SMPL-X có khung chuẩn. Chi phí cao (65 GB/mesh, remat, JAX) và không có lợi ích tương xứng.
- Mô hình **single-example**: không hợp với bài toán nhiều signer, và train mỗi signer 2 ngày là không thực tế.
- DDPM 1000 bước: ngược với mục tiêu thời gian thực của H2.
- Chuyển texture bằng sample model trên mesh khác: GReg đã có tương ứng chính xác, không cần.

#### 7.4 Kết luận cho luận văn ⚠️

Bài này **không phải** hướng chính cho GReg. Giá trị của nó là (1) **lập luận** rằng biểu diễn ngoại hình **trên bề mặt** tránh được bất nhất giữa các view và tránh được seam của UV, và (2) bằng chứng định lượng rằng **decoder theo tọa độ cục bộ** thắng nội suy barycentric (Bảng 1). Trong luận văn có thể dùng bài này làm trích dẫn nền khi đề xuất một **bộ nhớ texture SMPL-X** làm điều kiện phụ cho H1 (mục 7.2-A), nhằm giải quyết giới hạn 3 (nhất quán thời gian) và một phần giới hạn 1 (vùng bị che). Với phần sinh vùng chưa từng thấy, hãy đọc **DINAR [51]** và **Grigorev 2019 [20]** trước.

---

## Bài 4 — Dexterous World Models (CVPR 2026)

Quy ước: ✅ = đã kiểm chứng trong paper (mục/bảng) hoặc code (`file:dòng`, đường dẫn tương đối tính từ gốc repo `dwm/`); ⚠️ = suy luận / phỏng đoán của người viết, cần kiểm tra lại.

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Byungjun Kim*, Taeksoo Kim* (đóng góp ngang nhau), Junyoung Lee, Hanbyul Joo (tác giả liên hệ). Seoul National University + RLWRLD ✅ |
| Paper | CVPR 2026 (bản Open Access của CVF), arXiv 2512.17907 ✅ (README) |
| Project | https://snuvclab.github.io/dwm/ |
| Code | https://github.com/snuvclab/dwm, giấy phép **Apache-2.0** ✅. Bản clone chỉ có 1 commit (bản phát hành sạch, không có lịch sử) ✅ |
| Checkpoint | `byungjun-kim/DWM-CogVideoX-Fun-5b-LoRA` và `byungjun-kim/DWM-Wan2.1-Fun-14b-LoRA` trên HF ✅ (`training/cogvideox/README.md:161`, `training/wan/README.md:89`). Tập eval "Real-World Dynamic" (48 mẫu Aria) phát hành trên HF ngày 26/10/2026 ✅ (README) |
| Backbone | Paper: CogVideoX-Fun (inpainting) [1]. Code: `alibaba-pai/CogVideoX-Fun-V1.1-5b-InP` (bản chính) và thêm `Wan2.1-Fun-V1.1-14B-InP` (bản WAN, **không có trong paper**) ✅ |
| Fine-tune | LoRA rank 64 + huấn luyện toàn phần lớp `patch_embed.proj` đã mở rộng kênh ✅ (`training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml:13-31`) |
| Compute train | **Paper không nêu.** Code: "5B thường cần GPU lớp 80 GB" ✅ (`README.md`, Notes). Config: batch 7/GPU × accum 2, mặc định slurm 4 GPU (⚠️ nếu đúng là 4 GPU thì batch hiệu dụng = 56), `max_train_steps: 15000`; checkpoint phát hành có tên `..._checkpoint-8000` ✅ (`training/cogvideox/examples/infer_realworld.sh:7`) → ⚠️ mô hình trong paper có lẽ là bước 8000 |
| Compute infer | 50 bước, CFG 6.0, 49 khung 480×720, 8 fps ✅ (`training/cogvideox/inference.py:155-160`). Không có số thời gian |
| Supplementary | PDF chỉ có bài chính (8 trang + tham khảo); phụ lục không nằm trong file ✅ |

---

### 1. TL;DR

- Bài toán: cho **video cảnh tĩnh** render theo quỹ đạo camera ego + **video render mesh tay** theo cùng camera + câu mô tả, sinh video tương tác (tay cầm, mở, di chuyển vật) mà vẫn giữ cảnh và chuyển động camera ✅ (§3.2, Hình 2).
- Ý chính: khởi tạo từ một **mô hình video inpainting** (CogVideoX-Fun-InP). Đưa video cảnh tĩnh vào khe "video đã che" và đặt mask = "biết hết". Khi đó mô hình gốc gần như là **hàm đồng nhất có prior sinh**. Fine-tune chỉ cần học **phần dư** do tay gây ra ✅ (§3.2, Eq. 6).
- Điều kiện tay: render mesh tay (trái xanh lá, phải đỏ, shading Phong, nền đen), **VAE-encode rồi ghép kênh** vào latent. Cột trọng số mới của patch-embed **khởi tạo 0** ✅ (code).
- Dữ liệu lai: TRUMANS tổng hợp (có cặp tĩnh/động/tay khớp tuyệt đối) + TASTE-Rob thật (camera cố định → video tĩnh = **lặp khung đầu**, tay từ HaMeR) ✅ (§3.3).
- Với GReg: đây là **bản thiết kế gần nhất với H1** đã được kiểm chứng. Có thể gần như bê nguyên kiến trúc "InP backbone + khe masked-video + kênh concat zero-init + LoRA" sang Wan2.1-Fun-1.3B-InP. Thay "video tĩnh" bằng **video nguồn đã warp bằng mesh (A1)**, và (khác DWM) dùng **mask nhìn thấy thật** thay vì mask toàn 1.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Đầu vào** ✅ (§3.2, Eq. 5):
- `Π(S0; C_1:F)`: video cảnh tĩnh render theo quỹ đạo camera ego.
- `Π(H_1:F; C_1:F)`: video render mesh tay (chỉ bàn tay) theo cùng quỹ đạo.
- `T`: câu mô tả.

**Đầu ra:** video `V_1:F = Π(S0 + ΔS_1:F; C_1:F)`, tức cảnh sau khi bị tay tác động, nhìn từ cùng camera.

**Lập luận hình thức** ✅ (§3.1, Eq. 2–4): các world model kiểu image-to-video `p(V | I0, A)` phải **vừa** sinh lại cảnh tĩnh S0 (từ một khung) **vừa** sinh động lực. Hai việc bị trộn vào nhau nên mô hình dễ "sửa" luôn cả phần cảnh không bị tác động. DWM điều kiện trực tiếp lên S0 (qua render) nên chỉ còn phải mô hình hóa ΔS.

**Vì sao khó:**
1. Không có dữ liệu thật dạng cặp (cảnh tĩnh + video tương tác cùng quỹ đạo camera ego) ✅ (§3.3).
2. Text không đủ chính xác để chỉ định cấu hình tay và thời điểm ✅ (§1).
3. Phải vừa đồng nhất (giữ cảnh) vừa sinh (vùng vật bị di chuyển, vùng bị tay che/lộ) ✅ (§3.2).
4. ⚠️ Render tay không bị cảnh che khuất (xem §5.3 mục 6), nên mô hình phải tự suy ra tay nằm trước hay sau vật.

---

### 3. Phương pháp chi tiết

#### 3.1 Biểu diễn
- **Cảnh tĩnh:** video RGB render cảnh không có thay đổi trạng thái vật, cùng quỹ đạo camera ✅ (§3.3). Với dữ liệu camera cố định: lặp khung đầu `V0` F lần ✅ (Eq. 8).
- **Tay:** video render mesh tay, mỗi tay một màu, trên nền đen ✅ (Hình 2, code `data_processing/trumans/hand_rendering_common.py:349-352`). Paper ghi tham chiếu SMPL-X/MANO [32, 37]. **Code TRUMANS lại render mesh nhân vật `CC_Base_Body`** (Character Creator), chỉ giữ các vertex group bàn tay ✅ (`hand_rendering_common.py:20, 50-65`).
- **Text:** caption ngôi thứ nhất do InternVL2 sinh **từ video GT**, sau đó viết lại bằng Llama-3.1-8B ✅ (`data_processing/video_caption/prompt/caption_ego.txt`, `caption_rewrite.py:31`).

#### 3.2 Kiến trúc
Theo Hình 2 và code CogVideoX ✅:

```
video tĩnh ──VAE──► c_s (16 kênh)  ┐
video tay  ──VAE──► c_h (16 kênh)  ├─ concat theo kênh ─► patch_embed (Conv2d 2×2, 49→3072) ─► DiT CogVideoX-5B (LoRA) ─► VAE dec
z_t (16) + mask (1, toàn 1×0.7)    ┘                         ▲ text tokens (T5, 226)
```

- Thứ tự kênh khi train: `[noisy(16), mask(1), static(16), hand(16)]` = 49 kênh ✅ (`training/cogvideox/train_dwm_cogvideox.py:798-802`).
- CogVideoX-Fun-InP gốc có `in_channels` = 33 (16 nhiễu + 1 mask + 16 masked video). DWM nối thêm 16 kênh tay. Cột trọng số mới = 0 ✅ (`training/cogvideox/models/cogvideox_fun_transformer_with_conditions.py:1275-1281`).
- **Video tĩnh đi vào đúng khe "masked video" của mô hình inpainting.** Mask = toàn "đã biết". Lúc bắt đầu fine-tune, mô hình vì vậy chép lại video tĩnh (đồng nhất có prior) ✅ (§3.2 + pipeline `training/cogvideox/pipeline/cogvideox_fun_static_to_video_pose_concat_pipeline.py:2116-2146`).

#### 3.3 Công thức chính
- Eq. 3: `p(V | S0, A) = p(V | S0, {C, H})`.
- Eq. 5: xấp xỉ bằng render ego: `p(V | Π(S0;C), Π(H;C), T)`.
- Eq. 6: `V = Π(S0;C) + ΔV`, tức học phần dư trong không gian ảnh.
- Eq. 7 (paper): `L = E||ε − ε_θ(z_t, t | c_s, c_h)||²`, các latent điều kiện concat theo kênh với `z_t`.

**Code thực tế (CogVideoX) dùng v-prediction**, không phải ε (xem §5.4).

#### 3.4 Huấn luyện ✅ (config + code)
- LoRA r=64, α=64 trên `to_q, to_k, to_v, to_out.0` (`training/cogvideox/static_hand_utils.py:217-223`). `patch_embed.proj` train toàn phần (`dwm_cogvideox_5b_lora.yaml:29-30`).
- AdamW lr 1e-4, β=(0.9, 0.95), wd 0.01, cosine_with_restarts (1 chu kỳ), warmup 100, clip grad 1.0, bf16, gradient checkpointing, 15000 bước tối đa.
- 49 khung, 480×720. Cache latent và text embedding trước (`load_tensors: true`).
- Timestep lấy đều trên {0..999} (`train_dwm_cogvideox.py:778-784`).
- Không dropout prompt/tay (`prompt_dropout_prob: 0.0`, `hand_dropout_prob: 0.0`), không flip.

#### 3.5 Dữ liệu ✅ (§3.3 + `dataset_files/`)

| Nguồn | Cách tạo | Số clip train (code) |
|---|---|---|
| TRUMANS (tổng hợp) | Blender 4.1.1 render 3 luồng: video tương tác ego, video tĩnh (replay camera, không đổi trạng thái vật), video tay (PyTorch3D). Camera gắn đầu, FOV 90°. Clip 49 khung, stride 25, frame-skip 3, 8 fps, 720×480 | 13 721 clip, 59 cảnh, 522 chuỗi hành động |
| TASTE-Rob (thật, camera cố định) | Chỉ tập `DoubleHand`. Resize + pad 720×480, 8 fps, 49 khung. Video tĩnh = lặp khung đầu. Tay từ HaMeR | 3 289 clip |
| Aria (thật, camera động, **chỉ eval**) | Quay quét cảnh trước → dựng 3DGS (gsplat) → render theo quỹ đạo SLAM của lúc tương tác | 60 mẫu thu, 48 dùng để test |

#### 3.6 Suy luận
- Sampler mặc định của checkpoint CogVideoX-Fun, 50 bước, CFG 6 (chỉ CFG theo text: latent tĩnh và tay được **nhân đôi y nguyên** cho nhánh uncond) ✅ (pipeline `:2022-2024`, `:2206-2208`).
- Thử nghiệm Hình 5: bỏ điều kiện tay thì mô hình thành "chỉ điều hướng" (navigation only) ✅ (§4.2).

#### 3.7 Đánh giá hành động (§3.4)
Mô phỏng N ứng viên hành động rồi xếp hạng:
- theo **VideoCLIP-XL** cosine với text mục tiêu (Eq. 9), hoặc
- theo **−LPIPS** giữa khung cuối và ảnh mục tiêu (Eq. 10–11).

Chỉ có minh họa định tính (Hình 7), không có số liệu tổng hợp ✅.

---

### 4. Kết quả & ablation quan trọng

**Benchmark** ✅ (§4): 144 mẫu = 48 TRUMANS (tổng hợp, camera động) + 48 TASTE-Rob (thật, camera tĩnh) + 48 Aria (thật, camera động). Mỗi mẫu sinh 3 seed rồi lấy trung bình. Chỉ số: PSNR, SSIM, LPIPS, DreamSim.

**Bảng 1 — so sánh** ✅

| Phương pháp | Syn-Dyn PSNR/SSIM/LPIPS/DS | RW-Static | RW-Dynamic |
|---|---|---|---|
| CVX SDEdit (noise 0.75, 50 bước) | 19.424 / 0.675 / 0.464 / 0.257 | 16.194 / 0.586 / 0.446 / 0.224 | 19.154 / 0.507 / 0.676 / 0.492 |
| CVX-Fun fine-tuned (không có tay) | 20.541 / 0.767 / 0.370 / 0.175 | 18.951 / 0.780 / 0.265 / 0.089 | 18.129 / 0.472 / 0.591 / 0.328 |
| InterDyn (ControlNet + mask tay, từ khung đầu) | – | 19.331 / 0.744 / 0.240 / 0.135 | – |
| **DWM** | **25.031 / 0.844 / 0.289 / 0.086** | **21.547 / 0.816 / 0.227 / 0.057** | **21.654 / 0.550 / 0.557 / 0.225** |

**Bảng 2 — thành phần dữ liệu** ✅

| | Syn-Dyn | RW-Static | RW-Dynamic |
|---|---|---|---|
| Chỉ TRUMANS | 24.151 / 0.834 / 0.304 / 0.093 | 17.959 / 0.766 / 0.304 / 0.124 | 20.654 / 0.520 / **0.543** / 0.273 |
| TRUMANS + TASTE-Rob | 25.031 / 0.844 / 0.289 / 0.086 | 21.547 / 0.816 / 0.227 / 0.057 | 21.654 / 0.550 / 0.557 / 0.225 |

**Bảng 3 — cách điều kiện tay** ✅ (bảng không ghi tập đánh giá)

| Điều kiện | PSNR | SSIM | LPIPS | DreamSim |
|---|---|---|---|---|
| AdaLN (global) — tham số MANO 2 tay | 21.962 | 0.806 | 0.306 | 0.127 |
| AdaLN (per frame) | 22.789 | 0.827 | **0.288** | 0.110 |
| Mask tay nhị phân (VAE-encode) | 22.876 | 0.797 | 0.338 | 0.137 |
| **Render mesh tay** | **24.151** | **0.834** | 0.304 | **0.094** |

**Nhận xét:**
- Hàng "Ours" của Bảng 3 trùng gần hết với hàng "Chỉ TRUMANS / Syn-Dyn" của Bảng 2 (DreamSim 0.094 so với 0.093). ⚠️ Vậy ablation điều kiện tay được train chỉ trên TRUMANS và đo trên tập tổng hợp.
- Trong Bảng 3, **AdaLN per-frame có LPIPS tốt hơn render mesh** (0.288 < 0.304). Paper không bàn tới điều này ✅. Render mesh thắng rõ ở PSNR/SSIM/DreamSim, nghĩa là tốt hơn về căn chỉnh không gian.
- Mask nhị phân có PSNR ngang AdaLN per-frame nhưng LPIPS tệ nhất. Silhouette thiếu thông tin khớp ngón, khớp với lập luận của paper.
- Bảng 2: thêm TASTE-Rob làm RW-Dynamic **xấu đi ở LPIPS** (0.543 → 0.557) dù tốt hơn ở 3 chỉ số còn lại. Paper viết "most metrics" nên vẫn trung thực ✅.
- Mức tuyệt đối trên RW-Dynamic còn thấp (SSIM 0.55, LPIPS 0.557). Thế giới thật với camera động vẫn khó.
- RW-Static: mô hình lai được train trên chính các nhóm cảnh của TASTE-Rob (xem §6). Mức tăng 17.96 → 21.55 PSNR vì vậy **một phần là do in-distribution**, không thuần là khả năng khái quát.
- Định tính ✅ (Hình 4–6): DWM nhắm đúng vật theo cấu hình tay. Baseline "CVX-Fun fine-tuned không tay" ảo giác vật. InterDyn khớp tay nhưng vật không chuyển động.

---

### 5. Đọc code

#### 5.1 Bản đồ repo
Khoảng 67k dòng Python. Phần lớn là code vendored hoặc thử nghiệm.

| Thư mục / file | Vai trò | Của tác giả? |
|---|---|---|
| `training/cogvideox/` | **Đường chính của paper**: `train_dwm_cogvideox.py` (898 dòng), `inference.py`, `static_hand_dataset.py`, `static_hand_utils.py`, wrapper `models/static_hand_concat_transformer.py`, `pipeline/static_hand_concat.py` | ✅ của tác giả (wrapper mỏng) |
| `training/cogvideox/models/cogvideox_fun_transformer_with_conditions.py` (4710 dòng) | Gom mọi biến thể đã thử: Concat (bản cuối), Adapter v1/v2/v3, CondToken, TripleAttention, JointGeneration, CrossTransformer, AdaLN-Pose (global/per-frame) | của tác giả, phần lõi dựa trên VideoX-Fun |
| `training/cogvideox/pipeline/cogvideox_fun_static_to_video_pose_concat_pipeline.py` (3934 dòng) | Pipeline InP của VideoX-Fun, mở rộng thêm static/hand/smpl_pos_map/raymap | chỉnh từ VideoX-Fun |
| `training/wan/` | Bản WAN: `train_dwm_wan.py` (3558 dòng), `models/wan_transformer3d_with_conditions.py` (concat), `wan_transformer3d_vace.py` (VACE), pipeline hand_concat/hand_vace, `utils/fm_solvers*` (vendored từ Wan) | trộn |
| `data_processing/trumans/` | Script Blender: `blender_ego_video_render*.py`, `blender_ego_static*.py`, `blender_ego_hand.py`, `hand_rendering_common.py`, disparity/raymap | ✅ của tác giả |
| `data_processing/hands/` | HaMeR cho TASTE-Rob/custom (`original_extractor.py`). `mano_overlay_video.py` được import nhưng **không có trong repo** (⚠️ có lẽ nằm trong submodule `third_party/hamer-mediapipe`, không được checkout) | ✅ |
| `data_processing/taste_rob/`, `custom/`, `video_caption/`, `encode_with_*.py` | Tiền xử lý, caption, cache latent | ✅ |
| `finetrainers/`, `train.py` gốc, `docs/training/*` | Bản sao HF finetrainers cũ, đường DWM không dùng | vendored |
| `configs/pipelines/*.yaml` | Config cho mọi biến thể: concat, adaln, adaln_perframe, adapter, posmap_concat/adapter, wan concat/vace, wan2.2 | ✅ |
| `scripts/blender/*` | Render hình minh họa trong paper | không quan trọng |

#### 5.2 Đi qua các file cốt lõi

**(a) `training/cogvideox/train_dwm_cogvideox.py`**
- `main()` (`:473`): load config → `build_pipeline_from_config` (`:523`, tạo transformer đã mở rộng kênh) → `setup_training_mode` (`:248-280`).
- Với LoRA: freeze toàn bộ → `add_adapter(LoraConfig)` → bật lại grad cho các tham số khớp regex `patch_embed.proj`.
- Save hook (`:291-318`) lưu riêng `pytorch_lora_weights.safetensors` và `non_lora_weights.safetensors` (patch-embed). Inference nạp cả hai (`inference.py:284-287`).
- Vòng train (`:719-856`):
  1. `encode_or_sample_video` cho video GT, video tay, video tĩnh (`:729-749`).
  2. Lấy `t ~ U{0..999}` và `add_noise`.
  3. Concat `[noisy, mask, static, hand]`.
  4. Transformer dự đoán v; `get_velocity` đổi sang x0; loss có trọng số `1/(1−ᾱ_t)` (`:776-816`).
- `run_validation` (`:388-470`) chạy pipeline đầy đủ 50 bước mỗi 500 bước train.

**(b) `models/cogvideox_fun_transformer_with_conditions.py`**
- `CogVideoXFunTransformer3DModel.forward` (`:658-774`): concat `inpaint_latents` và `control_latents` theo kênh **trước** `patch_embed` (`:704-709`), rồi chạy DiT CogVideoX bình thường.
- Không có nhánh điều kiện riêng nào. Mọi điều kiện chỉ đi vào qua lớp patch-embed.
- `CogVideoXFunTransformer3DModelWithConcat._setup_condition_channels` (`:1249-1290`): tạo Conv2d mới với `in = 33 + 16`, chép trọng số cũ, cột mới = 0, giữ bias.
- `from_pretrained` khi khởi tạo từ base (`:1366-1410`): load CogVideoXTransformer3DModel → copy sang class Fun → tạo model mới với `condition_channels=0`, `load_state_dict`, rồi mới mở rộng proj.

**(c) Pipeline `__call__` (`:1715-2255`)**
- `preprocess_static_conditions` và `preprocess_hand_conditions` chỉ nội suy số khung (`:1387-1428`, `:1655-1696`).
- `prepare_latents` (`:1216-1385`) VAE-encode tĩnh/tay bằng `.sample()` (không phải `.mode()`). Có cả đường SDEdit: nếu `strength < 1`, khởi tạo latent từ video tĩnh cộng nhiễu (`:1252-1267`).
- Nhánh không có `mask_video` (`:2116-2146`): `mask = zeros` → `mask_latents = resize_mask(1 − mask) × scaling_factor`, tức toàn 1×0.7, khớp với lúc train (`train:798`). `masked_video_latents = static_videos_latents`.

**(d) Dữ liệu**
- `StaticHandConcatDataset` (`static_hand_dataset.py:55-136`) ưu tiên `.pt`. Nếu thiếu latent tay/tĩnh thì đọc mp4 và encode tại chỗ (`:79-107`).
- Lấy khung: `step = max(1, n // 49)`, cắt 49 khung (`static_hand_utils.py:97-104`). Không có crop hay augmentation.

**(e) Render tay**
- `hand_rendering_common.py` / `blender_ego_hand.py:555-674`: nhân bản body CC → dùng modifier VERTEX_WEIGHT_MIX + MASK để chỉ giữ vertex group bàn tay → lấy mesh đã deform → biến đổi world→camera (Blender→PyTorch3D) → SoftPhongShader với đèn điểm đặt ở tâm camera → z-buffer giữa hai tay → nền đen.
- TASTE-Rob: HaMeR (detector ViTDet + ViTPose → crop → MANO) từng khung, không làm mượt theo thời gian (`original_extractor.py:246-344`). Tay trái lấy bằng cách lật trục x của mesh MANO phải (`:310-312`).

**(f) WAN (`training/wan/train_dwm_wan.py`)**
- `conditioning_mode == "dwm"` (`:3147-3165`): `mask_latents = ones(B, 4, F', H', W')`, `y = cat[mask(4), static(16)]`. Tay được concat vào `y` trong `WanTransformer3DModelWithConcat.forward` (`wan_transformer3d_with_conditions.py:231-251`). Tổng 16 + 4 + 16 + 16 = 52 kênh vào `patch_embedding` (Conv3d) đã mở rộng với cột zero (`:135-178`).
- Flow matching: `x_t = (1−σ)·x0 + σ·ε`, target `ε − x0` (`:3359-3371`). Timestep lấy đều theo **chỉ số** của lịch FlowMatchEuler đã shift (shift = 5.0 trong `configs/pipelines/wan2.1_fun_inp_hand_concat.yaml:38`) (`:3349-3355`).
- Loss là MSE, weighting `none`, bỏ qua phần tử có |diff| > 50 (`:376-387`, `:3432-3433`).
- LoRA kiểu VideoX-Fun: áp lên **mọi Linear** trong DiT, rank 64, α 64 (`:2320-2328`, `utils/lora_utils.py:157-210`). `patch_embedding` train toàn phần với lr đầy đủ (`:2333-2354`).

#### 5.3 Những điểm hay nhất trong code

**1. Dùng khe "masked video" của mô hình inpainting cho điều kiện cảnh, mask = "biết hết"** — `train_dwm_cogvideox.py:797-802`

```python
noisy_model_input = scheduler.add_noise(videos_latents, noise, timesteps)
mask_input = torch.ones_like(noisy_model_input[:, :, :1]) * vae_scaling_factor
transformer_input = torch.cat(
    [noisy_model_input, mask_input, static_videos_latents, hand_videos_latents], dim=2)
```

Vì sao hay:
- Không thêm module nào cho điều kiện cảnh. Cảnh tận dụng đúng kênh mà backbone đã được pretrain để "chép vùng đã biết".
- Ở bước 0, mô hình đã sinh lại video tĩnh với chất lượng cao. Khi chưa có tín hiệu khác, đó là kết quả an toàn nhất, vì vậy LoRA chỉ phải học phần dư.
- Mask đưa vào ở dạng `1 × scaling_factor`, khớp quy ước `resize_mask(1 − mask) × scaling_factor` của VideoX-Fun lúc suy luận (pipeline `:2129-2133`). Lệch quy ước này thì train và infer sẽ khác nhau.

**2. Mở rộng kênh patch-embed với cột trọng số khởi tạo 0** — `cogvideox_fun_transformer_with_conditions.py:1275-1281`, và tương tự với Conv3d của Wan ở `wan_transformer3d_with_conditions.py:161-167`

```python
new_proj.weight[:, :original_in_channels] = original_proj.weight
new_proj.weight[:, original_in_channels:] = 0.0
new_proj.bias.data = original_proj.bias.data
```

Vì sao hay: đây là "zero-conv" của ControlNet nhưng **không tốn thêm tham số hay FLOP đáng kể** (chỉ thêm 16 × 4 × 3072 tham số với CogVideoX). Lúc khởi tạo, kênh tay không ảnh hưởng gì, nên mô hình đúng bằng InP gốc. Gradient vẫn chảy vào cột zero vì đầu vào khác 0. Dễ port sang bất kỳ DiT nào.

**3. Huấn luyện `patch_embed` toàn phần nhưng phần còn lại chỉ dùng LoRA** — `train_dwm_cogvideox.py:266-278` + yaml `trainable_parameter_patterns: ["patch_embed.proj"]`

Vì sao hay: lớp duy nhất "nhìn thấy" điều kiện mới phải học từ 0. LoRA trên một ma trận vốn bằng 0 sẽ chỉ có hạng thấp, quá hẹp. Lớp này lại rất nhỏ, nên train full gần như miễn phí. Lưu tách `non_lora_weights.safetensors` (`static_hand_utils.py:226-237`) để checkpoint vẫn nhẹ.

**4. Loss v-prediction viết dưới dạng x0 có trọng số** — `train_dwm_cogvideox.py:811-816`

```python
model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)  # = x0_hat
weights = 1 / (1 - alphas_cumprod[timesteps])
loss = mean(weights * (model_pred - videos_latents) ** 2)
```

Vì sao đáng biết: với `x0 = √ᾱ·x_t − √(1−ᾱ)·v`, ta có `x0_hat − x0 = √(1−ᾱ)·(v − v_hat)`. Nhân với `1/(1−ᾱ)` ra đúng `||v − v_hat||²`, tức loss chuẩn của CogVideoX (v-pred + zero-terminal-SNR). Viết dưới dạng x0 giúp dễ gắn loss theo vùng hoặc loss ảnh lên `x0_hat` (⚠️ một hướng mở rộng cho GReg: thêm loss vùng tay trên `x0_hat`).

**5. Cache tham số phân phối VAE và lấy mẫu lại mỗi bước (CogVideoX)** — `data_processing/encode_with_cogvideox.py:64-73` (`vae._encode` trả moments 32 kênh) và `train_dwm_cogvideox.py:374-383`

```python
if channels <= 6:   # video pixel → encode
    ...
else:               # đã cache moments (mean, logvar)
    latents = DiagonalGaussianDistribution(value).sample() * vae_scaling_factor
```

Vì sao hay: tiết kiệm VAE encode mà vẫn giữ được nhiễu posterior, một dạng augmentation nhẹ cho cả target lẫn điều kiện. Bản WAN thì cache **một mẫu cố định** (`encode_with_wan_impl.py:118`), nên mất lợi ích này.

**6. Render tay "dễ đọc cho VAE"** — `hand_rendering_common.py:321-363`

```python
lights = PointLights(location=[[0,0,0]], ambient=(0.7,)*3, diffuse=(0.6,)*3, specular=(0,)*3)
color = (0.5, 0.8, 0.5) if "CC_Hand_L" in obj.name else (0.8, 0.4, 0.4)
closer = valid & (zbuf < final_z)   # z-test giữa hai tay
```

Vì sao hay:
- Đèn đặt tại camera, không có specular, ambient cao → shading chỉ phụ thuộc góc pháp tuyến so với tia nhìn. Đây thực chất là một **normal map được nén thành độ sáng**, cho thông tin hướng ngón tay.
- Hai màu tách trái/phải, giúp xử lý tay chéo nhau.
- Chỉ lấy vertex group bàn tay (cắt ở cổ tay) → điều kiện chỉ chứa phần mô hình phải "điều khiển".
- Chú ý: **không có z-test với cảnh**. Tay luôn vẽ đè lên nền đen, nên mô hình phải tự suy ra bị che hay không ✅ (không có cảnh trong rasterizer).

**7. Hand-dropout có "target thay thế" (WAN)** — `train_dwm_wan.py:3364-3369`

```python
target = noise - latents.clone()
target[dropout_mask] = noise[dropout_mask] - static_latents[dropout_mask]
```

Vì sao hay: khi bỏ điều kiện tay, mô hình được dạy rằng **không có tay thì cảnh đứng yên**, tức sinh lại đúng video tĩnh. Nhờ vậy "không tay = chỉ điều hướng" (Hình 5) có cơ sở huấn luyện, và có thể dùng CFG theo điều kiện tay. ⚠️ Cả hai config công khai đặt `hand_dropout_prob: 0.0` nên mô hình phát hành có lẽ không dùng cơ chế này.

**8. Mask theo thời gian của Wan-Fun-InP: 4 kênh = 4 khung pixel ứng với 1 khung latent** — `pipeline_wan_fun_inpaint_hand_concat.py:357-366`

```python
mask_condition = torch.concat([repeat_interleave(mask[:, :, 0:1], 4, dim=2), mask[:, :, 1:]], dim=2)
mask_condition = mask_condition.view(bs, F // 4, 4, H, W).transpose(1, 2)
mask_latents = resize_mask(1 - mask_condition, masked_video_latents, True)
```

Vì sao quan trọng cho GReg: đây là cách **duy nhất đúng** để đưa một mask nhìn thấy **theo từng khung** vào Wan-Fun-InP. Mask được "pixel-shuffle" theo thời gian thành 4 kênh, khung đầu lặp 4 lần cho VAE nhân quả. DWM dùng mask toàn 1 nên không cần tới, nhưng GReg với `V_geo,t` thì phải dùng đúng đường này.

**9. Lấy mẫu timestep phân tầng theo rank (Wan)** — `training/wan/utils/discrete_sampler.py:39-50` (từ VideoX-Fun/CogVideo)

Mỗi GPU lấy t trong một khoảng con riêng. Với batch 1/GPU, một bước tối ưu vẫn phủ đều dải noise, nên phương sai gradient giảm. Config Wan dùng batch 1 nên điều này có ý nghĩa ✅ (`dwm_wan_14b_lora.yaml:18`).

**10. Ablation AdaLN-Zero cho tham số tay** — `cogvideox_fun_transformer_with_conditions.py:3674-3757`

`PoseEmbedding`: MLP theo khung → Conv1d theo thời gian → global pool → LayerNorm. Kết quả cộng vào shift/scale/gate của mọi block qua `pose_linear` khởi tạo 0. `pose_dim = 102` = 2 tay × (45 pose + 3 orient + 3 transl) ✅ (`configs/pipelines/cogvideox_fun_static_to_video_pose_adaln.yaml`). Đây là baseline sạch để chứng minh render căn chỉnh pixel tốt hơn vector tham số. GReg có thể dùng lại khi cần ablation "điều kiện bằng tham số SMPL-X vs bằng render".

#### 5.4 Khác biệt paper vs code

| # | Paper | Code | Mức độ |
|---|---|---|---|
| 1 | Loss ε-prediction (Eq. 7) | CogVideoX: v-prediction (`train:811-816`). Wan: flow matching, target `ε − x0` (`train_dwm_wan.py:3371`) | nhỏ (cùng họ), nhưng cần biết khi tái hiện |
| 2 | Tay = render mesh SMPL-X/MANO [32, 37] | TRUMANS: mesh `CC_Base_Body` (Character Creator) cắt theo vertex group (`hand_rendering_common.py:20, 50-65`). TASTE-Rob: MANO từ HaMeR → **hai loại mesh khác nhau** giữa dữ liệu tổng hợp và thật | ⚠️ có domain gap mà paper không nói |
| 3 | Chỉ có CogVideoX-Fun | Có thêm Wan2.1-Fun-14B-InP (checkpoint phát hành), Wan2.2-Fun-5B, biến thể VACE (`wan_transformer3d_vace.py`, zero-init `before_proj/after_proj` `:79-88`, hint ở mỗi layer chẵn `:315`), kênh disparity cảnh/tay, raymap Plücker, SMPL pos map (`configs/pipelines/cogvideox_fun_static_to_video_posmap_concat.yaml`) | các thử nghiệm không công bố |
| 4 | Hình 5 "không tay → chỉ điều hướng" | Cơ chế hand-dropout→target-tĩnh có trong Wan nhưng prob = 0. CogVideoX dropout = gán latent 0 (`train:771-774`); latent 0 ≠ latent của video đen | ⚠️ chưa rõ Hình 5 làm bằng cách nào |
| 5 | Không nêu hyperparameter và compute | Có trong yaml (§5.5) | – |
| 6 | Tập test "unseen" | Test TRUMANS gồm 48 clip từ 18 cảnh, **cả 18 cảnh đều có trong train**, chỉ chuỗi hành động là khác (0 chuỗi trùng) ✅ (đếm từ `dataset_files/`). Test TASTE-Rob dùng cùng 4 nhóm cảnh với train | ảnh hưởng cách hiểu chữ "unseen" (Hình 3a) |
| 7 | Prompt là "semantic guidance" | Prompt sinh **từ video GT** và mô tả cả kết quả, ví dụ "...place the cushion on the left side... The cushion rests on the blanket" (`examples/realworld/realworld_dynamic/prompts_rewrite/*`) | ⚠️ thông tin kết quả lọt vào điều kiện, có lợi cho mọi phương pháp dùng text |
| 8 | – | Wan train ở chế độ `dwm` truyền `clip_fea=None`, nên không có token CLIP. Lúc infer, pipeline truyền `clip_context = zeros_like(...)` qua `img_emb` → thêm 257 token hằng (`pipeline_wan_fun_inpaint_hand_concat.py:443-446`, `wan_transformer3d.py:1171-1173`) | ⚠️ train/infer lệch nhẹ; tác động chưa rõ |
| 9 | Quy ước vùng bị che trong masked video | CogVideoX-Fun tô −1 (đen) (`pipeline:2078`). Wan-Fun nhân 0 (xám giữa) (`pipeline_wan_fun_inpaint_hand_concat.py:344`) | quan trọng nếu dùng mask thật (GReg) |

#### 5.5 Hyperparameter thực tế

| Tham số | CogVideoX (bản chính) | WAN 14B |
|---|---|---|
| Base | CogVideoX-Fun-V1.1-5b-InP | Wan2.1-Fun-V1.1-14B-InP |
| Kênh vào | 16 + 1 + 16 + 16 = 49 | 16 + 4 + 16 + 16 = 52 |
| LoRA | r64/α64, chỉ q,k,v,out (`static_hand_utils.py:217-223`) | r64/α64, mọi Linear (VideoX-Fun) |
| Train full | `patch_embed.proj` | `patch_embedding` |
| lr / opt | 1e-4, AdamW (0.9, 0.95), wd 0.01 | như CogVideoX (Wan dùng lr/2 cho nhóm LoRA text encoder, nhưng mặc định không train text encoder `args.py:51`) |
| Lịch lr | cosine_with_restarts, 1 chu kỳ, warmup 100 | như CogVideoX |
| Batch | 7/GPU × accum 2 (slurm 4 GPU) | 1/GPU × accum 1 |
| Số bước | 15 000 (checkpoint phát hành: 8000 ⚠️) | 10 000 |
| Timestep | đều trên {0..999}, DDPM v-pred | đều theo chỉ số lịch FlowMatch, shift 5, phân tầng theo rank |
| Độ phân giải | 49 × 480 × 720, 8 fps | như CogVideoX |
| Precision | bf16, grad ckpt, VAE slicing/tiling, TF32 | như CogVideoX |
| Dropout | prompt 0, tay 0 | prompt 0, tay 0 |
| Infer | 50 bước, CFG 6 (chỉ text), seed 42 + index | 50 bước, CFG 6, negative "bad detailed" (`wan/inference.py:1419-1421`) |
| Dữ liệu | 13 721 TRUMANS + 3 289 TASTE-Rob = **17 010 clip** × 49 khung | như CogVideoX |

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Không có compute, thời gian train hay thời gian infer** trong paper. Người đọc không ước lượng được chi phí tái hiện.
2. **Baseline yếu:** chỉ có SDEdit, CVX-Fun fine-tune *không có điều kiện tay*, và InterDyn (chỉ đo ở RW-Static). Thiếu baseline "cùng backbone + điều kiện tay qua ControlNet/VACE" để tách đóng góp của "InP-as-identity" khỏi đóng góp của "render tay". ⚠️ **Không có ablation "khởi tạo từ InP vs từ T2V/I2V"**, dù đây là luận điểm trung tâm của §3.2.
3. **Rò thông tin qua text:** prompt mô tả kết quả, lấy từ GT. Bảng 1 vì vậy đo trong điều kiện có gợi ý mạnh. ⚠️ Không có thí nghiệm prompt rỗng hoặc prompt chỉ nói ý định.
4. **"Unseen"** chỉ đúng ở mức clip/chuỗi hành động, không đúng ở mức cảnh (TRUMANS, TASTE-Rob). Chỉ Aria là thật sự lạ.
5. **Bảng 3 lẫn lộn:** AdaLN per-frame thắng LPIPS. Bảng không ghi rõ tập đánh giá.
6. **Domain gap mesh tay** (CC mesh so với MANO/HaMeR), render không có che khuất bởi cảnh, và HaMeR chạy từng khung không làm mượt → tay giật ở dữ liệu thật ⚠️.
7. **8 fps, 49 khung (~6 s):** quá thưa cho thao tác nhanh. Paper không nói về rollout dài.
8. **TASTE-Rob tĩnh = lặp khung đầu**, mà khung đầu có thể đã có tay → điều kiện "tĩnh" chứa tay ⚠️. Mô hình phải tự xóa hoặc di chuyển tay đó.
9. **Action evaluation** (§3.4) chỉ minh họa trên vài ví dụ, không có độ chính xác xếp hạng.
10. Câu hỏi mở: mask toàn 1 có khiến mô hình **quá bảo thủ** (khó tạo thay đổi lớn) không? Sẽ ra sao nếu dùng mask mềm quanh vùng tay để cho phép sinh mạnh hơn? ⚠️

---

### 7. Liên hệ với GReg

#### 7.1 Ánh xạ trực tiếp DWM → H1 ("warp-then-diffuse")

| DWM | GReg-H1 tương ứng | Ghi chú |
|---|---|---|
| Video cảnh tĩnh `Π(S0;C)` vào khe masked-video | **Video nguồn đã warp bằng mesh `I_st = warp(I_s, F_geo,t)`** (A1) | Mạnh hơn bản "lặp khung đầu" của TASTE-Rob, vì hình học đã đúng vị trí |
| Mask = toàn 1 ("biết hết") | **Mask = `V_geo,t · C_t`** (nhìn thấy và đáng tin) | Đúng nghĩa gốc của kênh mask trong InP: 1 = giữ, 0 = sinh. DWM không dùng được điều này, GReg thì có |
| Render tay L/R màu, Phong đèn-camera, VAE-encode, concat 16 kênh zero-init | **Render normal map SMPL-X** (cả thân + tay) **và/hoặc render tay+cẳng tay L/R** từ mesh đích, VAE-encode, concat | Bảng 3 ủng hộ render hơn mask/tham số ✅. GReg nên **giữ cả cẳng tay** vì mặt trong cẳng tay là vùng yếu |
| CogVideoX-Fun-5B-InP / Wan2.1-Fun-14B-InP | **Wan2.1-Fun-V1.1-1.3B-InP** (cùng họ, cùng API với bản 14B của DWM) | `WanTransformer3DModelWithConcat` (~270 dòng, Apache-2.0) dùng lại gần như nguyên |
| LoRA r64 + patch-embed full | LoRA r16–64 + patch-embed full | Data GReg ~4.6k clip, ít hơn 17k của DWM → cân nhắc r thấp hơn, dừng sớm ⚠️ |
| Text prompt từ GT | **Prompt cố định hoặc rỗng** (ví dụ "a person performing sign language, front view") | Không lấy caption GT, tránh rò thông tin như DWM ⚠️ |

→ **Kết luận quan trọng nhất:** mục 1.4 của `docs/diffusion_improvement.md` ghi Wan2.1-Fun-InP "không có kênh điều kiện hình học". DWM cho thấy **chỉ cần mở rộng `patch_embedding` với cột zero** là có kênh đó, và phương án này đã được kiểm chứng ở quy mô CVPR. Hơn nữa, **khe InP sẵn có chính là khe cho "video warped + mask nhìn thấy"**. Vì vậy với GReg, InP có thể khớp bài toán hơn cả VACE hay Control: không cần thêm nhánh nào cho phần warp, chỉ thêm kênh cho normal map. ⚠️ Nên đưa "InP-1.3B + concat" thành phương án chính hoặc ngang hàng với VACE trong ablation.

#### 7.2 Kỹ thuật chuyển được (theo độ khó)

**Dễ (vài ngày):**
1. **Zero-init mở rộng kênh patch-embed + train full lớp đó** (§5.3 mục 2–3). Port nguyên `_extend_patch_embedding`.
2. **Quy ước mask Wan-Fun-InP:** đóng gói 4 kênh theo thời gian + `1 − mask` + vùng che tô **0** trước khi VAE-encode (§5.3 mục 8, §5.4 mục 9). Sai quy ước thì prior inpainting không kích hoạt đúng. Đây là cách đưa `V_geo,t` vào.
3. **Cache latent** của video đích, video warped, normal map. Trên Wan, ⚠️ nên cache moments thay vì một mẫu (như bản CogVideoX, §5.3 mục 5) để có augmentation.
4. **Công thức train khởi điểm:** lr 1e-4, AdamW β2 0.95, wd 0.01, cosine, warmup 100, bf16, grad ckpt, timestep đều với shift 5 (Wan), loss MSE flow-matching, clip |diff| > 50.
5. **Render điều kiện kiểu "normal map nén thành độ sáng" + mã màu trái/phải** (§5.3 mục 6). Với GReg có thể render thẳng normal RGB, nhưng **giữ mã màu L/R** cho tay (ngôn ngữ ký hiệu có nhiều cảnh tay chéo nhau). Khác DWM: GReg **nên có z-test** với thân, vì mesh toàn thân có sẵn.

**Trung bình (1–2 tuần):**

6. **Dropout điều kiện có target thay thế** (§5.3 mục 7), áp vào GReg: khi drop normal map, target vẫn là GT, nhưng mô hình chỉ còn video warped. Điều này cho phép **CFG theo điều kiện hình học**: `ε_uncond = model(warp, mask, ∅)` và `ε_cond = model(warp, mask, normal)`, rồi tăng thang CFG để tay bám normal map chặt hơn. DWM chỉ CFG theo text; GReg cần CFG theo hình học hơn ⚠️.
7. **Loss vùng trên `x0_hat`** (§5.3 mục 4): với flow matching, `x0_hat = x_t − σ·v_hat`. Nhân trọng số vùng tay/mặt (B1) trong latent, kiểu SignRefine ×10 ⚠️.
8. **Ablation AdaLN vs render** (§5.3 mục 10): tái hiện cho SMPL-X (tham số tay 2 × 45 + orient) so với normal map, để chứng minh lựa chọn điều kiện pixel-aligned trong luận văn.

**Khó / tốn compute:**

9. Backbone 5B/14B (DWM cần 80 GB). Với 24 GB local chỉ nên làm 1.3B. Nếu muốn so với bản lớn hơn thì thuê H100 ⚠️.
10. **Ablation "InP-init vs I2V-init vs T2V-init"**, thí nghiệm mà DWM bỏ qua. Nếu GReg làm được thì đây là đóng góp phương pháp luận rõ ràng ⚠️.

#### 7.3 Liên hệ với H2 (refiner một bước)
- Ý "InP như hàm đồng nhất có prior" áp thẳng được vào H2: đưa **output GReg** vào khe masked-video với mask = 1 (hoặc mask = 1 ở vùng nhìn thấy, 0 ở vùng tay/che khuất), target = GT. Mô hình học **phần dư sửa lỗi**. Đây là H2 phiên bản video, có tính nhất quán thời gian ⚠️.
- DWM **không** có phần chưng cất một bước: 50 bước + CFG. Code có hỗ trợ LoRA LightX2V (chưng cất ít bước) nhưng chỉ cho chế độ i2v (`train_dwm_wan.py:1018`), không thuộc mô hình DWM. Với H2 vẫn phải dựa vào OSEDiff/pix2pix-turbo.
- ⚠️ Mask = 1 ở mọi nơi có thể quá bảo thủ để sửa vùng mờ. Nên đặt mask mềm, = 0 ở vùng tay (B1) hoặc vùng `V_geo = 0`, để prior được phép vẽ lại.

#### 7.4 KHÔNG nên lấy, và vì sao
1. **Text prompt mô tả kết quả từ GT:** với GReg, đó là rò GT và không có ở bài toán thật. Dùng prompt cố định.
2. **8 fps / frame-skip:** tay ký hiệu chuyển động nhanh; 8 fps làm mất hình dạng tay ở các khung trung gian. GReg nên giữ 16–25 fps (xem `diffusion_improvement.md` §1.6).
3. **"Lặp khung đầu" làm điều kiện tĩnh** (cách TASTE-Rob): đó là phương án dự phòng khi không có hình học. GReg có A1, nên phải dùng video warped. Ablation "lặp I_s" vẫn nên giữ làm baseline (tương đương "copy nguồn", L1 0.0387).
4. **HaMeR từng khung không làm mượt:** GReg có SMPL-X fit theo thời gian (sai số ~2.5 px), tốt hơn hẳn. Không thay bằng HaMeR.
5. **Chỉ render bàn tay, cắt ở cổ tay:** với GReg, cánh tay và cẳng tay là vùng lỗi chính → render cả cẳng tay, hoặc normal map toàn thân.
6. **Render không z-test với cảnh/thân:** DWM không có mesh cảnh động nên chấp nhận được. GReg có mesh toàn thân, nên render điều kiện cần che khuất đúng (tay sau thân phải bị che).
7. **CFG chỉ theo text:** với GReg, text gần như hằng nên CFG text vô nghĩa. Dùng CFG theo điều kiện hình học (mục 6 ở trên) hoặc tắt CFG để giảm một nửa thời gian infer ⚠️.

#### 7.5 Việc cụ thể đề xuất (thêm vào M0–M2 của H1)
- **M0':** đọc `training/wan/pipeline/pipeline_wan_fun_inpaint_hand_concat.py` + `models/wan_transformer3d_with_conditions.py`. Chạy Wan2.1-Fun-1.3B-InP **zero-shot** với `masked_video = I_st` (vùng `V=0` tô 0) và mask = `V·C` (đóng gói 4 kênh), chưa có normal map. Đây là bản đo "prior inpainting + warp A1" thuần, rẻ và rất có tính thông tin ⚠️.
- **M1':** thêm 16 kênh normal map (VAE-encode) bằng patch-embed zero-init, LoRA r32 + patch-embed full, 49 khung ở 320–480 px trên 24 GB; cache latent.
- **M2':** ablation mask toàn 1 (đúng như DWM) vs mask `V·C` (đề xuất GReg). Đây là câu hỏi mới, DWM chưa trả lời.

---

## Bài 5 — Generative Rendering: Controllable 4D-Guided Video Generation with 2D Diffusion Models (CVPR 2024)

Quy ước: ✅ = kiểm chứng từ bài báo/code (có dẫn mục, hình, bảng hoặc file:dòng/URL); ⚠️ = suy luận, phỏng đoán hoặc thiết kế đề xuất của mình.

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Shengqu Cai (Stanford, thực tập Adobe), Duygu Ceylan, Matheus Gadelha, Chun-Hao Paul Huang, Tuanfeng Yang Wang (Adobe Research), Gordon Wetzstein (Stanford) ✅ |
| Venue | CVPR 2024, tr. 7611–7620 ✅ (bản Open Access CVF) |
| arXiv | 2312.01409. Bản arXiv có **phụ lục A–E** mà bản CVPR không có: bảng nhất quán theo khoảng cách khung, độ dài video, ablation thêm, hạn chế ✅ (https://arxiv.org/html/2312.01409) |
| Project | https://primecai.github.io/generative_rendering/ (chỉ có video, PDF, BibTeX) ✅ |
| Code | **Không có code chính thức.** Project page không có link code ✅. Bài EG4D gọi Generative Rendering là "concurrent work, with no code available". Có bằng sáng chế Mỹ US 12,664,751 "Systems and techniques to perform 4D-guided video generation with diffusion models"; các claim (theo snippet tìm kiếm) khớp đúng với bài: keyframe từ depth + UV map, đặc trưng pre/post-attention, "fill a texel … if not filled before", "mean of inpainted texture and average texture", "Gaussian noise texture for each object". Bản PDF USPTO là ảnh scan, không đọc được text ⚠️. Lý do không có code nhiều khả năng là IP của Adobe ⚠️ |
| Backbone | Stable Diffusion v1.5 + ControlNet depth, 512×512, scheduler Karras, 50 bước khử nhiễu ✅ (§4 "Implementation details") |
| Huấn luyện | **Không.** Zero-shot, training-free ✅ (§5 "zero-shot pipeline") |
| Compute | **Không báo thời gian chạy và không nêu loại GPU.** Phụ lục C: video dài hơn 200 khung chạy với bộ nhớ GPU dưới 12 GB, vì chỉ keyframe phải xử lý song song ✅ |

---

### 1. TL;DR

- **Bài toán.** Cho một cảnh 3D động không có texture (mesh rig Mixamo, mô phỏng vật lý, camera quay) và một prompt văn bản, sinh video stylized nhất quán theo thời gian. Mô hình sinh chỉ là **T2I 2D** (SD1.5 + ControlNet depth) ✅.
- **Ý tưởng.** Mesh cho **tương ứng ground-truth** giữa mọi khung qua không gian UV chính tắc. Tương ứng này được tiêm vào ba chỗ ✅:
  1. **Khởi tạo nhiễu trong UV.** Lấy mẫu một "texture nhiễu Gaussian" trong UV của từng vật, rồi chiếu sang từng khung. Cùng một điểm bề mặt sẽ nhận cùng giá trị nhiễu ở mọi khung (§3.3).
  2. **Pre-attention.** Mỗi khung tính self-attention với K/V ghép từ chính nó và một nhóm **keyframe** lấy mẫu ngẫu nhiên ở mỗi bước (extended attention) (§3.1, Eq. 5).
  3. **Post-attention.** Đầu ra self-attention của keyframe được chiếu lên UV, hợp nhất thành một "texture đặc trưng", rồi render lại về từng khung. Kết quả trộn với đầu ra attention của khung đó theo hệ số α (Eq. 6).
- Thêm một bước **AdaIN trên latent**, thống kê chỉ trên pixel nền, để chống nhấp nháy màu (§3.4) ✅.
- **Kết quả.** CLIP frame consistency 0.9845, tốt nhất trong nhóm dùng T2I; chỉ thua Gen-1 (0.9907), một mô hình T2V độc quyền. Prompt fidelity 0.3227, gần bằng per-frame (0.3233) (Bảng 1) ✅. Ablation (Hình 6) chỉ là định tính.
- **Ý nghĩa cho GReg.** A1 đã cho đúng loại tương ứng mà bài này cần, và còn chính xác hơn vì có kiểm tra che khuất. Có thể mượn ba thứ:
  - cơ chế post-attention qua tương ứng mesh để giữ nhất quán thời gian cho refiner per-frame (H2);
  - ý "nhiễu gắn vào bề mặt" (chỉ hữu ích cho sampling nhiều bước);
  - dùng làm baseline training-free.
  
  Khác biệt lớn: GReg **có ảnh nguồn thật**, nên keyframe tự nhiên là I_s và texture UV có thể khởi tạo từ ảnh thật thay vì từ text ⚠️.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Đầu vào** ✅ (§3):
- Chuỗi depth map 𝒟 = [D¹…D^N].
- Chuỗi UV map 𝒰𝒱 = [UV¹…UV^N], mỗi map có 3 kênh RGB = (u, v, object ID).
- Prompt văn bản c.

Các map này được render từ cảnh 3D động. **Đầu ra:** N khung RGB 512².

**Vì sao khó** ✅ (§1, §3.3, §4.1):
1. Sinh từng khung độc lập bằng ControlNet cho mỗi khung đẹp nhưng **nhấp nháy** (Fig. 3, hàng "Per-Frame").
2. Các phương pháp video editing zero-shot (Pix2Video, TokenFlow, Rerender…) dựa vào **DDIM inversion** của video nguồn để có nhiễu khởi tạo nhất quán và đặc trưng tương đồng giữa các khung. Ở đây không có video nguồn, chỉ có render không texture, nên inversion "constantly fails" (Phụ lục D).
3. **Fixed noise** (cùng một nhiễu cho mọi khung) gây **texture sticking**: texture dính vào toạ độ màn hình thay vì đi theo bề mặt (§3.3; minh hoạ trong supplementary web).
4. Các mô hình T2V năm 2023 khó điều khiển và chỉ sinh vài giây.
5. Tương ứng hình học đúng **không đảm bảo đặc trưng tương ứng giống nhau** trong không gian đặc trưng của UNet (§4.1 ablation). Vì vậy không thể copy đặc trưng một cách thô bạo.

---

### 3. Phương pháp chi tiết

#### 3.0 Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| i | Chỉ số khung |
| l | Bước diffusion (bài dùng l trong §3, t trong Alg. 1) |
| f^(i,l) | Đặc trưng **vào** self-attention ("pre-attention") |
| F^(i,l) | Đặc trưng **ra** self-attention ("post-attention") |
| π_{i,j} | Phép chiếu lại khung i → khung j qua tương ứng pixel từ UV map |
| π_{𝒰𝒱} | Phép chiếu UV → ảnh |
| π_{𝒰𝒱}^{-1} | Phép chiếu ảnh → UV |

✅ (§3.1–3.2, Alg. 1)

#### 3.1 Self-attention và hai kiểu tiêm đặc trưng ✅ (§3.1)

- Eq. 1: F = Softmax(QKᵀ/√d)·V.
- Eq. 2: Q = W^Q f^(i,l), K = W^K f^(i,l), V = W^V f^(i,l).
- **Pre-attention injection:** thay đổi K và V. Dạng đầy đủ là extended attention trên **toàn bộ** N khung, K = W^K[f^(1,l)…f^(N,l)] (Eq. 3). Cách này quá đắt khi N lớn, nên bài chỉ dùng nó cho keyframe.
- **Post-attention injection:** chép thẳng đầu ra, F^(j,l) = π_{i,j}(F^(i,l)) (Eq. 4).
  - Ưu điểm: nhất quán.
  - Nhược điểm theo tác giả: "tends to create blurry results". Nó cũng **chặn các hiệu ứng tương tác** như bóng và phản xạ, vì ghi đè lên đầu ra attention.

#### 3.2 UV-space feature blending ✅ (§3.2)

Đây là cách tạo texture đặc trưng hợp nhất T từ đặc trưng của một tập khung:

1. Chiếu feature map của từng khung lên UV chính tắc. Văn bản viết: "for each texel in the UV space, we find its closest point correspondence in each frame".
2. Nếu chỉ lấy trung bình (có trọng số) như TokenFlow thì "consistently leads to sub-optimal blurry results".
3. Vì vậy bài làm thêm **blending tuần tự**: duyệt các khung theo thứ tự; một texel nhận đặc trưng của pixel tương ứng trong khung hiện tại **chỉ khi nó chưa được điền trước đó**. Kết quả là "inpainted texture" T_inp, kiểu first-come-first-served.
4. Texture hợp nhất là T = ½(T_inp + T_avg), với T_avg là texture trung bình.
5. Chiếu T ngược về từng khung để được F̄^(i,l).

**Điểm không rõ, phải tự quyết khi tái hiện** ⚠️:
- (a) Có dùng trọng số trong T_avg (ví dụ theo góc nhìn, như SyncMVD) hay không: không nói.
- (b) Thứ tự duyệt trong T_inp: Alg. 1 có `shuffle`, nên nhiều khả năng là thứ tự keyframe đã xáo, đổi ở mỗi bước. Như vậy mỗi bước một keyframe khác "thắng", và nhiễu này trung bình hoá dần qua các bước.
- (c) **Texel không thấy ở keyframe nào** không được xử lý tường minh (Phụ lục E chỉ nói đến độ phân giải texture). Hợp lý nhất là dùng mặt nạ: ở pixel có texel trống thì đặt α = 0, tức chỉ giữ F̂.
- (d) Dùng nearest hay bilinear, và hạ UV map xuống từng độ phân giải đặc trưng 64/32/16/8 bằng cách nào: không nói.
- (e) **Hướng chiếu thực tế có lẽ là scatter (pixel → texel), không phải gather.** Phụ lục E: texture **quá thấp** (2048²) "will create corrupted regions" (Fig. 7, artefact trên mặt đất), còn **quá cao** thì "UV coordinates too far apart" nên "no blending effect". Hiện tượng này chỉ hợp lý nếu mỗi pixel đặc trưng (lưới 64×64) ghi vào **một texel**:
  - texture cao → các khung hiếm khi ghi trùng texel → không có blending, mỗi pixel đọc lại đúng giá trị của chính nó;
  - texture thấp → nhiều pixel dồn vào một texel → hỏng.
  
  Nếu là gather thì độ phân giải cao không gây lỗi "không blend". Bài dùng 2048² và 3072² (Fig. 7) cho latent 64² ✅. Đây là một **siêu tham số nhạy** mà tác giả tự thừa nhận "tricky".

#### 3.3 Vòng lặp sinh ✅ (§3.2, Alg. 1, Fig. 2)

Ở mỗi bước khử nhiễu t:

1. **Chọn keyframe:** KF = shuffle({i₁…i_k}). Lấy ngẫu nhiên m keyframe, **khác nhau ở mỗi bước**. Văn bản dùng m, Alg. 1 dùng k. Bài **không nêu giá trị**.
2. **Lượt keyframe:** chạy UNet + ControlNet trên batch keyframe với **extended attention giữa các keyframe** (Eq. 3 giới hạn trong KF). Lưu lại f_KF (đầu vào) và F_KF (đầu ra) của self-attention, ở mọi lớp self-attention và ở bước hiện tại.
3. **Hợp nhất:** T̄_KF = blend(π_{𝒰𝒱}^{-1}(F_KF)) theo §3.2.
4. **Lượt sinh lại cho từng khung i, tuần tự (kể cả keyframe):**
   - Tính F̄^i = π_{UV^i}(T̄_KF).
   - Chạy UNet trên z^i. Trong mỗi self-attention:
     - K = W^K[f^(i,l), f_KF^l] và V = W^V[f^(i,l), f_KF^l], tức khung i nhìn **chính nó + mọi keyframe** (Eq. 5). Đầu ra là F̂^(i,l).
     - Trộn: **F_out = α·F̄^(i,l) + (1−α)·F̂^(i,l)** (Eq. 6). Giá trị α **không được công bố**.
   - Lấy z^i_{t−1}. Kết quả của lượt keyframe (bước 2) **bị bỏ**: "in case of keyframes, the diffusion step is simply repeated".
5. **Lưu ý về Alg. 1.** Dòng z^i_{t−1} = Φ(z^i, 𝒟_KF, c, f_KF, F̄^i) ghi điều kiện là 𝒟_KF. Gần như chắc đây là lỗi đánh máy và phải là D^i (ControlNet depth của khung i) ⚠️.

**Lớp và bước nào được tiêm?** Văn bản nói "self-attention modules" nói chung, ở **mọi bước** diffusion. Không có lịch tắt/bật theo bước, cũng không chọn lớp ✅ (§3.2, theo bản arXiv). Có tiêm vào self-attention của ControlNet hay không thì không nói ⚠️.

**Pseudo-code tổng thể** (dựng lại từ Alg. 1; ⚠️ ở các chỗ bài không nêu):
```python
z_uv = randn(C=4, H_tex, W_tex)                # texture nhiễu Gaussian theo từng vật (Fig.2 "initial noise for each object")
z = [render_uv(z_uv, UV[i], size=64) for i in range(N)]   # π_UV: lấy mẫu texture tại uv của pixel latent (nearest ⚠️)
for t in timesteps:                            # 50 bước Karras
    KF = random.sample(range(N), m)            # m chưa biết ⚠️
    store = run_unet(z[KF], D[KF], c, mode="extended_among_KF", record=True)  # f_KF, F_KF ở mọi lớp self-attn
    T = {l: blend_uv([unproject(store.F[l][j], UV[KF[j]]) for j in range(m)]) for l in layers}
    for i in range(N):
        Fbar = {l: project(T[l], UV[i]) for l in layers}           # + mask texel hợp lệ ⚠️
        eps  = run_unet(z[i], D[i], c, mode="inject", kv_extra=store.f, Fbar=Fbar, alpha=α)
        z_new[i] = scheduler.step(eps, t, z[i])
        z_new[i] = adain_bg(z_new[i], ref=z_new[0], bg_mask=bg[i])  # §3.4; bài nói "predicted latents", chưa rõ x0 hay z_t ⚠️
    z = z_new
frames = [vae.decode(zi) for zi in z]
```

#### 3.4 UV noise initialization ✅ (§3.3, Phụ lục D)

- Lấy mẫu z_𝒰𝒱 ~ 𝒩(0, I) trong UV, rồi z_T = π_𝒰𝒱(z_𝒰𝒱) (Alg. 1). Mỗi vật có texture nhiễu riêng, phân biệt bằng object ID (Fig. 2).
- Lấy cảm hứng từ PYoCo [9] và "Coherent noise for NPR" (Kass & Pesare 2011) [19].
- Không nói cách lấy mẫu (nearest/bilinear) và cách xử lý pixel không có UV ✅ (bài không nêu).

**Lưu ý lý thuyết** ⚠️ (từ How I Warped Your Noise, ICLR 2024, và Go-with-the-Flow, CVPR 2025):
- Lấy mẫu nearest từ texture nhiễu ra lưới latent có hai cách hỏng:
  - texture **thấp hơn** mật độ pixel → nhiều pixel trùng một giá trị → nhiễu không còn i.i.d. trong không gian;
  - **bilinear** → phương sai giảm, nhiễu bị làm mờ, ảnh ra mờ/phẳng.
- Texture **rất cao** (3072² cho latent 64²) giữ được tính i.i.d. Nhưng khi đó cùng một điểm bề mặt ở hai khung hiếm khi rơi đúng cùng một texel, nên tương quan thời gian yếu đi.
- Đây chính là mâu thuẫn mà ∫-noise giải quyết: mỗi pixel = tích phân của trường nhiễu vô hạn phân giải trên diện tích pixel. Bài GR không bàn đến chuyện này.

#### 3.5 Latent normalization ✅ (§3.4, Phụ lục D)

- Áp AdaIN lên latent dự đoán để phân bố của mọi khung khớp với **khung đầu tiên**.
- Thống kê **chỉ trên pixel nền**, theo Cross-image attention [1], vì thống kê toàn ảnh rất nhạy với chuyển động.
- Lý do: VAE decoder khuếch đại các lệch màu nhỏ trong latent.

---

### 4. Kết quả & ablation quan trọng

**Dữ liệu** ✅ (§4):
- Ba loại cảnh: camera quay 360° quanh vật tĩnh; mô phỏng vật lý (bóng nảy, va chạm rigid-body); nhân vật rig Mixamo nhảy.
- **Không nêu số cảnh hay số clip**, không có test set chuẩn.

**Baseline** ✅:
- Per-frame (ControlNet depth).
- Pix2Video\*: mỗi khung attend tới anchor + khung trước.
- TokenFlow\*: extended attention + token propagation, nhưng **dùng tương ứng GT từ 3D thay NN-matching**.
- Gen-1 (T2V độc quyền).
- Cả Pix2Video\* và TokenFlow\* dùng **nhiễu giống hệt nhau** cho mọi khung, vì không inversion được.

**Bảng 1** ✅ (chỉ số CLIP; consistency = cos trung bình giữa các cặp khung liên tiếp; fidelity = CLIP ảnh–prompt):

| Phương pháp | Frame consistency ↑ | Prompt fidelity ↑ |
|---|---|---|
| Per-Frame | 0.9547 | **0.3233** |
| Pix2Video\* | 0.9630 | 0.2983 |
| TokenFlow\* | 0.9822 | 0.2737 |
| Gen1 | **0.9907** | 0.3029 |
| **Ours** | 0.9845 | 0.3227 |

**Phụ lục B, Bảng 2** ✅ (consistency theo khoảng cách khung 1/5/10/15/20):

| | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Ours | 0.9845 | 0.9815 | 0.9738 | 0.9749 | 0.9727 |
| Gen1 | 0.9907 | 0.9715 | 0.9582 | 0.9624 | 0.9601 |

Từ khoảng cách 5 trở đi, GR **hơn** Gen-1. Tương ứng 3D cho nhất quán **dài hạn**; prior video chỉ cho nhất quán **ngắn hạn**. Đây là điểm đáng trích cho luận văn.

**Ablation** (Fig. 6, **chỉ định tính**, nhân vật "the Joker dancing") ✅ (§4.1):
- **No pre-attn** (chỉ chiếu post-attn qua UV): "chaotic", artefact nặng ở mặt và quần áo. Giải thích: khi chuyển động lớn, điểm khớp hình học **không** khớp trong không gian đặc trưng. Chép đặc trưng thẳng qua hình học thì hỏng; phải cho đặc trưng đi qua attention (pre-attn) trước để chúng "hài hoà".
- **No post-attn** (chỉ extended attention): áo quần Joker đổi theo khung. Attention chỉ khớp theo độ tương đồng nên **không đảm bảo nhất quán từng pixel**.
- **Fixed noise** (thay UV noise): artefact khi blend, mặt không nhận ra ở khung thứ 3.
- **DDIM inversion và latent normalization:** chỉ có video so sánh trên web supplementary, không có số (Phụ lục D).

**Qualitative** ✅ (Fig. 3–5):
- TokenFlow\* mờ, vì blend đặc trưng không qua attention + tương ứng chỉ chính xác ở 64×64, nhất là quanh biên vật.
- Pix2Video\* đẹp nhưng không nhất quán.
- GR sinh được bóng (hàng 3, 7 của Fig. 5) và thay đổi ánh sáng (hàng 1). Hỗ trợ camera quay quanh vật tĩnh (Fig. 4).

**Đánh giá phản biện** ⚠️:
- Chỉ có hai chỉ số CLIP, rất thô: không FVD, không warping error, không user study.
- Không có ablation định lượng, không báo runtime.
- Prompt fidelity của "Ours" thấp hơn per-frame, tức consistency đổi lấy một chút fidelity.

---

### 5. Code

#### 5.1 Tình trạng
- **Không có code chính thức** ✅ (project page). Không tìm thấy bản tái hiện không chính thức trên GitHub.
- Tex4D (arXiv 2410.10821, github.com/ZqlwMatt/Tex4D) là bài gần nhất về chủ đề "UV aggregation trong video diffusion cho chuỗi mesh". Repo hiện chỉ có README + requirements; checklist "Release inference code" **chưa tick** ✅ (commit a5c9c45). Không dùng được.

#### 5.2 Code mở gần nhất về cơ chế: SyncMVD (**KHÔNG phải code của Generative Rendering**)

Repo: https://github.com/LIU-Yuxin/SyncMVD (Liu et al., "Text-Guided Texturing by Synchronized Multi-View Diffusion", arXiv 2311.12891), commit `faa9562`. Đây là texturing mesh tĩnh nhiều view chứ không phải video. Tuy vậy nó có đủ hai mảnh kỹ thuật dùng chung với GR: (i) texture nhiễu trong UV rồi render ra các view, (ii) hợp nhất latent qua UV ở mỗi bước, cộng với self-attention mở rộng sang các view tham chiếu. Điểm cốt lõi ✅:

- **Thay processor self-attention** (`src/syncmvd/attention.py:17-22`): chỉ thay các module `attn1` (self-attn) trong UNet diffusers. Đây là đúng điểm cắm để làm pre-attention injection.
- **Extended attention theo từng mẫu** (`attention.py:100-147`):
  - Mỗi view b tính `query_b` với K/V ghép từ các view lân cận (`custom_attention_mask[b_idx]`, gồm chính nó và hai view kề).
  - Song song tính attention với K/V của các **view tham chiếu** (`ref_attention_mask`).
  - Trộn `(h_a + ref_weight·h_ref)/(1+ref_weight)`.
  - Tương đương Eq. 5 của GR, chỉ khác là trộn hai lượt attention thay vì nối chung K/V.
- **Lịch bật/tắt theo bước** (`pipeline.py:595-598`): `ref_weight=1` chỉ trong 20% bước đầu (`ref_attention_end=0.2`). Comment trong `configs.py`: "Lower→better quality; higher→better harmonization". GR **không** có lịch như vậy; đây là một nút đáng thêm khi tái hiện ⚠️.
- **Khởi tạo nhiễu UV** (`src/renderer/project.py:189-194`, `pipeline.py:465-469`):
  - `torch.normal(0,1,(C,)+tex_size)` làm texture.
  - Render ra các view với `sampling_mode="nearest"` (`pipeline.py:241`).
  - Nền lấy nhiễu riêng và ghép bằng mask (`composite_rendered_view`, `pipeline.py:87-99`).
- **Hợp nhất trong UV ở mỗi bước** (`src/syncmvd/step.py:75-118`):
  - "Bake" cả x̂₀ dự đoán lẫn z_t của mọi view vào texture.
  - Làm bước DDPM **trong không gian texture**: `prev_tex = c0·x0_tex + ct·tex + σ·noise`.
  - Render lại ra các view.
  - Đây là "đồng bộ latent", mạnh hơn GR (GR chỉ đồng bộ đặc trưng attention và nhiễu ban đầu, còn latent mỗi khung vẫn đi riêng).
- **Bake = phép kề (adjoint) của render** (`project.py:434-475`):
  - Tối ưu một bước SGD với lr = 1 trên texture rỗng. Gradient của ∑‖render(tex) − view‖² chính là scatter pixel → texel.
  - Chia cho `gradient_maps` (số pixel rơi vào mỗi texel, `project.py:348-368`) để ra trung bình.
  - Texel trống thì lấp bằng **Voronoi/JFA** (`voronoi_solve`).
  - Trộn các view theo trọng số `visible_triangles · cos(góc nhìn)^exp`, với `exp` tăng tuyến tính từ 0 lên 6 theo bước (`pipeline.py:634`, `configs.py:38-39`: "low value encourage consistency … high value encourage sharper results").
  - → Đây là một lời giải rõ ràng cho câu hỏi "texel trống" và "blend mờ" mà GR để ngỏ ⚠️.
- **Độ phân giải:** latent view 96², latent texture 512² (paper gốc 1536²) (`configs.py`). Cùng bậc với 2048–3072² của GR cho latent 64². Cả hai bài đều chọn texture **lớn hơn nhiều** so với lưới latent ✅.
- **Các nút khác:** `mvd_end=0.8`, tức tắt đồng bộ UV ở 20% bước cuối để lấy lại độ sắc nét ✅ (`configs.py`, `pipeline.py:319,635`).

#### 5.3 Code liên quan khác (chỉ tham khảo, không đọc sâu)
- **Go-with-the-Flow** (CVPR 2025, https://github.com/Eyeline-Labs/Go-with-the-Flow): `make_warped_noise.py` warp nhiễu theo optical flow mà vẫn giữ Gaussianity, real-time. Nhưng để mô hình dùng được nhiễu warped thì **phải fine-tune** (CogVideoX) ✅ (README/paper).
- **How I Warped Your Noise** (ICLR 2024): ∫-noise. Có bản không chính thức https://github.com/mskwak01/How-I-Warped-Your-Noise-Unofficial ✅.
- **TokenFlow, Pix2Video, Text2Video-Zero:** đều có code mở; là nguồn của extended attention và token propagation.

#### 5.4 Pseudo-code PyTorch/diffusers đủ để tái hiện phần attention ⚠️ (thiết kế của mình, viết theo kiểu `attention.py` của SyncMVD)

```python
class GRSelfAttnProcessor:
    """Thay cho attn1. state dùng chung giữa các lớp: mode ∈ {"record","inject"}."""
    def __init__(self, name, state): self.name, self.S = name, state
    def __call__(self, attn, x, encoder_hidden_states=None, attention_mask=None, temb=None):
        B, L, C = x.shape                                   # x = f^(i,l) (đã norm1 bên ngoài block)
        q = attn.to_q(x)
        if self.S.mode == "record":                         # lượt keyframe: extended attention giữa các KF
            kv_src = x.reshape(1, B * L, C).expand(B, -1, -1)
            self.S.f_kf[self.name] = x.detach()             # pre-attn của KF
        else:                                               # lượt khung i: K,V = [f_i, f_KF] (Eq.5)
            fkf = self.S.f_kf[self.name].reshape(1, -1, C).expand(B, -1, -1)
            kv_src = torch.cat([x, fkf], dim=1)
        k, v = attn.to_k(kv_src), attn.to_v(kv_src)
        h = sdpa(split_heads(q), split_heads(k), split_heads(v))
        F_hat = attn.to_out[1](attn.to_out[0](merge_heads(h)))      # F̂ (đầu ra attention, trước residual)
        if self.S.mode == "record":
            self.S.F_kf[self.name] = F_hat.detach()
            return F_hat
        # post-attn injection (Eq.6): F̄ đã chiếu từ texture UV hợp nhất, kèm mask texel hợp lệ
        F_bar, m = self.S.F_bar[self.name]                  # (B,L,C), (B,L,1)
        a = self.S.alpha * m                                # α=0 nơi texel trống ⚠️
        return a * F_bar + (1 - a) * F_hat

def unproject(F, uv_idx, valid, tex_hw):                    # scatter pixel → texel (kiểu SyncMVD bake)
    T = F.new_zeros(tex_hw[0] * tex_hw[1], F.shape[-1])
    cnt = F.new_zeros(tex_hw[0] * tex_hw[1], 1)
    T.index_add_(0, uv_idx[valid], F[valid])
    cnt.index_add_(0, uv_idx[valid], torch.ones_like(F[valid, :1]))
    return T, cnt

def blend(list_T_cnt):                                      # §3.2: ½(first-filled + mean)
    T_inp = torch.zeros_like(list_T_cnt[0][0]); filled = torch.zeros_like(list_T_cnt[0][1])
    S = 0; N = 0
    for T, c in list_T_cnt:                                 # thứ tự = thứ tự KF đã shuffle
        mean_j = T / c.clamp(min=1)
        new = (c > 0) & (filled == 0)
        T_inp = torch.where(new, mean_j, T_inp); filled = filled + new.float()
        S = S + T; N = N + c
    T_avg = S / N.clamp(min=1)
    return 0.5 * (T_inp + T_avg), (N > 0)
```
- `uv_idx`: chỉ số texel phẳng của mỗi token ở **từng độ phân giải** (64/32/16/8 cho SD1.5). Tính bằng cách hạ UV map xuống mỗi độ phân giải bằng nearest, lấy tâm pixel ⚠️.
- **Lưu ý thực thi:** lượt "record" chạy batch m keyframe một lần; lượt "inject" lặp N khung.
- **Chi phí** ⚠️: khoảng (N + m) lượt UNet + ControlNet mỗi bước, nhân 2 nếu dùng CFG, nhân 50 bước. Với N = 100, m = 4 là khoảng 10⁴ lượt UNet ở 512², tức **hàng phút đến hàng chục phút mỗi clip**. Bài không báo số liệu.

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Không đảm bảo nhất quán tuyệt đối và không giữ chi tiết.** Nguyên nhân là tương ứng UV ở 64×64 không chính xác ✅ (§5 bản arXiv). Phụ lục E: "A large portion of our inconsistency comes from the VAE decoder", vì decoder khuếch đại lệch nhỏ trong latent ✅.
2. **Độ phân giải texture rất khó chỉnh:** thấp thì hỏng vùng, cao thì không blend ✅ (Phụ lục E, Fig. 7).
3. **Không tổng quát cho thay đổi môi trường lớn hay góc nhìn thay đổi mạnh,** do "highly overlapping pixel matches and poor feature projections". Tác giả gợi ý fine-tune hoặc thêm module video ✅ (Phụ lục E).
4. **Chậm.** Nhiều bước, không real-time; gợi ý dùng consistency model ✅ (§5 arXiv).
5. **Thiếu chi tiết tái hiện:** α, m, lớp, bước, cách lấy mẫu, xử lý texel trống, thứ tự fill, AdaIN trên x̂₀ hay z_t đều không có ✅ (không nêu). Không có code ✅.
6. **Đánh giá yếu:** chỉ hai chỉ số CLIP, ablation định tính, không nêu số cảnh ✅.
7. **Ngoại hình tĩnh trong UV:** blend post-attn kéo mọi thứ về một texture chung. Bài nói pre-attn để lại chỗ cho bóng, phản xạ, nhưng tương tác phụ thuộc thời gian (nếp vải) có lẽ bị dập ⚠️. Nếp vải là thứ GReg quan tâm.
8. **Câu hỏi mở cho mình** ⚠️: GR cho thấy pre-attn + post-attn là cần (Fig. 6). Nhưng với một mô hình **đã fine-tune trên dữ liệu có điều kiện warp** (H1/H2), có còn cần pre-attn không? Có thể mô hình đã học tự "hài hoà" nên post-attn đơn thuần là đủ.

---

### 7. Liên hệ với GReg

#### 7.0 Khác biệt nền tảng (đọc trước khi mượn bất cứ thứ gì) ⚠️

| | Generative Rendering | GReg (+A1, B1) |
|---|---|---|
| Nguồn ngoại hình | Text prompt (tự do, stylized) | **Ảnh nguồn thật I_s**, phải giữ danh tính ✅ |
| Tương ứng | UV map render, không xử lý che khuất tường minh ngoài z-buffer | A1: target → source theo barycentric + **visibility** (winding, kề mặt, depth) + độ tin cậy C (`greg/geometry/mesh_flow.py:1-47`) ✅ |
| Không gian chính tắc | Atlas UV của từng vật | Hiện **không có atlas UV** trong repo (grep không thấy `smplx_uv`/`vt`). Không gian chính tắc tự nhiên = **mặt phẳng ảnh của I_s** cho phần nhìn thấy; phần bị che cần atlas UV SMPL-X (`smplx_uv.obj` từ bản tải SMPL-X) hoặc biểu diễn (face_id, barycentric) |
| Độ phân giải | 512², latent 64² | Cache 256² → latent SD 32², latent Wan 32² và nén 4× theo thời gian. **Bàn tay chỉ còn vài pixel latent** |
| Mô hình | T2I đóng băng, zero-shot | H1: video DiT + LoRA; H2: SD-Turbo một bước + LoRA |

→ Hệ quả quan trọng: lý do GR phải dùng UV noise ("inversion không dùng được trên render không texture") **không đúng với GReg**. I_s là ảnh thật nên invert được. Phiên bản hợp lý cho GReg là:
- texel/pixel **thấy được trong nguồn**: nhiễu/latent lấy từ I_s đã invert (hoặc VAE-encode + nhiễu ở mức t) rồi warp bằng A1;
- texel **bị che** (lòng bàn tay, mặt trong cẳng tay): nhiễu Gaussian gắn vào bề mặt (UV hoặc face/bary), dùng chung giữa các khung đích.

#### 7.1 (a) Khởi tạo nhiễu nhất quán theo bề mặt cho H1/H2 ⚠️

- **H2 (pix2pix-turbo một bước): gần như vô nghĩa.** pix2pix-turbo nhận ảnh làm đầu vào, chạy tất định một bước, không có nhiễu khởi tạo (`docs/diffusion_improvement.md` §2.10). Đừng tốn công.
- **H1 (Wan2.1, flow-matching nhiều bước): có thể thử nhưng rủi ro.**
  1. Latent Wan nén 4× theo thời gian, nên một latent frame ứng với 4 khung. "Nhiễu gắn bề mặt" chỉ khớp được với một khung đại diện, ví dụ khung giữa nhóm.
  2. Go-with-the-Flow cho thấy video DiT **phải fine-tune** trên nhiễu warped mới khai thác được tương quan đó ✅. Với LoRA nhỏ trên 4.6k clip thì có thể học được, nhưng phải train **cùng** kiểu nhiễu đó (cả lúc train, không chỉ lúc inference).
  3. Phải giữ Gaussianity. Nếu lấy mẫu nearest từ texture UV, dùng texture ≥ 4–8× mật độ pixel latent, hoặc dùng ∫-noise / thuật toán của Go-with-the-Flow. **Tuyệt đối không bilinear.**
- **Cách làm tối thiểu** nếu muốn: tạo noise texture trên atlas SMPL-X, render bằng rasterizer sẵn có (`greg/render/rasterize.py`, `interpolate_vertex_attribute`) với thuộc tính đỉnh = UV. Nền (camera tĩnh) dùng **cùng một nhiễu nền cho mọi khung**: nền không đổi thì fixed noise lại là đúng.
- **Ablation đề xuất cho H1:** i.i.d. vs nhiễu gắn bề mặt vs nhiễu gắn bề mặt + latent I_s warp ở vùng nhìn thấy. Đo warping error theo flow mesh và FVD.
- **Độ khó:** trung bình (rasterize UV có sẵn; cần atlas UV SMPL-X và xử lý đường nối UV).

#### 7.2 (b) Chiếu đặc trưng attention qua tương ứng mesh để refiner per-frame (H2) nhất quán thời gian ⚠️

Đây là phần **đáng mượn nhất**.
- **Không cần atlas UV.** A1 `correspond()` đã nhận hai render cùng topology bất kỳ, nên gọi được cho cặp (khung đích t−1 → t) hoặc (I_s → t). Kết quả cho π_{i,j} kèm mặt nạ V và độ tin cậy C (`greg/geometry/mesh_flow.py:128`).
- **Thiết kế** (inference-time, có thể training-free trên H2 đã train):
  - **Pre-attn:** K/V của mỗi khung t = [f_t, f_{I_s}, f_{t−1}].
    - I_s là "keyframe" cố định, giữ danh tính, giống reference-attention.
    - t−1 là anchor thời gian.
    - Thay cho việc lấy keyframe ngẫu nhiên như GR. Clip ký hiệu ngắn và có một khung nguồn rõ ràng.
  - **Post-attn:** F_out = α·V_{t−1→t}·π(F_{t−1}) + (1 − α·V)·F̂_t, chỉ ở các self-attention có độ phân giải cao nhất (32² ở cache 256²/512²). Với UNet SD-Turbo, một bước nghĩa là "mỗi bước" chỉ còn **một** lần tiêm, nên phải tiêm ở các lớp có độ phân giải cao.
  - Ở vùng không thấy (V = 0, ví dụ lòng bàn tay mới lộ ra lần đầu): **không tiêm**, để prior sinh.
  - Thay "first-filled + mean" của GR bằng trọng số theo V·C, hoặc theo cos góc nhìn như SyncMVD.
- **Lý do kỳ vọng hiệu quả:** GR Bảng 2 cho thấy tương ứng 3D giữ nhất quán **dài hạn** tốt hơn cả T2V. Ablation Fig. 6 cho thấy post-attn sửa đúng loại lỗi "quần áo đổi màu theo khung", là kiểu flicker dự kiến ở H2.
- **Rủi ro:**
  1. Ở latent 32², bàn tay chỉ còn 2–4 token. Tiêm qua tương ứng sẽ ổn định **màu áo, mặt, nền**, không giúp ngón tay; thậm chí có thể làm ngón tay mờ. Nên **tắt tiêm ở vùng tay** (mask B1) hoặc chỉ dùng α nhỏ.
  2. Tiêm theo chuỗi (t−1 → t) tích luỹ sai số và trôi. Nên neo vào I_s và chỉ dùng t−1 với trọng số nhỏ.
  3. Tiêm training-free vào mô hình một bước chưa có tiền lệ đã kiểm chứng.
  
  Phương án an toàn hơn: dùng cùng phép chiếu π làm **loss thời gian lúc train** (đã có trong §2.6 của file diffusion), còn tiêm attention chỉ là thử nghiệm phụ.
- **Độ khó:** thấp–vừa. Viết một attention processor kiểu `SamplewiseAttnProcessor2_0` của SyncMVD, cộng với hạ (grid, V) của A1 xuống từng độ phân giải đặc trưng (`F.grid_sample` nearest cho đặc trưng, max-pool / AND cho V).

#### 7.3 (c) Baseline training-free cho chương thực nghiệm ⚠️

- **"GR-for-signing":**
  - SD1.5 + ControlNet (normal map SMPL-X qua ControlNet-normal của SD1.5, hoặc depth);
  - keyframe cố định = I_s, được invert DDIM để lấy đặc trưng tham chiếu;
  - nhiễu: latent I_s warp bằng A1 ở vùng thấy, nhiễu gắn bề mặt ở vùng che;
  - post-attn qua A1;
  - AdaIN theo nền (nền tĩnh, mask nền đã có từ mesh).
- **Giá trị:** trả lời câu hỏi reviewer "prior diffusion training-free + tương ứng mesh đã đủ chưa, cần gì train?". Kỳ vọng ⚠️: nhất quán tốt nhưng **danh tính, độ khớp ngón tay kém** so với H1/H2 đã train, và chậm (50 bước × (N + m) lượt).
- **Độ khó:** vừa (≈ 3–5 ngày). Code xương sống có thể dựa trên cấu trúc pipeline SyncMVD (ControlNet + thay attn1 + bake/render), thay renderer PyTorch3D bằng tương ứng A1.
- **Khuyến nghị:** chỉ làm nếu còn thời gian sau H2. Đây là baseline phụ, không phải đóng góp chính.

#### 7.4 Những gì KHÔNG nên lấy ⚠️
- **Điều kiện bằng text / stylization.** GReg cần độ trung thực với I_s, không cần phong cách.
- **Keyframe lấy ngẫu nhiên mỗi bước.** Với GReg, I_s là keyframe đúng nghĩa và luôn có sẵn; lấy ngẫu nhiên chỉ thêm phương sai.
- **Blend "first-filled + mean".** Phụ thuộc thứ tự, là heuristic. A1 đã có V và C, nên dùng trọng số theo độ tin cậy (và/hoặc cos góc nhìn, Voronoi fill như SyncMVD) rõ ràng hơn.
- **Texture UV độ phân giải cực cao (2048–3072²) làm canonical cho đặc trưng.** Với GReg, canonical tốt hơn là mặt phẳng ảnh I_s cho phần thấy, cộng tương ứng cặp đôi π_{i,j} của A1. Tránh được siêu tham số "texture resolution" mà chính tác giả gọi là tricky.
- **Toàn bộ pipeline T2I zero-shot 50 bước làm hướng chính.** Chậm, chất lượng tay ở 256² kém, và H1 (video DiT) đã có prior thời gian tự nhiên. GR tự gợi ý "fine-tuning or adding a video module" cho hạn chế của mình (Phụ lục E).
- **Chép post-attn thẳng không qua attention** (kiểu TokenFlow): Fig. 6 "no pre-attn" chứng minh là hỏng.

#### 7.5 Câu để trích trong luận văn
- "Tương ứng ground-truth từ mesh, khi tiêm vào cả đầu vào lẫn đầu ra self-attention, cho nhất quán dài hạn vượt mô hình T2V (Bảng 2 phụ lục B của Cai et al. 2024)". Đây là lập luận ủng hộ việc dùng tương ứng SMPL-X của A1 làm tín hiệu thời gian cho thành phần diffusion.
- "Tương ứng hình học không đồng nghĩa với tương ứng đặc trưng; chép thẳng đặc trưng gây hỏng (Fig. 6)". Đây là lập luận cho việc đưa ảnh warped vào làm **điều kiện** (H1) thay vì dán đè, và cho ablation GenWarp đã nêu ở §1.3 của `docs/diffusion_improvement.md`.

---
**Nguồn:**
- CVPR OA PDF: `ref papers/GenerativeRendering_CVPR2024.pdf`
- arXiv HTML (phụ lục A–E): https://arxiv.org/html/2312.01409
- Project: https://primecai.github.io/generative_rendering/
- SyncMVD: https://github.com/LIU-Yuxin/SyncMVD @faa9562
- Tex4D: https://github.com/ZqlwMatt/Tex4D @a5c9c45
- Go-with-the-Flow: https://github.com/Eyeline-Labs/Go-with-the-Flow
- HIWYN (không chính thức): https://github.com/mskwak01/How-I-Warped-Your-Noise-Unofficial
- Patent: US 12,664,751 (claims theo snippet tìm kiếm; PDF scan không đọc được)

---

## Bài 6 — Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models (ICLR 2026)

Quy ước: ✅ = kiểm chứng từ bài báo (có trích mục/bảng/phương trình); ⚠️ = suy luận/đánh giá của người viết, chưa có trong bài.

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Marc Benedí San Millán, Angela Dai, Matthias Nießner (TUM) ✅ |
| Venue | ICLR 2026 (poster) ✅. arXiv 2503.15996. OpenReview `DIPeQTxpe7` |
| Project page | https://marcb.pro/atu/ (có paper, supp, video YouTube, trang "more results") |
| Code | **Chưa có.** Ngày 2026-10-29 project page ghi nút Code là "🚧Coming soon🚧" và trỏ tới https://github.com/marcbenedi (trang hồ sơ, không có repo riêng). Phụ lục A có link ẩn danh `https://anonymous.4open.science/r/aj93jkfjl8/`, nhưng API trả về `{"error":"repository_expired"}` → link này đã chết. |
| Video model | **Kling AI** (đóng nguồn, thương mại) cho mọi kết quả chính ✅ §4.1, A.4. Có thử thêm CogVideoX (Fig. 7) và Wan2.2 (Fig. 8, A.4.1) |
| Thư viện | PyTorch 2.0.1, CUDA 11.7, PyTorch3D 0.7.7 (render khả vi), SMPL + VPoser, MediaPipe Pose, DINOv2 ✅ §4.1 |
| Compute | Toàn bộ tối ưu khoảng **1.5 giờ / chuỗi** trên **RTX 2080 Ti 12 GB**, RAM 16 GB, 4 nhân CPU Xeon Gold 6240 ✅ §4.1. Thử với Wan2.2 thì dùng A100 80 GB ✅ A.4.1. Thời gian sinh video bằng Kling không được báo cáo. |
| Mô hình thân người | **SMPL** (không phải SMPL-X): không có khớp ngón tay, không có biểu cảm mặt ✅ §3.1 |

---

### 1. TL;DR

- **Bài toán:** cho một mesh người tĩnh (không rig, tư thế tùy ý) và một câu mô tả hành động, sinh chuỗi 4D để mesh làm đúng hành động đó. ✅
- **Cách làm:** (1) render mesh thành ảnh xám, dùng ảnh này làm **khung đầu** cho một video diffusion image+text→video (Kling), **không fine-tune**; (2) fit SMPL vào mesh, rồi "neo" từng đỉnh mesh vào mặt SMPL gần nhất bằng tọa độ barycentric cộng khoảng cách theo pháp tuyến; (3) **track** video sinh ra bằng SMPL, tối ưu qua render khả vi với ba tín hiệu: landmark 2D MediaPipe (Geman-McClure), silhouette (BCE), đặc trưng dày DINOv2 (cosine). Kèm theo các regularizer: VPoser, phạt gập khuỷu/gối, mượt theo thời gian, ARAP. Tham số pose/translation/rotation theo thời gian được **sinh bởi MLP nhỏ nhận positional encoding của chỉ số khung**, không tối ưu trực tiếp. ✅ §3
- **Kết quả:** trên 26 chuỗi CAPE render không texture, MPJPE 0.036 / PVE 0.041 / Accel 1.49, tốt hơn SMPLify-X, WHAM, Multi-HMR ✅ Tab. 1. Trong user study 30 người, họ được ưa thích hơn MDM, FineMoGen, MotionDiffuse, ReMoDiffuse, MotionCLIP ✅ Fig. 4.
- **Với GReg:** bài này đi **ngược chiều** với mình (video sinh → SMPL). Giá trị chính cho mình: (a) một **bộ fit SMPL trên video sinh** có thể dùng làm **chỉ số đánh giá hình học** cho video ký hiệu sinh ra; (b) công thức **neo đỉnh vào mặt SMPL** (eq. 4) để mở rộng flow A1 ra quần áo/tay áo; (c) bằng chứng định lượng rằng **tham số hóa theo thời gian bằng MLP + L_temp** làm giảm rung mạnh (Accel 2.65 → 1.50, Tab. 2). Phần conditioning video diffusion **rất thô** (chỉ khung đầu + prompt), gần như không dùng được cho H1.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Đầu vào** ✅ §3:
- Mesh người `S` không rig, không skeleton, tư thế bất kỳ. Phần lớn ví dụ là nhân vật Mixamo có texture, nhưng **ảnh đưa vào VDM là ảnh render xám** (prompt có "Light grey person"; Fig. 2, 6, 7, 8 đều cho thấy video xám).
- Prompt văn bản `P` mô tả hành động.

**Đầu ra:** tham số biến dạng `Θ = {Θ_t}_{t=0}^{F−1}`, với `Θ_t = (s, β, θ_t, T_t, R_t, Δ_t)`. Áp `Θ` lên `S` cho ra chuỗi mesh `S_t`. ✅ §3.3

**Vì sao khó** (theo bài + bổ sung):
1. Dữ liệu mocap 4D có cặp text–motion rất ít, nên các mô hình text-to-motion (MDM…) khái quát kém ✅ §1. Ý tưởng của họ là "mượn" prior chuyển động từ VDM huấn luyện trên dữ liệu video lớn.
2. Video do VDM sinh ra là **ngoài phân phối** với các bộ ước lượng pose kiểu regression (HMR, WHAM, Multi-HMR), vì đó là người xám, không texture, có artefact. Do đó họ chọn **tối ưu hóa** thay cho regression ✅ §2, §4.2.
3. Tracking đơn mắt vốn thiếu ràng buộc (mơ hồ độ sâu), và VDM có thể **morph** (lật mặt trước thành mặt sau) ✅ §4.4, Fig. 6.
4. Mesh đầu vào không có mô hình biến dạng, nên phải dựng một proxy (SMPL) và gắn mesh vào proxy đó ✅ §3.3.1.

---

### 3. Phương pháp chi tiết

#### 3.1 Tiền xử lý và sinh video ✅ §3.2

1. Chuẩn hóa `S` về unit scale, dời tâm về gốc tọa độ. Hàm render `P(S, C) → I ∈ R^{H×W×3}`.
2. **Sinh video:** điều kiện VDM bằng **ảnh render khung đầu `I_0^RGB` + prompt `P`**. Khung đầu của video sinh ra trùng với `I_0^RGB`, nên `D(S, Θ_0) = S`. Không fine-tune, không có điều kiện hình học nào cho các khung sau ✅ §2 ("does not require any fine-tuning").
3. **Prompt đầy đủ** ✅ A.2 (ACTION được thay bằng hành động):
   > "An award winning documentary about a person ACTION. Energetically ACTION. Dynamic movement. Light grey person. Realistic movement. Realistic motion. Realistic human body. Wide angle shot showcasing the man, dynamic movement, this video is incredibly detailed and high resolution, the uniform light is impressive, a masterpiece. Clear illumination. Bright light. No dark light. Fixed camera. No zoom in."

   Mục đích từng cụm, theo tác giả: "dynamic" để video không đứng yên; "realistic human body" để tránh morph thân; "fixed camera / no zoom" để camera cố định; "bright light" để tránh ánh sáng tối ✅ A.2.
4. **Video model:** Kling AI, video 5 giây ở **768×1280**, được **downsample về 384×640** khi tối ưu vì giới hạn bộ nhớ ✅ §4.1. fps và số khung F **không được nêu** ⚠️.

#### 3.2 Đặc trưng trích từ khung hình ✅ §3.2

| Tín hiệu | Cách lấy |
|---|---|
| Landmark 2D | MediaPipe Pose Landmarker, **33 landmark** mỗi khung, mỗi điểm `(j_i, ω_i)` = (tọa độ pixel chuẩn hóa, độ tin cậy). Sắp lại theo thứ tự khớp SMPL. Có **lọc làm mượt** để giảm nhiễu (loại bộ lọc không nêu ⚠️). |
| Đặc trưng dày video | DINOv2 per-pixel → `{I_t^φ}`. Biến thể DINOv2 (ViT-S/B/L/G), tầng, và cách upsample **không nêu** ⚠️. |
| Đặc trưng trên đỉnh mesh | Theo **Diff3F** (Dutt et al. 2024): render mesh từ **100 camera phân bố đều trên mặt cầu**, trích DINOv2 mỗi view, back-project và cộng dồn lên đỉnh → `V^φ = {v_i^φ}`. Sau đó **PCA giữ 64 thành phần** (do bộ nhớ). |
| Silhouette | Ngưỡng hóa nền trắng của khung video → `I^sil`. |

#### 3.3 Tham số mô hình ✅ §3.3

- `s`: scale; `β ∈ R^10`: shape. Hai tham số này **dùng chung cho mọi khung**.
- Theo từng khung: `θ_t` (pose), `T_t` (translation), `R_t` (global rotation), `Δ_t` (offset theo đỉnh).
- Pose dùng **VPoser**: tối ưu latent `Z ∈ R^32`, decode ra `θ ∈ R^{23×3}`. Bài dùng chung ký hiệu `θ` cho cả hai ✅ §3.1. (⚠️ VPoser thực ra decode 21 khớp thân; bài ghi 23×3, có lẽ là cách viết gọn.)
- **Tham số hóa neural:** `θ_t = θ_MLP(PE(t))`, `T_t = T_MLP(PE(t))`, `R_t = R_MLP(PE(t))`. Mỗi MLP có **4 tầng, 128 hidden units**, đầu vào là **positional encoding sinusoid 64 chiều** của chỉ số khung ✅ §3.3, §4.1. `Δ_t` có vẻ là tensor tối ưu trực tiếp, không qua MLP (Fig. 2 vẽ `Δ_{t=k}` như một vector tham số) ⚠️.
- Lý do dùng MLP: thiên kiến quy nạp của MLP với PE làm mượt nhiễu landmark ✅ §3.3, Tab. 2.

#### 3.4 Giai đoạn 1: đăng ký SMPL vào mesh (SMPL registration) ✅ §3.3.1

Tối ưu `s, β, θ_0, T_0, R_0`:

- **3D joint:** `L_J = (1/N) Σ_i ω_i ‖Ĵ_i − J_i‖²₂` (eq. 1). `J` là khớp 3D thu được bằng cách **triangulate landmark 2D MediaPipe trên nhiều view render quanh mesh**. `Ĵ` là khớp SMPL.
- **Point-to-point:** `L_p2p(V) = (1/|V|) Σ_{v∈V} ‖v − NN(v, V^SMPL)‖²₂` (eq. 2). Phương trình đi theo chiều đỉnh mesh đầu vào → đỉnh SMPL gần nhất, trong khi lời văn lại mô tả chiều ngược lại (SMPL → mesh). Đây là **mâu thuẫn nhỏ trong bài**, chỉ là loss một chiều ⚠️.
- **Prior:** `L_β = ‖β‖²₂`, `L_θ = ‖θ_t‖²₂` (eq. 3). Vì `θ` là latent VPoser, `L_θ` thực chất là prior Gaussian trên latent ⚠️.
- **Lịch tối ưu:** Adam, lr **0.01**, batch size 8, **1000 iteration** hoặc tới khi hội tụ ✅ §4.1.
- **Trọng số giữa các loss: KHÔNG được báo cáo.**

#### 3.5 SMPL làm proxy biến dạng: "neo" đỉnh ✅ §3.3.1, eq. 4

Sau khi đăng ký, với mỗi đỉnh `v` của mesh đầu vào, tìm mặt SMPL gần nhất `f^SMPL = (v1, v2, v3)` rồi biểu diễn:

```
v = γ1·v1^SMPL + γ2·v2^SMPL + γ3·v3^SMPL + d·n          (eq. 4)
```

- `γ_i`: tọa độ barycentric của hình chiếu `v` lên mặt.
- `d`: khoảng cách có dấu tới mặt phẳng tam giác.
- `n`: pháp tuyến của mặt SMPL.

Hàm khả vi này ký hiệu là `Γ: V^SMPL → V`. Khi SMPL đổi pose, `(γ, d)` giữ nguyên, còn `v_i^SMPL, n` thay đổi, nên đỉnh mesh "đi theo" mặt SMPL.

**Lọc outlier** (đỉnh gắn sai), kết hợp ba tiêu chí ✅:
1. Khoảng cách tuyệt đối giữa đỉnh mesh và SMPL (ngưỡng không nêu ⚠️);
2. Góc giữa pháp tuyến đỉnh và pháp tuyến mặt SMPL gần nhất phải **≤ 45°**;
3. Thống kê lân cận: trung bình và độ lệch chuẩn khoảng cách giữa các đỉnh, để loại các điểm lệch hẳn so với hàng xóm.

Bài **không nói** đỉnh bị loại thì được animate thế nào (nội suy từ hàng xóm? gán mặt khác? hay chỉ bỏ khỏi loss?) ⚠️.

#### 3.6 Giai đoạn 2: tracking video và chuyển chuyển động ✅ §3.3.2

Mô hình biến dạng (eq. 5). Giữ cố định `β, s`; tối ưu `T_t, R_t, θ_t, Δ_t`:

```
D(S, Θ_t) = Γ( s · R_t · SMPL(β, θ_t) + T_t ) + Δ_t          (eq. 5)
```

**Hàm mục tiêu** (eq. 6), tổng bắt đầu từ t = 1 vì khung 0 đã cố định:

```
L_total = (1/F) Σ_{t=1}^{F−1} ( L_j + L_sil + L_φ ) + L_regs
```

**Data terms:**
- **Landmark (eq. 7):** `L_j = (1/N) Σ_i w_i · ρ(ĵ_i − j_i)`. `ĵ` là khớp SMPL chiếu xuống 2D, `w` là confidence của MediaPipe, `ρ` là **Geman-McClure** (robust). Tham số σ của GMoF không nêu ⚠️.
- **Silhouette (eq. 8):** BCE theo pixel giữa silhouette render khả vi `Î^sil` và silhouette video `I^sil`, chia cho `N = HW`.
- **Dense feature (eq. 9):** `L_φ = (1/N) Σ_i (1 − cos(Î^φ_{t,i}, I^φ_{t,i}))`. `Î^φ` là ảnh đặc trưng có được khi rasterize đặc trưng đỉnh `V^φ` (sau feature mapper) tại tư thế hiện tại.

**Regularizers** (`L_regs`):
- **Temporal (eq. 10):** `L_temp(x) = Σ_t ‖x_t − x_{t−1}‖₂` (chuẩn L2, **không bình phương** theo công thức), áp cho **translation, rotation, pose và vị trí khớp 3D** ✅.
- **Phạt gập quá mức (eq. 11):** `L_ex.ben.(θ) = Σ_{i∈{khuỷu, gối}} exp{θ_i}`, lấy cảm hứng từ SMPLify (Bogo 2016). Lời văn lại nói "tổng **bình phương góc**", không khớp công thức exp. Trong SMPLify gốc là `exp(±θ_i)` có dấu theo hướng gập ⚠️.
- **Prior VPoser:** `L_θ` (eq. 3).
- **ARAP** (Sorkine & Alexa 2007) trên biến dạng mesh, để mượt và giữ cấu trúc nội tại. Chủ yếu ràng buộc `Δ_t` ⚠️. Không có công thức trong bài.

**Feature mapper:** một **phép chiếu học được** biến đặc trưng đỉnh `V^φ` sang không gian khớp hơn với `I^φ` của video. Lý do: ngoại hình mesh và video lệch dần theo thời gian. Phép chiếu được tối ưu "tự giám sát". Kiến trúc (tuyến tính 64→64? MLP?), có chia sẻ giữa các khung hay không, và loss của nó **không được mô tả** ⚠️.

**Lịch tối ưu:** Adam, lr **0.001**, batch size 8, **4000 iteration** hoặc tới khi hội tụ ✅ §4.1. "Batch 8" có lẽ là 8 khung mỗi bước ⚠️. Không rõ 8 khung đó liên tiếp hay ngẫu nhiên; với `L_temp` thì cần các cặp khung kề nhau.

**Trọng số λ cho mọi loss ở giai đoạn 2 cũng KHÔNG được báo cáo.** Đây là lỗ hổng tái lập lớn nhất của bài.

---

### 4. Kết quả & ablation quan trọng

#### 4.1 Tracking có ground truth: CAPE ✅ §4.2, Tab. 1

Thiết lập: 26 chuỗi CAPE chọn ngẫu nhiên, tổng 7,000 khung. Họ **render mesh SMPL không texture** (không phải mesh quần áo) và dùng tham số SMPL của registration SMPL+D làm GT. Baseline WHAM và SMPLify-X được **Procrustes-align** với mesh khung đầu. SMPLify-X* là SMPLify-X cộng bộ lọc làm mượt.

| Method | MPJPE | PVE | Accel |
|---|---|---|---|
| SMPLify-X | 0.054 | 0.057 | 20.57 |
| SMPLify-X* | 0.053 | 0.056 | 2.18 |
| WHAM | 0.051 | 0.054 | 8.61 |
| Multi-HMR | 0.045 | **0.510** | 9.03 |
| **Ours** | **0.036** | **0.041** | **1.49** |

Nhận xét ⚠️:
- Đơn vị không ghi. Nhiều khả năng là mét (3.6 cm) cho MPJPE/PVE; đơn vị của Accel cũng không rõ.
- PVE 0.510 của Multi-HMR gần như chắc chắn là lỗi đánh máy (gấp 10 lần MPJPE của chính nó).
- Thiết lập này **rất thuận lợi** cho họ: video là SMPL render sạch, không có morph, và mesh đầu vào chính là SMPL, nên phép neo Γ gần như đồng nhất. Họ cũng biết mesh khung đầu, trong khi các baseline không được dùng thông tin đó (chỉ được align sau).
- Các baseline regression bị thiệt vì video không texture nằm ngoài phân phối huấn luyện. Bản thân tác giả cũng thừa nhận điều này ✅.

#### 4.2 Mixamo ✅ A.6.1, Tab. 3
50 animation Mixamo ngẫu nhiên, render rồi track. PVE: SMPLify-X 0.099, WHAM 0.066, **Ours 0.058**.

#### 4.3 Ablation ✅ A.1, Tab. 2 (cùng thiết lập CAPE)

| Biến thể | MPJPE | PVE | Accel |
|---|---|---|---|
| w/o L_θ (VPoser prior) | 0.0486 | 0.0556 | 1.7121 |
| w/o L_ex.ben. | 0.0393 | 0.0453 | 1.5699 |
| w/o L_φ (DINOv2) | 0.0392 | 0.0447 | 1.5895 |
| w/o L_temp | 0.0403 | 0.0458 | **3.2005** |
| Opt. Parameters (bỏ MLP, tối ưu trực tiếp) | 0.0453 | 0.0533 | 2.6494 |
| **Full** | **0.0362** | **0.0411** | **1.4981** |

Đọc bảng:
- **Prior pose (VPoser) quan trọng nhất cho độ chính xác** (+34% MPJPE khi bỏ).
- **MLP theo thời gian** là thành phần quan trọng thứ hai (+25% MPJPE, Accel ×1.77). Đây là bằng chứng rằng "neural parameterization" có tác dụng như một prior mượt mạnh.
- **L_temp** chủ yếu cải thiện độ mượt (Accel ×2.14 khi bỏ).
- **DINOv2 (L_φ) chỉ giúp ~8% MPJPE** trong thiết lập này ⚠️. Thiết lập lại là SMPL xám đồng màu, nơi DINOv2 ít thông tin nhất; với mesh có texture thì có thể giúp nhiều hơn, nhưng bài không đo.
- **Không có ablation** cho silhouette, ARAP, feature mapper, `Δ_t`, hay bộ lọc outlier. Cũng **không có ablation trên video sinh thật** (chỉ có video render).

#### 4.4 Video sinh thật: user study ✅ §4.3, Fig. 4, A.3
- 30 người tham gia, 17 cặp chuyển động (front + side view), thứ tự ngẫu nhiên. Danh sách 17 prompt có ở A.3.1 (ví dụ "taking cover and shooting", "nunchaku", "tying her shoes", "brushing her hair"…).
- Ba câu hỏi: Q1 khớp prompt, Q2 tự nhiên, Q3 thích tổng thể; cộng thang Likert 0–5 cho realism và prompt adherence.
- Đọc từ biểu đồ (giá trị **xấp xỉ**, bài không in số) ⚠️: tỉ lệ chọn Ours so với MDM khoảng 60–64% ở cả ba câu; so với MotionCLIP khoảng 87–89%. Likert của Ours: khoảng 3.8 (adherence) và 4.1 (natural); MotionDiffuse cao thứ hai, khoảng 3.2 / 3.7.
- Lời văn ghi "Figure 3 shows that participants prefer…", nhưng đúng ra phải là Fig. 4 (lỗi trong bài).
- **Không có metric định lượng nào cho video sinh**: không FID/FVD, không đo độ giữ danh tính hay hình học giữa video và mesh.

#### 4.5 Về chất lượng conditioning của VDM (quan sát từ hình) ⚠️
- **Kling:** video xám giữ dáng tổng thể khá tốt (Fig. 6). Tuy vậy ở khung cuối của "bouldering", VDM **không còn biểu diễn đúng mesh** (xuất hiện tảng đá, người co lại) ✅ Fig. 6 caption.
- **CogVideoX** (Fig. 7, "The person is running"): 5 khung gần như **đứng yên**, tức chuyển động rất yếu ⚠️. Điều này khớp với việc tác giả viết thêm "Dynamic movement / Energetically" vào prompt.
- **Wan2.2** (Fig. 8, "playing tennis"): người xám, VDM **tự thêm vợt tennis** và đổi phần đầu (mũ/tóc tối màu). Tracking vẫn chạy được ⚠️.
- **Failure chính** ✅ §4.4.1, Fig. 6: VDM morph mặt trước thành mặt sau. Tracker "giải thích" việc này bằng một **phép xoay 180°** mượt của mesh. Nhờ MLP + temporal, mesh **không sụp**, nhưng chuyển động sai ngữ nghĩa.

---

### 5. Code

Không có code công khai (xem §0). Dưới đây là **pseudo-code PyTorch** dựng lại từ bài. Những chỗ bài không nêu (λ, σ, ngưỡng) được đánh dấu `# ⚠️` và chỉ là giá trị giả định.

#### 5.1 Neo đỉnh vào mặt SMPL (eq. 4) + lọc outlier

```python
import torch, torch.nn.functional as F
from pytorch3d.ops import knn_points
# pytorch3d.ops.closest point-to-face: dùng point_mesh_face_distance nội bộ / hoặc tự cài

def face_normals(V, Fc):                       # V:(Ns,3) Fc:(Nf,3) long
    a, b, c = V[Fc[:,0]], V[Fc[:,1]], V[Fc[:,2]]
    n = torch.cross(b - a, c - a, dim=-1)
    return F.normalize(n, dim=-1)

def closest_face_barycentric(P, V, Fc):
    """P:(Np,3) điểm mesh đầu vào; trả về face id, (γ1,γ2,γ3), d có dấu."""
    # Lọc ứng viên bằng tâm mặt (k gần nhất) rồi chiếu chính xác
    cen = V[Fc].mean(1)                                        # (Nf,3)
    k = knn_points(P[None], cen[None], K=16).idx[0]            # (Np,16)
    tri = V[Fc[k]]                                             # (Np,16,3,3)
    n = face_normals(V, Fc)[k]                                 # (Np,16,3)
    d = ((P[:,None] - tri[:,:,0]) * n).sum(-1)                 # khoảng cách có dấu tới mặt phẳng
    proj = P[:,None] - d[...,None] * n                         # chiếu lên mặt phẳng
    g = barycentric(proj, tri)                                 # (Np,16,3), có thể ra ngoài [0,1]
    g_cl = clamp_to_triangle(g)                                # chiếu vào trong tam giác
    q = (g_cl[...,None] * tri).sum(-2)
    dist = (P[:,None] - q).norm(dim=-1)
    j = dist.argmin(1)
    ar = torch.arange(len(P))
    return k[ar, j], g_cl[ar, j], d[ar, j], dist[ar, j]

def anchor(P, Pn, V_smpl, F_smpl, ang_thr=45., dist_thr=0.05, z_thr=3.0):   # ⚠️ dist_thr, z_thr
    fid, gam, d, dist = closest_face_barycentric(P, V_smpl, F_smpl)
    fn = face_normals(V_smpl, F_smpl)[fid]
    ok = dist < dist_thr                                                   # tiêu chí (1)
    ok &= (F.cosine_similarity(Pn, fn, -1) > torch.cos(torch.deg2rad(torch.tensor(ang_thr))))  # (2) ≤45°
    # (3) thống kê lân cận: d của đỉnh lệch nhiều so với trung bình hàng xóm
    nb = knn_points(P[None], P[None], K=8).idx[0]
    mu, sd = d[nb].mean(1), d[nb].std(1) + 1e-6
    ok &= ((d - mu).abs() / sd) < z_thr
    return dict(fid=fid, gam=gam, d=d, valid=ok)

def Gamma(V_smpl_posed, F_smpl, A):
    """Γ: V^SMPL → V (khả vi). A = kết quả anchor()."""
    tri = V_smpl_posed[F_smpl[A['fid']]]                    # (Np,3,3)
    n = face_normals(V_smpl_posed, F_smpl)[A['fid']]
    return (A['gam'][...,None] * tri).sum(1) + A['d'][:,None] * n
```

#### 5.2 Tham số hóa theo thời gian bằng MLP

```python
def sinusoidal_pe(t, dim=64, max_period=10000.):            # t: (B,) chỉ số khung
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)

class MLP(torch.nn.Module):                                 # 4 tầng × 128 hidden ✅
    def __init__(s, out, d_in=64, h=128):
        super().__init__()
        s.net = torch.nn.Sequential(
            torch.nn.Linear(d_in,h), torch.nn.ReLU(), torch.nn.Linear(h,h), torch.nn.ReLU(),
            torch.nn.Linear(h,h), torch.nn.ReLU(), torch.nn.Linear(h,out))
    def forward(s, pe): return s.net(pe)

class Track(torch.nn.Module):
    def __init__(s, F_frames, Nv, z0, R0, T0):
        super().__init__()
        s.zmlp = MLP(32)      # latent VPoser
        s.rmlp = MLP(6)       # ⚠️ biểu diễn xoay (axis-angle 3 hay 6D?) không nêu
        s.tmlp = MLP(3)
        s.delta = torch.nn.Parameter(torch.zeros(F_frames, Nv, 3))   # Δ_t trực tiếp ⚠️
        s.register_buffer('z0', z0); s.register_buffer('R0', R0); s.register_buffer('T0', T0)
        s.feat_map = torch.nn.Linear(64, 64)                         # feature mapper ⚠️ kiến trúc không nêu
    def forward(s, t):
        pe = sinusoidal_pe(t)
        # ⚠️ bài không nói cách ép khung 0 = registration; một cách: residual + offset bằng 0 tại t=0
        z = s.z0 + s.zmlp(pe) - s.zmlp(sinusoidal_pe(torch.zeros_like(t)))
        R = rot6d_to_mat(s.rmlp(pe)) @ s.R0      # tương tự cho R, T
        T = s.T0 + s.tmlp(pe)
        return z, R, T, s.delta[t]
```

#### 5.3 Vòng tối ưu tracking (eq. 5–11)

```python
def gmof(x, sigma=100.):                                     # Geman-McClure; σ ⚠️ (SMPLify dùng 100 px)
    x2 = (x**2).sum(-1)
    return sigma**2 * x2 / (sigma**2 + x2)

lam = dict(j=1., sil=1., phi=1., temp=1., bend=1., pose=1., arap=1.)   # ⚠️ KHÔNG có trong bài

opt = torch.optim.Adam(model.parameters(), lr=1e-3)                    # ✅
for it in range(4000):                                                 # ✅
    t = sample_window(F_frames, B=8)                                   # ⚠️ 8 khung liên tiếp để có L_temp
    z, R, T, Dt = model(t)
    theta = vposer.decode(z)                                           # (B,21,3)
    Vs = smpl(betas=beta_fixed, body_pose=theta).vertices              # β, s cố định
    Vs = s_fixed * (Vs @ R.transpose(-1,-2)) + T[:,None]
    Vin = Gamma(Vs, F_smpl, A) + Dt                                    # eq.5
    J3d = smpl_joints(Vs)

    # data terms
    j2d = cam.project(J3d)
    L_j   = (w_mp[t] * gmof(j2d - j_mp[t])).mean()                     # eq.7
    sil   = soft_silhouette(Vin, F_in, cam)                            # PyTorch3D SoftSilhouetteShader
    L_sil = F.binary_cross_entropy(sil.clamp(1e-4,1-1e-4), Isil[t])    # eq.8
    feat  = rasterize_vertex_features(Vin, F_in, model.feat_map(Vphi), cam)  # Î^φ
    m = (sil > 0.5) & (Isil[t] > 0.5)                                  # ⚠️ mask vùng tính cosine
    L_phi = (1 - F.cosine_similarity(feat, Iphi[t], dim=1))[m].mean()  # eq.9

    # regularizers
    L_temp = sum((x[1:] - x[:-1]).norm(dim=-1).sum() for x in (T, R.flatten(1), theta.flatten(1), J3d.flatten(1)))  # eq.10
    L_bend = torch.exp(theta[:, ELBOW_KNEE_IDX, BEND_AXIS] * BEND_SIGN).sum()  # eq.11 (dấu kiểu SMPLify ⚠️)
    L_pose = (z**2).sum(-1).mean()                                     # eq.3
    L_arap = arap_energy(Vin, V_rest=V_input, F=F_in)                  # Sorkine & Alexa

    L = lam['j']*L_j + lam['sil']*L_sil + lam['phi']*L_phi \
      + lam['temp']*L_temp + lam['bend']*L_bend + lam['pose']*L_pose + lam['arap']*L_arap
    opt.zero_grad(); L.backward(); opt.step()
```

#### 5.4 Đặc trưng đỉnh kiểu Diff3F (rút gọn)

```python
@torch.no_grad()
def vertex_dino_features(mesh, n_views=100, pca_dim=64):
    acc = torch.zeros(Nv, Cdino); cnt = torch.zeros(Nv)
    for cam in fibonacci_sphere_cameras(n_views):             # 100 view đều trên mặt cầu ✅
        img = render_rgb(mesh, cam)
        f = upsample(dino(img), img.shape[-2:])               # (C,H,W)
        pix2face = rasterize(mesh, cam).pix_to_face
        vis_v, uv = visible_vertices_and_pixels(pix2face, mesh, cam)
        acc[vis_v] += f[:, uv[:,1], uv[:,0]].T; cnt[vis_v] += 1
    Vf = acc / cnt.clamp(min=1)[:,None]
    return pca_fit_transform(Vf, pca_dim)                     # PCA 64 ✅ (video features chiếu cùng basis ⚠️)
```

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Không tái lập được:** VDM chính là Kling (đóng nguồn); không có trọng số loss; feature mapper không được mô tả; code chưa phát hành, link ẩn danh đã hết hạn.
2. **Đánh giá tracking không phản ánh bài toán thật:** CAPE/Mixamo là video **render sạch**, không phải video sinh. Trên video sinh chỉ có user study (30 người, 17 cặp), không có metric hình học. Baseline regression bị đặt vào bất lợi (video không texture), còn phương pháp của họ lại được biết mesh khung đầu.
3. **Conditioning VDM rất yếu:** chỉ có khung đầu + prompt. Không có ràng buộc hình học cho các khung sau, nên xuất hiện morph, vật thể lạ (vợt, đá), và người co lại. Tracker chỉ "hấp thụ" lỗi chứ không sửa được (lật 180°, Fig. 6).
4. **SMPL không có tay/mặt:** không ngón tay, không biểu cảm. MediaPipe Pose 33 điểm chỉ có cổ tay và vài điểm bàn tay. Hoàn toàn không phù hợp cho ngôn ngữ ký hiệu ⚠️.
5. **Một số mâu thuẫn trong ký hiệu:** L_p2p một chiều so với lời văn; L_ex.ben. ghi "squared angles" nhưng công thức dùng exp; "Figure 3" thay vì Fig. 4; PVE của Multi-HMR 0.510.
6. **Chi phí:** 1.5 giờ / chuỗi 5 s trên 2080 Ti. Dùng làm metric cho hàng nghìn clip là quá đắt (xem §7).
7. **Câu hỏi mở:** độ mượt do MLP + L_temp có làm **mất chuyển động nhanh** không? Bài chỉ đo Accel error trên CAPE (chuyển động chậm/vừa). Với chuyển động nhanh như ký hiệu tay, rủi ro over-smoothing là thực ⚠️.

---

### 7. Liên hệ với GReg

Khác biệt nền tảng: GReg **biết trước** chuyển động (SMPL-X đích từ SignAvatars) và cần **sinh ngoại hình** đúng. ATU làm ngược lại: biết ngoại hình (mesh) và cần **trích chuyển động** từ video sinh. Vì vậy những gì lấy được chủ yếu nằm ở **đánh giá** và **hình học proxy**, không nằm ở kiến trúc sinh.

#### 7.1 Nên lấy

**(a) Metric "re-fit consistency": video sinh có thực sự theo SMPL-X đích không** ⚠️. Độ khó: trung bình.
- Ý tưởng: chạy một bộ fit/track trên video GReg (hoặc H1/H2) sinh ra, rồi so với SMPL-X đích đã dùng làm điều kiện → MPJPE/PVE và Accel error (Accel lấy đúng định nghĩa ATU/WHAM).
- **Không chạy nguyên pipeline của ATU** (1.5 giờ/clip × 1146 clip test ≈ 1,700 giờ GPU 2080 Ti). Hai cách rẻ hơn:
  1. *Bản 2D, rẻ:* chiếu khớp SMPL-X đích (camera đã có trong SignAvatars) rồi so với keypoint phát hiện trên khung sinh (RTMPose/DWPose, gồm 21 điểm mỗi tay). Dùng **GMoF + trọng số confidence** như eq. 7 để metric bền với outlier. Đây chính là "AKD có trọng số theo SMPL-X". Nó bổ sung cho kế hoạch đo MPJPE tay bằng RTMPose trong `diffusion_improvement.md` §1.8, và tránh được metric vòng tròn vì điều kiện bằng SMPL-X còn đo bằng RTMPose.
  2. *Bản 3D:* re-fit SMPL-X ngắn (khởi tạo bằng SMPL-X đích, khoảng 100–200 iteration, chỉ data term keypoint + prior) trên một tập con cố định khoảng 100 clip. Báo cáo độ lệch so với khởi tạo. ⚠️ Cần có baseline "re-fit trên video thật" để biết sàn nhiễu của chính bộ fit (SignAvatars đã sai khoảng 2.5 px).
- Lợi ích: định lượng được "sai hình dạng tay = sai nghĩa", điều mà L1/LPIPS không bắt được.

**(b) Silhouette / part-mask consistency làm metric hoặc loss phụ** ⚠️. Độ khó: thấp.
- Ta đã có mask phần thân từ mesh (B1). Có thể đo IoU giữa mask tay render từ SMPL-X đích và mask tay phát hiện trên khung sinh (bằng segmenter, ví dụ SAM/Sapiens-seg) → "hand silhouette IoU".
- Làm **loss** thì rủi ro hơn: cần segmenter khả vi, và quần áo/tay áo làm silhouette người thật ≠ silhouette SMPL-X. Chỉ nên áp cho **vùng bàn tay** (không có quần áo che), đúng tinh thần BCE eq. 8.

**(c) DINOv2 làm loss tương ứng ngữ nghĩa / tri giác** ⚠️. Độ khó: thấp, lợi ích không chắc.
- Khi huấn luyện GReg, ta **có khung đích GT**. Vì thế phiên bản hợp lý là `1 − cos(DINOv2(Î_t), DINOv2(I_t))` theo patch, có mask B1, như eq. 9. Thực chất đây là một perceptual loss kiểu DINO, không cần tới phần render đặc trưng đỉnh.
- Lưu ý độ phân giải: DINOv2 dùng patch 14. Ở 256² chỉ có lưới khoảng 18×18, bàn tay chiếm khoảng 2–3 patch, nên **vô dụng cho ngón tay**. Chỉ hữu ích cho tổng thể thân/áo/danh tính. Ablation của chính ATU cũng cho thấy L_φ chỉ đóng góp khoảng 8% MPJPE.
- Biến thể đúng tinh thần ATU hơn, dùng được cho **cross-identity reenactment (không có GT)**: gán đặc trưng DINOv2 của ảnh nguồn lên các đỉnh SMPL-X nhìn thấy ở khung nguồn, rasterize lên tư thế đích (dùng chính hạ tầng flow A1), rồi so cosine với DINOv2 của khung sinh trên vùng `V_geo = 1`. Kết quả là metric "giữ danh tính theo tương ứng hình học".

**(d) Neo kiểu eq. 4 để mở rộng flow A1 ra ngoài mesh (tay áo, tóc, quần áo rộng)** ⚠️. Độ khó: trung bình–cao.
- A1 hiện chỉ có tương ứng **trên bề mặt SMPL-X**. Pixel áo/tóc nằm ngoài silhouette SMPL-X ở khung nguồn sẽ không có flow (hoặc bị đánh dấu `C` thấp).
- Analog 2.5D của eq. 4: với mỗi pixel nguồn thuộc người (mask segmentation) nhưng ngoài silhouette SMPL-X, tìm **mặt SMPL-X gần nhất trong không gian ảnh** (ưu tiên mặt của phần thân hợp lý theo B1, ví dụ cẳng tay cho tay áo), lưu `(face id, barycentric của điểm chiếu, offset theo pháp tuyến/độ sâu)`, rồi chuyển sang tư thế đích như một "đỉnh ảo" → flow cho cả vùng quần áo.
- Lấy luôn **bộ lọc outlier 3 tiêu chí** (khoảng cách, góc pháp tuyến ≤ 45°, thống kê lân cận) để hạ `C` ở những chỗ neo không đáng tin.
- Rủi ro: quần áo không dính cứng vào thân (vải rủ), nên chỉ dùng cho offset nhỏ, ví dụ ngưỡng vài pixel; vùng xa hơn để diffusion (H1) tự sinh.

**(e) Làm mượt SMPL-X đích bằng MLP(PE(t)) + L_temp trước khi tính flow** ⚠️. Độ khó: thấp. Đáng thử như tiền xử lý.
- SignAvatars fit **từng khung**, nên có thể có jitter. Jitter này đi thẳng vào flow A1 và gây nhấp nháy. Tab. 2 của ATU cho thấy tham số hóa MLP giảm Accel 2.65 → 1.50, còn L_temp giảm 3.20 → 1.50.
- Cách làm: fit lại SMPL-X của mỗi clip bằng `θ_t = MLP(PE(t))`, loss = khoảng cách tới tham số SignAvatars (hoặc keypoint 2D gốc) + `L_temp`.
- **Cẩn thận:** ký hiệu tay có chuyển động nhanh (fingerspelling, chuyển động lặp). Nên tăng số tần số PE / giảm λ_temp cho khớp ngón tay, và kiểm tra bằng đo **sai số reprojection trước/sau** để không làm mất chuyển động thật. Đây là ablation nhỏ cho A1: "flow từ mesh thô vs mesh đã làm mượt".

**(f) Mẹo prompt cho H1 (nếu backbone Wan có nhánh text)** ✅ A.2 → ⚠️ áp dụng.
- Các cụm "Fixed camera. No zoom in. Bright light / uniform light" được tác giả dùng có chủ đích để chống zoom và ánh sáng tối. Video ký hiệu là camera tĩnh, nền đơn, nên một prompt cố định kiểu "a person signing, fixed camera, no zoom, plain background, uniform lighting" là baseline hợp lý cho ablation 1 (zero-shot) trong `diffusion_improvement.md` §1.9.
- Bằng chứng gián tiếp từ ATU: VDM **zero-shot chỉ với khung đầu** cho ra morph, vật thể lạ, và (với CogVideoX) chuyển động gần như đứng yên. Điều này **củng cố** lập luận rằng H1 cần điều kiện hình học dày theo từng khung (video warped + visibility + normal) và LoRA, không thể chỉ dựa vào I2V + text. Nhận định này khớp với rủi ro SignMimic đã ghi ở `diffusion_improvement.md` §1.10.

#### 7.2 KHÔNG nên lấy

- **Toàn bộ pipeline text-to-motion** và việc dùng **Kling**: không tái lập được, và bài toán khác hẳn (mình đã có chuyển động).
- **SMPL + VPoser + MediaPipe Pose 33 điểm**: không có tay/ngón. Nếu dùng tracker cho đánh giá thì bắt buộc dùng **SMPL-X + keypoint tay** (RTMPose-wholebody/HaMeR); prior cho tay dùng PCA tay của SMPL-X hoặc prior tay riêng.
- **Phạt gập khuỷu exp(θ)**: khuỷu tay trong ký hiệu thường gập mạnh và đó là tư thế hợp lệ. Áp nguyên term này sẽ sai lệch ⚠️.
- **Đặc trưng đỉnh kiểu Diff3F 100 view**: cần mesh có texture đầy đủ 3D, trong khi mình chỉ có **một ảnh nguồn**. Chỉ dùng được biến thể "back-project từ một view" như ở (c).
- **Tối ưu 4000 iteration/clip làm thành phần trong vòng huấn luyện**: quá chậm. Chỉ dùng offline cho đánh giá trên tập con hoặc cho tiền xử lý mesh (e).
- **Đừng trích dẫn Tab. 1 như bằng chứng rằng "tối ưu hóa tốt hơn regression trên video sinh"**: thí nghiệm của họ chạy trên video render SMPL, không phải video sinh.

#### 7.3 Tóm tắt ưu tiên cho luận văn ⚠️

| # | Việc | Loại | Độ khó | Giá trị |
|---|---|---|---|---|
| 1 | Keypoint-reprojection consistency (SMPL-X đích ↔ RTMPose trên khung sinh, GMoF + conf) | Metric | Thấp | Cao: đo đúng "đúng hình dạng tay" |
| 2 | Làm mượt SMPL-X bằng MLP(PE(t)) + L_temp trước A1 | Tiền xử lý | Thấp | Trung bình: giảm flicker, ablation dễ |
| 3 | Hand-silhouette IoU từ mask B1 | Metric | Thấp | Trung bình |
| 4 | DINOv2 perceptual/correspondence loss có mask | Loss | Thấp | Thấp–trung bình (không giúp ngón tay) |
| 5 | Neo pixel quần áo vào mặt SMPL-X (mở rộng A1) | Hình học | Trung bình–cao | Trung bình: giảm vùng `C` thấp quanh tay áo |
| 6 | Re-fit SMPL-X 3D trên tập con | Metric | Trung bình | Trung bình (cần đo sàn nhiễu trên video thật) |

---

## Bài 7 — Towards 3D-Aware Video Diffusion Models: Render-Free Human Motion Control with Mesh Tokenization ("MeshToken", arXiv 2606.02000, 06/2026)

Quy ước: ✅ = đã kiểm chứng từ paper (ghi mục/bảng) hoặc code (ghi `file:dòng`); ⚠️ = suy luận/đoán của người viết, cần kiểm lại.
Đường dẫn code trong note này tính từ gốc repo Uni3C (github.com/alibaba-damo-academy/Uni3C, commit `75ed6e2`).

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Jingyun Liang, Min Wei, Shikai Li, Yizeng Han, Hangjie Yuan, Lei Sun, Weihua Chen, Fan Wang (DAMO Academy Alibaba, Hupan Lab, ZJU, INSAIT) ✅ |
| Paper | arXiv:2606.02000v1, 1/6/2026, 11 trang, **không có phụ lục/supplementary** trong PDF ✅ |
| Project | https://jingyunliang.github.io/MeshToken. Trang chỉ có mô tả ngắn + demo walking/jogging. Các mục "Ablation", "Failure Cases" chỉ có tiêu đề, chưa có nội dung ✅ (đã fetch 29/10/2026) |
| Code | https://github.com/jingyunliang/MeshToken: **chỉ có 1 file README.md, nội dung "Repo for MeshToken."**, không có code, không có license ✅ (đã fetch) |
| Code gần nhất | **Uni3C** (SIGGRAPH Asia 2025), https://github.com/alibaba-damo-academy/Uni3C, Apache-2.0 ✅ (`LICENSE`). Cùng nhóm DAMO. Jingyun Liang là đồng tác giả thứ 4, không phải tác giả chính (tác giả chính là Chenjie Cao) ✅ (README). **Uni3C chỉ có code suy luận, không có code huấn luyện** ✅ (grep không thấy loss/optimizer trong code của tác giả) |
| Backbone | Wan-2.1 I2V (DiT, D = 5120 ⇒ bản 14B), đóng băng ✅ (§3.3, §4.1) |
| Kích thước | Tổng 20.33B tham số ✅ (§4.1). ⚠️ Ước lượng: 40 block × 4×5120² (q,k,v,o của motion cross-attn) ≈ 4.2B tham số mới được train, cộng ~16.1B của Wan-I2V-14B (đã gồm image cross-attn) ≈ 20.3B, khớp với con số 20.33B. Nghĩa là phần "adapter" nặng khoảng **4B**, không hề nhẹ |
| Train | ~300K clip nội bộ, mesh lấy từ GVHMR, caption lấy từ LLaVA. 97 khung @16 fps. Cascade 240p (160k iter, bs 64) → 480p (80k, bs 32) → 720p (40k, bs 8). AdamW, lr 1e-5, wd 1e-4. **~20 ngày**. Số GPU không nêu ✅ (§4.1) |
| Infer | CFG 5.0, "DDIM" 40 bước. ~4 phút cho video 97 khung 720p **trên 8×H100** ✅ (§4.1) |

---

### 1. TL;DR

- Bài lập luận rằng điều khiển người bằng **mesh render ra 2D** (depth/normal/skeleton/SMPL-color) là phụ thuộc góc nhìn, làm mất phần bị che/mặt sau. Điều đó biến bài toán thành "dịch video-sang-video". ✅ (§1)
- Thay vào đó: mesh SMPL (6890 đỉnh) ở **dạng chính tắc** (bỏ tịnh tiến, bỏ hướng toàn cục, bỏ β) được nén bằng **VQ-VAE mesh của VQ-HPS** thành **54 token × 9 chiều**/khung. Sau đó qua 5 block self-attention rồi chiếu lên 5120. Tịnh tiến và hướng (cả world lẫn camera space) được MLP thành một vector, rồi **cộng broadcast** vào cả 54 token. ✅ (§3.2, §4.1)
- Token được đưa vào Wan-2.1-I2V đóng băng bằng **cross-attention theo từng latent-frame**, chèn sau mỗi text cross-attn. Lớp chiếu cuối khởi tạo 0. Có dropout điều kiện p = 0.1 để dùng CFG. ✅ (§3.3, §4.1)
- Kết quả: tương đương RealisDance-DiT (render-based, cùng backbone) về lỗi quỹ đạo. Tốt hơn chút về PSNR/LPIPS/FVD và VBench. Ổn định hơn khi **sửa chuyển động** (flicker, smoothness). **Không có ablation đối chứng "render vs token" trong cùng điều kiện huấn luyện**, và **không mô hình hóa ngón tay** (SMPL, không phải SMPL-X). ✅
- Với GReg: luận điểm "render-free" **không áp dụng trực tiếp**, vì bài toán của mình cần căn pixel chính xác, camera cố định và chi tiết ngón tay. Những gì đáng lấy là các *cơ chế*: per-frame cross-attn cho token 3D, zero-init, dropout + CFG riêng cho điều kiện chuyển động, backbone đóng băng. Ngoài ra, **Uni3C chính là một bản "warp-then-diffuse" có code**, gần như khớp với H1.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

- **Đầu vào**: ảnh tham chiếu I^r, chuỗi mesh người M_1..T (SMPL), prompt văn bản ✅ (§3.1).
- **Đầu ra**: video T khung, trong đó người ở I^r chuyển động theo M, dưới góc camera tương ứng, với nền và cảnh do mô hình tự sinh ✅.
- **Mục tiêu khoa học** mà bài tự đặt: dùng điều khiển người làm "phép thử" xem video diffusion có thật sự "hiểu 3D" hay không. Nếu điều kiện đã là ảnh 2D thì mô hình có thể đi tắt bằng cách bắt chước ảnh điều kiện ✅ (§1).
- **Vì sao khó** (theo bài):
  1. Render 2D là view-dependent, mất mặt sau/bị che ✅.
  2. Khi **sửa chuyển động** (ghép quỹ đạo của clip này với pose của clip khác), điều kiện render thường lệch với cảnh: chân lơ lửng, xuyên vật, trượt chân. Mô hình render-based "làm theo quá cứng" ✅ (§1, §4.3).
  3. Token phải mang đủ thông tin quỹ đạo và góc nhìn, vì không còn ảnh căn pixel nữa. Mô hình phải *tự suy ra phép chiếu* ✅ (§1 "the model needs to infer how the same underlying 3D motion should appear under different viewpoints").
- ⚠️ Lưu ý khác biệt then chốt với GReg: ở đây **không có ground-truth pixel-aligned** quan trọng (PSNR tuyệt đối chỉ ~16.8). Bài toán thiên về "sinh hợp lý", không phải "tái tạo chính xác".

---

### 3. Phương pháp chi tiết

#### 3.1 Biểu diễn chuyển động: tách quỹ đạo và pose

- SMPL F(γ, φ, θ, β): γ∈R³ tịnh tiến, φ∈R³ hướng toàn cục, **θ∈R^{21×3}** pose thân, β∈R^{10}. Mesh có 6890 đỉnh, 13776 mặt ✅ (§3.2).
  → θ chỉ gồm 21 khớp thân, **không có khớp ngón tay, không có biểu cảm mặt** ✅. Đây là SMPL thuần, không phải SMPL-X.
- **Nhánh pose chính tắc** ✅ (§3.2):
  - Dịch mesh về gốc, bỏ hướng toàn cục, **bỏ β** khi dựng mesh ("largely irrelevant to human motion").
  - Encoder là fully convolutional mesh autoencoder (Zhou et al. NeurIPS 2020, spatially-varying kernels), dùng trọng số pretrained của **VQ-HPS** [Fiche et al., ECCV 2024].
  - V∈R^{6890×3} → N = 54 vector latent z_n∈R^L với L = 9. Mỗi vector ứng với **một vùng cơ thể cục bộ** ✅. Mỗi vector được lượng tử hóa về entry gần nhất của codebook S×L (§3.2, Eq. 1).
  - ⚠️ Kích thước codebook S không ghi trong bài. Theo trí nhớ về VQ-HPS thì S = 512, cần kiểm lại.
  - Encoder và codebook **đóng băng** trong lúc train ✅ (§3.3).
  - Mesh từng khung được mã hóa độc lập → {Z^p_t}. Sau đó **reshape theo thời gian** để khớp latent video (Wan nén 4× thời gian: 97 khung → 25 latent frame) ✅ (§3.2, §4.1).
  - ⚠️ Bài không nói cách xử lý khung đầu: Wan nén 1 + 4k, nên 97 = 1 + 96. Cũng không nói 4 khung được gộp vào kênh (54 × 36) hay lấy mẫu.
- **Tinh chỉnh pose** (Eq. 2): Ẑ^p = Proj(Z^p + SA(Z^p)) ✅. Cụ thể: 9 → 512, rồi **5 block self-attention 16 head**, rồi chiếu lên 5120 ✅ (§4.1).
  - ⚠️ Bài không nói 54 token có positional/part embedding hay không. Nếu không có thì cross-attn hoàn toàn bất biến hoán vị trên 54 token. Khi đó danh tính "token nào là tay trái" chỉ còn nằm trong nội dung codebook (vốn dùng chung giữa các vị trí trong VQ-HPS). Đây là câu hỏi mở quan trọng.
- **Nhánh chuyển động toàn cục** (Eq. 3): Z^t = MLP([γ_w, vec(R_w), γ_c, vec(R_c)]) ✅.
  - Hướng biểu diễn bằng ma trận quay 3×3. Dùng **cả world space** (chuyển động không phụ thuộc góc nhìn) **lẫn camera space** (chuyển động phụ thuộc góc nhìn, khớp ảnh) ✅.
  - Chiều thời gian được reshape vào kênh trước MLP. MLP gồm 2 Linear + GELU, ra 5120 ✅ (§4.1).
  - ⚠️ **Không có intrinsics (focal) nào** trong điều kiện. Mô hình phải tự đoán phép chiếu từ ảnh tham chiếu.
- **Token chuyển động** (Eq. 4): Z^m = Z^t + Ẑ^p, tức là **broadcast cộng** vector quỹ đạo vào cả 54 token ✅. Kết quả là chuỗi Z^m_{1..T/s} ∈ R^{T/s × N × D}.
- Sai số tái tạo mesh qua tokenizer: **8.0 mm** (Trajectory100), **7.1 mm** (RealisDance-Val) per-vertex ✅ (§4.2).

#### 3.2 Đưa vào DiT

- Wan-2.1 I2V đóng băng. Token ảnh tham chiếu được **concat** với token latent nhiễu (theo Fig. 1: "reference image tokens concatenated with latent noise tokens") ✅. Text đi qua text cross-attn như cũ ✅.
- **Motion cross-attention** được chèn **sau mỗi text cross-attn**, ở **tất cả** các block ✅. Cấu hình giống text cross-attn ✅ (§3.3, §4.1). Công thức (Eq. 5):
  `Z̃^{v,l}_t = Z^{v,l}_t + CA(Z^{v,l}_t, Z^m_t)`
  Trong đó Z^{v,l}_t∈R^{M×D} là token thị giác của latent-frame t ở block l. **Mỗi latent-frame chỉ attend tới 54 token của đúng frame đó** ✅. Lợi ích: ép căn chỉnh thời gian và rẻ hơn global cross-attn ✅.
  - ⚠️ Cách cài đặt hợp lý: reshape (B, F·H·W, D) → (B·F, H·W, D), K/V là (B·F, 54, D). Không cần mask attention.
- **Zero-init** lớp chiếu cuối (`to_out`) của mỗi motion cross-attn, để lúc đầu không phá backbone ✅ (§3.3).

#### 3.3 Loss, lịch train, suy luận

- Loss: **rectified flow matching**, giống Wan; scheduler cũng giữ nguyên như Wan-2.1 I2V ✅ (§3.3, §4.1).
- Chỉ train module mới (motion CA + tokenizer head: SA, Proj, MLP). Tokenizer VQ và backbone đóng băng ✅.
- Dropout điều kiện p = 0.1: thay text bằng negative prompt cố định, **đặt quỹ đạo và pose token = 0** ✅. Mục đích là vừa tăng độ bền khi sửa chuyển động, vừa để dùng CFG.
  - ⚠️ Bài không nói dropout chung một lần hay từng điều kiện độc lập. Tuy vậy ablation có "w/o CFG on Trajectory" và "w/o CFG on Pose" tách riêng, nên có lẽ là **CFG đa điều kiện**, và công thức kết hợp không được công bố.
- Vì module mới không phụ thuộc độ phân giải, họ train cascade theo độ phân giải (xem §0) ✅.
- Suy luận: CFG 5.0, 40 bước "DDIM" ✅. ⚠️ Với mô hình flow-matching thì "DDIM" có lẽ chỉ là cách gọi sampler xác định; Uni3C dùng UniPC.

#### 3.4 Dữ liệu & đánh giá

- Train: 300K clip nội bộ, nhiều độ phân giải/fps. Lấy mẫu đều 97 khung @16 fps, **khung đầu làm ảnh tham chiếu** ✅.
- Benchmark: **Trajectory100** (từ RealisMotion [22]) và **RealisDance-Val** ✅.
- Metric: VBench-I2V, PSNR, SSIM, LPIPS, FID, FVD. Thêm Translation/Rotation Error: chạy GVHMR trên video sinh ra rồi so với chuyển động đích ✅.
- **Không có metric riêng cho tay hay mặt** ✅.

---

### 4. Kết quả & ablation quan trọng

#### 4.1 Bảng 1: Trajectory100 ✅ (đọc từ ảnh trang 6)

| Method | Trans.Err (m)↓ | Rot.Err (°)↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ |
|---|---|---|---|---|---|---|---|
| Wan-2.1-I2V | 10.349 | 0.418 | 14.96 | 0.4763 | 0.3260 | 33.06 | 1421.87 |
| Tora | 5.667 | 0.355 | 16.56 | 0.5195 | 0.2501 | **21.51** | 957.81 |
| RealisDance-DiT (render, cùng backbone) | 1.706 | **0.167** | 16.17 | 0.4892 | 0.2481 | 23.02 | 758.08 |
| *RealisMotion\** (render + **có nền video**) | *1.198* | *0.101* | *22.57* | *0.7664* | *0.0686* | *12.00* | *314.59* |
| **MeshToken** | **1.697** | 0.173 | **16.78** | **0.5245** | **0.2451** | 22.13 | **695.62** |

Nhận xét:
- So với RealisDance-DiT: lỗi tịnh tiến gần như bằng (1.697 so với 1.706). **Lỗi quay tệ hơn** (0.173 so với 0.167). PSNR +0.6 dB, LPIPS −0.003, FVD −62 ✅.
- RealisMotion (cũng của Jingyun Liang, render-based nhưng được cho nền) bỏ xa tất cả: PSNR 22.57, LPIPS 0.0686. Bài loại nó khỏi so sánh trực tiếp ✅.
- ⚠️ Con số này cho thấy: khi có tín hiệu căn pixel (nền), fidelity tăng vọt. Với GReg, nơi gần như toàn bộ khung là "nền + người đã biết", đây là bằng chứng *ủng hộ* điều kiện pixel-aligned, không phải chống lại.

#### 4.2 Bảng 2: RealisDance-Val (VBench) ✅

| Method | I2V Subj | I2V BG | Subj Cons | BG Cons | Temp Flicker | Motion Smooth | Dyn Deg | Aesthetic |
|---|---|---|---|---|---|---|---|---|
| Animate-X | **96.06** | 96.59 | 93.83 | 94.83 | 97.40 | 98.52 | 53 | 55.22 |
| ControlNeXt | 92.91 | 93.92 | 91.41 | 93.57 | 96.91 | 98.05 | 63 | 55.57 |
| MimicMotion | 92.79 | 93.80 | 91.10 | 93.20 | 96.78 | 98.20 | 59 | 53.31 |
| MooreAA | 92.33 | 93.35 | 93.12 | 93.77 | 95.20 | 96.74 | 68 | 56.08 |
| MusePose | 92.24 | 93.01 | 93.88 | 94.88 | 97.88 | 98.57 | 57 | 56.28 |
| RealisDance-DiT | 95.97 | 96.57 | 93.91 | 95.83 | 97.76 | 98.71 | 66 | 57.93 |
| **MeshToken** | 96.03 | **97.18** | **94.04** | **96.23** | **98.32** | **98.79** | **68** | **58.14** |

- Chênh với RealisDance-DiT đều < 0.6 điểm. ⚠️ Với VBench, khác biệt cỡ này thường nằm trong nhiễu; bài không báo độ lệch chuẩn hay số seed. Bảng này cũng không có metric fidelity (PSNR...).

#### 4.3 Bảng 3: chuyển động đã sửa ✅ (50 video Trajectory100, sửa thành "Walking" hoặc "Jogging" lấy từ action bank của [22])

| Cat. | Method | I2V Subj | I2V BG | Subj Cons | BG Cons | Flicker | Smooth | Dyn | Aesth |
|---|---|---|---|---|---|---|---|---|---|
| Walking | Wan2.1-I2V | 96.02 | 96.21 | 93.52 | 93.74 | 96.94 | 98.65 | 61 | **56.33** |
| | RealisDance-DiT | **96.09** | 96.37 | 93.69 | 94.42 | 96.31 | 97.88 | **63** | 55.77 |
| | MeshToken | 96.03 | **96.86** | **93.71** | **95.79** | **97.32** | **98.79** | 60 | 56.12 |
| Jogging | Wan2.1-I2V | 95.94 | 96.02 | **93.66** | 93.19 | 96.46 | 97.31 | 65 | 56.32 |
| | RealisDance-DiT | 96.01 | 96.18 | 93.46 | 93.87 | 95.85 | 96.67 | **69** | 54.01 |
| | MeshToken | **96.11** | **96.63** | 93.41 | **95.16** | **97.58** | **98.34** | 68 | **56.48** |

- Mức tăng rõ nhất: Temporal Flicker (+1.0 và +1.7) và Motion Smooth (+0.9 và +1.7) so với RealisDance-DiT; BG Consistency +1.4 và +1.3 ✅.
- ⚠️ Dynamic Degree thấp hơn chút. Smooth/flicker cao có thể một phần chỉ do chuyển động "ít bám" điều kiện hơn. Đây là trade-off kinh điển của VBench.

#### 4.4 Bảng 4: ablation (PSNR / LPIPS; dòng "ours" trùng Bảng 1 ⇒ ablation làm trên Trajectory100) ✅

| Biến thể | PSNR | LPIPS | Ý nghĩa |
|---|---|---|---|
| No Trajectory Input | 15.32 | 0.2941 | bỏ toàn bộ quỹ đạo: tụt mạnh |
| Camera-Space Trajectory Only | 16.17 | 0.2781 | thiếu world space thì camera rung |
| World-Space Trajectory Only | 16.32 | 0.2688 | thiếu camera space thì đứng im quá mức (Fig. 5) |
| w/o CFG on Trajectory | 16.31 | 0.2521 | CFG riêng trên quỹ đạo có ích |
| No Pose Input | 15.61 | **0.3131** | bỏ pose: LPIPS tệ nhất trong nhóm |
| No Pose Refinement (bỏ 5 SA block) | 16.38 | 0.2520 | SA giữa các token giúp nhất quán giữa các bộ phận |
| w/o CFG on Pose | 16.19 | 0.2548 | CFG riêng trên pose có ích (ví dụ cảnh lau mặt bằng khăn, Fig. 5) |
| **MLP for Mesh Tokenization** | **14.79** | **0.3271** | MLP trực tiếp trên đỉnh mesh: **tệ nhất toàn bảng** |
| **SMPL Parameters as Motion** | 15.91 | 0.2899 | dùng tham số SMPL trực tiếp: kém |
| No Per-Frame Cross Attention (global CA) | 16.49 | 0.2591 | bỏ căn chỉnh thời gian: kém hơn |
| Unfreeze the Video Backbone | 16.81 | 0.2452 | ≈ đóng băng; vì vậy họ chọn đóng băng |
| **MeshToken** | 16.78 | 0.2451 | |

**Điểm phải nói thẳng:** bài lấy "render-free" làm luận điểm chính, nhưng **Bảng 4 không có dòng "rendered condition" nào** được train cùng dữ liệu, cùng lịch và cùng backbone đóng băng ✅. Bằng chứng duy nhất về "render vs token" là so với **RealisDance-DiT**, một mô hình khác (khác dữ liệu, khác cách train, nó fine-tune backbone). Ở đó MeshToken chỉ ngang hoặc nhỉnh hơn chút. ⚠️ Vì vậy lập luận "render làm mất thông tin" chủ yếu là lập luận định tính kèm hình (Fig. 2–4), chưa phải bằng chứng có đối chứng.

#### 4.5 Bàn tay
- Không có mục nào nói về tay hay ngón. Không có metric tay. SMPL không có khớp ngón ✅.
- Fig. 3 và Fig. 5 chỉ nói tới tương tác tay–vật (tay không xuyên qua con chó, lau mặt bằng khăn) ở mức cánh tay ✅.
- ⚠️ Với ký hiệu tay, phương pháp này **không cung cấp bất kỳ thông tin hình dạng bàn tay nào**.

---

### 5. Đọc code (Uni3C làm đại diện)

#### 5.0 MeshToken dựa trên gì?
- MeshToken **không** xây trên Uni3C. Uni3C chỉ được trích ở Related Work như ví dụ dùng point cloud để điều khiển camera ([3], §2.2) ✅.
- MeshToken kế thừa trực tiếp từ **RealisMotion [22]** (cùng tác giả chính): ý tưởng tách quỹ đạo / hướng / pose, benchmark Trajectory100 và action bank ✅ (§3.3 cuối, §4.2, §4.3).
- Baseline chính là **RealisDance-DiT [55]**, cũng chính là backbone mà Uni3C dùng cho nhánh người ✅ (`uni3c_inference.py:69`).
- ⚠️ Do đó, Uni3C là đại diện tốt cho phía "**render-based**" mà MeshToken phản biện. Đọc nó cho thấy chính xác cái mà MeshToken thay thế.

#### 5.1 Bản đồ repo

Tổng ~56k dòng Python, nhưng **code của tác giả chỉ khoảng 3.6k dòng** ✅ (`wc -l`):

| Đường dẫn | Dòng | Vai trò | Nguồn |
|---|---|---|---|
| `src/models/uni3c.py` | 517 | `RealisDanceDiT` + controlnet + shifted RoPE + SP | sửa từ RealisDance-DiT (header copyright RealisDance-DiT team) |
| `src/models/pcd_controller.py` | 239 | `PCDController` (Wan-I2V + controlnet), `MaskCamEmbed` | tác giả |
| `src/models/controlnet.py` | 300 | `WanXControlNet`, block attention không cross-attn, processor SP | tác giả |
| `src/pipelines/pipeline_uni3c.py`, `pipeline_pcd.py` | 838 / 747 | pipeline suy luận | sửa từ `diffusers` WanImageToVideoPipeline / RealisDance |
| `alignment.py` | 498 | căn SMPL-X (GVHMR) + HaMeR vào point cloud của ảnh tham chiếu, render mọi điều kiện | tác giả (lõi "3D") |
| `cam_render.py`, `cam_control.py`, `uni3c_inference.py` | 182/158/127 | entry point | tác giả |
| `src/pointcloud.py`, `render.py`, `camera.py`, `dataset.py`, `utils.py`, `fsdp.py` | ~720 | render point cloud, Plücker, load dữ liệu, FSDP | tác giả (FSDP lấy từ Wan) |
| `third_party/depth_pro`, `third_party/GeoCalib` | lớn | depth đơn ảnh + focal; hiệu chỉnh trọng lực | vendored nguyên |
| `third_party/GVHMR_realisdance` | lớn | GVHMR, có sửa `extract_pose.py` để xuất SMPL-X vertices, COCO-17 3D, cs_map | vendored + sửa |

Cấu hình thực (`config.json` trên HF `ewrfcas/Uni3C`, đã fetch) ✅:
```json
{"nframe": 81, "camera_embedding": true, "grad_checkpoint": true,
 "controlnet_cfg": {"conv_out_dim": 5120, "time_embed_dim": 5120, "dim": 1024, "ffn_dim": 4096,
                    "num_heads": 16, "num_layers": 20, "add_channels": 7, "mid_channels": 256}}
```

#### 5.2 Đi qua các file cốt lõi

**(a) Tiền xử lý người: `third_party/GVHMR_realisdance/extract_pose.py`**
- GVHMR (SMPL-X "supermotion") trên video nguồn. Xuất **COCO-17 3D ở world** và **toàn bộ đỉnh SMPL-X global** cho Uni3C ✅ (`extract_pose.py:213-221`).
- Render 2 video điều kiện "incam": `cs_map` (màu theo `smplx_color.pt`) và `canonical_norm` (màu theo `smplx_canonical_norm.pt`) ✅ (`extract_pose.py:313-333`).
  - ⚠️ "cs" gần như chắc là "canonical space": mỗi đỉnh mang màu bằng toạ độ chính tắc của nó. Về bản chất đây là **bản đồ tương ứng dày (dense correspondence) mã hóa thành RGB**, rất gần với ý tưởng "UV/coord map" ở mục 1.3 của `docs/diffusion_improvement.md`. (Không mở file .pt vì là pickle không tin cậy.)
- GVHMR chỉ cho tay ở tư thế mặc định. Tay chi tiết lấy từ **HaMeR** (`hamer_info.pkl`) ✅ (`alignment.py:334`). README ghi "Hamer alignment" vẫn là TODO, tức là chưa phát hành script sinh `hamer_info.pkl` ✅.

**(b) Căn người vào cảnh: `alignment.py`**
1. Ảnh tham chiếu được resize về 1 trong 5 khung hình [480×768, 512×720, 608×608, 720×512, 768×480] theo tỉ lệ gần nhất ✅ (`:146-153`).
2. Depth-Pro cho depth (mét) và focal. Unproject toàn ảnh thành point cloud ✅ (`:169-193`). GeoCalib cho vector trọng lực, dùng focal của Depth-Pro làm prior ✅ (`:176-180`).
3. YOLO11-pose lấy 17 keypoint 2D của người lớn nhất, bỏ điểm conf < 0.7 và điểm có depth > 3× median ✅ (`:196-225`).
4. **Procrustes có scale, có trọng số** giữa COCO-17 3D của GVHMR và 17 điểm 3D tương ứng trên point cloud Depth-Pro ✅ (`:260-261`). Áp lên toàn bộ đỉnh SMPL-X (`:289`).
5. **Xoay sửa trọng lực**: trục y của SMPL (sau R) được xoay cho trùng trọng lực thế giới ước từ GeoCalib ✅ (`:291-297`).
6. Resample chuỗi về 16 fps, độ dài 4k+1 ✅ (`:270-284`).
7. Quỹ đạo camera có thể "bám người" (`smpl_follow`: cộng dịch chuyển tích luỹ của tâm mesh vào vị trí camera) ✅ (`:45-48`, `:72-73`).
8. Render 5 video: `render.mp4` (point cloud môi trường), `render_mask.mp4` (lỗ), `smpl_render.mp4`, `hand_render.mp4`, `combined_render.mp4` ✅ (`:494-498`).

**(c) Cách điều kiện đi vào DiT: `src/models/uni3c.py`**
Có ba đường riêng biệt ✅:
1. **Pose (render SMPL + HaMeR)** → VAE latent (16 + 16 = 32 kênh, `pipeline_uni3c.py:485`) → `add_conv_in` (Conv3d patch 1×2×2) → **`add_proj` zero-init** → **cộng** vào token nhiễu (`uni3c.py:295-298`, `:404-410`). Đây là kiểu "ControlNet-lite": cộng tại đầu vào, căn theo pixel.
2. **Ảnh tham chiếu** → VAE latent → `attn_conv_in` → **nối thêm vào chuỗi token** (in-context), với **RoPE dịch chuyển** ✅ (`:299-301`, `:405-412`, `:128-185`). Kênh I2V gốc của Wan bị **vô hiệu hoá**: mask 4 kênh = 0 và video điều kiện = toàn 0 ✅ (`pipeline_uni3c.py:434-437`, `:479-483`).
3. **PCDController (camera)**: một transformer phụ gồm 20 block, dim 1024, **không có cross-attn**. Đầu vào là [latent nhiễu + 4 kênh mask] ⊕ latent của render point cloud, cộng thêm embedding của (mask lỗ + Plücker 6 kênh). Đầu ra được **cộng residual vào 20 block đầu** (trong tổng 40 block) của DiT, qua `proj_out` zero-init ✅ (`uni3c.py:387-398`, `:470-480`; `controlnet.py:250-300`).

**(d) Pipeline: `src/pipelines/pipeline_uni3c.py`**
- Khung đầu của `render_video` được thay bằng ảnh tham chiếu chất lượng cao ✅ (`dataset.py:31`; `alignment.py:437`; `pointcloud.py:169-170`, kèm `render_masks[0:1] = 0`).
- CFG chỉ đổi text (negative prompt rỗng), **giữ nguyên pose, ref và render** ở nhánh uncond ✅ (`:776-790`, comment "we use different CFG strategy compared to RealisDance-DiT"). Có `guidance_scale = 5.0` (`uni3c_inference.py:119`) và 40 bước (`:529`). Scheduler là UniPC; README ghi đây là bản sửa lỗi ngày 2025-08-06, thay cho FlowMatchEuler ✅.

#### 5.3 Những điểm hay nhất trong code

**1. Cùng một controller dùng được cho hai backbone ("plug-and-play")** ✅
`cam_control.py:74-76` nạp `PCDController.from_pretrained("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")` rồi gọi `build_controlnet("controlnet.pth")`. `uni3c_inference.py:72-74` nạp `RealisDanceDiT.from_pretrained("theFoxofSky/RealisDance-DiT")` rồi gọi **cùng** `build_controlnet("controlnet.pth")`, với cùng `config.json`.
→ Controller được train trên Wan-I2V gốc vẫn gắn được vào RealisDance-DiT (một bản fine-tune của Wan) mà không cần train chung.
⚠️ Hệ quả cho GReg: có thể train "nhánh warp" trên Wan gốc, rồi gắn vào một Wan đã LoRA cho ký hiệu, hoặc ngược lại. Hai nhánh này tách rời được.

**2. Controller "biết trạng thái nhiễu"** ✅ (`uni3c.py:387`)
```python
render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)  # noisy(16)+mask(4) ⊕ render(16)
```
Controller không chỉ nhìn render mà nhìn cả x_t, nên nó tự quyết bao nhiêu thông tin render cần bơm vào ở mỗi mức nhiễu. Đây là ý của ControlNet, nhưng làm rất gọn trong không gian token.

**3. Mask + Plücker ở độ phân giải pixel, khớp lưới latent 1 + 4k** ✅ (`pcd_controller.py:19-41`)
```python
self.mask_padding = [0, 0, 0, 0, 3, 0]  # pad 3 khung ở đầu (I2V)
self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8)), ...)
self.mask_zero_proj = zero_module(nn.Conv3d(mid_channels, conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
```
81 khung + 3 khung pad = 84 = 21 × 4, nên khớp đúng 21 latent frame của Wan. Mask được học nén thời gian, không bị nearest-downsample thô.
⚠️ Áp dụng thẳng cho **mặt nạ nhìn thấy V_geo và độ tin cậy C** của A1: đưa chúng vào ở độ phân giải pixel, không downsample tay về 60×60.

**4. Zero-init ở cả hai cửa** ✅
`controlnet.py:272-277` đặt `proj_out = zero_module(nn.Linear(dim, 5120))` cho mỗi tầng controller. `uni3c.py:298` đặt `add_proj = zero_module(...)` cho pose. MeshToken cũng zero-init `to_out` của motion CA. Đây là quy tắc chung của cả họ mô hình này: cấy điều kiện mới vào DiT đóng băng mà không làm hỏng prior.

**5. Ảnh tham chiếu là token in-context với RoPE dịch chuyển** ✅ (`uni3c.py:161-179`)
```python
cond_freqs_f = ori_freqs[0][shift_f_size:shift_f_size + cond_ppf]   # f-index bắt đầu từ 81
cond_freqs_h = ori_freqs[1][pph:pph + cond_pph]                    # h-index nằm sau lưới video
cond_freqs_w = ori_freqs[2][ppw:ppw + cond_ppw]
```
Token tham chiếu được đặt "ngoài" lưới không–thời gian của video. Nhờ vậy mô hình không coi ảnh tham chiếu là khung 0 căn pixel; nó lấy ngoại hình qua self-attention. Comment của tác giả: "This solution is ugly. We will design new RoPE…" (`:162`). Khi xuất ra, các token ref bị cắt bỏ (`:502`). Tín hiệu controller chỉ cộng vào phần token video, không cộng vào token ref (`:476-478`).
⚠️ Với GReg: I_s không trùng tư thế với khung đích, nên cách "ref token + shifted RoPE" hợp lý hơn kiểu "ref làm khung đầu" của Wan-I2V. Tuy vậy, **ảnh đã warp** thì *phải* căn pixel, nên nên đi qua đường controller/cộng.

**6. Lọc "điểm bay" ở biên depth trước khi warp** ✅ (`pointcloud.py:19-29`, dùng ở `alignment.py:410-414`)
```python
alpha = torch.exp(-beta * sobel_mag(normalized_disparity))
return alpha < sobel_threshold   # True = biên → bỏ điểm
```
Ngoài ra còn nén depth xa (`pointcloud.py:103-104`): `d > 8·median → 2m − m²/d`, giống contraction của mip-NeRF 360.
⚠️ Với GReg: pixel warped gần **biên silhouette/biên che khuất** của mesh cũng là pixel kém tin cậy (mesh lệch vài px so với ảnh thật). Nên hạ C ở dải biên này thay vì tin như vùng trong.

**7. Tay HaMeR được căn vào tay SMPL-X từng khung + vá cổ tay** ✅ (`alignment.py:357-375`, `:388-405`)
- Procrustes có scale giữa 778 đỉnh MANO (HaMeR) và các đỉnh tay SMPL-X tương ứng (`MANO_SMPLX_vertex_ids.pkl`). Nếu khung đó HaMeR thất bại (không đủ 2 tay, hoặc 2 tay cùng bên) thì quay về tay SMPL-X.
- Lọc ra các mặt có cả 3 đỉnh thuộc tay, rồi thêm **14 mặt khâu cổ tay viết tay** (`faces_new`) để mesh tay kín.
- Tay trái/phải tô màu cố định khác nhau (`:328-329`).

**8. Che tay bằng z-test, nhưng chỉ khi tay đến từ SMPL-X** ✅ (`alignment.py:466-473`)
```python
hand_mask = (hand_depth <= smpl_depth).float().unsqueeze(-1)
hand_mask = hand_mask * (1 - valid_hand) + torch.ones_like(hand_mask) * valid_hand
```
⚠️ Khi HaMeR hợp lệ, tay **luôn được vẽ đè**, kể cả khi thân che tay. Có lẽ vì tay HaMeR lệch khỏi silhouette thân (pixel ngoài thân có zbuf = −1 nên luôn trượt z-test). Đây là một heuristic. Với ký hiệu, tay thường *trước* thân nên ít ảnh hưởng; còn mesh SMPL-X fit sẵn của mình thì không cần trick này.

**9. Kênh tay riêng biệt** ✅ (`pipeline_uni3c.py:485`)
`latent_pose = cat(latent_smpl, latent_hamer)`: render tay là một video riêng, VAE-encode riêng, ghép kênh. Tay không bị "chìm" trong render toàn thân.
⚠️ Tương đương B1 (mặt nạ tay từ parts) nhưng ở mức điều kiện đầu vào.

**10. Điểm lạ/lỗi nhỏ** ✅ `camera.py:23-24`: `x = arange(w) - 0.5` nhưng `y = arange(h) + 0.5`. Lệch tâm pixel không đối xứng trong Plücker. Không quan trọng với mình, nhưng là ví dụ cho thấy code có chỗ cẩu thả.

#### 5.4 Uni3C vs MeshToken: khác gì

| Khía cạnh | Uni3C (render-based) | MeshToken (render-free) |
|---|---|---|
| Mô hình người | SMPL-X (GVHMR) + tay HaMeR ✅ | SMPL 6890 đỉnh, θ 21 khớp, bỏ β ✅ |
| Dạng điều kiện người | Video render SMPL-X tô màu chính tắc + render tay, qua VAE, cộng vào token (căn pixel) ✅ | 54 token VQ/khung + vector quỹ đạo, qua cross-attn từng latent frame ✅ |
| Camera | Render point cloud + mask lỗ + Plücker qua controller 20 tầng ✅ | Chỉ có γ_c, R_c (camera-space) trong MLP, không có intrinsics ✅ |
| Ngoại hình | Ref token in-context + shifted RoPE + CLIP image ✅ | Ref token concat với latent nhiễu (Fig. 1) ✅ |
| Backbone | RealisDance-DiT (Wan-14B đã fine-tune) + controller ✅ | Wan-2.1 I2V gốc đóng băng ✅ |
| Phần train | controller dim 1024 × 20 tầng ✅ (cấu hình); ⚠️ ước ~1B tham số (xem dưới) | motion CA ~4.2B ⚠️ ước + head nhỏ |
| Chi tiết tay | Có (kênh riêng) ✅ | Không ✅ |
| Căn pixel | Có ✅ | Không (theo thiết kế) ✅ |
| Code | Có (chỉ suy luận) ✅ | Không ✅ |

⚠️ Ước lượng tham số controller: mỗi block ≈ 4·1024² (attn) + 2·1024·4096 (FFN) + 2 × adaLN (5120 → 3·1024) ≈ 4.2M + 8.4M + 31M ≈ 44M; × 20 ≈ 0.87B, cộng `proj_out` 20 × 1024 × 5120 ≈ 0.1B và patch conv 36 → 5120. Tổng **~1B**. Phần adaLN chiếm nhiều vì time_embed_dim = 5120.

#### 5.5 Hyperparameter thực tế

| | Uni3C (code/config) | MeshToken (paper) |
|---|---|---|
| Số khung | 81 @16 fps ✅ | 97 @16 fps ✅ |
| Độ phân giải | max_area 768², 5 khung hình cố định ✅ | 240p → 480p → 720p ✅ |
| Bước suy luận | 40, UniPC ✅ | 40, "DDIM" ✅ |
| CFG | 5.0, chỉ trên text ✅ | 5.0, có CFG riêng cho quỹ đạo/pose ✅ |
| Controller/adapter | 20 tầng, dim 1024, 16 head, FFN 4096, add_channels 7 (1 mask + 6 Plücker), mid 256 ✅ | tokenizer 9 → 512, 5 SA × 16 head → 5120; MLP 2 lớp GELU → 5120; CA ở cả 40 block ✅ |
| Point raster | radius 0.008, 8 điểm/pixel; mesh raster 1 mặt/pixel ✅ (`alignment.py:156-166`) | không render |
| Bộ nhớ/thời gian | H20: 50.8 GB + offload ~20 phút; 4×H20 SP+FSDP <5 phút ✅ (README) | 8×H100 ~4 phút / 97 khung 720p ✅ |
| Train | không công bố code ✅ | lr 1e-5, wd 1e-4, AdamW, drop p = 0.1, ~20 ngày ✅ |

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Không có ablation đối chứng render vs token** trong cùng điều kiện huấn luyện ✅. Luận điểm trung tâm chỉ dựa trên so sánh với một mô hình khác (RealisDance-DiT) và trên hình minh hoạ.
2. **Không có ngón tay hay mặt** ✅. SMPL 21 khớp; tokenizer VQ-HPS được train trên SMPL. Muốn mở rộng sang SMPL-X (10475 đỉnh) thì phải train lại mesh VQ-VAE ⚠️.
3. **Bỏ β** ✅: dáng người (cao/thấp, tỉ lệ tay) chỉ còn đến từ ảnh tham chiếu. Với retarget thì ổn, nhưng với tái tạo chính xác (GReg) là mất thông tin ⚠️.
4. **Fidelity tuyệt đối thấp**: PSNR ~16.8 ✅. RealisMotion có nền đạt 22.57 ✅. ⚠️ Render-free không đem lại căn pixel; mọi chi tiết vị trí phải "tự suy" qua một phép chiếu ngầm không có intrinsics.
5. Lượng tử hóa: 7–8 mm/đỉnh ✅. Chấp nhận được cho thân, nhưng ⚠️ với ngón tay (đường kính ~15–20 mm) thì sai số này đã đáng kể, nếu có ai đó mở rộng sang tay.
6. Chưa rõ: positional/part embedding của 54 token; cách gộp 4 khung vào một latent frame và xử lý khung đầu; công thức CFG đa điều kiện; kích thước codebook ⚠️.
7. Tốn kém: ~4B tham số train, 300K clip, ~20 ngày ✅/⚠️. Không có code. Không có checkpoint.
8. Metric: chủ yếu VBench (bão hòa, khác biệt < 1 điểm) và lỗi quỹ đạo đo bằng GVHMR (một mô hình ước lượng khác, có thể thiên lệch về chính phân phối GVHMR dùng để tạo điều kiện) ⚠️.
9. Trade-off ngầm: "ít bám điều kiện hơn khi điều kiện xấu" (Bảng 3) cũng là "ít bám điều kiện hơn" nói chung. Dynamic Degree giảm nhẹ ✅/⚠️. Với ký hiệu, bám *chính xác* tư thế tay là yêu cầu cứng.
10. Phía Uni3C: không có code train; alignment phụ thuộc chuỗi 4 mô hình (Depth-Pro, GeoCalib, YOLO, GVHMR) và HaMeR chưa phát hành script ✅.

---

### 7. Liên hệ với GReg

#### 7.1 Render-based hay token-based cho SMPL-X trong ngôn ngữ ký hiệu?

Các điều kiện khiến lập luận của MeshToken **không chuyển sang được** GReg:

| Giả định của MeshToken | Thực tế GReg |
|---|---|
| Camera thay đổi, cần bất biến góc nhìn | Camera cố định, người nhìn thẳng. Phụ thuộc góc nhìn **không phải vấn đề** ⚠️ |
| Sửa chuyển động gây lệch quỹ đạo–cảnh | Mình tái tạo hoặc dịch ký hiệu với mesh đã fit; quỹ đạo gần như cố định (người đứng yên) ⚠️ |
| Mục tiêu là "sinh hợp lý" (PSNR ~16.8) | Mục tiêu là tái tạo đúng pixel (L1 ~0.033, cần PSNR cao); nền và áo **đã biết** từ I_s |
| Thân người, không có ngón | Ngón tay là thông tin ngôn ngữ chính |

→ **Kết luận** ⚠️: với GReg, điều kiện **pixel-aligned** (ảnh warp bằng mesh + visibility + normal) vẫn phải là xương sống của H1. Bảng 1 thậm chí ủng hộ điều này một cách gián tiếp: RealisMotion có tín hiệu căn pixel (nền) đạt PSNR 22.57 so với 16.78 ✅.

Nhưng MeshToken chỉ ra đúng một **điểm yếu thật** của render: **thông tin bị che hoặc ở mặt sau bị mất**. Ở GReg, chính vùng này đang mờ (lòng bàn tay, mặt trong cẳng tay, ngón bị che). Normal map đích chỉ cho mặt nhìn thấy, nên khi tay nắm hay ngón chồng nhau, phần ngón bị che hoàn toàn biến mất khỏi điều kiện. **Token 3D bổ sung đúng phần đó.**

#### 7.2 Thiết kế đề xuất: "H1 + hand mesh tokens" (lai, không thay thế) ⚠️

```
Đường 1 (căn pixel, như H1 hiện tại + học từ Uni3C):
  [I_st warped, normal N_t] → VAE latent ┐
  [noisy latent x_t]  ───────────────────┼→ controller nhẹ (kiểu PCDController, 8–15 tầng, dim 512–768)
  [V_geo·C, M_hand/face] ở độ phân giải pixel → Conv3d stride (4,8,8) + pad 3 khung đầu (Uni3C MaskCamEmbed)
                                          → cộng residual zero-init vào nửa đầu số block của Wan-1.3B
Đường 2 (token 3D, học từ MeshToken):
  Mỗi khung: đỉnh tay SMPL-X (2 × 778 đỉnh MANO-subset) trong hệ toạ độ cổ tay (canonical)
          → gom theo phần (ví dụ 16 phần/tay theo skinning weight lớn nhất: lòng bàn tay + 15 đốt)
          → mỗi phần: mean/PCA đỉnh hoặc PointNet nhỏ → token, + part embedding học được (bắt buộc, xem §6.6)
  + vector toàn cục: vị trí và quay cổ tay trong camera space (6D rot), cộng broadcast (Eq. 4)
  4 khung của một latent frame → ghép vào kênh (Eq. 3 kiểu MeshToken)
  → 2–3 SA block (như "pose refinement") → per-latent-frame cross-attn sau text-CA,
    zero-init to_out, có thể mask query theo vùng tay (B1) + sink token (SignRefine)
Huấn luyện: LoRA/adapter, backbone đóng băng; drop từng điều kiện độc lập p≈0.1; CFG riêng cho token tay.
```

Lý do chọn từng thành phần:
- **Per-latent-frame CA** rẻ và căn thời gian. Ablation cho thấy tốt hơn global CA: 16.78/0.2451 so với 16.49/0.2591 ✅.
  - ⚠️ Chi phí trên Wan-1.3B (D = 1536, 30 block): 4·1536² ≈ 9.4M/block. Nếu CA ở mọi block thì ~283M; chỉ dùng block chẵn thì ~140M. Vẫn vừa 24 GB khi train LoRA ở 480², 49 khung (cần đo).
- **Không dùng MLP trên đỉnh thô hay tham số pose trực tiếp**: Bảng 4 cho thấy cả hai tệ hơn rõ (14.79 và 15.91 so với 16.78) ✅. Tokenizer cần cấu trúc cục bộ theo vùng cơ thể (mesh conv / theo phần).
  - ⚠️ Không có sẵn VQ-VAE cho tay SMPL-X; gom theo phần + embedding là phương án rẻ nhất. VQ thì không cần thiết (chỉ là nén; liên tục có lẽ tốt hơn cho chi tiết ngón).
- **Đóng băng backbone ≈ fine-tune toàn bộ** (16.78 so với 16.81) ✅ ủng hộ thêm cho hướng adapter của H1.
- **CFG riêng cho điều kiện chuyển động** có ích (pose: 16.19 → 16.78) ✅. Nên train với dropout độc lập cho (warp, normal, hand-token, text), để khi suy luận điều chỉnh được trọng số tay.
- Từ Uni3C: controller nhìn cả x_t (`uni3c.py:387`); mask đưa vào ở độ phân giải pixel (`pcd_controller.py:30-32`); khung đầu là ảnh thật (`dataset.py:31`); lọc biên (`pointcloud.py:19-29`) → hạ C ở dải biên silhouette của mesh warp.
- Giữ I_s như **ref token in-context** (kiểu RealisDance/Uni3C) để lấy ngoại hình cho vùng không thấy. Ảnh warp đi đường căn pixel riêng.

#### 7.3 Ablation nên chạy (trả lời đúng câu hỏi MeshToken đặt ra, trong bối cảnh ký hiệu) ⚠️

1. (a) Chỉ warp + V + normal (H1 gốc); (b) (a) + hand tokens; (c) chỉ hand/body tokens, không warp (≈ MeshToken-style); (d) (a) nhưng thay normal bằng render "canonical-color" của SMPL-X (kiểu `cs_map` của RealisDance/Uni3C, một bản đồ tương ứng mã hóa RGB).
2. Metric theo vùng: PSNR/LPIPS **trong mặt nạ tay** và **trong vùng V = 0** (bị che ở nguồn); lỗi keypoint tay (HaMeR hoặc MediaPipe trên output so với GT); FVD.
3. Kỳ vọng ⚠️: (c) thua xa (a) về PSNR toàn khung. (b) chỉ thắng (a) ở vùng tay bị che hoặc ngón chồng nhau. Nếu (b) không thắng ở các vùng đó thì bỏ đường token, giữ thiết kế đơn giản.

#### 7.4 Độ khó & chi phí ⚠️
- Đường 1 (controller) là phần H1 vốn đã lên kế hoạch. Uni3C cho mẫu code gần như dùng được ngay (nhưng là 14B, diffusers 0.33.1); cần port sang Wan-1.3B (D 1536, 12 head, 30 block).
- Đường 2: khoảng 300–500 dòng code (tokenizer theo phần + CA hook vào `WanTransformerBlock`), cộng thêm việc tiền tính đỉnh tay SMPL-X cho 4.6k clip (đã có mesh, chỉ cần xuất đỉnh). Độ khó trung bình.
- Rủi ro chính: dữ liệu nhỏ (4.6k clip) so với 300K của MeshToken. Token "không căn pixel" cần nhiều dữ liệu để học phép chiếu ngầm. Vì vậy **phải đi kèm** đường căn pixel, không được dùng một mình.

#### 7.5 Những gì KHÔNG nên lấy
- Không bỏ điều kiện căn pixel (render-free hoàn toàn): sai mục tiêu tái tạo, và bằng chứng của bài cũng không đủ mạnh (§4.4).
- Không dùng tokenizer VQ-HPS / SMPL: không có ngón, không có mặt, bỏ β.
- Không cần quỹ đạo world-space hay camera-space phức tạp: camera cố định. Một vector vị trí/quay cổ tay theo camera là đủ.
- Không theo quy mô của họ (Wan-14B, 4B tham số CA, 20 ngày). Wan-1.3B + LoRA + CA thưa là phù hợp với 24 GB / A100 thuê.
- Không chép trick "HaMeR luôn đè lên thân" (`alignment.py:472`); mình đã có SMPL-X fit có ngón và z-buffer đúng từ A1.
- Không coi VBench là metric chính. Dùng metric theo vùng tay và vùng bị che như trên.

#### 7.6 Câu để đưa vào luận văn (phần related work / thảo luận) ⚠️
> "MeshToken (Liang et al., 2026) lập luận rằng điều kiện render 2D từ mesh làm mất thông tin 3D bị che và nên được thay bằng token mesh. Tuy nhiên, bài không có thí nghiệm đối chứng render-vs-token trong cùng điều kiện huấn luyện, chỉ mô hình hóa SMPL (không có ngón tay), và đạt PSNR thấp (16.8 dB) so với phương pháp render-based có tín hiệu căn pixel (22.6 dB). Trong bài toán tái tạo ký hiệu với camera cố định, chúng tôi giữ điều kiện căn pixel (warp bằng mesh + visibility) và chỉ bổ sung token 3D cho bàn tay, nhằm bù đúng phần thông tin bị che mà render không mang được."

---

## Bài 8 — Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video (CVPR 2026)

Quy ước: ✅ = đã kiểm chứng từ paper (mục/bảng) hoặc code (`file:dòng`); ⚠️ = suy luận / phỏng đoán của người viết.
Đường dẫn code tương đối so với gốc repo `Mesh4D/` (bản clone commit `30269e3`, message "clean", 2026-01-09).

---

### 0. Thông tin nhanh

| Mục | Nội dung |
|---|---|
| Tác giả | Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, Andrea Vedaldi (VGG Oxford, NTU, Naver Labs Europe) ✅ |
| Venue | CVPR 2026 (bản Open Access CVF, trang 14524–14535) ✅. arXiv 2601.05251 ✅ (README) |
| Project / code | mesh-4d.github.io; github.com/jzr99/Mesh4D ✅ |
| Trạng thái code | **Chỉ có inference** (`hy3dshape/infer.py`) + checkpoint (deform VAE, denoiser qua gdown). README ghi "benchmark and evaluation code will come soon", "visualization code will come soon" ✅. **Không có script train**, không có config train VAE, data loader train đã bị cắt (chỉ load ảnh) ✅ (`hy3dshape/dataset/custom_dataloader.py:793-843`). Tuy vậy các LightningModule train (`vae_pl_batt.py`, `flow_matching_sit_video_bone_batt.py`) vẫn còn → đọc được loss và luồng train. |
| License | Repo là fork của Hunyuan3D-2.1 → chịu **TENCENT HUNYUAN 3D 2.1 COMMUNITY LICENSE** (không áp dụng ở EU, UK, Hàn Quốc; có Acceptable Use Policy) ✅ (`hy3dpaint/LICENSE:1-3,17`). Không có LICENSE riêng ở gốc repo ✅. Phần SiT transport là MIT ✅ (`.../transport/__init__.py:1-24`). |
| Backbone | Hunyuan3D 2.1 (shape VAE + shape DiT + Paint PBR) ✅ §3.1 |
| Compute train | **Paper không báo cáo** GPU/giờ train ✅. Gợi ý gián tiếp: comment kế thừa từ HY3D "Overfitting successfully cost 68G memory" (`hy3dshape/configs/infer.yaml:2`) ✅; tên thư mục thí nghiệm lộ ra trong code `..._lr1e4_motion_2k_pretrain_bone_resume_5_batt/ckpt-step=00004000` và `..._batt6_new_40000` (`hy3dshape/infer.py:102`, `hy3dshape/obj_to_glb.py:37`) ⚠️ (có thể là lr 1e-4, ~40k step, nhưng không chắc). |
| Compute infer | HY3D 2.1 shape (khung 1) + HY3D-Paint (6 view, 512) + DiT deform 50 bước Euler, CFG 5.0 (batch ×2), fp16 ✅ (`infer.py:89-111`, `pipelines_video_newvae_all_nonalign_infer.py:890-894,1077`). Không có số thời gian ✅. DiT cỡ HY3D 2.1 (hidden 2048, depth 21, MoE 6 lớp × 8 expert) ✅ config; ⚠️ khoảng 3B+ tham số. |

---

### 1. TL;DR

- Bài toán: từ **video đơn mắt** của một vật thể động → **một mesh hoàn chỉnh (cả phần không thấy) + trường biến dạng** cho mọi khung, giữ nguyên topology → có tracking dày (dense correspondence) miễn phí ✅ §3.
- Cách làm: (1) HY3D 2.1 dựng mesh tĩnh từ **khung đầu**; (2) một **Deformation VAE** mới mã hóa *cả chuỗi* biến dạng vào latent gọn (6 khung × 256 token × 64 kênh), encoder dùng **attention không gian / thời gian / toàn cục** và **thông tin xương (skinning weights + bone) chỉ lúc train**; (3) một **DiT flow-matching** (khởi tạo từ HY3D DiT) sinh latent biến dạng, điều kiện bằng đặc trưng DINO theo khung + latent hình dạng của mesh khung đầu + positional embedding theo vị trí 3D chính tắc ✅ §3.2–3.3, Fig. 2–3.
- Kết quả: trên benchmark 50 chuỗi Objaverse tự xây, thắng HY3D-từng-khung, L4GM, GVFD về IoU/P2S/Chamfer/ℓ2-Corr và NVS (FVD 572.7 vs 692.2 của HY3D) ✅ Tab. 1–2. Ablation: bỏ xương làm ℓ2-Corr 0.0117 → 0.0138; bỏ attention thời gian + toàn cục → 0.0160 ✅ Tab. 3.
- Bài học chính cho GReg: **tách hình học (đã có mesh, tương ứng qua barycentric) khỏi ngoại hình** cho tính nhất quán thời gian tốt hơn hẳn sinh từng khung độc lập; và một loạt mẹo "cấy lớp mới vào mô hình pretrained mà không phá nó" (zero-init, phẫu thuật trọng số input) dùng thẳng được cho Hướng 1/2.

---

### 2. Bài toán, đầu vào/đầu ra, vì sao khó

**Đầu vào:** video RGB $\mathcal{I}=\{I_t\}_{t=1}^T$, $I_t\in\mathbb{R}^{H\times W\times 3}$, vật thể đã tách nền (RGBA, nền trắng) ✅ §3, README.
**Đầu ra:** mesh khung đầu $\mathcal{M}_1=\langle\mathcal{V}_1,\mathcal{F}_1\rangle$ và trường biến dạng $\mathcal{T}_{1\to t}$; mesh tại $t$ là $\mathcal{M}_t=\langle\mathcal{V}_1+\mathcal{T}_{1\to t}(\mathcal{V}_1),\mathcal{F}_1\rangle$ ✅ Eq. (1).

**Vì sao khó** ✅ §1, §2:
1. Video đơn mắt chỉ thấy một mặt; phải "bịa" phần không thấy cả về hình lẫn chuyển động → cần prior 3D học từ dữ liệu.
2. Các phương pháp feed-forward kiểu DUSt3R/MonST3R/St4RTrack chỉ dựng điểm **nhìn thấy** và chuyển động **giữa hai khung**, không ra cấu trúc 4D đầy đủ.
3. Dựng từng khung độc lập (HY3D từng khung) → pose sai, hình dạng nhảy, texture nhấp nháy, không có correspondence.
4. Phương pháp 3D-GS (L4GM, GVFD) dựa trên loss ảnh, không có ràng buộc topology → "ghost artifacts" khi chuyển động lớn ✅ §4.3.
5. Thiếu dữ liệu 4D có nhãn → phải khởi đầu từ mô hình 3D tĩnh pretrained và lọc Objaverse ✅ §1, §4.1.

⚠️ Điểm tương đồng với GReg: (3) chính là bệnh "sinh từng khung, không prior thời gian" của GReg; (2)/(4) tương ứng với việc warp học được (TPSMM) chỉ đúng phần nhìn thấy.

---

### 3. Phương pháp chi tiết

#### 3.1 Nền: latent 3D reconstruction (Hunyuan3D 2.1) ✅ §3.1
- Shape VAE kiểu VecSet: lấy mẫu điểm $P\in\mathbb{R}^{M\times3}$ + pháp tuyến $n$ trên mesh → $z^s=\mathcal{E}^s(P;n)$; decoder $\mathcal{D}^s(Q;z^s)$ trả SDF tại lưới truy vấn $Q$, marching cubes ra mesh.
- DiT flow matching:
$$\min_\theta \mathbb{E}_{(z^s,I),t,\epsilon}\big\|v^s - v^s_\theta(z^s_t,t,I)\big\|_2^2,\quad z^s_t=t z^s+(1-t)\epsilon,\; v^s=z^s-\epsilon,\; t\sim\mathcal{U}(0,1)$$
  (Eq. 2). Lấy mẫu bằng Euler ODE.
- Texture: mô hình Paint riêng (ảnh vào + normal/coord map đa view) → PBR. Chỉ dùng cho khung đầu.

#### 3.2 Deformation VAE ✅ §3.2, Fig. 2

**(a) Mã hóa điểm theo thời gian.** Giả định các mesh trong chuỗi **cùng topology** (đỉnh thứ i tương ứng đỉnh thứ i). Lấy mẫu $\mathcal{P}_1$ trên $\mathcal{M}_1$, biểu diễn mỗi điểm bằng **tọa độ barycentric** trong tam giác chứa nó, rồi dựng lại vị trí ở khung $t$ từ $\mathcal{V}_t$ → chuỗi point cloud $\{\mathcal{P}_t\}$ tương ứng nhau. Đặc trưng:
$$h_t = f_l\big(\mathrm{PE}(\mathcal{P}_1)\oplus n_1\oplus \mathrm{PE}(\mathcal{P}_t)\oplus n_t\big)\in\mathbb{R}^{M\times c}\quad(3)$$
→ ghép kênh cặp (khung 1, khung t) giúp mạng "thấy" chuyển động.

**(b) Tiêm thông tin xương (privileged, chỉ lúc train).** Skinning weights $w\in\mathbb{R}^{M\times B_{max}}$ ($B_{max}=64$). Self-attention giữa các điểm với mask:
$$\hat h_t=\mathrm{softmax}\Big(\tfrac{h_t h_t^\top+M^s}{\sqrt c}\Big)h_t+h_t,\quad M^s=\begin{cases}0 & ww^\top>\tau^s\\ -\infty & \text{ngược lại}\end{cases}\quad(4)$$
Xương: đầu/đuôi $b^h_t,b^t_t\in\mathbb{R}^{B_{max}\times3}$, $h^b_t=f^b_l(\mathrm{PE}(b^t_1)\oplus\mathrm{PE}(b^h_1)\oplus\mathrm{PE}(b^t_t)\oplus\mathrm{PE}(b^h_t))$; cross-attention điểm→xương với mask $M^b$ ($w>\tau^b$) (Eq. 5).

**(c) Nén + attention không–thời gian.** FPS xuống $N\ll M$ token, cross-attention $h''_t$ (query) với $h'_t$ (key/value) (Eq. 6). Sau đó $L=8$ lớp, mỗi lớp **temporal → global → spatial** (Fig. 2 phải); temporal và global dùng **RoPE 1D theo trục thời gian**. Chiếu tuyến tính ra mean/var, $c_o\ll c$, lấy mẫu $z^d$.

**(d) Decoder.** Chiếu $z^d$ lên $c$, 16 lớp spatio-temporal, cuối cùng **cross-attention với đỉnh chính tắc $\mathcal{V}_1$ làm query** → độ dời từng đỉnh.

**(e) Loss:**
$$\mathcal{L}_{VAE}=\sum_{t=1}^{T}\big\|(\mathcal{V}_t-\mathcal{V}_1)-\mathcal{D}^d_t(\mathcal{V}_1;z^d)\big\|_2^2+\lambda L_{KL}\quad(7)$$
chỉ tính trên một tập con đỉnh ngẫu nhiên cho nhanh. Paper **không** cho giá trị $\lambda$, $\tau^s$, $\tau^b$, $c$, $c_o$, $N$ ✅ (code cho một phần, xem §5.5).

#### 3.3 Deformation diffusion ✅ §3.3, Fig. 3
- Mở rộng shape DiT của HY3D: thêm **temporal embedding**, **spatial embedding** $p_1\in\mathbb{R}^{N\times3}$ lấy từ $\mathcal{M}_1$, và thêm lớp attention trong mỗi block DiT.
- Thứ tự trong block (Fig. 3): Spatial Attn → Temp Attn → Cross-Attn (DINO, cùng khung) → Cross-Attn (latent HY3D ShapeVAE của mesh chính tắc) → MoE; có skip connection dài kiểu U-Net.
- Điều kiện video: DINO (paper ghi **DINO-Giant**) trích **từng khung**, token latent khung t cross-attend vào đặc trưng khung t.
- Spatial embedding: lúc train dùng **vị trí FPS** của các token VAE (để mỗi latent mục tiêu khớp với embedding của nó); lúc infer FPS trên mesh chính tắc được dựng ra. "Giống GVFD, embedding không gian cho nhiễu ban đầu nhận biết vị trí".
- Loss: giống Eq. (2).

#### 3.4 Dữ liệu ✅ §4.1
- Objaverse-1.0 bản đã lọc của Diffusion4D → trích skeleton, skinning, chuỗi mesh có đỉnh tương ứng → lọc vật có quá nhiều đỉnh/xương → **~9k** instance. Render video **nhìn thẳng (frontal)**, tối đa 100 khung.
- Test: 50 chuỗi rời rạc, chuyển động lớn, texture tốt; render 4 view (azimuth 0°, 90°, 180°, 270°), 1 view làm input, 3 view cho NVS.
- Baseline: HY3D 2.1 chạy từng khung **với nhiễu dùng chung** (shared noise) cho nhất quán ✅ (code: `infer.py:154` truyền `preset_latent=init_latents`); L4GM, GVFD lấy tâm Gaussian có opacity > 0.01 làm point cloud. Căn chỉnh bằng **Coherent Point Drift** (scale + rigid) trên mesh khung đầu.
- Metric: IoU thể tích, P2S, Chamfer; ℓ2-Corr (khoảng cách Euclid giữa điểm tương ứng, tương ứng = láng giềng gần nhất ở khung đầu); NVS: PSNR/SSIM/LPIPS/CLIP + FVD.

#### 3.5 Inference ✅ (code `hy3dshape/infer.py`, `pipelines_video_newvae_all_nonalign_infer.py`)
1. HY3D 2.1 shape từ khung đầu → mesh; HY3D-Paint tô texture (`infer.py:146-172`).
2. Đổi trục (y-up → z-up), chuẩn hóa: `loc = bbox_min`, `scale = max_extent/0.9` (`pipelines_...:919-934`).
3. Shape condition: lấy 2048 điểm (random 8192 → FPS ¼) trên đỉnh mesh + pháp tuyến + kênh 0 → ShapeVAE HY3D → latent (`:1050-1060`).
4. Canonical points: FPS 256 điểm từ **đỉnh** mesh (`:1071`).
5. 50 bước Euler, sigma `linspace(0,1,50)`, CFG 5.0 (`:1077-1120`).
6. VAE decode tại **mọi đỉnh** của mesh sinh ra → 6 mesh cùng faces; ghép GLB animation (`:1145-1162`).

---

### 4. Kết quả & ablation quan trọng

#### 4.1 Hình học + tracking (Tab. 1) ✅
| Phương pháp | IoU ↑ | P2S ↓ | Chamfer ↓ | ℓ2-Corr ↓ |
|---|---|---|---|---|
| HY3D 2.1 (từng khung) | 0.3071 | 0.0376 | 0.0370 | N/A |
| L4GM | N/A | 0.0459 | 0.0505 | N/A |
| GVFD | N/A | 0.0345 | 0.0378 | 0.0514 |
| **Ours** | 0.3731 | 0.0287 | 0.0273 | 0.0384 |
| **Ours (Aligned)** | **0.3949** | **0.0261** | **0.0243** | **0.0338** |

"Aligned" = mesh chính tắc đã căn CPD **trước khi** đưa vào diffusion; "Ours" = căn ngay trước khi đánh giá ✅ §4.1. Repo phát hành bản **non-align** (đoạn CPD bị comment: `pipelines_...:945-953`) ✅.

#### 4.2 Novel view synthesis (Tab. 2) ✅
| Phương pháp | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | FVD ↓ |
|---|---|---|---|---|---|
| HY3D 2.1 | 19.14 | 0.8976 | 0.1195 | **0.9174** | 692.2 |
| L4GM | 18.07 | 0.8939 | 0.1453 | 0.8954 | 747.3 |
| GVFD | 17.31 | 0.8912 | 0.1459 | 0.8802 | 905.0 |
| Ours | 19.67 | 0.9018 | 0.1087 | 0.9141 | 601.9 |
| Ours (Aligned) | **19.88** | **0.9030** | **0.1052** | 0.9141 | **572.7** |

Nhận xét của tác giả: CLIP thấp hơn HY3D vì texture chỉ sinh **một lần từ khung đầu**, còn HY3D sinh texture mới mỗi khung ✅ §4.3. ⚠️ Đây là đánh đổi "nhất quán (FVD) vs khớp từng khung (CLIP)" — y hệt đánh đổi GReg sẽ gặp khi chuyển texture từ ảnh nguồn qua warp.

#### 4.3 Ablation VAE (Tab. 3, dùng mesh khung đầu GT) ✅
| Biến thể | IoU ↑ | P2S ↓ | Chamfer ↓ | ℓ2-Corr ↓ |
|---|---|---|---|---|
| w/o temp & global attention | 0.6328 | 0.0153 | 0.0113 | 0.0160 |
| w/o skeleton information | 0.6704 | 0.0148 | 0.0107 | 0.0138 |
| **Ours** | **0.7039** | **0.0144** | **0.0099** | **0.0117** |

- Bỏ xương → biến dạng cứng bị "xoắn" (cây gậy bị vặn) ✅ Fig. 6.
- Bỏ attention thời gian/toàn cục → rung (jitter), sai số lớn ở bàn chân ✅ Fig. 6.
- ⚠️ Ablation chỉ đo **tái tạo của VAE** (encode GT → decode), **không** có ablation cho diffusion (ví dụ bỏ spatial embedding, bỏ shape cross-attn, DINO-Large vs Giant, số khung T). Không có số về độ nhạy với chất lượng mesh khung đầu ngoài cặp Ours/Aligned.

#### 4.4 Định tính ✅ §4.2–4.3, Fig. 4–5
Bốn loại lỗi của đối thủ: pose sai (HY3D), texture nhấp nháy (HY3D), mesh chính tắc sai (GVFD, L4GM), ghost của 3D-GS.

---

### 5. Đọc code

#### 5.1 Bản đồ repo

~41k dòng Python, phần lớn là vendored. Phân loại:

| Thư mục / file | Nguồn | Vai trò |
|---|---|---|
| `hy3dpaint/` | Hunyuan3D-2.1 Paint (vendored) | Tô texture PBR khung đầu |
| `hy3dshape/im2mesh/` | Occupancy Flow / Motion2VecSets (vendored) | Evaluator IoU/Chamfer/correspondence, libmesh, torchdiffeq |
| `hy3dshape/hy3dshape/models/diffusion/transport/` | SiT (MIT) | Flow matching (path Linear, velocity) |
| `hy3dshape/hy3dshape/models/autoencoders/{model.py, attention_blocks.py, ...}` | HY3D 2.1 | ShapeVAE tĩnh (dùng làm shape condition) |
| `hy3dshape/hy3dshape/pipelines.py`, `schedulers.py`, `conditioner.py`, `moe_layers.py` | HY3D 2.1 (sửa nhẹ) | Pipeline HY3D, Euler FM scheduler, DINO encoder, MoE |
| **`autoencoders/model_deform_batt.py`** | **Tác giả** | Deformation VAE `DeformVAEHY` (bản cuối, "batt" ⚠️ = bone attention) |
| **`autoencoders/attention_blocks_deform_batt.py`** | **Tác giả** | Encoder điểm + skin/bone attention + block spatio-temporal |
| **`autoencoders/vae_pl_batt.py`** | **Tác giả** | LightningModule train VAE (loss) |
| **`diffusion/flow_matching_sit_video_bone_batt.py`** | **Tác giả** | LightningModule train DiT deform |
| **`denoisers/hunyuandit_video_scale_batt.py`** | **Tác giả** (sửa từ `hunyuandit.py`) | DiT video `HunYuanDiTPlainVideo` |
| **`pipelines_video_newvae_all_nonalign_infer.py`** | **Tác giả** | Pipeline infer + đánh giá |
| **`dataset/custom_dataloader.py`** | Tác giả (gốc Motion2VecSets) | Dataset (bản phát hành chỉ load ảnh) |
| `*_deform.py`, `*_deform_shape.py`, `*_batt_norm.py`, `hunyuandit_video.py`, `_scale.py`, `flow_matching_sit_video{,_bone}.py` | Tác giả | **Các biến thể/ablation cũ** (không xương; có `global_layers`; thêm normal vào query decoder; 320 token = 256 điểm + 64 xương) — không dùng ở bản cuối |

Config duy nhất: `hy3dshape/configs/infer.yaml` (model) và `configs/OBJVERSE/train/infer.yaml` (dataset; nhiều key là di sản Motion2VecSets) ✅.

#### 5.2 Đi qua các file cốt lõi

**(1) `model_deform_batt.py` — `DeformVAEHY`**
- Kế thừa khung VecSet của HY3D: `FourierEmbedder`, `PointCrossAttentionEncoder`, `pre_kl`/`post_kl`, `Transformer`, `CrossAttentionDecoder(out_channels=3)` (`:303-350`).
- RoPE 1D tính sẵn: `rotary_emb` theo T (cho temporal attn) và `global_rotary_emb` theo chỉ số khung của từng token trong T·N (cho global attn) (`:364-369`).
- `encode` (`:416-444`): lấy khung 0 làm nguồn, lặp T lần, **ghép kênh** `[src, tgt]` → (B,T,N,2C) → encoder; `DiagonalGaussianDistribution` với logvar kẹp **[-10, 2]** (`:66`, HY3D gốc là [-30, 20] — dòng 65 bị comment).
- `decode` (`:446-452`): `post_kl` → 16 lớp spatio-temporal → `decode_shape` (`:395-414`): query = Fourier(canonical xyz) cross-attend latent của từng khung, **cộng residual `+ query_points`** (`:413`) → mạng dự đoán **độ dời**, đầu ra là vị trí tuyệt đối.

**(2) `attention_blocks_deform_batt.py`**
- `ResidualAttentionBlock.forward` (`:447-485`): temporal attn (reshape (B·N,T,C), RoPE) → global attn (B, T·N, C, RoPE theo khung) → spatial attn + MLP (B·T, N, C). Temporal/global **không có MLP riêng**; chỉ spatial có MLP (giữ nguyên từ HY3D).
- `PointCrossAttentionEncoder.sample_points_and_latents_triposg` (`:890-936`): chọn token bằng `_sample_features_index` (random 4×num_tokens điểm rồi FPS tỉ lệ ¼, `:852-887`) **tính trên khung 0** rồi dùng **cùng chỉ số cho mọi khung** (`:901-912`) → token thứ k là cùng một điểm vật lý ở mọi thời điểm.
- `forward` (`:939-1008`): `input_proj` → skin self-attn trên dense points (mask từ skinning) → bone cross-attn (dense points query, 64 bone token làm KV) → cross-attn FPS query ↔ dense points → Transformer 8 lớp → `ln_post`.

**(3) `vae_pl_batt.py` — train VAE**
- bf16 autocast; `loss_def = MSE(deformed, points_tgt)`; `loss_kl = sum(kl)/B * kl_weight` (`:179-193`).
- **Batch ngoại lai bị "tắt"**: nếu loss > `clip_loss_threshold` (mặc định 0.01) thì ghi tên mẫu vào `large_loss_log.txt` và nhân loss với 1e-8 (`:198-214`).
- `on_train_epoch_start` gọi `train_dataset.set_epoch` (`:169-177`) để xoay vòng tập con 5000 chuỗi mỗi epoch (`custom_dataloader.py:638-647`).

**(4) `flow_matching_sit_video_bone_batt.py` — train DiT**
- Video cond: `image[:,1:]` (bỏ phần tử 0 vì dataset nhân đôi khung đầu, xem §5.3-9), DINO từng khung, reshape (B,T,1370,1024) (`:282-289`).
- Target: `first_stage_model.encode(..., sample_posterior=True)` (`:297`) — **lấy mẫu posterior** chứ không dùng mean.
- Shape cond: ShapeVAE HY3D encode `surface_src[:,0]` + kênh 0 (`:307-308`), `sample_posterior=False`, lặp T lần.
- Loss: `transport.training_losses(model, latents, {contexts, shape_latent_cond, canonical_points})` (`:366`); `canonical_points` = vị trí FPS trả từ encoder VAE.

**(5) `transport/transport.py` (SiT)**: path Linear, velocity; `t ~ U(0,1)` **một giá trị cho cả clip** (shape `(B,)`) (`:145-147`); loss `mean_flat((v_pred - u_t)^2)` (`:181-182`). Có sẵn nhánh `logit_normal` + shift (`:149-154`) nhưng config không bật ✅.

**(6) `hunyuandit_video_scale_batt.py` — `HunYuanDiTPlainVideo`**
- `forward` (`:799-856`): `x_embedder` → + `temp_pos_embed` sin-cos cố định theo khung (`:710-713, 806-808`) → + **spatial embedding sin-cos của canonical xyz** (`AbsolutePositionEmbedder`, `:38-78, 810-824`), giống nhau cho mọi khung → **timestep là một token prepend** vào mỗi khung (`:842-844`, kiểu HY3D 2.1, không AdaLN) → 21 block, skip dài từ nửa đầu sang nửa sau (`:752, 848-853`), MoE ở 6 block cuối (`:757`) → `FinalLayer` bỏ token thời gian (`:557-563`).
- `HunYuanDiTBlock.forward` (`:447-499`): skip-linear → spatial self-attn (B·T,N) → temporal attn (B·N,T, **không RoPE**, không zero-init) → cross-attn DINO khung tương ứng → cross-attn shape latent (`shape_cond_proj: 64→1024`, `:707`) → FFN/MoE.

**(7) `conditioner.py` — `SingleImageEncoder`**: dropout điều kiện theo **từng ảnh** (= từng khung, vì input là (B·T,…)) với `drop_ratio=0.1` (`:252-262`); uncond = tensor 0 (`:108-119`).

**(8) `pipelines_video_newvae_all_nonalign_infer.py`**: xem §3.5. CFG chỉ áp lên **điều kiện video**; shape latent và canonical points giữ nguyên ở cả hai nhánh (`:1096-1116`).

**(9) `custom_dataloader.py`**: crop theo **hộp bao hợp của mọi khung** (`:244-265`), pad vuông tỉ lệ 1.15 nền trắng (`:46-88`), stride thời gian ngẫu nhiên (`:423-435`), cửa sổ trượt cách 40 khung (`interval_between_sequence`), truncate skinning về 64 xương đầu (`:788-789`).

#### 5.3 Những điểm hay nhất trong code

**1. Phẫu thuật trọng số để tái dùng encoder/decoder của VAE tĩnh** — `model_deform_batt.py:205-215`
```python
except Exception as e:
    ...
    new_state_dict['encoder.input_proj.weight'] = new_state_dict['encoder.input_proj.weight'].repeat(1, 2) / 2
    new_state_dict['geo_decoder.output_proj.weight'] = new_state_dict['geo_decoder.output_proj.weight'].repeat(3, 1)
    new_state_dict['geo_decoder.output_proj.bias'] = new_state_dict['geo_decoder.output_proj.bias'].repeat(3)
```
Khi nạp ShapeVAE HY3D (input = 1 point cloud, output = 1 SDF) vào VAE biến dạng (input = cặp [nguồn, đích] → gấp đôi kênh; output = 3 kênh độ dời): **nhân đôi ma trận chiếu vào và chia 2** → với input `[x, x]` (khung đích = khung nguồn) đầu ra y hệt mô hình gốc; **lặp đầu ra 3 lần** cho 3 trục. Hay vì: mọi trọng số pretrained khác nạp nguyên, và lúc t=0 (biến dạng bằng 0) mạng tương đương mô hình gốc. ⚠️ Đây là cơ chế bắt lỗi `load_state_dict` (try/except), hơi "hack", nhưng ý tưởng rất đáng dùng.

**2. Zero-init mọi khối mới cấy vào backbone pretrained** — `attention_blocks_deform_batt.py:429-430, 443-444, 662-663, 673-676, 685-688`
```python
self.attn_t.c_proj.weight.data.zero_(); self.attn_t.c_proj.bias.data.zero_()     # temporal
self.attn_g.c_proj.weight.data.zero_(); self.attn_g.c_proj.bias.data.zero_()     # global
nn.init.zeros_(self.bone_linear.weight); nn.init.zeros_(self.bone_linear.bias)   # bone embed
self.skin_cross_attn.mlp.c_proj.weight.data.zero_(); ... attn.c_proj ... zero_()
```
Mọi nhánh mới đều là residual có output projection = 0 → lúc khởi đầu mạng **chính xác bằng** VAE HY3D. Đáng chú ý: ở DiT, temporal attn `attn0` và shape attn **không** zero-init (dòng zero-init bị comment, `hunyuandit_video_scale_batt.py:387-388, 422-423`) ⚠️ có thể vì DiT được fine-tune toàn bộ với lr nhỏ, hoặc khởi tạo từ checkpoint DiT bản trước.

**3. Mask attention từ skinning weights (privileged info)** — `attention_blocks_deform_batt.py:965-990`
```python
attn_mask = (attn_mask > 0.1).to(torch.float32)               # w > 0.1 -> bone "ảnh hưởng"
skin_attn_mask = (attn_mask @ attn_mask.permute(0, 2, 1)) > 0 # hai điểm chung >= 1 xương
bone_attn_mask = attn_mask > 0.1                              # điểm chỉ nhìn xương ảnh hưởng nó
data = self.skin_cross_attn(data, data, attn_mask=skin_attn_mask)
data = self.bone_cross_attn(data, query_bone, attn_mask=bone_attn_mask)
```
Hay vì biến cấu trúc động học (điểm nào di chuyển cùng nhau) thành **mask attention cứng**, không cần loss phụ; chỉ encoder nhận, decoder/diffusion không cần → latent "được dạy" chuyển động cứng theo đoạn. Khác paper: paper viết $ww^\top>\tau^s$ trên trọng số thực; code **nhị phân hóa trước** ở ngưỡng 0.1 rồi xét tích > 0 ✅. ⚠️ Rủi ro: điểm không có xương nào > 0.1 (hoặc chỉ chịu ảnh hưởng của xương thứ >64 bị cắt, `custom_dataloader.py:788-789`) sẽ có hàng mask toàn False → SDPA ra NaN.

**4. Token = cùng một điểm vật lý ở mọi khung** — `attention_blocks_deform_batt.py:901-916`
```python
x32 = x[:,0].to(torch.float32)                         # chỉ khung nguồn
indices, sampled_indices = self._sample_features_index(x32, num_tokens, seed)
selected_points = x.reshape(B*T, -1, C)[:, indices]   # cùng indices cho mọi khung
...
sampled_x = selected_points[sampled_indices]
```
FPS chạy trên khung chính tắc, áp cùng chỉ số cho mọi khung → attention thời gian theo trục T thực chất là attention dọc **quỹ đạo của một điểm**. Đây là lý do temporal attention ở đây hiệu quả (ℓ2-Corr 0.0160 → 0.0117, Tab. 3). Lấy mẫu "random 4N rồi FPS ¼" (`:852-887`) rẻ hơn FPS trên toàn bộ và có `random_start=self.training` → augmentation tự nhiên.

**5. Spatial embedding = PE của vị trí chính tắc, dùng chung mọi khung** — `hunyuandit_video_scale_batt.py:810-824`
```python
spatial_emb = self.spatial_embed(canonical_points)        # [B, N, D], sin-cos của xyz
spatial_emb = spatial_emb.reshape(B, 1, N, -1).repeat(1, T, 1, 1)
x = x + spatial_emb
```
Latent VecSet vốn **không có thứ tự**; nhiễu Gaussian ban đầu không biết token nào là điểm nào. Gắn PE của vị trí 3D chính tắc giúp mỗi token "biết mình là ai" và khớp đúng với latent đích (lúc train dùng chính vị trí FPS của encoder, `flow_matching_sit_video_bone_batt.py:297,366`). Đơn giản (sin-cos tần số `1/10000^(i/d)`, pad 0 cho đủ chiều, `:38-78`) mà hiệu quả.

**6. Timestep làm token prepend cho mỗi khung + bỏ đi ở cuối** — `hunyuandit_video_scale_batt.py:842-844, 557-563`
```python
c = c.unsqueeze(dim=1).repeat(1, T, 1, 1)   # [B, T, 1, D]
x = torch.cat([c, x], dim=2)                # [B, T, 1+N, D]
...
x = x[:, :, 1:]                             # FinalLayer bỏ token thời gian
```
Giữ nguyên cơ chế của HY3D 2.1 (in-context conditioning) để không phải thêm AdaLN → nạp được trọng số pretrained. Token thời gian cũng đi qua temporal attention.

**7. Dropout điều kiện theo từng khung** — `conditioner.py:252-262`
```python
random_p = torch.rand(len(image), device='cuda')   # len = B*T khung
outputs['main'] *= (random_p > self.drop_ratio).view(-1,1,1)
```
Vì ảnh được flatten (B·T), mỗi khung bị xóa điều kiện **độc lập** với xác suất 0.1 → vừa phục vụ CFG, vừa ép mô hình **suy chuyển động của khung thiếu từ các khung lân cận** qua temporal attention. ⚠️ Lưu ý quirk: `disable_drop=True` mặc định (`:250`), chỉ chuyển sang False sau lần đầu callback gọi `test_overfit` (`flow_matching_sit_video_bone_batt.py:431-463`, callback mỗi 100 step `configs/infer.yaml` phần `callbacks`) → dropout thực tế phụ thuộc vào việc bật logger.

**8. "Giảm câm" batch ngoại lai thay vì crash/NaN** — `vae_pl_batt.py:198-214`
```python
if loss_value > self.clip_loss_threshold:   # 0.01
    ... f.write(f"{model_name}: {loss_value}\n")
    loss = loss * 1e-8                       # giữ graph, gần như không cập nhật
```
Với dữ liệu Objaverse bẩn (rig hỏng, biến dạng vô lý), một mẫu xấu có thể phá cả VAE. Mẹo: nhân 1e-8 (không phải 0) để DDP vẫn đồng bộ gradient, đồng thời **log tên mẫu** để lọc dữ liệu sau. ⚠️ Bug nhỏ: `model_name` chỉ được gán trong nhánh `output_folder is not None` nhưng được dùng trong `print` ngoài nhánh (`:207-211`).

**9. Khung đích đầu tiên = chính khung nguồn (neo identity)** — `custom_dataloader.py:808-814`
```python
sampled_idx = range(start_idx, end_idx + 1, stride)
sampled_idx = [start_idx] + sorted(sampled_idx)   # phần tử 0 = nguồn, 1..T = đích (đích[0] == nguồn)
```
Phần tử 0 là canonical; T khung đích bắt đầu bằng **chính nó** → biến dạng của khung đích đầu luôn = 0, neo latent và giúp decoder học "đứng yên"; video điều kiện `[:,1:]` cũng bắt đầu bằng ảnh đã dùng để dựng mesh.

**10. Augmentation tốc độ bằng stride + crop hợp nhất theo clip** — `custom_dataloader.py:423-435, 244-265`
```python
avai_stride_list = [1,2] if length_sequence > 10 else [1,2,3,4]   # train
x_min_all, ... = min(x_min_all, x_min), ...                        # bbox hợp của MỌI khung
```
Stride 1–4 với T=6 cho phép mô hình thấy chuyển động nhanh/chậm khác nhau (6 khung phủ 6–21 khung gốc). Crop theo bbox hợp của cả clip (không crop từng khung) **giữ nguyên chuyển động tịnh tiến trong ảnh**, vốn là tín hiệu cho deformation.

**11. Phẫu thuật nhỏ cho kênh "sharp edge"** — `attention_blocks_deform_batt.py:894-896`; `flow_matching_sit_video_bone_batt.py:307`
HY3D 2.1 nhận `point_feats=4` (normal + cờ cạnh sắc). Mesh4D không có cờ này nên **chèn một kênh 0** để giữ đúng shape input pretrained thay vì đổi kiến trúc.

**12. LR gần như hằng số do `steps` cực lớn** — `configs/infer.yaml:6, 177`; `utils/trainings/lr_scheduler.py` (`LambdaWarmUpCosineFactorScheduler`)
`steps: 10_0000_0000` (1e9) làm cosine decay gần như không giảm → thực tế là warmup 500 step rồi LR hằng 1e-5. ⚠️ Có thể cố ý (fine-tune lâu, dừng tay theo val).

#### 5.4 Khác biệt paper vs code

| # | Paper | Code | Ghi chú |
|---|---|---|---|
| 1 | Video feature = **DINO-Giant** (§3.3) | `facebook/dinov2-large`, `context_dim 1024`, ảnh 518 (`configs/infer.yaml:89-94`) | ✅ khác biệt rõ ràng. ⚠️ Có thể checkpoint phát hành dùng Large |
| 2 | Mask skin: $ww^\top>\tau^s$ trên trọng số thực | Nhị phân hóa ở 0.1 rồi tích > 0 (`attention_blocks_deform_batt.py:968-971`) | ✅ τ = 0.1 không có trong paper |
| 3 | Thứ tự VAE block: temporal → global → spatial (Fig. 2) | Đúng như vậy (`:447-485`) | ✅ khớp; temporal/global **không có MLP** |
| 4 | DiT: "temporal embeddings" | Sin-cos cố định cộng vào token (`use_temp_pe: true`), **không RoPE** (`use_temporal_rope: false`) | ✅ trong khi VAE dùng RoPE |
| 5 | Không nói số khung | **T = 6** khung mỗi cửa sổ (`latent_shape: [6,256,64]`, `num_frames: 6`) | ✅ quan trọng: "mô hình hóa toàn bộ chuỗi" thực ra là 6 khung (stride 1–4) |
| 6 | Không nói số token | N = 256 token/khung, $c=1024$, $c_o=64$ | ✅ `model_deform_batt.py:292`, config |
| 7 | KL weight λ | `first_stage_config.kl_weight`, **không có giá trị** trong repo | ✅ thiếu |
| 8 | "Ours (Aligned)" dùng CPD | Pipeline phát hành bỏ CPD (comment `:945-953`) | ✅ repo tái tạo hàng "Ours" |
| 9 | Diffusion train trên latent | Latent = **mẫu posterior** (`sample_posterior=True`), không phải mean | ✅ không nói trong paper |
| 10 | Chỉ nói điều kiện bằng video | Dropout điều kiện **theo khung** 10%, CFG 5.0 lúc infer | ✅ không nói trong paper |
| 11 | Canonical points lúc infer: FPS trên mesh | FPS trên **đỉnh** mesh (không phải điểm lấy mẫu bề mặt như lúc train) | ✅ `:1071`; ⚠️ lệch phân phối nhẹ nếu mesh có đỉnh phân bố không đều |
| 12 | Loss VAE trên độ dời $(\mathcal{V}_t-\mathcal{V}_1)$ | MSE trên **vị trí tuyệt đối**, mạng xuất query + offset (`model_deform_batt.py:413`, `vae_pl_batt.py:190`) | ✅ tương đương toán học |
| 13 | Xương: head/tail của khung 1 và khung t | Có, nhưng bone token **chỉ làm KV** cho dense points rồi bị bỏ, không vào latent (`attention_blocks_deform_batt.py:988-994`) | ✅ bản cũ `_scale.py` từng giữ 320 = 256+64 token |
| 14 | Train/test split, filter | Data loader train thực sự (đọc mesh, normal, skinning) **không phát hành** | ✅ chỉ còn `get_deform_ae_tripoSG` cũ (có biến `loc`/`scale` chưa định nghĩa, `:700-702`) |

#### 5.5 Hyperparameter thực tế (từ code/config)

| Nhóm | Giá trị | Nguồn |
|---|---|---|
| VAE | width 1024, heads 16, encoder 8 lớp, decoder 16 lớp, embed_dim 64, num_freqs 8, include_pi false, qk_norm true, qkv_bias false | `configs/infer.yaml:48-70` ✅ |
| VAE token | 256 FPS token/khung (random 1024 → FPS ¼); $B_{max}$ 64 | `model_deform_batt.py:290-293`, `attention_blocks_deform_batt.py:967` ✅ |
| VAE logvar clamp | [-10, 2] | `model_deform_batt.py:66` ✅ |
| VAE outlier threshold | 0.01 (comment cũ 0.002) | `vae_pl_batt.py:30,198` ✅ |
| Skin/bone mask τ | 0.1 | `attention_blocks_deform_batt.py:968,975` ✅ |
| Dataset (di sản M2VS) | pc_size 2048, num_samples 1024, length_sequence 6, stride {1..4}, interval_between_sequence 40, 5000 chuỗi/epoch | `configs/OBJVERSE/train/infer.yaml`, `custom_dataloader.py:131,641` ✅ |
| DiT | hidden 2048, depth 21, heads 16, context_dim 1024, qk_norm RMS, MoE 6 lớp cuối × 8 expert top-2, in_channels 64, T=6 | `configs/infer.yaml:97-124` ✅ |
| Shape cond | HY3D ShapeVAE, 2048 điểm, downsample 8 → 256 token × 64 | `configs/infer.yaml:74-83` ✅ |
| Image cond | DINOv2-Large, 518², CLS token, drop 0.1 | `configs/infer.yaml:86-95` ✅ |
| Flow matching | Linear path, velocity, t ~ U(0,1) chung cho clip | `configs/infer.yaml` (scheduler_cfg), `transport.py:145-147` ✅ |
| z_scale_factor | 1.00395 (= scale factor của HY3D, gần như 1) | `configs/infer.yaml:40` ✅ |
| Optim | AdamW β=(0.9, 0.99), eps 1e-6, wd 0.01; base_lr 1e-5; warmup 500; bf16; grad clip 1.0 (norm) | `configs/infer.yaml:5-12,166-180` ✅ |
| Sampling | 50 bước Euler, sigma linspace(0,1), CFG 5.0, fp16 | `pipelines_...:890-894,1077` ✅ |
| Ảnh | pad 1.15, nền trắng, mean/std 0.5 → DINO norm | `configs/infer.yaml:28-32`, `conditioner.py:93-99` ✅ |

---

### 6. Hạn chế / điểm yếu / câu hỏi mở

1. **Phụ thuộc hoàn toàn vào mesh khung đầu** (tác giả tự nêu) ✅ §5. Sai ở khung đầu (ví dụ chân dính nhau) không sửa được về sau; không đổi được topology ✅.
2. **Cửa sổ chỉ 6 khung** (T=6) ✅ config. Paper không nói cách xử lý video dài (nối cửa sổ? trôi?). ⚠️ "Encode cả chuỗi" trong paper nên hiểu là "cả cửa sổ 6 khung".
3. **Texture tĩnh từ khung 1** → không thể hiện thay đổi ngoại hình, phần sau lưng là texture "bịa" của HY3D-Paint ✅ §4.3 (CLIP thấp hơn).
4. **Không vật lý / không rất mềm**: khó với vật rất phi cứng (tác giả nêu) ✅ §5. Prior xương thiên về biến dạng cứng từng đoạn.
5. **Benchmark tự xây, chỉ 50 chuỗi, video frontal, tổng hợp** ✅ §4.1. In-the-wild chỉ định tính (3 ví dụ Consistent4D trong `DATA/`).
6. Ablation chỉ cho VAE; **không có ablation diffusion** (spatial PE, shape cross-attn, số khung, CFG) ⚠️.
7. Không báo cáo compute, thời gian infer ✅; DiT ~cỡ HY3D 2.1 nặng (⚠️ vài B tham số).
8. Code: thiếu train script, thiếu kl_weight, mask có thể NaN (§5.3-3), dropout CFG phụ thuộc callback (§5.3-7) ⚠️.
9. License Tencent HY3D: phi thương mại theo điều khoản cộng đồng, loại trừ EU/UK/Hàn Quốc ✅ → không nên đem backbone này vào sản phẩm.
10. Câu hỏi mở: tương ứng từ FPS-token là cố định theo khung đầu — nếu vùng bị che ở khung đầu thì sao? (Ở đây không sao vì mesh hoàn chỉnh; nhưng ở 2D thì là vấn đề lớn.)

---

### 7. Liên hệ với GReg

#### 7.1 Bài học tổng quát (mức "luận điểm" để viết luận văn)

- **Tách hình học khỏi ngoại hình + tương ứng tường minh ⇒ nhất quán thời gian.** Mesh4D (mesh cố định + trường biến dạng, texture từ khung 1) có FVD 572.7–601.9 so với 692.2 của HY3D dựng từng khung (dù đã dùng chung nhiễu) ✅ Tab. 2. Đây là bằng chứng hỗ trợ cho lập luận A1 + Hướng 1: dùng tương ứng SMPL-X để **mang texture từ ảnh nguồn**, phần sinh chỉ lo vùng che. ⚠️ Tương tự, có cái giá: CLIP (khớp từng khung) giảm nhẹ 0.9174 → 0.9141 — GReg sẽ gặp "texture nguồn không chứa vùng mới lộ".
- **Tương ứng barycentric qua topology chung** chính là cơ chế A1 đang dùng (Mesh4D §3.2: lấy mẫu điểm ở khung 1, biểu diễn barycentric, dựng lại ở khung t) ✅. Có thể trích Mesh4D làm tiền lệ (CVPR 2026) cho việc dùng tương ứng barycentric từ mesh cùng topology làm tín hiệu học.
- **Attention dọc quỹ đạo**: Mesh4D làm temporal attention hiệu quả vì token k luôn là cùng một điểm vật lý. Video DiT (Wan) có token theo lưới pixel cố định → điểm vật lý trượt qua token. ⚠️ Đây là lý do nên **đưa bản đồ tương ứng (canonical coordinate map) vào làm điều kiện** để mô hình tự liên kết token qua thời gian.

#### 7.2 Kỹ thuật chuyển được

| # | Kỹ thuật Mesh4D | Áp dụng vào GReg | Hướng | Độ khó | Ghi chú |
|---|---|---|---|---|---|
| T1 | **Spatial PE từ vị trí chính tắc** (§5.3-5) | Rasterize, với mỗi pixel đích, **tọa độ chính tắc của SMPL-X** (xyz đỉnh ở T-pose, nội suy barycentric — có sẵn từ A1) → 3 kênh; sin-cos rồi cộng vào token, hoặc concat làm kênh điều kiện. Đây chính là ablation 6 "UV/coordinate map kiểu GenWarp" của §1.9, nhưng dùng tọa độ **bề mặt cơ thể** thay vì tọa độ ảnh nguồn | H1 (chính), H2 (kênh phụ) | Thấp–vừa | Tọa độ chính tắc giống nhau giữa các khung cho cùng điểm cơ thể → temporal attention của Wan "biết" token nào là cùng điểm. Tốt hơn tọa độ (u,v) ảnh nguồn ở vùng che (vẫn có giá trị cho lòng bàn tay dù nguồn không thấy) ⚠️ |
| T2 | **Zero-init mọi khối mới** (§5.3-2) | Kênh vào mới của Wan (warp, V·C, normal, map T1), adapter tay/mặt, conv điều kiện trong pix2pix-turbo: zero-init output projection | H1, H2 | Thấp | Đã có trong §1.4 của kế hoạch; Mesh4D là thêm một tiền lệ. Lưu ý Mesh4D **không** zero-init temporal/shape attn trong DiT → zero-init không bắt buộc nếu lr nhỏ, nhưng là mặc định an toàn |
| T3 | **Phẫu thuật trọng số input "nhân đôi /2"** (§5.3-1) | Khi patch-embed của Wan phải nhận thêm latent **cùng loại** (ví dụ latent VAE của video warped, cùng không gian với latent nhiễu): khởi tạo trọng số kênh mới bằng **bản sao trọng số latent gốc** thay vì 0, rồi chia đôi cả hai ⚠️ | H1 | Thấp | Cẩn thận: khác zero-init, nó thay đổi hành vi ngay từ bước 0 (mô hình nhận trung bình nhiễu+warp). Đề xuất: làm ablation "zero-init vs copy/2" cho kênh warp; có thể hội tụ nhanh hơn vì kênh warp "đã được hiểu" |
| T4 | **Mask attention từ skinning weights** (§5.3-3) | SMPL-X có sẵn LBS weights (55 khớp) cho mọi đỉnh → rasterize **bản đồ phần cơ thể / top-k joint** cho từng pixel; dùng làm (a) mask cho cross-attention của adapter tay/mặt (chỉ token thuộc tay mới nhìn crop tay), (b) kênh điều kiện thêm. Với GReg, thông tin này **không phải privileged** — có cả lúc test | H1, B1 | Vừa | Tab. 3: thêm xương giảm ℓ2-Corr 15% (0.0138 → 0.0117) ✅ — nhưng đó là 3D, ⚠️ chưa biết lợi ở 2D. Khi làm, nhớ bẫy NaN: hàng mask toàn False → thêm "sink token" (đã có trong ý tưởng SignRefine §1.4) |
| T5 | **Dropout điều kiện theo từng khung** (§5.3-7) | Khi train H1, xóa ngẫu nhiên (p≈0.1) điều kiện warp/normal của **từng khung riêng** → mô hình học nội suy khung thiếu từ lân cận → bền với khung fit SMPL-X hỏng, khung fade-in | H1 | Thấp | Đồng thời cho CFG trên điều kiện hình học. Có thể giữ ảnh tham chiếu I_s trong cả hai nhánh CFG (Mesh4D giữ shape cond ở cả hai nhánh, `pipelines_...:1096-1116`) |
| T6 | **Stride ngẫu nhiên 1–4** (§5.3-10) | Lấy cửa sổ khung với stride ngẫu nhiên khi train LoRA Wan → giải quyết một phần câu hỏi fps 25 vs 16 (§1.6) và đa dạng tốc độ ký hiệu | H1 | Thấp | Giới hạn stride ≤2 cho tay nhanh ⚠️ (stride lớn làm tay nhòe chuyển động) |
| T7 | **Neo khung 0 = nguồn** (§5.3-9) | Đặt I_s làm khung đầu của cửa sổ video đích (Wan I2V/VACE vốn hỗ trợ); khung 0 có loss ≈ identity → neo danh tính | H1 | Thấp | Khớp giao thức reconstruction của GReg (khung nguồn = khung hợp lệ đầu) |
| T8 | **"Giảm câm" batch ngoại lai + log tên clip** (§5.3-8) | Trong train GReg/H2: clip có loss vượt ngưỡng (SMPL-X fit hỏng, cắt cảnh) → nhân 1e-8 và ghi tên vào file để lọc dữ liệu | GReg hiện tại, H1, H2 | Rất thấp | Ngưỡng phải đặt theo phân vị loss (ví dụ > p99.5 trượt) chứ đừng cố định ⚠️ |
| T9 | **Crop theo bbox hợp của cả clip** (§5.3-10) | Kiểm tra lại pipeline cache 480/512: crop phải cố định theo clip (không recenter từng khung), nếu không warp và chuyển động toàn cục bị phá | Dữ liệu (cả hai hướng) | Rất thấp | ⚠️ cần đối chiếu `scripts/02_preprocess_hamnosys.py` |
| T10 | **Metric ℓ2-Corr** | Đánh giá chất lượng flow A1 bằng tương ứng từ mesh: so flow dự đoán (TPSMM) với flow mesh trên vùng nhìn thấy — một chỉ số "tracking" cho bảng ablation A1 | A1 | Thấp | Bổ sung cho L1/LPIPS; trả lời câu "warp đúng chỗ hơn bao nhiêu" |
| T11 | **Temporal attention zero-init chèn vào mạng per-frame** | Nếu muốn GReg (hoặc refiner H2) có prior thời gian rẻ: thêm temporal attention (B·HW, T, C) ở bottleneck, zero-init, train trên clip ngắn 4–6 khung | H2 / GReg | Vừa | Tab. 3: bỏ temporal+global → ℓ2-Corr 0.0117 → 0.0160 ✅ (3D). Cho H2 đây là phương án thay "loss thời gian bằng flow" ở §2.6 |

#### 7.3 Ý tưởng mới gợi ra (⚠️ chưa kiểm chứng)

- **Texture UV + inpainting** thay vì inpainting ảnh: Mesh4D tô texture một lần cho toàn mesh (phần không thấy do HY3D-Paint sinh), rồi chỉ biến dạng hình học → nhất quán tuyệt đối. Cho GReg: chiếu I_s lên UV SMPL-X (nhờ A1 đã có rasterize + visibility), **inpaint UV texture** cho vùng không thấy (lòng bàn tay, mặt trong cẳng tay) bằng một mô hình diffusion một lần cho mỗi người ký, rồi render theo mesh đích và để GReg/H2 chỉ sửa phần SMPL-X không mô hình hóa (tóc, áo rộng, ánh sáng). Khó: quần áo/tóc không thuộc SMPL-X, ánh sáng phụ thuộc pose, UV tay SMPL-X nhỏ. Có thể là một ablation "texture prior vs video prior" thú vị.
- **Canonical coordinate map làm "ngôn ngữ chung" giữa H1 và A1**: cùng một map (T1) vừa là điều kiện cho DiT, vừa dùng để đo nhất quán thời gian (cùng tọa độ chính tắc ở hai khung phải cùng màu ở vùng nhìn thấy cả hai) → một loss/metric thời gian không cần optical flow.

#### 7.4 Không nên lấy (và vì sao)

1. **Toàn bộ pipeline 3D (Deformation VAE + DiT latent 3D + HY3D)**: GReg đã **có** mesh SMPL-X fit sẵn cho mọi khung — hình học là đầu vào, không phải thứ cần dự đoán. Bài toán của GReg là ngoại hình. Dựng lại 4D mesh là thừa.
2. **Backbone HY3D 2.1 / license Tencent**: phi thương mại, loại trừ EU/UK/KR; DiT nặng (comment 68 GB khi overfit) — không vừa 24 GB để fine-tune ⚠️.
3. **Điều kiện bằng DINO từng khung đích**: ở GReg khung đích chính là thứ phải sinh → không có ảnh đích làm điều kiện. Chỉ DINO/CLIP của **ảnh nguồn** là dùng được (nhánh tham chiếu đã có trong Wan).
4. **Mask attention xương dạng cứng trong encoder như nguyên bản**: ở 2D, pixel thuộc nhiều phần (biên tay–thân, tay che mặt) → mask cứng dễ cắt ngữ cảnh cần thiết; nên dùng mềm (bias) hoặc chỉ trong adapter ⚠️.
5. **Cửa sổ 6 khung không có cơ chế nối**: Mesh4D không giải quyết video dài; GReg/H1 cần cửa sổ chồng lấn + neo khung nguồn (đã có trong kế hoạch §1.6) — đừng kỳ vọng Mesh4D cho lời giải.
6. **LR hằng do `steps=1e9`**: là quirk, không phải khuyến nghị; với LoRA trên 4.6k clip nên có lịch giảm LR + dừng sớm theo val.

#### 7.5 Ưu tiên đề xuất cho luận văn (⚠️)
1. (Rẻ, ngay) T8 + T9 + T10 vào pipeline GReg/A1 hiện tại.
2. (Khi bắt đầu H1) T2 + T5 + T6 + T7 là mặc định; **T1 làm ablation chính** đối chiếu với "chỉ ảnh warped" và "UV ảnh nguồn kiểu GenWarp" — trả lời phản biện GenWarp bằng một biến thể có cơ sở 3D.
3. (Nếu còn thời gian) T4 cho adapter tay; T11 cho H2.
4. Trích dẫn Mesh4D trong phần related work: "dense correspondence từ mesh cùng topology + texture từ khung tham chiếu cho nhất quán thời gian tốt hơn dựng từng khung (Tab. 2)".
