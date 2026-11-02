# Ghi chú 3 bài SLVG/SLP: SignRefine (2026), SignViP (NeurIPS 2025), Signs as Tokens / SOKE (ICCV 2025)

Ngày lập: 2026-11-02. PDF nằm trong `ref papers/`. Đã đọc **toàn bộ** ba PDF (SignRefine 21 trang; SignViP 27 trang gồm checklist + phụ lục A–G; SOKE 11 trang, bản CVF không kèm phụ lục).

Quy ước (giữ như các file docs khác):
- ✅ = đã kiểm chứng trong paper (ghi mục/bảng/phương trình) hoặc trên trang web/repo chính thức (ghi URL).
- ⚠️ = suy luận, thông tin không có trong paper, hoặc chưa kiểm chứng; **phải kiểm tra trước khi viết vào luận văn**.

**Kết luận chung quan trọng:** cả ba bài đều **không** báo cáo số liệu của GReg (Shi et al., ICCV 2025) hay PGMM (Shi et al., ECCV 2024). PGMM chỉ được trích dẫn trong danh mục tham khảo (SignRefine [50], SignViP [69], SOKE [53]); GReg không được trích dẫn ở bài nào ✅ (đã grep toàn văn ba PDF). Cũng không bài nào dùng SignAvatars hay HamNoSys làm dữ liệu huấn luyện/đánh giá ✅ (chi tiết ở từng bài).

---

## Bảng so sánh nhanh

| | SignRefine | SignViP | SOKE (Signs as Tokens) |
|---|---|---|---|
| Venue | arXiv 2609.08496 (8/9/2026); trang dự án ghi ECCV 2026 ✅ | NeurIPS 2025 ✅ (arXiv 2506.15980) | ICCV 2025 ✅ (arXiv 2411.17799) |
| Bài toán | **pose → video** (giai đoạn 2 của SLP) ✅ §1, §4 | **text → video** (SLVG), p(X \| x*, T) ✅ §3.2, Eq. 3–4 | **text → chuỗi SMPL-X 3D** (avatar), không sinh video ✅ §3.1 |
| Điều kiện vào bộ sinh | Ảnh tham chiếu + skeleton 2D toàn thân + 3 render keypoint vùng (mặt, tay T, tay P) 256² + toạ độ CoordConv ✅ §4, §4.1 | Ảnh tham chiếu + embedding đa điều kiện từ (fine-grained pose 2D + 3D hands) ✅ §3.2–3.4 | Text + thẻ ngôn ngữ + token motion của từ tra từ điển ✅ §3.3–3.4 |
| Backbone | Wan2.1-1.3B-Fun-Control DiT (đóng băng) ✅ §4, §5.1 | SD v1.5 U-Net + temporal attn AnimateDiff + ReferenceNet; tổng 2777.92M tham số ✅ §4.1, Bảng 8 | mBART-large-cc25 (fine-tune) + 3 VQ-VAE ✅ §4 |
| Dữ liệu | NVSign (mới, BSL, 78.1 h) train; test NVSign/Phoenix14T/BSL Corpus ✅ §3, §5.2 | PHOENIX-2014T (272×224) + How2Sign (512×512) ✅ §4.1 | How2Sign + CSL-Daily + Phoenix-2014T (SMPL-X) ✅ §4 |
| Compute | 4× GH200, 30k iter ✅ §5.1 | 4× RTX A6000, 3×50k bước ✅ §4.1 | 6× RTX 3090 ✅ §4 |
| Inference | "vài phút" cho clip 81 khung ✅ §6; số bước không ghi ⚠️ | DDIM 50 bước, CFG 3.5; 1.2398 s/khung ✅ §4.1, Bảng 8 | 1.55 s/video (RTX 3090) ✅ Bảng 2 |
| Module tay/mặt | Local adapters + masked cross-attn + sink token; loss vùng w_h = w_f = 10, w_b = 1 ✅ Eq. 2–4 | Điều kiện 3D hands; metric Hand SSIM ✅ §4.1, Bảng 5 | Tokenizer tách rời body/LH/RH; multi-head decoding ✅ §3.2–3.3 |
| Code / checkpoint | **Không tìm thấy** (trang dự án không có link code/data) | **Có**: github.com/umnooob/signvip; ckpt RWTH-T trên HF; How2Sign "coming soon" | **Có**: github.com/2000ZRL/SOKE; ckpt tokenizer + mBART (Drive); ckpt generator không thấy |

---

## Bài 1 — SignRefine: Adapting Foundational Video Models for Sign Language Generation

Tác giả: Pelykh, Fish, Mercanoglu Sincan, Bowden (CVSSP, Surrey). arXiv 2609.08496 v1, 8/9/2026 ✅. Trang dự án ghi "ECCV 2026", BibTeX ghi "proceedings entry pending" ✅ (https://cogvis-cvssp.github.io/papers/signrefine/); trang arXiv không ghi venue ✅.

### 1.1 Bài toán và I/O
- Tác giả nói rõ chỉ làm **giai đoạn 2** của SLP: "generating ... sign language video across diverse appearances given the driving skeleton sequence" ✅ §1 (đoạn "Modern sign language production approaches typically consist of two stages").
- Đầu vào: (a) **ảnh tham chiếu** định nghĩa ngoại hình người ký, (b) **chuỗi keypoint 2D toàn thân** ✅ §4 câu đầu; Hình 2.
- Điều kiện vùng: keypoint của **mặt, tay trái, tay phải** được render thành video 256×256 riêng cho mỗi vùng, qua 3 encoder độc lập ✅ §4.1. Thêm 2 kênh toạ độ tuyệt đối chuẩn hoá [−1, 1] theo kiểu CoordConv (Eq. 1) ✅ §4.1.
- Bộ phát hiện keypoint: đánh giá dùng **133 keypoint toàn thân từ RTMPose-X** ✅ §5.2; pipeline điều kiện "rely on off-the-shelf pose estimators such as RTMPose, WiLoR" ✅ §6. Bài không nói rõ detector nào tạo keypoint điều kiện lúc train ⚠️ (suy ra là RTMPose-X vì cùng pipeline).
- Text: backbone có T5 nhưng "varying the text prompt has a negligible effect", nên **cache latent của một prompt chung** ✅ §5.1. Text không phải điều kiện có nghĩa.
- 3D mesh: **không dùng mặc định**. Ablation "Dense conditions" thay render keypoint bằng **face normals + render MANO 3D** (Bảng 5), nhưng chọn sparse làm mặc định ✅ §5.4. SMPL/MANO/FLAME chỉ là **annotation kèm dataset NVSign** ✅ §3.

### 1.2 Backbone, phần train/đóng băng, tham số
- Backbone: **Wan2.1-1.3B-Fun-Control** (DiT 30 block, hidden 1536), gồm T5 text encoder, CLIP image encoder (ảnh tham chiếu), 3D VAE — **toàn bộ đóng băng** ✅ §4, §5.1, §4.2.
- Train: 3 encoder điều kiện vùng (ResNet-style, GroupNorm + SiLU, 256² → lưới 16×16, conv 1D nhân quả nén T → T/4, linear 512 → 1536) + 2 adapter (mặt, tay) ✅ §5.1. Output projection khởi tạo gần 0 (kiểu ControlNet) ✅ §4.3.
- Vị trí tiêm: adapter mặt ở block {0, 6, 12, 18, 24}, adapter tay ở {2, 8, 14, 20, 26} (5 block mỗi adapter) ✅ §4.2. **Mâu thuẫn nội bộ** ⚠️: §5.1 lại ghi "each including 6 cross-attention blocks"; mỗi block 12 head × dim 128, RMSNorm trên Q/K ✅ §5.1.
- Tay trái + tay phải được **nối (concat)** trước khi tiêm để xử lý che khuất giữa hai tay ✅ §4.2.
- **Số tham số của adapter: không ghi** ⚠️. Chỉ có "1.3B-parameter DiT backbone" ✅ §5.4.

### 1.3 Dữ liệu
- **NVSign** (mới): BSL, nguồn web, 44,759 đoạn ký từ 317 video, 78.1 giờ, ~2,184 người ký, ~40 % cảnh hội thoại ≥ 2 người, vocab 12K ✅ §3, Bảng 1. Annotation: transcript tiếng Anh căn thời gian, keypoint 2D, điểm hoạt động từng người ký, depth, tham số **SMPL body, MANO hand, FLAME face** ✅ §3. Ba partition (chi tiết ở supplementary, không có trong PDF) ⚠️.
- Train: ~40,000 clip từ NVSign train ✅ §5.1. Khung gốc **832×480** ✅ §5.4; clip **81 khung** ✅ §6; dữ liệu quay ở **25 fps** ✅ §6.
- Test: NVSign (appearance-independent), Phoenix14T (210×260 ✅ §5.3), BSL Corpus — **100 video ngẫu nhiên mỗi bộ** ✅ §5.2. Comprehension study: 10 clip BSL từ YouTube-SL-25 ✅ §5.3.
- SignAvatars: chỉ trích dẫn [76] như một phương pháp giai đoạn 1 (text → motion token) ✅ §1; **không dùng**. How2Sign, CSL-Daily: chỉ trong Bảng 1 so sánh dataset ✅. HamNoSys: không xuất hiện ✅.

### 1.4 Metric và số liệu (Bảng 2, §5.2)
Đánh giá **theo vùng** (crop từ bbox tính từ keypoint GT): MPJPE (RTMPose-X, 133 kp), LPIPS (VGG), SSIM ✅ §5.2.

| Bộ | Phương pháp | Face MPJPE↓ / LPIPS↓ / SSIM↑ | RH MPJPE↓ / LPIPS↓ / SSIM↑ | LH MPJPE↓ / LPIPS↓ / SSIM↑ |
|---|---|---|---|---|
| NVSign | SignGAN | 10.424 / 0.564 / 0.302 | 24.774 / 0.627 / 0.388 | 21.852 / 0.620 / 0.384 |
| | SignViP | 10.576 / 0.534 / 0.356 | 23.957 / 0.581 / 0.428 | 24.383 / 0.588 / 0.424 |
| | Wan-VACE | 4.067 / 0.422 / 0.367 | 15.904 / 0.525 / 0.380 | 15.431 / 0.503 / 0.407 |
| | Wan-FC | 1.835 / 0.197 / 0.713 | 9.565 / 0.364 / 0.583 | 9.341 / 0.349 / 0.597 |
| | **Ours** | **1.770 / 0.194 / 0.714** | **6.602 / 0.330 / 0.629** | **6.638 / 0.321 / 0.640** |
| Phoenix14T | SignGAN | 5.457 / 0.585 / 0.343 | 28.507 / 0.707 / 0.469 | 25.373 / 0.657 / 0.470 |
| | SignViP | 7.194 / 0.456 / 0.276 | 10.345 / 0.534 / 0.425 | 14.405 / 0.566 / 0.376 |
| | Wan-VACE | 2.162 / 0.383 / 0.413 | 11.717 / 0.551 / 0.367 | 13.214 / 0.546 / 0.299 |
| | Wan-FC | 1.151 / 0.223 / 0.688 | 5.754 / 0.402 / 0.629 | 5.719 / 0.374 / 0.642 |
| | **Ours** | **0.916 / 0.214 / 0.701** | **4.582 / 0.386 / 0.660** | **5.101 / 0.366 / 0.668** |
| BSL Corpus | SignGAN | 9.415 / 0.634 / 0.330 | 22.726 / 0.720 / 0.406 | 16.858 / 0.689 / 0.390 |
| | SignViP | 28.222 / 0.626 / 0.289 | 23.085 / 0.611 / 0.381 | 22.111 / 0.593 / 0.372 |
| | Wan-VACE | 3.172 / 0.436 / 0.380 | 17.524 / 0.567 / 0.351 | 15.385 / 0.557 / 0.338 |
| | Wan-FC | 1.790 / 0.221 / 0.685 | 8.232 / 0.389 / 0.559 | 7.002 / 0.373 / 0.563 |
| | **Ours** | **1.524 / 0.222 / 0.690** | **6.314 / 0.365 / 0.608** | **5.406 / 0.353 / 0.607** |

Ghi chú về baseline ✅ §5.2: SignViP dùng **checkpoint Phoenix14T duy nhất mà tác giả công bố**, token điều kiện xây từ GT; SignGAN dùng checkpoint xin trực tiếp từ tác giả, một ngoại hình cố định, không nhận ảnh tham chiếu. Đơn vị MPJPE không ghi (⚠️ có lẽ pixel ở độ phân giải crop).

- **User study** (15 người, trình độ BSL khác nhau, ~200 quan sát/mô hình) — Bảng 3 mean rank (1 = tốt nhất): Overall Ours 1.22 / Wan-FC 1.89 / Wan-VACE 2.46 / SignViP 3.56 / SignGAN 3.41; NVSign 1.16 / 1.95 / 2.41 / 3.81 / 3.23; Phoenix14T 1.41 / 1.89 / 2.60 / 2.65 / 3.95; Text Ref. 1.16 / 1.63 / 2.44 / 3.76 / 3.22 ✅. Wilcoxon + Holm-Bonferroni, p < 0.001 ✅ §5.3. Head-to-head: Ours thắng mọi baseline > 83 % ✅ Hình 4.
- **Comprehension study** (6 người ký BSL thành thạo, 10 clip, 30 phản hồi/mô hình) — Bảng 4: Keyword Recall 0.54 vs Wan-FC 0.30; BLEURT-20 0.31 vs 0.19; LLM adequacy (Claude Opus 4.8, thang 0–4) 2.07 vs 1.50; Failure Rate 0.17 vs 0.27 ✅. Chỉ Keyword Recall và BLEURT có ý nghĩa thống kê (p < 0.05) ✅ §5.3.
- **Ablation** (Bảng 5, NVSign; Face / LH / RH MPJPE↓, LPIPS↓): Dense 1.692, 0.192 / 6.665, 0.319 / 6.713, 0.329; Sparse 1.770, 0.194 / 6.638, 0.321 / 6.602, 0.330; Sparse w/o attention masking 1.949, 0.214 / 7.309, 0.353 / 7.270, 0.363; Sparse w/o regional coords 1.847, 0.202 / 6.926, 0.335 / 6.888, 0.344 ✅. Nhận xét tác giả: dense (MANO + face normals) chỉ hơn ở mặt, **không hơn ở tay**; yếu tố quyết định là **phóng đại crop lên 256²**, không phải mật độ điều kiện; MANO dự đoán có "structured noise" ✅ §5.4.

### 1.5 Compute
- Train: 4× NVIDIA GH200, 30,000 iteration, Adam lr 1e-4, gradient accumulation 8, batch 1/GPU, mixed precision ✅ §5.1. **Số giờ không ghi** ⚠️.
- Inference: "generating an 81-frame clip takes minutes even on high-end GPUs" ✅ §6. Số bước denoise, CFG: **không ghi** ⚠️. s/khung: không ghi ⚠️.

### 1.6 Module tay/mặt
- 3 encoder vùng riêng (mặt / tay T / tay P) ở 256² ✅ §4.1; 2 adapter (mặt, tay) cross-attention tiêm xen kẽ để không block nào nhận 2 residual cùng lúc ✅ §4.2.
- **Masked cross-attention với sink token** (Eq. 2–3): query nền (m_i = 0) chỉ attend vào sink token, query trong bbox attend vào token motion → không "rò" đặc trưng tay ra nền ✅ §4.2.
- **Loss MSE có trọng số vùng** (Eq. 4): L = (w_h N_h L_h + w_f N_f L_f + w_b N_b L_b) / (w_h N_h + w_f N_f + w_b N_b), **w_h = w_f = 10, w_b = 1**; tay ưu tiên hơn mặt khi chồng lấn; thân/cánh tay ngoài bbox tính vào nền; tay + mặt chiếm ~15 % diện tích latent ✅ §4.3.

### 1.7 Hạn chế tác giả thừa nhận (§6) ✅
1. Chậm: vài phút cho 81 khung, không real-time.
2. Tái tạo cả motion blur của dữ liệu 25 fps → nhoè chi tiết tay khi chuyển động nhanh.
3. Phụ thuộc pose estimator (RTMPose, WiLoR): keypoint nhiễu làm hỏng điều kiện và làm nhiễu metric đánh giá.

### 1.8 Code / checkpoint / dataset
- Trang dự án https://cogvis-cvssp.github.io/papers/signrefine/: liên kết thực tế chỉ có arXiv và template Nerfies; **không có URL code, checkpoint hay NVSign** ✅ (đã trích xuất toàn bộ `href` của trang ngày 2026-11-02). arXiv abs không có link code ✅. Danh sách VIPL-SLP/Awesome-Sign-Language-Processing ghi "[paper pending]", không có code ✅.
- Kết luận: **code: not found; checkpoint: not found; NVSign: not found** (tính đến 2026-11-02).
- Backbone Wan2.1-Fun-1.3B-Control (V1.0 và V1.1) **công khai** trên HF (`alibaba-pai/Wan2.1-Fun-1.3B-Control`, `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control`), Apache-2.0, hỗ trợ control Canny/Depth/Pose/MLSD + ảnh tham chiếu (V1.1) ✅ (HF).

---

## Bài 2 — SignViP: Advanced Sign Language Video Generation with Compressed and Quantized Multi-Condition Tokenization

Tác giả: Wang, Deng, Jiang, Yin, Shen, Cheng, Ge, Gan, Gu (Nanjing Univ., NUS). NeurIPS 2025 ✅.

### 2.1 Bài toán và I/O
- SLVG: text ngôn ngữ nói T + ảnh tham chiếu x* → video F khung, p_θ(X | x*, T) ✅ §3.2, Eq. 3–4.
- Điều kiện vào bộ sinh video: ảnh tham chiếu (qua ReferenceNet + CLIP) + **embedding đa điều kiện** e_{0:F−1} từ **fine-grained poses** (thân + biểu cảm mặt) và **3D hands** ("detailed and accurate descriptions of hand movements, even in the presence of occlusions") ✅ §3.2. Encoder đa điều kiện: 8 lớp conv SiLU, mỗi lớp giảm 2× H và W ✅ Phụ lục B.
- **Bộ phát hiện pose/3D hand lúc train: paper không nêu tên** ⚠️. Repo có thư mục `scripts/hamer/` (HaMeR, 3D hand) và `scripts/sk/` ("SK (DWPose)") ✅ (README GitHub) → ⚠️ suy ra 3D hands = render HaMeR/MANO, fine-grained pose = DWPose.
- Đánh giá pose back-translation: OpenPose + ánh xạ 2D→3D [87] ✅ §4.2.
- Text encoder: CLIP đa ngôn ngữ `sentence-transformers/clip-ViT-B-32-multilingual-v1` → GPT-2 ✅ §3.6, §4.1.
- Không dùng SMPL-X toàn thân, không có mesh thân ✅.

### 2.2 Backbone, train/đóng băng, tham số
- Denoising U-Net: kiến trúc **SD v1.5** + temporal attention khởi tạo từ **AnimateDiff**; ReferenceNet = SD v1.5; Condition Guider nhẹ (conv + temporal attn), cộng vào các block downsampling ✅ §3.4, §4.1.
- Ba giai đoạn train (§3.3): I) diffusion + multi-condition encoder, loss denoise Eq. 5; II) FSQ Autoencoder, loss L2 Eq. 6; III) Token Translator (GPT-2), cross-entropy Eq. 7 + scheduled sampling 40 % ✅ §3.5–3.6.
- Paper **không nói** có đóng băng phần nào của U-Net/ReferenceNet ở giai đoạn I ⚠️ (chỉ ghi "initialized with SD v1.5"); so với AnimateAnyone, ⚠️ nhiều khả năng train toàn bộ.
- FSQ: 4 kênh latent, mỗi kênh 5 mức → **vocab 625** ✅ §4.1; tỷ lệ nén 8 ✅ Bảng 10.
- **Tham số: 2777.92M** (so với SignGAN+AnimateAnyone 2575.85M, SignGen 2217.38M) ✅ Bảng 8.
- Condition augmentation: thay khung điều kiện bằng khung clip khác với xác suất p = 1e-3 ✅ §3.4, §4.1.

### 2.3 Dữ liệu
- **RWTH-2014T (PHOENIX-2014T)**: 8,257 video; 7,096 / 519 / 642; khung **260×210 → 272×224** cho chia hết 8 ✅ §4.1.
- **How2Sign**: 2,456 video, cắt theo timestamp thành **31,128 / 1,741 / 2,322** câu; khung **512×512** ✅ §4.1.
- fps: không ghi ⚠️. Độ dài: "sign language videos often exceed 100 frames" ✅ §3.6.
- SignAvatars, HamNoSys, CSL-Daily: **không dùng, không nhắc** ✅.

### 2.4 Metric và số liệu
Metric ✅ §4.1: back-translation (BLEU-1..4, ROUGE-L, COMET; huấn luyện SLT [6] riêng, Phụ lục C), FID, CLIP-FID, FVD, IDS (YOLO5Face + Arc2Face), PSNR, SSIM, LPIPS, **Hand SSIM** (SSIM trên vùng tay). Không có MPJPE/AKD ✅.

**Bảng 1 — video back-translation** (BLEU-4 / ROUGE / COMET):
| | RWTH-2014T | How2Sign |
|---|---|---|
| Ground-Truth | 11.90 / 34.27 / 0.6157 | 7.53 / 21.43 / 0.5882 |
| MoMP + ControlNet | 3.61 / 21.54 / 0.5033 | 2.31 / 13.72 / 0.5122 |
| MoMP + AnimateAnyone | 3.72 / 21.68 / 0.5091 | 2.25 / 14.15 / 0.5208 |
| SignGAN | 3.16 / 19.64 / 0.4977 | 1.97 / 11.76 / 0.5104 |
| SignGAN w/ AnimateAnyone | 3.23 / 19.70 / 0.4928 | 1.92 / 12.12 / 0.5135 |
| SignGen | 0.51 / 16.13 / 0.4086 | 0.41 / 9.54 / 0.4127 |
| **SignViP** | **8.65 / 28.85 / 0.5608** | **5.04 / 16.99 / 0.5524** |

**Bảng 2 — pose back-translation** (BLEU-4 / ROUGE / COMET): GT 9.87 / 31.02 / 0.5978 (RWTH), 7.91 / 24.88 / 0.6250 (H2S); ProTran 4.07 / 20.97 / 0.5091 ; 3.42 / 17.32 / 0.5549; Adversarial 4.18 / 21.15 / 0.5127 ; 3.48 / 17.84 / 0.5618; MDN 4.52 / 21.44 / 0.5251 ; 3.67 / 18.21 / 0.5685; **MoMP 5.14 / 23.75 / 0.5466 ; 4.16 / 19.38 / 0.5802**; SignGAN 2.85 / 14.79 / 0.5123 ; 2.62 / 13.19 / 0.5673; SignGen 0.38 / 12.68 / 0.4324 ; 1.36 / 8.69 / 0.4413; SignViP 4.61 / 22.67 / 0.5347 ; 4.42 / 18.23 / 0.5738 ✅ (MoMP hơn SignViP ở vài metric, tác giả thừa nhận §4.2).

**Bảng 3 — chất lượng video** (FID↓ / CLIP-FID↓ / FVD↓ / IDS↑):
| | RWTH-2014T | How2Sign |
|---|---|---|
| SignGAN | 547.90 / 167.70 / 1431.38 / 0.463 | 667.44 / 210.11 / 2766.97 / 0.538 |
| SignGAN w/ AnimateAnyone | 595.99 / 161.97 / 1330.54 / 0.462 | 679.41 / 215.05 / 2484.39 / 0.533 |
| SignGen | 644.06 / 184.66 / 1715.32 / 0.515 | 815.69 / 186.32 / 3538.49 / 0.539 |
| **SignViP** | **508.91 / 154.10 / 1025.45 / 0.571** | **575.67 / 109.61 / 2207.67 / 0.624** |

**Bảng 4 — chỉ bộ sinh video, điều kiện GT** (FVD↓ / SSIM↑ / PSNR↑ / LPIPS↓ / Hand SSIM↑) — **đây là bảng so sánh được với GReg nhất** (cùng setting pose-to-video):
| | RWTH-2014T | How2Sign |
|---|---|---|
| ControlNet | 556.63 / 0.784 / 19.50 / 0.137 / 0.483 | 427.22 / 0.826 / 21.32 / 0.116 / 0.657 |
| AnimateAnyone | 365.42 / 0.794 / 20.06 / 0.121 / 0.505 | 293.18 / 0.821 / 21.54 / 0.103 / 0.663 |
| **Sign Video Diffusion Model (SignViP)** | **275.22 / 0.829 / 22.91 / 0.089 / 0.614** | **210.63 / 0.855 / 23.11 / 0.074 / 0.752** |

- **Bảng 5 — ablation điều kiện** (RWTH, FVD / Hand SSIM / BLEU-4 / ROUGE-L): w/o 3D Hands 382.64 / 0.477 / 5.39 / 23.98; w/o Fine-Grained Poses 461.98 / 0.488 / 3.13 / 19.41; cả hai 275.22 / 0.614 / 8.65 / 28.85 ✅.
- Bảng 6 (Phụ lục D, 10 người không chuyên, chọn video giống GT nhất): SignViP 70.0 % (RWTH), 81.8 % (H2S) ✅. Bảng 9: không khởi tạo pretrained → FID 2278.23, FVD 3277.20 ✅. Hình 3(c): bỏ FSQ → FVD 1932.26 ✅. Hình 4(d): FSQ dùng > 97 % codebook ✅.

### 2.5 Compute
- Train: **4× NVIDIA RTX A6000**, Adam, mỗi giai đoạn 50,000 bước; batch 2 / 16 / 16; lr 1e-5 / 5e-5 / 1e-6 ✅ §4.1. Back-translation model: 1× A6000 ✅ Phụ lục C. **Số giờ không ghi** ⚠️ (checklist nói "claim the compute resources in Appendix" nhưng phụ lục chỉ ghi GPU).
- Inference: DDIM, **50 bước**, CFG 3.5 ✅ §4.1; **1.2398 s/khung** ✅ Bảng 8 (GPU đo không ghi ⚠️).

### 2.6 Module tay/mặt
- Không có adapter vùng hay loss vùng; tay được xử lý bằng **điều kiện 3D hands** riêng và metric **Hand SSIM** ✅. Không có trọng số loss kiểu ×10 ✅.
- Bằng chứng tay: bỏ 3D hands làm Hand SSIM 0.614 → 0.477 và FVD 275 → 383 ✅ Bảng 5.

### 2.7 Hạn chế (Phụ lục A) ✅
1. Mỗi ngôn ngữ ký hiệu phải train một mô hình riêng.
2. Denoise lặp nhiều bước chậm; đề xuất consistency model / DPM-Solver.
- ⚠️ Không nhắc: phụ thuộc detector pose/3D hand; chỉ 2 bộ dữ liệu studio; SignRefine (§5.2) cho thấy checkpoint Phoenix **sụp đổ ngoài miền** (hand MPJPE gấp ~2).

### 2.8 Code / checkpoint
- Code: **https://github.com/umnooob/signvip** ✅ (ghi trong abstract và README). Có script cả 3 giai đoạn (`train_stage1_multi_cond.py`, `train_stage2_multi_cond.py`, `train_compress_vq_multicond.py`, `train_multihead_t2vqpgpt.py`), tiền xử lý RWTH-T/How2Sign, sinh video `eval_compress_vq_video.py` ✅ (README).
- Checkpoint: **RWTH-T: https://huggingface.co/umnooob/signvip/tree/main/RWTH** ✅; How2Sign: "comming soon" ✅. Khớp với SignRefine §5.2: "checkpoint trained on Phoenix14T ... the only checkpoint made available".
- License: **không có** file LICENSE ✅ (README). Repo chỉ 8 commit ⚠️ (theo kết quả tìm kiếm).

---

## Bài 3 — Signs as Tokens (SOKE): A Retrieval-Enhanced Multilingual Sign Language Generator

Tác giả: Zuo, Potamias, Ververas, Deng, Zafeiriou (Imperial College London). ICCV 2025 ✅.

### 3.1 Bài toán và I/O
- **Text → chuỗi motion SMPL-X 3D** (avatar), **không sinh video RGB** ✅ §3.1, Hình 2. Đầu ra S ∈ R^{T×133}: 11 khớp thân trên, 30 khớp tay, 10 tham số biểu cảm ✅ §3.1.
- Điều kiện: text (mBART encoder) + thẻ ngôn ngữ <ASL>/<CSL>/<DGS> + **token motion của các từ tra được trong từ điển ký hiệu** (retrieval) nối vào prompt ✅ §3.3–3.4, Hình 4.
- Không có ảnh tham chiếu, không keypoint 2D làm điều kiện ✅.
- Nguồn SMPL-X: How2Sign dùng pose của NSA [2]; **CSL-Daily và Phoenix-2014T tự fit**: OSX (thân) + WiLoR (tay) + tinh chỉnh xoay khớp thân trên bằng loss tái chiếu so với keypoint 2D **Mediapipe** ✅ §3.1.

### 3.2 Backbone, train/đóng băng, tham số
- **DETO** (decoupled tokenizer): 3 VQ-VAE (thân trên / tay T / tay P), encoder 1D-CNN, codebook N_B = 96, N_LH = N_RH = 192, dim C = 512; loss rec + emb + commitment ✅ §3.2, §4.
- **AMG**: **mBART-large-cc25** (12 lớp enc-dec, dim 1024) được **fine-tune** với vocab mở rộng (token text + token motion + thẻ ngôn ngữ) ✅ §3.3, §4. Paper không nói có đóng băng lớp nào ⚠️ (đọc là full fine-tune).
- Multi-head decoding: 3 head LM (FC) dự đoán 3 token/bước; embedding đầu vào = (1−2λ)E_B + λE_LH + λE_RH, λ = 1/3 ✅ §3.3, §4. Greedy decoding ✅.
- **Số tham số không ghi** ⚠️ (mBART-large ≈ 610M ⚠️ theo kiến thức chung).

### 3.3 Dữ liệu
- How2Sign (ASL, 35K cặp), CSL-Daily (CSL, 20K), Phoenix-2014T (DGS, 8K) gộp thành một bộ đa ngôn ngữ ✅ §4. Từ điển: MS-ASL, WLASL, DW-DGS, … + nguồn online, fit SMPL-X rồi token hoá ✅ §3.4.
- Độ phân giải/fps: không áp dụng (dữ liệu là chuỗi SMPL-X) ✅. Số khung: không ghi ⚠️.
- **SignAvatars** [72]: trích dẫn là bài dùng "costly gloss and HamNoSys notations" để train semantic adapter và là nguồn metric PA-MPJPE ✅ §2, Bảng 3; **không dùng làm dữ liệu**. **HamNoSys**: chỉ nhắc như notation tốn kém ✅ §2.

### 3.4 Metric và số liệu
Metric ✅ §4: DTW-PA-JPE và DTW-JPE (thân / tay), back-translation BLEU-4 (B-T), latency s/video, PA-MPJPE tái tạo của tokenizer, user study.

**Bảng 1** (DTW-PA-JPE Body / Hand ; DTW-JPE Body / Hand ; B-T BLEU-4):
| | How2Sign | CSL-Daily | Phoenix-2014T |
|---|---|---|---|
| NAR | 13.94 / 11.80 ; – ; 5.75 | – | – |
| Prog. Trans.* | 14.15 / 11.57 ; 14.74 / 30.17 ; 2.75 | 15.98 / 12.91 ; 16.30 / 32.63 ; 3.07 | 13.67 / 11.95 ; 15.01 / 31.77 ; 4.94 |
| Text2Mesh* | 13.99 / 13.47 ; 15.50 / 32.97 ; 7.51 | 13.47 / 12.10 ; 13.76 / 30.37 ; 5.11 | 13.48 / 12.06 ; 14.04 / 31.64 ; 5.81 |
| Adv. Train. | 13.78 / 11.17 ; – ; 6.21 | – | – |
| T2S-GPT* | 11.48 / 6.39 ; 12.65 / 18.44 ; 11.20 | 11.94 / 5.93 ; 12.32 / 15.43 ; 8.94 | 10.38 / 6.47 ; 11.65 / 19.09 ; 9.06 |
| NSA | 7.83 / 7.33 ; – ; 13.12 | – | – |
| S-MotionGPT* | 11.23 / 4.39 ; 12.41 / 13.74 ; 11.45 | 10.81 / 3.78 ; 11.58 / 11.31 ; 8.82 | 9.45 / 3.41 ; 10.42 / 9.08 ; 9.68 |
| **SOKE** | **6.82 / 2.35 ; 7.75 / 10.08 ; 14.48** | **6.24 / 1.71 ; 7.38 / 9.68 ; 11.30** | **4.77 / 1.38 ; 6.04 / 7.72 ; 11.87** |
(* = tái hiện bởi tác giả ✅.)

- **Bảng 2** (decoding × retrieval; DTW-PA-JPE Avg / Body / Hand ; latency s/video trên 1× RTX 3090): Sequential 4.64 / 8.55 / 3.53 ; 3.26 — Parallel 5.06 / 9.27 / 3.86 ; 1.39 — Multi-head 4.17 / 7.91 / 3.10 ; 1.46 — **Multi-head + retrieval 3.34 / 6.82 / 2.35 ; 1.55** (How2Sign) ✅. Retrieval giảm DTW trung bình 19.9 / 19.6 / 23.9 % ✅ §4.2.
- **Bảng 3** (scalability): train đủ 3 bộ → PA-MPJPE tái tạo tokenizer H2S 19.37 / 6.65 (body / hand), CSL 23.52 / 5.13, Ph-T 25.79 / 6.78 ✅.
- **Bảng 4** (tokenizer): whole-body 1 VQ-VAE → H2S hand DTW 3.56; body + hands → 3.13; body + L-hand + R-hand → **2.35** ✅.
- User study (5 người ký ASL, 4 CSL, 15 mẫu, thang 1–10): SOKE **5.32 / 6.45** vs S-MotionGPT 1.36 / 2.70 (H2S / CSL) ✅ §4.3, Hình 7. GT không ghi số trong text ⚠️ (chỉ trên Hình 7).
- Không có metric video (FVD/LPIPS/SSIM) ✅; không có GReg/PGMM ✅.

### 3.5 Compute
- **6× RTX 3090** ✅ §4. DETO: batch 320/GPU, 500 epoch, AdamW lr 2e-4 cosine; LM: batch 32/GPU, 150 epoch ✅ §4. **Số giờ không ghi** ⚠️.
- Inference: 1.55 s/video (multi-head + retrieval), 3.26 s sequential, 1.39 s parallel; đo trên 1× RTX 3090 ✅ Bảng 2. Không có "s/khung" (không sinh ảnh) ✅.

### 3.6 Module tay/mặt
- **Tokenizer tách rời** body / LH / RH với codebook tay lớn gấp đôi thân (192 vs 96) ✅ §4. Multi-head decoding giữ fusion giữa thân và tay ✅ §3.3. Mặt chỉ có 10 tham số biểu cảm trong vector body ⚠️ (không có codebook mặt riêng).
- Không có loss có trọng số vùng ✅.

### 3.7 Hạn chế
- Bản CVF **không có mục Limitations** ✅ (đã đọc toàn bộ 11 trang; supplementary không nằm trong PDF ⚠️).
- ⚠️ Hạn chế suy ra từ nội dung: phụ thuộc pipeline fit SMPL-X (OSX + WiLoR + Mediapipe) — chất lượng GT 3D không được kiểm chứng trong thân bài (hứa ở supplementary); cần từ điển ký hiệu ngoài; greedy decoding; không sinh video nên không đánh giá được ngoại hình/ảnh.

### 3.8 Code / checkpoint / dữ liệu
- Code: **https://github.com/2000ZRL/SOKE** ✅ (train, test, vis; dựa trên MotionGPT, ProgressiveTransformer, WiLoR, OSX).
- Checkpoint: tokenizer DETO trên Google Drive (`drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk`), mBART-large-cc25 + deps trên Google Drive (`drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi`) ✅ (README). **Checkpoint generator SOKE: không tìm thấy** ✅ (README chỉ có lệnh train/test; trang dự án ghi "Code, models, and data will be made publicly available").
- SMPL-X poses: trang dự án https://2000zrl.github.io/soke/ có link Data How2Sign (Dropbox), CSL-Daily (Dropbox), Phoenix (Google Drive) ✅; không ghi điều kiện truy cập ✅. License: có `license.txt` trong repo nhưng nội dung chưa đọc ⚠️.

---

## Ý nghĩa cho GReg

Bối cảnh GReg (theo `README.md`, `docs/decisions.md`): pose-to-video trên **SignAvatars HamNoSys**, 256², điều kiện = **normal map SMPL-X** render từ `.pkl` (shape tối ưu, camera từng khung) + ảnh nguồn (style) + flow TPSMM (nay thay bằng mesh flow A1) + prior gradient vùng (RE); metric = L1 / PSNR / SSIM / LPIPS ảnh, region-L1 (box 64×64 cố định), AKD (HRNet-W32 COCO-WholeBody), TCD, FVD (I3D), hand_l1 / face_l1 / arm_l1.

### A. Tín hiệu điều kiện của họ mà GReg KHÔNG dùng
| Tín hiệu | Ai dùng | GReg | Nhận xét |
|---|---|---|---|
| Keypoint 2D toàn thân (RTMPose-X 133 kp / DWPose) làm điều kiện | SignRefine ✅, SignViP ✅ | **Không** (HRNet chỉ cho AKD) ✅ decisions §2 | GReg thay bằng normal map SMPL-X; khác hẳn nhau về "mật độ" điều kiện |
| Crop vùng tay/mặt **phóng đại lên 256²** đưa qua encoder riêng + CoordConv | SignRefine ✅ §4.1 | Không: RE nhận crop 64×64 tại 256² (tay chỉ ~15 % box) ✅ memory | Đây là yếu tố SignRefine coi là quyết định (§5.4), đáng thử trong B1 |
| Ảnh tham chiếu qua CLIP image encoder / ReferenceNet | SignRefine ✅, SignViP ✅ | Không: style encoder CNN 5 lớp tự train ✅ decisions §6 | |
| Backbone video diffusion pretrain (Wan 1.3B / SD1.5 + AnimateDiff) | SignRefine ✅, SignViP ✅ | Không: GAN + warping train từ đầu | Khoảng cách chất lượng lớn; là hướng `docs/diffusion_improvement.md` |
| Render **3D hand (MANO/HaMeR)** làm điều kiện riêng | SignViP ✅; SignRefine chỉ ablation ✅ Bảng 5 | Có gián tiếp: normal map SMPL-X đã gồm tay ⚠️ | GReg "giàu" hơn ở tay nhưng không tách kênh |
| Text / token rời rạc / retrieval từ điển | SignViP ✅, SOKE ✅ | Không (GReg là giai đoạn 2) ✅ | Không cần cho luận văn |
| Loss có **trọng số vùng ×10** | SignRefine ✅ Eq. 4 | Có λ_struct = 10 cho prior gradient, nhưng không có trọng số vùng cho loss tái tạo chính ✅ decisions §4 | ⚠️ Dễ thêm: nhân mask tay/mặt ×10 vào L_per/L1 — tiền lệ rõ ràng để trích dẫn |
| Masked cross-attention + sink token | SignRefine ✅ Eq. 2–3 | Không (GRM attention toàn ảnh) | ⚠️ Ý tưởng cho GRM: giới hạn key/value theo bbox |

### B. Metric phải thêm để so sánh được
1. **Regional MPJPE / LPIPS / SSIM trên crop tay-trái, tay-phải, mặt từ bbox GT** (SignRefine Bảng 2). GReg có AKD (HRNet, chuẩn hoá theo kích thước ảnh, gộp mặt + tay) và region-L1; cần **tách trái/phải/mặt** và thêm **LPIPS + SSIM theo crop**. Detector: SignRefine dùng RTMPose-X (133 kp) — ⚠️ để so sánh số tuyệt đối phải cùng detector và cùng đơn vị, nhưng vì khác dataset nên chỉ so **xu hướng tương đối** (ví dụ % cải thiện so với baseline).
2. **Hand SSIM** (SignViP Bảng 4): SSIM trên vùng tay — gần như có sẵn từ region-L1; thêm một dòng trong `greg/eval/metrics.py` ⚠️.
3. **FVD + SSIM + PSNR + LPIPS** toàn ảnh (SignViP Bảng 4): GReg đã có ✅ README; FVD cần I3D (đã có script 12) ✅.
4. **FID / CLIP-FID / IDS** (SignViP Bảng 3): chưa có; IDS cần YOLO5Face + Arc2Face ⚠️ (nặng, có thể bỏ qua vì GReg tái tạo cùng người).
5. **Back-translation BLEU/ROUGE/COMET**: không khả thi trên SignAvatars HamNoSys (không có SLT model và câu tiếng nói tương ứng) ⚠️; nếu cần, chỉ làm được nếu chuyển sang Phoenix14T.
6. **User study** (mean rank + Wilcoxon-Holm như SignRefine §5.3; hoặc "chọn video giống GT nhất" như SignViP Phụ lục D — không cần người ký chuyên nghiệp). Nên dùng protocol SignViP vì rẻ ⚠️.
7. **Chi phí**: s/khung và tham số (SignViP Bảng 8); GReg đã log `sec_per_step`/`vram_gb` ✅ README — thêm s/khung inference.

### C. Có thể so sánh "cùng backbone, keypoint-2D vs mesh" với SignRefine không?
- **Chạy lại SignRefine: không thể** — code, checkpoint và NVSign đều chưa công bố (mục 1.8) ✅.
- **Backbone Wan2.1-Fun-1.3B-Control công khai** (Apache-2.0, HF) ✅, và chính là baseline mạnh nhất của họ (Wan-FC, chỉ thua SignRefine ở tay ~30 %). Vậy có thể làm phiên bản "gần đúng":
  - (i) Wan-FC zero-shot với control = render keypoint 2D (DWPose/RTMPose) — đúng như baseline "Wan-FC" của SignRefine ✅ §5.2.
  - (ii) Cùng Wan-FC với control = **normal map SMPL-X** của GReg. ⚠️ Fun-Control chỉ được train với Canny/Depth/Pose/MLSD; normal map không phải loại control chính thức → zero-shot sẽ không công bằng, phải fine-tune (LoRA / adapter) trên SignAvatars. Khi đó so sánh (i)-fine-tune-keypoint vs (ii)-fine-tune-normal mới là "cùng backbone, chỉ khác điều kiện".
  - (iii) Tự cài lại adapter của SignRefine (mô tả khá đầy đủ ở §4.1–4.3, §5.1, trừ mâu thuẫn 5 vs 6 block) với đầu vào crop normal map thay vì crop keypoint. Chi phí tham khảo: SignRefine cần 4× GH200 × 30k iter (batch hiệu dụng 32) ⚠️ — vượt ngân sách vast.ai hiện tại của GReg; cần giảm (ít iter, 480p → 256², 1–2 GPU 80 GB).
- **Lưu ý khi lập luận cho mesh**: SignRefine Bảng 5 đã thử dense (MANO + face normals) và kết luận **không hơn ở tay**, chỉ hơn nhẹ ở mặt, với lý do (a) crop 256² đã đủ và (b) MANO dự đoán từ WiLoR có nhiễu ✅ §5.4. GReg dùng SMPL-X **tối ưu sẵn của SignAvatars** (ít nhiễu hơn) — đây là điểm khác biệt cần nêu rõ nếu luận văn muốn bảo vệ mesh-conditioning ⚠️. Thí nghiệm (i) vs (ii) trên cùng backbone sẽ là bằng chứng trực tiếp.
- **So với SignViP**: checkpoint RWTH-T công khai nên có thể chạy Stage-I generator với điều kiện GT trên Phoenix14T (như SignRefine đã làm) — nhưng chỉ có ý nghĩa nếu GReg cũng train trên Phoenix14T ⚠️; trên SignAvatars thì SignViP sẽ ngoài miền (SignRefine cho thấy sụp đổ khi đổi miền).
- **So với SOKE**: không so sánh được (text → SMPL-X, không có ảnh); chỉ hữu ích như nguồn SMPL-X fit cho Phoenix/CSL-Daily nếu GReg mở rộng dataset ⚠️.

### D. Việc nên làm ngay (ngắn)
1. Thêm metric: hand-L/R/face LPIPS + SSIM theo crop bbox GT; Hand SSIM; s/khung inference.
2. Thêm trọng số vùng ×10 (tay, mặt) vào loss tái tạo — trích SignRefine Eq. 4 làm tiền lệ.
3. Thử ablation "phóng đại crop tay lên 256²" cho RE/B1 — trích SignRefine §5.4.
4. Nếu đi hướng diffusion: lấy Wan2.1-Fun-1.3B-Control làm backbone để có thể đối chiếu trực tiếp với Wan-FC trong Bảng 2 SignRefine.
