# Học gì từ các hướng diffusion + mesh 3D: những điểm hay nhất áp dụng cho GReg

Ngày lập: 2026-10-29.

Nguồn:
- `docs/tong_hop_bai_bao_diffusion_mesh.md`: note đọc kỹ 8 bài (PartDiffuser, CraftsMan3D, Field Latents, DWM, Generative Rendering, Animating the Uncaptured, MeshToken/Uni3C, Mesh4D) và code của các bài có code. PDF trong `ref papers/`.
- `docs/diffusion_improvement.md`: hai hướng H1 (warp-then-diffuse, video DiT) và H2 (refiner diffusion một bước). File này **bổ sung và sửa** kế hoạch đó, không thay thế.
- `docs/A1_B1_kien_truc.md`: A1 (flow + visibility từ mesh SMPL-X) và B1 (vùng tay/mặt từ mesh).

Quy ước:
- ✅ = đã kiểm chứng (paper/code, có trích dẫn trong file tổng hợp).
- ⚠️ = suy luận, thiết kế đề xuất, chưa đo. **Phải đo trước khi viết vào luận văn.**
- 🛠 = phần tự chỉnh sửa/đề xuất mới, không có nguyên bản trong bài nào.

---

## 0. Bốn kết luận lớn (thay đổi so với `diffusion_improvement.md`)

### K1. Backbone H1: chuyển ưu tiên sang **Wan2.1-Fun-1.3B-InP + concat kênh** (công thức DWM)

- `diffusion_improvement.md` §1.4 khuyến nghị thử VACE trước và ghi rằng Fun-InP "không có kênh điều kiện hình học".
- DWM (CVPR 2026) cho thấy hai điều ✅:
  1. Thêm kênh hình học chỉ cần mở rộng `patch_embedding` với cột trọng số khởi tạo 0, train full lớp đó, phần còn lại dùng LoRA r64 (`training/wan/models/wan_transformer3d_with_conditions.py:135-178`, Apache-2.0, khoảng 270 dòng).
  2. Khe **masked video + mask** có sẵn của mô hình inpainting là chỗ đặt điều kiện "đã biết". Ở bước 0, mô hình gần như chép lại nó (hàm đồng nhất có prior), nên LoRA chỉ phải học phần dư.
- Với GReg, khe đó **khớp đúng nghĩa**: masked video = ảnh nguồn đã warp bằng mesh `I_st`, mask = `V_geo·C` (1 = giữ, 0 = sinh). DWM chỉ dùng được mask toàn 1, còn GReg có mask nhìn thấy thật nhờ A1. Đây là điểm mới so với DWM.
- → VACE trở thành **phương án so sánh**, không còn là lựa chọn đầu. ⚠️

### K2. Điều kiện **căn pixel** là xương sống; thêm **bản đồ tọa độ chính tắc** làm kênh thứ hai

- MeshToken lập luận nên bỏ render và dùng token mesh, nhưng không có thí nghiệm đối chứng cùng điều kiện huấn luyện. Phương pháp render-based có tín hiệu căn pixel đạt PSNR 22.57 so với 16.78 ✅. DWM Bảng 3: render mesh tay (PSNR 24.15) tốt hơn mask nhị phân (22.88) và tham số MANO qua AdaLN (21.96–22.79) ✅. Với bài toán **tái tạo, camera cố định** của GReg, căn pixel là đúng.
- Hai bài độc lập dùng cùng một thứ: Mesh4D (positional embedding từ vị trí chính tắc 3D, dùng chung qua các khung) và RealisDance/Uni3C (render SMPL-X tô màu theo tọa độ chính tắc, `cs_map`). Ở GReg đó là **bản đồ tọa độ T-pose cho từng pixel đích (CCM)**. 🛠 A1 đã có face map + barycentric của khung đích, nên tính CCM gần như miễn phí:
  `CCM_t(p) = Σ_k b_k(p) · v_Tpose[f_t(p)_k]` (dùng `greg/render/rasterize.py:139 interpolate_vertex_attribute`).
- CCM trả lời phản biện GenWarp (`diffusion_improvement.md` §1.3) **tốt hơn** bản đồ (u, v) ảnh nguồn ⚠️, vì:
  1. Có giá trị cả ở vùng **không thấy trong nguồn** (lòng bàn tay vẫn có tọa độ T-pose), còn (u, v) nguồn thì không.
  2. Cùng một điểm cơ thể giữ cùng giá trị qua mọi khung. Temporal attention của DiT nhờ vậy "biết" token nào là cùng một điểm vật lý; đây chính là lý do Mesh4D làm temporal attention hiệu quả.

### K3. Nhất quán thời gian cho H2 lấy **trực tiếp từ tương ứng của A1**

- Generative Rendering: tương ứng mesh tiêm vào **trước** (K/V mở rộng) và **sau** self-attention (chiếu đặc trưng) cho nhất quán dài hạn tốt hơn cả mô hình T2V ✅ (phụ lục B, Bảng 2: ở khoảng cách 20 khung, 0.9727 so với 0.9601 của Gen-1).
- `correspond()` của A1 đã cho tương ứng giữa hai khung bất kỳ (I_s → t, t−1 → t) kèm `V`, `C`, nên **không cần atlas UV** như bài gốc. 🛠
- Lưu ý ✅ (Fig. 6): chép đặc trưng thẳng qua tương ứng mà không qua attention thì ảnh hỏng. Đây cũng là lập luận cho việc đưa ảnh warped vào làm **điều kiện** (H1) thay vì dán đè.

### K4. Diffusion "làm nét" làm **trôi tần số thấp**: không blend cứng ở bước cuối

- CraftsMan3D chỉ ra rằng tinh chỉnh bằng diffusion âm thầm đổi tần số thấp (Fig. 7/10) ✅ và phải sửa bằng Poisson blending.
- Với ảnh người, trôi tần số thấp nghĩa là **trôi màu áo, tông da**, lỗi điển hình của refiner 2D.
- → Thay `Î = V·I_st + (1−V)·Î_diff` ở cuối H1/H2 bằng **blend miền gradient** hoặc **tách tần số** (tần số thấp từ nguồn tin cậy, tần số cao từ diffusion). Thêm một chỉ số "trôi tần số thấp". 🛠

---

## 1. Bảng tổng: kỹ thuật → nguồn → áp dụng vào đâu

Ưu tiên: **P0** = làm ngay/rẻ; **P1** = mặc định khi dựng H1/H2; **P2** = ablation chính; **P3** = nếu còn thời gian.

| # | Kỹ thuật | Nguồn | Áp dụng cho GReg | Thành phần | Độ khó | Ưu tiên |
|---|---|---|---|---|---|---|
| 1 | Khe masked-video của InP làm điều kiện "đã biết" | DWM | `I_st` vào khe masked video, `V·C` vào mask | H1 | Thấp | **P1** |
| 2 | Mở rộng patch-embed, cột mới = 0, train full lớp đó + LoRA phần còn lại | DWM, Mesh4D, Uni3C | Thêm kênh normal `N_t`, CCM | H1 | Thấp | **P1** |
| 3 | Đóng gói mask theo thời gian 4 kênh/latent frame, `1−mask`, vùng che tô 0 | DWM (code Wan-Fun-InP) | Đưa `V_geo,t` từng khung vào Wan | H1 | Thấp, nhưng **sai là hỏng** | **P1** |
| 4 | Mask ở độ phân giải pixel → Conv3d stride (4,8,8) + pad 3 khung đầu → cộng zero-init | Uni3C | Đưa `M_t` (tay trái/phải/mặt) mà không qua VAE | H1 | Thấp | P1 |
| 5 | Bản đồ tọa độ chính tắc (CCM) | Mesh4D, RealisDance/Uni3C | Kênh điều kiện thứ hai; cũng dùng làm metric thời gian | A1 → H1, H2, eval | Thấp–vừa | **P2** |
| 6 | Khung 0 = ảnh tham chiếu thật | Uni3C, Mesh4D | Khung 0 của video warped = `I_s` (warp đồng nhất), mask = 1 | H1 | Rất thấp | P1 |
| 7 | Dropout điều kiện **độc lập** từng loại + **từng khung** (p≈0.1) | Mesh4D, MeshToken, CraftsMan (làm sai: theo batch) | CFG theo điều kiện hình học; bền với khung fit SMPL-X hỏng | H1 | Thấp | P1 |
| 8 | Loss vùng tính trên `x0_hat` | DWM (v-pred viết dạng x0) | `x0_hat = x_t − σ·v_hat`; trọng số tay/mặt (B1) và vùng `V=0` | H1 | Thấp | P1 |
| 9 | Công thức train: lr 1e-4, AdamW (0.9, 0.95), wd 0.01, cosine, warmup 100, bf16, cache latent, shift 5, timestep phân tầng | DWM | Điểm khởi đầu cho LoRA Wan-1.3B | H1 | Rất thấp | P1 |
| 10 | Thay latent vùng đã biết bằng latent gốc re-noise mỗi bước (RePaint-style) | Field Latents Eq. 14 | Giữ vùng `V≈1` trong latent khi infer | H1 | Thấp | P2 |
| 11 | Blend Poisson / tách tần số + chỉ số trôi tần số thấp | CraftsMan3D | Thay blend cứng ở đầu ra | H1, H2 | Thấp | **P1** |
| 12 | Tiêm attention qua tương ứng mesh (K/V = [f_t, f_Is, f_t−1], post-attn theo `V·C`) | Generative Rendering | Nhất quán thời gian cho refiner từng khung | H2 | Thấp–vừa | P2 |
| 13 | Dùng cùng phép chiếu làm **loss thời gian** khi train | Generative Rendering + `diffusion_improvement.md` §2.6 | Phạt chênh lệch `Î_t` với `warp(Î_t−1)` ở vùng nhìn thấy | H2 | Thấp | P1 |
| 14 | Temporal attention zero-init chèn vào mạng từng khung | Mesh4D (bỏ temporal attn: ℓ2-Corr 0.0117 → 0.0160 ✅) | Prior thời gian rẻ cho refiner H2 | H2 | Vừa | P3 |
| 15 | Baseline training-free: ControlNet-Tile (+ normal) img2img strength 0.3–0.5, giới hạn trong mask tay và `V=0` | CraftsMan3D (refiner), Generative Rendering | Có số liệu H2 trước khi train | H2 | Thấp | **P0** |
| 16 | K/V global + part cho adapter vùng | PartDiffuser (17.8 so với 29.1 / 54.7 ✅) | Adapter tay: K/V = [token toàn cục, token crop tay] | H1 | Vừa | P2 |
| 17 | Mask attention theo phần cơ thể + sink token | Mesh4D (skinning), PartDiffuser (padding mask), SignRefine | Query thuộc tay mới nhìn crop tay; crop vắng thì không attend | H1 | Vừa | P2 |
| 18 | Token mesh tay 3D (theo phần, hệ tọa độ cổ tay) qua cross-attention từng latent frame | MeshToken | Bù thông tin ngón tay bị che mà render không mang | H1 | Vừa–cao | P3 |
| 19 | Hạ `C` ở dải biên silhouette | Uni3C (Sobel trên disparity loại "flying points") | Giảm vệt warp ở viền tay/thân | A1 | Thấp | P0 |
| 20 | Phát hiện xuyên thấu tay–thân bằng winding number | CraftsMan3D | z-buffer sai khi tay chạm mặt/ngực → hạ `C` | A1 | Vừa | P2 (sau chẩn đoán) |
| 21 | Làm mượt SMPL-X theo thời gian | Animating the Uncaptured (gia tốc 2.65 → 1.50 ✅) | Tiền xử lý trước A1, giảm nhấp nháy flow | A1, dữ liệu | Thấp | P2 |
| 22 | Neo đỉnh vào mặt SMPL gần nhất + offset theo pháp tuyến | Animating the Uncaptured Eq. 4 | Mở rộng flow ra ngoài silhouette (tay áo, tóc) | A1 (A3) | Vừa–cao | P3 |
| 23 | Batch có loss ngoại lai: nhân 1e-8 + ghi tên clip | Mesh4D | Tìm clip SMPL-X fit hỏng | Dữ liệu (mọi giai đoạn) | Rất thấp | **P0** |
| 24 | Metric keypoint tay không vòng tròn | Animating the Uncaptured Eq. 7 | Khớp SMPL-X đích chiếu 2D so với RTMPose trên ảnh sinh (Geman-McClure × độ tin cậy) | Eval | Thấp | **P0** |
| 25 | Nhiễu khởi tạo gắn bề mặt + latent `I_s` warp ở vùng thấy | Generative Rendering (UV noise), Go-with-the-Flow | Nhất quán thời gian cho H1 | H1 | Vừa, rủi ro | P3 |
| 26 | Bộ nhớ texture trên bề mặt SMPL-X | Field Latents, Mesh4D (tô texture một lần) | Chiếu pixel đã thấy lên đỉnh/UV, render làm kênh điều kiện | H1 | Vừa–cao | P3 |

---

## 2. H1 sửa đổi ("H1′"): Wan2.1-Fun-1.3B-InP + concat, công thức DWM + A1

### 2.1 Sơ đồ ⚠️🛠

```
Mỗi khung đích t (khung 0 = chính I_s, warp đồng nhất, mask = 1):

 I_st = warp(I_s, F_geo,t), vùng V=0 tô 0 ──VAE──► c_warp (16)  ─┐  [khe masked-video gốc của InP]
 V_t·C_t (1 kênh/khung) ── đóng gói 4 khung → 4 kênh, 1−mask ──► m (4) ─┤  [khe mask gốc của InP]
 x_t (latent nhiễu) ──────────────────────────────────────────► (16) ─┤
 N_t normal map SMPL-X (z-test toàn thân) ──VAE──► c_N (16) ─────────┤  [kênh MỚI, cột patch-embed = 0]
 CCM_t tọa độ T-pose (RGB) ──VAE──► c_CCM (16) ─────────────────────┘  [kênh MỚI, cột = 0] (ablation)
                                         │ patch_embedding (Conv3d, train full)
 M_t tay T/P/mặt (pixel) ─Conv3d(4,8,8)+pad 3 khung─► zero-proj ─(+)─►  token
                                         ▼
                  Wan2.1-Fun-V1.1-1.3B-InP DiT (đóng băng) + LoRA r16–32 (mọi Linear)
                  + CLIP image(I_s), prompt CỐ ĐỊNH
                                         ▼
                  x0_hat ──► VAE dec ──► Î_diff ──► blend tách tần số / Poisson (K4) ──► Î
```

Kênh vào patch-embed: 16 + 4 + 16 + 16 (+16) = 52 (68 nếu có CCM). DWM bản Wan dùng đúng 52 kênh (thay normal bằng render tay) ✅.

### 2.2 Chi tiết lấy từ code (bắt buộc đúng)

1. **Quy ước mask Wan-Fun-InP** (DWM `pipeline_wan_fun_inpaint_hand_concat.py:357-366`) ✅:
   - Khung đầu lặp 4 lần.
   - Reshape thành `(B, F/4, 4, H, W)` rồi transpose → 4 kênh mask cho mỗi latent frame.
   - Dùng `resize_mask(1 − mask)`. Trong pipeline này "mask" nghĩa là **vùng cần sinh**, nên phải truyền `1 − V·C`; lật dấu sai thì prior inpainting làm ngược.
   - Vùng che trong masked video: Wan-Fun nhân 0 (`init_video * (mask < 0.5)`), còn CogVideoX-Fun tô −1. Phải dùng đúng quy ước của Wan.
2. **Mở rộng patch-embed** (`wan_transformer3d_with_conditions.py:161-167`) ✅: chép trọng số cũ, cột mới = 0, giữ bias. Lưu riêng `non_lora_weights.safetensors` cho patch-embed.
3. **Loss** (DWM Wan `train_dwm_wan.py:3359-3371`, `:376-387`) ✅:
   - `x_t = (1−σ)·x0 + σ·ε`, target `ε − x0`, MSE.
   - Bỏ các phần tử có |diff| > 50.
   - Timestep lấy đều theo chỉ số lịch FlowMatchEuler shift 5, phân tầng theo rank (`utils/discrete_sampler.py:39-50`).
4. **Loss vùng** 🛠: `x0_hat = x_t − σ·v_hat`; `L = mean(w ⊙ (v − v_hat)²)`.
   - `w = 1 + λ_h·M_hand↓ + λ_f·M_face↓ + λ_o·(1 − V)↓`, với `↓` là hạ mẫu về lưới latent (max-pool để không mất tay).
   - Khởi đầu: λ_h = λ_f = 9 (tương đương ×10 của SignRefine, cần xác nhận lại trong bài gốc), λ_o = 1 ⚠️.
5. **Cache latent**: DWM bản CogVideoX cache **moments** (mean, logvar) rồi lấy mẫu lại mỗi bước, được augmentation miễn phí; bản Wan cache một mẫu cố định ✅. Nên làm theo bản CogVideoX ⚠️.
6. **Prompt cố định**, ví dụ "a person signing, front view, fixed camera, uniform lighting" (theo lý do ở phụ lục A.2 của Animating the Uncaptured). **Không** sinh caption từ video GT như DWM (rò thông tin) ✅.

### 2.3 Dropout và CFG theo hình học 🛠

- Dropout độc lập p = 0.1 cho `c_N`, `c_CCM`, `M_t`, cộng dropout điều kiện **theo từng khung** p = 0.05 (Mesh4D). Giữ `c_warp`, mask và `I_s` ở mọi nhánh.
- Infer: `v = v(∅_geo) + w·(v(N, CCM) − v(∅_geo))`, w ≈ 1.5–3. Mục đích là tay bám normal map chặt hơn. DWM chỉ dùng CFG theo text, vô nghĩa với GReg vì prompt cố định. Nếu w = 1 thì tắt CFG, tiết kiệm một nửa thời gian.

### 2.4 Infer: giữ vùng nhìn thấy, sinh vùng che

Ba mức, đưa vào ablation (§6):

| Mức | Cách | Nguồn |
|---|---|---|
| a | Không làm gì thêm (tin LoRA + mask InP) | DWM |
| b | Mỗi bước: `z ← m↓·noise(z_warp, σ_t) + (1−m↓)·z`, chỉ ở vùng `V·C ≈ 1` đã co (erode) 1 latent px | Field Latents Eq. 14 ✅ / RePaint |
| c | Sau decode: blend tách tần số: `Î = LP(Î_diff) + HP(V'·I_st + (1−V')·Î_diff)`, `V'` là mask mềm; hoặc Poisson với biên = pixel warped tin cậy | CraftsMan3D 🛠 |

⚠️ Mức b có rủi ro mép mờ, vì mask sau khi hạ 8× rất thô; chỉ dùng với mask đã co.

### 2.5 Adapter tay (chỉ khi chỉ số vùng tay chững lại) ⚠️🛠

- Crop normal map (+ CCM) vùng tay 128–256² → encoder nhỏ → token tay.
- Masked cross-attention vào vài block DiT:
  - **K/V = [token toàn cục của khung (pool latent), token tay]**: PartDiffuser cho thấy bỏ global tệ hơn bỏ part (§1 #16).
  - Query: chỉ token thuộc tay (mask B1 hạ về latent, **giãn 1–2 token**).
  - Thêm sink token để hàng mask toàn False không gây NaN (Mesh4D, SignRefine).
  - Crop vắng (tay ra khỏi khung, bị che) → padding mask, không attend (PartDiffuser).
- `to_out` zero-init.
- Mở rộng P3: token mesh tay 3D (MeshToken lai) — đỉnh tay SMPL-X trong hệ tọa độ cổ tay, gom theo khoảng 16 phần/tay theo trọng số skinning, thêm part embedding học được, cộng vector pose cổ tay. Đưa vào qua cross-attention theo từng latent frame (MeshToken: per-frame CA 16.78/0.2451 tốt hơn global CA 16.49/0.2591 ✅). Chi phí trên Wan-1.3B khoảng 140–283M tham số ⚠️. **Chỉ giữ nếu thắng ở vùng tay bị che.**

### 2.6 Dữ liệu và độ phân giải

- Crop đã cố định cho cả clip (`scripts/02_preprocess_hamnosys.py:128-135`) ✅, đúng yêu cầu "crop theo bbox cả clip" mà Mesh4D nhấn mạnh. Không cần sửa.
- Cửa sổ 49 khung (4n+1); stride ngẫu nhiên {1, 2} (Mesh4D dùng 1–4; tay ký hiệu nhanh nên giới hạn ≤ 2 ⚠️). Đây cũng là cách trả lời một phần câu hỏi fps 25 vs 16 (`diffusion_improvement.md` §1.6).
- **Không** dùng 8 fps như DWM: quá thưa cho tay ký hiệu.
- Bắt đầu ở 320² trên 24 GB, sau đó 480² (thuê 80 GB nếu OOM).

---

## 3. H2 sửa đổi ("H2′")

### 3.1 Làm trước: baseline training-free (P0, 2–3 ngày) ⚠️🛠

- SD1.5 + ControlNet-Tile trên output GReg (upsample 512), thêm ControlNet-normal với `N_t` nếu cần.
- img2img strength 0.3–0.5, **chỉ trong** mask tay ∪ vùng `V=0` (B1, A1), dán lại bằng blend tách tần số (K4).
- Self-attention mở rộng K/V với đặc trưng `I_s` (đã invert hoặc VAE-encode + nhiễu ở mức t). CraftsMan3D Eq. 3 dùng chia sẻ K/V giữa các view; Generative Rendering dùng keyframe.
- Giá trị: có ngay số "prior SD không cần train làm được đến đâu" (ablation 1 của §2.9 trong file diffusion), trả lời câu hỏi reviewer "có cần train không?".

### 3.2 Refiner một bước (pix2pix-turbo), thêm thời gian từ A1

- **Loss thời gian khi train (P1)**: lấy cặp khung liên tiếp, `L_temp = ‖V_{t−1→t}·C·(Î_t − warp(Î_{t−1}, F_{t−1→t}))‖₁`, với `F`, `V`, `C` lấy từ `correspond()` của A1 giữa hai khung **đích**. Đã có trong `diffusion_improvement.md` §2.6; Generative Rendering cho thêm lý do: tương ứng mesh giữ nhất quán dài hạn.
- **Tiêm attention lúc infer (P2, thử nghiệm)** 🛠:
  - Pre-attn: K/V của khung t = [f_t, f_{I_s}, f_{t−1}].
  - Post-attn: `F_out = α·V·C·π(F_{t−1}) + (1 − α·V·C)·F̂_t`, chỉ ở self-attention có độ phân giải cao nhất.
  - **Tắt** ở vùng `V = 0` và trong mask tay: ở 32² latent, tay chỉ còn 2–4 token, tiêm vào chỉ làm ngón tay mờ ⚠️.
  - Neo vào `I_s`, trọng số `t−1` nhỏ để tránh trôi tích lũy.
- **Temporal attention zero-init (P3)**: thêm ở bottleneck UNet, train trên clip 4–6 khung (Mesh4D).

### 3.3 Biến thể video của H2 (tùy chọn) ⚠️🛠

- Công thức DWM áp thẳng: **output GReg** vào khe masked-video của Wan-Fun-1.3B-InP, mask = 1 ở vùng tin cậy và = 0 ở mask tay ∪ `V=0`; target = GT.
- Được nhất quán thời gian sẵn có, nhưng mất ưu thế tốc độ (50 bước; DWM không có chưng cất ít bước).
- Thực chất là "H1 với đầu vào là GReg thay vì `I_st`" → chỉ nên làm như một dòng ablation của H1.

---

## 4. Cải tiến rẻ cho A1/B1 và dữ liệu (làm ngay, không cần diffusion)

| # | Việc | Nguồn | Cách làm trong repo | Ghi chú |
|---|---|---|---|---|
| D1 | Xuất **CCM** cùng lúc với flow | Mesh4D, RealisDance | Trong `correspond()` đã có face map + barycentric đích → `interpolate_vertex_attribute(frag_t, v_Tpose, faces)` (`greg/render/rasterize.py:139`). `v_Tpose` = `SMPLXBody.model.v_template` (V, 3): T-pose trung tính, giống nhau cho mọi người ký nên CCM không lộ danh tính/hình thể ✅ (`greg/render/body.py:84`, thuộc tính chuẩn của `smplx`). Chuẩn hóa về [0, 1] theo bbox template trước khi coi như ảnh RGB | Vừa là điều kiện H1, vừa là metric (D6) |
| D2 | Hạ `C` ở dải biên silhouette | Uni3C `pointcloud.py:19-29` | Sobel trên depth z-buffer đích; nơi gradient lớn → nhân C với (1 − dải) | Thêm 1 khoá config, sweep độ rộng dải |
| D3 | Làm mượt SMPL-X theo thời gian | Animating the Uncaptured | Lọc Savitzky–Golay / One-Euro trên tham số pose (không lọc vị trí ngón quá mạnh) | **Rủi ro:** làm mờ chữ cái ngón tay (fingerspelling) nhanh; ablation trên bin chuyển động |
| D4 | Chẩn đoán xuyên thấu tay–thân | CraftsMan3D (winding number) | Kiểu E0: đếm tỉ lệ khung có đỉnh tay nằm trong mesh thân | Chỉ sửa nếu tỉ lệ đáng kể |
| D5 | Batch ngoại lai: loss ×1e-8 + ghi tên clip | Mesh4D | Trong `train_utils.py`: ngưỡng theo phân vị trượt (ví dụ > p99.5), không cố định | Có thể chạy ngay trên GReg hiện tại |
| D6 | Metric nhất quán thời gian không cần optical flow | Mesh4D (T10) 🛠 | Điểm cùng CCM ở hai khung, nhìn thấy cả hai → màu phải gần nhau | Dùng chung cho GReg, H1, H2 |

---

## 5. Đánh giá: những gì nên thêm

1. **Keypoint tay không vòng tròn** (P0) ✅ ý từ Animating the Uncaptured Eq. 7:
   - Chiếu khớp tay SMPL-X đích lên 2D, so với RTMPose-X / DWPose chạy trên ảnh sinh.
   - Lỗi tính theo Geman-McClure, nhân độ tin cậy keypoint.
   - Mình điều kiện bằng SMPL-X nhưng đo bằng RTMPose, nên tránh được lời phê bình "metric vòng tròn" nhắm vào SignRefine.
   - Phải báo kèm lỗi RTMPose trên **video thật** làm trần dưới.
2. **Chỉ số theo vùng**: L1/LPIPS **trong mask tay** và **trong vùng `V=0`** (bị che ở nguồn). Đây là nơi diffusion phải chứng minh giá trị; MeshToken cũng không đo vùng tay.
3. **Thời gian**: FVD; lỗi warp theo flow mesh giữa hai khung đích liên tiếp; D6.
4. **Trôi tần số thấp** 🛠: L1 giữa ảnh sinh và GT sau lọc Gaussian lớn (σ ≈ 8 px ở 256²), tính ngoài vùng tay. Bắt được trôi màu áo/da (K4).
5. **Tái fit SMPL-X 3D** trên video sinh: chỉ trên tập con cố định khoảng 100 clip. Chạy toàn bộ 1146 clip theo pipeline của Animating the Uncaptured tốn khoảng 1700 GPU-giờ ⚠️.
6. Nhắc lại nghịch lý perception–distortion (`diffusion_improvement.md` §1.8).

---

## 6. Ma trận ablation cập nhật cho H1′

Bổ sung cho bảng §1.9 của `diffusion_improvement.md`; các dòng **in đậm** là câu hỏi mới mà chưa bài nào trả lời.

| # | Biến thể | Câu hỏi |
|---|---|---|
| 0 | Wan-Fun-1.3B-InP **zero-shot**, khe masked = `I_st`, mask = `V·C` (chưa có kênh mới, chưa LoRA) | Prior inpainting + warp A1 thuần làm được đến đâu (M0′) |
| 1 | + LoRA, không kênh mới | Fine-tune giúp bao nhiêu |
| 2 | **mask toàn 1 (đúng DWM) so với mask `V·C`** | Mask nhìn thấy thật có ích không (DWM chưa trả lời) |
| 3 | + `c_N` (normal) | Điều kiện hình học đích |
| 4 | **+ `c_CCM` / thay `I_st` bằng CCM / thay bằng (u, v) nguồn kiểu GenWarp** | Phản biện GenWarp, có cơ sở 3D |
| 5 | + `M_t` qua Conv3d pixel (Uni3C) + loss vùng | Phần dành riêng cho tay/mặt |
| 6 | Infer mức a / b / c (§2.4) | Giữ chi tiết vùng thấy so với liền mạch |
| 7 | CFG hình học w ∈ {1, 2, 3} | Tay bám điều kiện chặt hơn có đổi lấy artefact không |
| 8 | **Backbone: InP + concat / VACE / Fun-Control** | Khe inpainting có hơn nhánh control không (DWM không ablation điều này) |
| 9 | Warp TPSMM thay A1 | Tách đóng góp A1 bên trong diffusion |
| 10 | Adapter tay: global+part / chỉ part / không adapter (P2); + token mesh tay (P3) | Kiểu PartDiffuser / MeshToken |
| 11 | Nhiễu khởi tạo gắn bề mặt (P3) | Kiểu Generative Rendering |

---

## 7. Lộ trình (thay §1.12 và §2.11 nếu đồng ý)

| Mốc | Việc | Chi phí ước tính ⚠️ |
|---|---|---|
| **M0** (song song A1+B1 đang chạy) | D2, D5, metric §5.1 + §5.2 + §5.4 vào `greg/eval/`; baseline H2 training-free (§3.1) trên 50 clip val | Vài ngày, 1 GPU 24 GB |
| **M0′** | Đọc `training/wan/pipeline/pipeline_wan_fun_inpaint_hand_concat.py` + `models/wan_transformer3d_with_conditions.py` của DWM. Ablation 0 (zero-shot) trên 50 clip val ở 480², 49 khung | Khoảng 50 × 3–5 phút ≈ 3–4 giờ trên 4090 |
| **M1′** | D1 (CCM). Port `WanTransformer3DModelWithConcat` sang 1.3B; dataloader cửa sổ 49 khung, cache moments. Ablation 1–3 ở 320² | 24 GB, LoRA r16–32 |
| **M2′** | Ablation 4–5; đo VRAM 480² → quyết định thuê 80 GB | – |
| **M3′** | Bản chính 480², ablation 6–9; refiner H2 có loss thời gian | A100/H100 thuê, vài ngày |
| **M4′** | Adapter tay / token tay (ablation 10) nếu chỉ số vùng tay chững lại | Tùy kết quả |

---

## 8. KHÔNG nên lấy (tổng hợp)

| Thứ | Từ bài | Lý do |
|---|---|---|
| Caption sinh từ video GT | DWM | Rò thông tin đích; GReg dùng prompt cố định |
| 8 fps, lặp khung đầu làm điều kiện tĩnh | DWM | Quá thưa cho ký hiệu; GReg có warp A1 tốt hơn lặp khung |
| Render chỉ bàn tay, cắt cổ tay, không z-test | DWM | Lỗi chính của GReg nằm ở cẳng tay; GReg có mesh toàn thân để che khuất đúng |
| Bỏ hẳn điều kiện căn pixel (render-free) | MeshToken | Bằng chứng yếu; mục tiêu GReg là tái tạo đúng pixel |
| Tokenizer VQ-HPS / SMPL | MeshToken | Không có ngón, không mặt, bỏ β |
| Pipeline 3D (VAE latent set, TSDF, dựng lại 4D mesh) | CraftsMan3D, Mesh4D | GReg **đã có** mesh; bài toán là ngoại hình |
| Copy code CraftsMan3D | CraftsMan3D | Nhiều lỗi, license mâu thuẫn |
| Backbone HY3D | Mesh4D | License Tencent phi thương mại, không vừa 24 GB |
| Toán số phức / đẳng biến trên mesh | Field Latents | Giải bài toán "không có hệ trục chuẩn"; SMPL-X topology cố định nên không cần |
| Train một mô hình cho mỗi mesh | Field Latents | 2 ngày × 8 V100 mỗi mesh |
| Tối ưu từng clip trong vòng train | Animating the Uncaptured | Khoảng 1.5 giờ/chuỗi |
| Phạt gập khuỷu tay | Animating the Uncaptured | Gập khuỷu mạnh là hợp lệ trong ký hiệu |
| DINOv2 cho chi tiết ngón | Animating the Uncaptured | Ở 256², patch 14 → tay chỉ 2–3 patch |
| Discrete diffusion / token mesh BPT / thứ tự BFS | PartDiffuser | Không phải bài toán của GReg; segmentation SMPL-X đã chính xác |
| Keyframe ngẫu nhiên, blend "first-filled + mean", atlas UV 2048² | Generative Rendering | GReg có `I_s` làm keyframe và `V·C` làm trọng số |
| Chép đặc trưng thẳng qua tương ứng (không qua attention) | Generative Rendering | Fig. 6 chứng minh là hỏng |

---

## 9. Câu hỏi còn mở (phải kiểm chứng)

1. Wan2.1-Fun-V1.1-**1.3B**-InP có cùng API/kênh (16 + 4 + 16) như bản 14B mà DWM dùng không? Có nhánh CLIP image không? ⚠️
2. VRAM thực khi train LoRA + patch-embed full ở 320²/480², 49 khung, với 52–68 kênh vào. ⚠️
3. ~~`SMPLXBody` có cho lấy `v_Tpose` không?~~ Có: `SMPLXBody.model.v_template` ✅. Còn mở: VAE của Wan có tái tạo đủ chính xác một ảnh tọa độ mượt (CCM) không, hay nên đưa CCM qua đường Conv3d pixel như `M_t`? ⚠️
4. Hệ số ×10 cho tay/mặt của SignRefine (vẫn đang lấy từ bản review). ⚠️
5. Lỗi RTMPose trên video ký hiệu thật (trần dưới cho metric §5.1). ⚠️
6. Mask toàn 1 của DWM có làm mô hình quá bảo thủ không, tức không dám vẽ lại vùng `V=0`? Ablation 2 trả lời. ⚠️

---

## 10. Đề xuất sửa `docs/diffusion_improvement.md` (chưa sửa, chờ đồng ý)

1. §1.4, bảng chọn backbone: dòng Wan2.1-Fun-1.3B-InP đổi "Không có kênh điều kiện hình học" thành "Thêm được bằng mở rộng patch-embed zero-init (DWM, CVPR 2026); khe masked-video + mask khớp đúng `I_st` + `V·C`". Khuyến nghị: **InP + concat trước**, VACE và Control làm so sánh.
2. §1.4, "Cách đưa điều kiện vào": thêm quy ước mask 4 kênh của Wan-Fun-InP (§2.2.1 file này).
3. §1.3: thêm CCM làm phương án (d) cho ablation GenWarp.
4. §1.9: gộp với ma trận §6 file này.
5. §2.4 / §1.4, công thức blend cuối: chuyển sang blend tách tần số (K4).
6. §6, tài liệu tham khảo: thêm 8 bài trong `ref papers/` và các repo DWM, Uni3C, Mesh4D.
