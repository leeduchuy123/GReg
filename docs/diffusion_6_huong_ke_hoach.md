# Sáu hướng đưa diffusion vào GReg: kế hoạch thiết kế, code, dữ liệu, thí nghiệm

Ngày lập: 2026-11-02. File chủ của chương trình thí nghiệm diffusion. Mỗi hướng là một nhánh `feat/diff-d<k>-*`, tách từ nhánh chung `feat/diff-common` (tách từ `feat/mesh-flow-regions`, vì cần `correspond()` / `V`, `C`, buffer mesh và bộ chỉ số vùng).

Tài liệu đi kèm (đọc trước khi sửa code):
- `docs/diffusion_improvement.md` (nhánh mesh-flow): kế hoạch H1/H2 gốc.
- `docs/bai_hoc_diffusion_mesh_cho_greg.md`: 26 kỹ thuật rút từ 8 bài (DWM, Uni3C, Mesh4D, Generative Rendering, CraftsMan3D, Field Latents, PartDiffuser, Animating the Uncaptured) với trích dẫn `file:dòng`.
- `docs/tong_hop_bai_bao_diffusion_mesh.md`: note đọc code của các bài trên.
- `docs/notes_signrefine_signvip_signs_as_tokens.md`, `docs/code_notes_pix2pix_turbo.md`, `docs/code_notes_diffusion_as_loss.md`: note đọc paper/code cho D1, D3 và bảng so sánh (đang được viết).
- `docs/A1_B1_kien_truc.md` (nhánh mesh-flow): API `correspond`, chỉ số `hand_l1`/`arm_l1`/`disocc_frac`/bin chuyển động.

Quy ước: ✅ đã kiểm chứng (paper/code/số đo, có trích dẫn) · ⚠️ suy luận hoặc ước lượng, phải đo · 🛠 thiết kế riêng của luận văn.

---

## 0. Tóm tắt cho người bận

| Mã | Hướng | Diffusion chen vào đâu | GPU | Chi phí ước tính ⚠️ | Đóng góp chính |
|---|---|---|---|---|---|
| **D1** | Refiner một bước, gate theo hình học | *Sau* GReg (run5) | 24 GB | ~12 USD | Hiệu quả: gần chất lượng DiT, chi phí ≈ GReg |
| **D2** | Regional Expert bằng diffusion | *Trong* GReg, thay RE | 24 GB | ~6 USD | Thành phần mới: prior đa mode cho tay bị che |
| **D3** | Diffusion làm loss huấn luyện | *Trong loss* GReg, không có diffusion lúc infer | 24 GB | ~10 USD | Ablation: critic diffusion thay/bổ sung GAN |
| **D4** | Tăng cường dữ liệu bằng diffusion | *Trước train*, nhất quán theo clip | 24 GB | ~3 USD | Bằng chứng cross-person |
| **D5** | LDM từng khung (SD1.5 + ControlNet) + attention theo tương ứng mesh | *Thay* bộ sinh, 24 GB | 24 GB | ~32 USD | **So sánh có kiểm soát: keypoint 2D vs normal vs normal+V·C+CCM, cùng backbone** |
| **D6** | Video DiT warp-rồi-diffuse (Wan2.1-Fun-1.3B-InP + concat) | *Thay* bộ sinh, video | 80 GB | ~110–200 USD | Trần chất lượng; xác nhận kết luận D5 trên backbone video |

Hạ tầng chung (một lần): ~5 USD. **Tổng ≈ 230–330 USD**; D1–D4 + hạ tầng ≈ 40 USD, nên nạp 100 USD trước, phần D5/D6 nạp sau khi thấy kết quả D1–D5.

Thứ tự chạy: hạ tầng → D1 (training-free rồi train) → D2 → D3 → D4 → D5 → D6. D1–D4 chia sẻ **một máy 4090** giữ sống suốt chương trình (cache 256² ở lại máy; stop khi rảnh, ~0.03 USD/h lưu đĩa).

Phân bổ lỗi đo được (val 100 clip, cấu hình run5, 15 epoch, `ablation_v2/base/val_metrics.json`) ✅:

| | toàn ảnh | tay | tay @large | cánh tay @large | LPIPS @small → @large |
|---|---|---|---|---|---|
| L1 / LPIPS | 0.0210 | 0.1382 | 0.1691 | 0.1036 | 0.0236 → 0.0982 |

`disocc_frac@large` = 0.388 ✅: ở chuyển động lớn, 39 % pixel tay + cánh tay **không có trong ảnh nguồn**, mọi phép warp đều phải để generator vẽ. Đó là chỗ diffusion phải chứng minh giá trị; các hướng đều đo **trước hết** bằng `hand_l1@large`, `hand_lpips@large`, `disocc_l1`, rồi mới đến L1/PSNR/SSIM/LPIPS/FVD/AKD toàn ảnh.

---

## 1. Nguyên tắc chung cho cả sáu nhánh

1. **Bài toán không đổi:** đầu vào = ảnh tham chiếu `I_s` + normal map SMPL-X `N_t` (+ hộp vùng); đầu ra = khung 256². Mọi chỉ số của GReg (`greg/eval/evaluate.py`, `09_eval_akd.py`, `12_eval_fvd.py`) giữ nguyên, tính ở 256² trên cùng split/`valid_range`/khung nguồn. Đầu ra ở độ phân giải khác phải **hạ mẫu** về 256² trước khi đo.
2. **Baseline cố định:** run5 (`GReg_runs/run5/outputs/greg_final/checkpoints/epoch_100.pth`, TPSMM đóng băng, `qk_norm=true`, `regional_perceptual=3`). Test: L1 0.01735, PSNR 29.20, SSIM 0.9151, LPIPS 0.04837, FVD 43.8, AKD 0.0102 ✅. Nhánh nào train lại GReg đều dùng đúng config run5 + đổi một yếu tố.
3. **Giao thức sweep 15 epoch** (`configs/sweep_greg.yaml`, val 100 clip × 16 khung) để so sánh; chỉ run thắng mới chạy 100 epoch + test đầy đủ.
4. **Ngẫu nhiên:** mọi mô hình diffusion lấy mẫu với seed cố định theo `(clip, row)`; bảng chính 1 seed, dòng thắng 3 seed.
5. **Chi phí phải vào bảng:** tham số train, GPU-giờ train, giây/khung infer trên cùng 4090 (SpeedMeter đã có).
6. **Code:** mọi thứ mới nằm trong `greg/diffusion/` + `scripts/2x_*.py`; không sửa hành vi mặc định của GReg (mọi tính năng mới sau khoá config, mặc định tắt); có unit test CPU cho từng module; `python -m pytest` phải xanh trước khi lên máy thuê.
7. **Dữ liệu:** cache 256² (5731 clip; 4585 train / 1146 test; val = 10 % train) + buffer mesh (`02c`) + `valid_range`. Sau khi tiền xử lý xong lần này, **nén cache thành tar theo shard và đẩy lên Drive** (`GReg_cache/hamnosys_256_v1/*.tar`, Drive còn 4.8 TiB ✅) để các máy sau chỉ tải về (~20–30 phút) thay vì render lại 3–4 giờ.

---

## 2. Hạ tầng chung (`feat/diff-common`)

### 2.1 Xuất đầu ra GReg và tín hiệu hình học — `scripts/20_export_greg.py` 🛠

Chạy một checkpoint GReg trên một split, ghi cho mỗi `(key, row)`:

```
outputs/export/<run>/<split>/<key>/<row>.png        I_hat (GReg)
                                      <row>_warp.png  I_st (TPSMM warp của run5)
                                      <row>_occ.png   O (TPSMM, 1 kênh)
                                      <row>_vis.png   V_mesh · C_mesh (1 kênh, từ correspond(), nguồn = khung nguồn của giao thức)
                                      <row>_ccm.png   CCM: tọa độ T-pose của pixel đích, RGB [0,1] (bai_hoc §K2, D1 bảng §4)
```

- `V_mesh` tính bằng `MeshCorrespondence` với `smplx_dir` **kể cả khi model là TPSMM** (API đã cho phép: `07 --smplx-dir`), nên gate của D1 không phụ thuộc vào A1 thắng hay thua.
- CCM: `interpolate_vertex_attribute(frag_t, v_template, faces)` từ face map đích (`faces/<key>/<row>.png`) và barycentric; chuẩn hoá theo bbox của `SMPLXBody.model.v_template` ✅ (thuộc tính có sẵn).
- Cần cho: D1 (đầu vào + gate), D5/D6 (điều kiện), metric CCM.
- Tập train của D1 dùng checkpoint **epoch_040** của run5 để ảnh thô "xấu như lúc test" (xem §3.4); val/test dùng epoch_100.

### 2.2 Đánh giá một thư mục khung — `scripts/21_eval_frames.py` 🛠

Tách đánh giá khỏi lớp `GReg`: nhận thư mục PNG (bất kỳ nhánh nào sinh ra) + cache → toàn bộ chỉ số. Bắt buộc vì D1/D5/D6 không phải `nn.Module` GReg. Dùng lại `greg/eval/evaluate.py` qua một `FrameDirModel` giả (trả `output` từ PNG).

### 2.3 Chỉ số mới — `greg/eval/metrics.py`, `greg/eval/evaluate.py`

| Chỉ số | Định nghĩa | Nguồn ý tưởng |
|---|---|---|
| `hand_lpips` | LPIPS trên crop tay (hộp thích ứng B1, phóng lên 128²), nanmean | bai_hoc §5.2 |
| `disocc_l1` | L1 trên pixel thân có `V_mesh = 0` (vùng phải vẽ) | bai_hoc §5.2 |
| `hkd` (hand keypoint distance) | RTMPose-X / DWPose (`rtmlib`, ONNX) trên ảnh sinh **và** ảnh thật; lỗi Geman-McClure × độ tin cậy, so với khớp tay SMPL-X chiếu 2D; báo kèm `hkd_real` (trần dưới) | Animating the Uncaptured Eq. 7; tránh "metric vòng tròn" nhắm vào SignRefine |
| `ccm_tc` | nhất quán thời gian: hai khung đích liên tiếp, điểm cùng CCM và nhìn thấy ở cả hai → chênh màu | Mesh4D (bai_hoc D6) |
| `lowfreq_drift` | L1 sau Gaussian σ = 8 px, ngoài vùng tay | CraftsMan3D (K4) |

AKD (HRNet qua mmpose, `09_eval_akd.py`) giữ nguyên để so với bảng của bài báo.

### 2.4 Môi trường
- `requirements-diffusion.txt`: `diffusers`, `transformers`, `peft`, `accelerate`, `xformers` (theo CUDA), `rtmlib`, `onnxruntime-gpu`. Pin phiên bản sau khi smoke test trên máy thuê.
- Trọng số tải một lần vào `/workspace/hf` (HF cache), không đẩy lên Drive (tải lại nhanh hơn).

### 2.5 Máy thuê (vast.ai)
- `scripts/vast/setup.sh` (đã có) + `scripts/vast/pipeline.sh` thêm stage `meshbuf`, `cache_archive`, `export_run5`, và `stage` cho từng D.
- Chọn offer: 4090, ≥250 GB đĩa, reliability > 0.97, `inet_down > 300`; **đo tốc độ Drive 2 phút trước khi commit** (bài học: máy Nhật/Việt Nam 1 file/s).
- Checkpoint và `val_metrics.json` sync lên Drive `GReg_runs/diff_<D>/` bằng `sync_loop.sh`.

---

## 3. D1 — Refiner một bước, gate theo hình học (`feat/diff-d1-refiner`)

### 3.1 Ý tưởng
Giữ GReg run5 làm "bộ dựng thô". Một mô hình diffusion **một bước** (SD-Turbo, công thức pix2pix-turbo ✅) nhận `[Î_GReg, N_t, V·C, M_tay/mặt]`, trả ảnh sắc nét; chỉ vùng `G = dilate(M_tay ∪ M_mặt) ∪ (1 − V)` được thay, phần còn lại giữ GReg. Loss thời gian qua tương ứng mesh giữa hai khung đích liên tiếp.

### 3.2 Kiến trúc 🛠 (chi tiết code theo `docs/code_notes_pix2pix_turbo.md`)

```
Î_GReg 256² ──bicubic ×2──► 512² ──VAE enc (LoRA)──► z_img (4, 64, 64) ──┐
N_t 256² ──×2──► VAE enc (đóng băng, no grad) ──► z_N (4, 64, 64) ──────┤ concat → 8 + 4 kênh
[V·C, M_face, M_lhand, M_rhand] ──avg-pool về 64² ──► m (4, 64, 64) ─────┘
                 │ UNet SD-Turbo conv_in mở rộng 4 → 12 kênh, cột mới = 0 (DWM, bai_hoc §1 #2)
                 │ LoRA r = 8 trên attention + conv (pix2pix-turbo mặc định), t = 999, một bước
                 ▼
          z_out ──VAE dec (LoRA + skip zero-conv từ encoder, pix2pix-turbo)──► Î_ref 512² ──↓2──► 256²
          Î = (1 − G_soft)·Î_GReg + G_soft·Î_ref      (G_soft: G làm mềm 4 px; ablation: blend tách tần số K4)
```

- Vì sao 512 chứ không 256: SD-Turbo được train ở 512; issue #93 của pix2pix-turbo báo **sụp đổ ở 256 px** ✅ (`diffusion_improvement.md` §2.3). Tốn ×4 compute nhưng an toàn; ablation 256 trực tiếp nếu còn thời gian.
- Prompt cố định: "a person signing, sharp hands, studio video". Không caption từ GT (rò thông tin, bai_hoc §8).
- Tất định: không nhiễu thêm (pix2pix-turbo paired mode) → không nhấp nháy do seed.

### 3.3 Loss
`L = L2 + λ_lp·LPIPS + λ_h·(L2 + LPIPS trên crop tay 128²) + λ_t·L_temp (+ λ_adv·GAN vision-aided, mặc định tắt vì không ổn định trên 24 GB)`

- Trọng số theo vùng `w = 1 + 9·M_tay/mặt + 1·(1 − V)` (SignRefine ×10 ⚠️ đang xác minh trong note SignRefine; DWM loss trên `x0_hat`).
- `L_temp = ‖V_{t−1→t}·C·(Î_t − warp(Î_{t−1}, F_{t−1→t}))‖₁` với `F, V, C` = `correspond()` giữa hai khung **đích** (bai_hoc §3.2). Dataset lấy cặp khung đích liên tiếp `(t−1, t)` cùng khung nguồn.
- Cảnh báo code: `correspond` cần `faces_{t−1}, faces_t, smplx, focal, princpt` → dataset D1 bật `with_mesh=True` và nạp thêm hàng `t−1`.

### 3.4 Dữ liệu (chỗ dễ sai nhất, `diffusion_improvement.md` §2.5)
- Cặp train = (xuất run5 **epoch_040** trên clip train, GT). Lý do: epoch_100 đã "nhớ" tập train nên ảnh thô đẹp hơn lúc test; epoch_040 có phân phối lỗi gần test hơn ⚠️ (kiểm bằng: `hand_l1` của epoch_040 trên train ≈ `hand_l1` của epoch_100 trên val?). Nếu lệch > 20 %, thêm suy giảm tổng hợp (blur cục bộ ở tay + trộn với ảnh warp).
- Val/test = xuất epoch_100 (đúng giao thức).
- Mẫu train: 4585 clip × 4 cặp `(t−1, t)` mỗi epoch; 10k–20k bước, batch 2 (+ tích luỹ 4), bf16, gradient checkpointing.

### 3.5 Thí nghiệm

| # | Run | Câu hỏi | Chi phí ⚠️ |
|---|---|---|---|
| 0 | **Training-free:** SD1.5 + ControlNet-Tile (+ ControlNet-normal) img2img strength 0.3/0.5 trong `G` | Prior SD không train làm được đến đâu; có cần train không | 2 h |
| 1 | D1 chỉ nhận `Î_GReg` | Prior một bước tự làm được gì | 4 h train + 3 h infer test |
| 2 | + `N_t` + mask (kênh mới) + loss vùng | Hình học giúp tay đúng hình không | 4 h |
| 3 | + `L_temp` | Nhấp nháy đổi lấy gì (`ccm_tc`, FVD) | 4 h |
| 4 | Dán lại: `G` cứng / `G_soft` / tách tần số (K4) | Trôi màu (`lowfreq_drift`) so với liền mạch | infer only |
| 5 | Cặp train từ epoch_040 / epoch_100 / + suy giảm | Lệch phân phối ảnh thô | 4 h |
| 6 | Refiner trên a1_res (mesh warp) thay run5 | Refiner có che được lỗi warp không (ablation 8 của H2) | 4 h |

Epoch: 15k bước ≈ 6.5 epoch trên 4585×4 cặp; đánh giá val mỗi 2.5k bước, dừng sớm theo `hand_lpips`.

### 3.6 Quyết định hiện thực (sau khi đọc code pix2pix-turbo, `docs/code_notes_pix2pix_turbo.md`)
- Repo gốc @`86f5414` cần `diffusers 0.25.1` + `transformers 4.35.2` vì **monkey-patch forward của VAE** để lấy skip activation ✅, và có ~20 lỗi ngầm (input [0,1] vs [−1,1]; `latent_dist.sample()` làm nhấp nháy; cast fp16 với lr 5e-6 làm mất cập nhật; transform ngẫu nhiên áp **độc lập** cho A và B; `max_train_steps` không dừng vòng lặp; filter `save_model` bỏ sót module mới) ✅. → **Viết lại tối thiểu** trong `greg/diffusion/turbo_refiner.py` trên diffusers/peft hiện hành (~300 dòng): VAE enc (LoRA r4) → UNet SD-Turbo `t=999` một bước (LoRA r8, `conv_in` train full và **mở rộng kênh zero-init**) → `DDPMScheduler.step` một bước → VAE dec (LoRA r4) + 4 skip-conv 1×1 zero-init từ down-block của encoder (ghi nhận bằng `register_forward_hook`, không monkey-patch). Trainable ≈ 9.6 M ✅ (+ cột `conv_in` mới).
- Dùng `latent_dist.mode()` (tất định); bf16 autocast với master fp32; AdamW lr 5e-6 → thử 2e-5 vì có cột mới; LPIPS-VGG λ = 5, L2 λ = 1 (mặc định repo ✅), CLIP-sim tắt (không có caption), GAN tắt ở v1.
- Unit test bắt buộc: (i) bước 0 với LoRA = 0 và skip = 0 cho ra ảnh ≈ VAE(dec(enc(x))) của SD-Turbo (identity có tổn thất VAE); (ii) cột `conv_in` mới = 0 → đổi `N_t`/mask không đổi đầu ra ở bước 0; (iii) transform đồng bộ A/B.

### 3.7 Rủi ro code
- `vision_aided_loss` thiếu trong repo gốc (issue #38 ✅) → tắt GAN ở v1.
- VAE skip-conv zero-init: nếu không zero-init thì bước đầu đã phá ảnh (pix2pix-turbo §3) → unit test "bước 0 ≈ identity".
- Mask 64² quá thô ở tay (tay ≈ 4–6 token) → mask tay giãn 1 token trước khi pool.
- Hạ mẫu 512→256 bằng area, không bilinear (giảm aliasing giả làm LPIPS tốt giả).


### 3.8 Đã hiện thực (nhánh `feat/diff-d1-refiner`)
- `greg/diffusion/turbo_refiner.py` (SD-Turbo một bước, skip VAE qua hook, `conv_in` mở rộng zero-init cho latent normal + 4 mặt nạ), `gating.py`, `refiner_data.py`, `refiner_losses.py`; `scripts/22_train_refiner.py` (train), `23_refine_frames.py` (infer + blend), `24_refine_training_free.py` (run 0: SD1.5 + ControlNet-Tile img2img), `25_collect_results.py`.
- Loss tính ở **256²** sau khi hạ mẫu (area) đầu ra 512 — đúng như lúc đánh giá; mục tiêu 512 không tồn tại trong cache 256 nên không dùng.
- Thứ tự chạy (`scripts/vast/stages_d1.sh`): smoke 20 bước → baseline GReg trên val → run 0 → run 3 (đầy đủ) → run 1 → run 2; đánh giá mọi blend cho run 3.

---

## 4. D2 — Regional Expert bằng diffusion (`feat/diff-d2-diffusion-re`)

### 4.1 Ý tưởng
`RegionalExpert` (`greg/models/regional_expert.py`) hồi quy một prior gradient 1 kênh 64² từ crop normal + style. Khi tay bị che/chồng, prior thật đa mode → hồi quy L1 cho trung bình mờ. Thay bằng diffusion có điều kiện trên crop 64²; GReg nhận prior lấy mẫu y như cũ (`prior_map`), nên GRM/generator không đổi.

### 4.2 Kiến trúc 🛠
- `DiffusionRegionalExpert(nn.Module)` trong `greg/diffusion/region_diffusion.py`:
  - Mạng: UNet nhỏ 64² (kênh 64-128-256, attention ở 16²), ~6 M tham số; đầu vào `cat[x_t (1), N_crop (3)]`; điều kiện `style (256) + t_emb` qua AdaGN (đã có `ModulatedConv2d`/`StyledBlock` trong `greg/models/style.py`, dùng lại).
  - Mục tiêu: **flow matching** (`x_t = (1−σ)x_0 + σε`, dự đoán `ε − x_0`, giống DWM/Wan ✅ bai_hoc §2.2.3), 1 mạng cho mặt, 1 chung cho hai tay (như paper).
  - Lấy mẫu: Euler 8 bước, nhiễu khởi tạo cố định theo `hash(key)` → tất định theo clip; **cùng nhiễu cho mọi khung của clip** để giảm nhấp nháy prior.
  - Dropout điều kiện style p = 0.1 → CFG w = 1.5 trên style (ablation).
- Giao diện giữ nguyên `RegionalPriorGenerator.forward(normal_target, boxes, style) -> {prior_patches, prior_map, normal_crops}` để `GReg.forward` không đổi; khoá config `model.rpg.type: regression | diffusion`.

### 4.3 Huấn luyện (hai giai đoạn, đúng tinh thần "separately train RE" + `detach_prior: true`)
1. **Giai đoạn RE:** trên crop (normal, style(I_s), gradient GT của crop đích) lấy từ cache (dataset có sẵn `crop_regions` + `gradient_map`); style encoder train cùng. 30k bước batch 64 crop ≈ 1.5 h ⚠️. Đo `struct_l1` trên val so với RE hồi quy, **và** đa dạng mẫu ở crop bị che (`disocc` cao).
2. **Giai đoạn GReg:** sweep 15 epoch với `rpg.type=diffusion` (RE + style đóng băng hoặc fine-tune nhỏ) so với `base`. Chi phí: lấy mẫu 8 bước × 3 crop 64² mỗi batch — không đáng kể so với generator 256².

### 4.4 Thí nghiệm
| # | Run | Câu hỏi |
|---|---|---|
| 1 | `struct_l1` RE hồi quy vs diffusion (1 mẫu, trung bình 8 mẫu) | Diffusion có prior sắc hơn không; trung bình nhiều mẫu có tụ về hồi quy không (kiểm tra "đa mode") |
| 2 | GReg + RE diffusion, 15 epoch | `hand_l1@large`, `hand_lpips`, AKD tay |
| 3 | Nhiễu cố định theo clip / theo khung | `ccm_tc`, nhấp nháy |
| 4 | Mục tiêu gradient / rgb (Table 2 của paper) | Loại prior |
| 5 | `detach_prior=false` + fine-tune RE trong GReg | Có cần end-to-end không |

### 4.5 Rủi ro code
- Prior gradient thưa (phần lớn ≈ 0): chuẩn hoá về [−1, 1] trước khi diffuse, loss MSE thuần; kiểm tra histogram mẫu không bị "xám đều".
- 8 bước Euler trong vòng train GReg phải chạy dưới `torch.no_grad()` khi RE đóng băng; nếu fine-tune thì chỉ bước cuối có grad (tránh 8× bộ nhớ).

### 4.6 Đã hiện thực (2026-11-02, nhánh `feat/diff-d2-diffusion-re`) — khác kế hoạch ở đâu
- **Không** viết UNet mới: `FlowExpert` = đúng `RegionalExpert` của GReg (cùng kênh 64-128-256-256, ~14 M/expert) với đầu vào `[x_t, N_crop]`, điều kiện `[style, MLP(t_emb)]` qua modulated conv, đầu ra không sigmoid. → So sánh "cùng mạng, chỉ đổi mục tiêu hồi quy → flow matching", sạch hơn.
- Nhiễu `fixed` = **một** tensor nhiễu (seed 0) cho mọi mẫu/khung (không theo `hash(key)`, vì `GReg.forward` không biết clip); `random` là ablation 3.
- Giai đoạn A (`scripts/26_train_region_diffusion.py`): EMA 0.999, 30k bước batch 32 cặp (96 crop), so trên **cùng crop val** với RE hồi quy của run5 (`--regression-checkpoint`): `diff_l1`, `diff_mean_l1` (trung bình 8 mẫu), `diff_std` (độ đa mode), từng vùng.
- Giai đoạn B: `configs/sweeps/d2_diffusion_re.yaml` (`base` = run5 `sweep_clean/ref` đánh giá lại, `d2_fixed`, `d2_random`); `06_train_greg.py` nạp `model.rpg.checkpoint` và đóng băng RE + style encoder.
- Ablation 4–5 (mục tiêu rgb, fine-tune end-to-end) chưa cấu hình; chỉ làm nếu `d2_fixed` thắng.

---

## 5. D3 — Diffusion làm loss huấn luyện (`feat/diff-d3-diffusion-loss`)

### 5.1 Ý tưởng
Không có diffusion lúc infer (tốc độ = GReg). Mô hình diffusion đóng băng làm critic/perceptual. Kết luận đọc code (`docs/code_notes_diffusion_as_loss.md`) ✅:
- **DMD/DMD2 bị loại:** loss DMD áp được lên đầu ra CNN bất kỳ (không cần nhiễu đầu vào của generator, `sd_guidance.py:168-255`), nhưng phần tốn kém là mạng "fake score" = **cả UNet SD 0.86 B train fp32**, cập nhật 5–10 lần/bước G (`train_sd.py:330-408`) → > 24 GB, và không còn là "critic đóng băng".
- **E-LatentLPIPS** (Kang et al. 2024, pip `elatentlpips`, CC BY-NC): VGG16-bn trên latent SD 4 kênh; 12 ms/+0.6 GB mỗi bước ✅; cần thêm VAE-enc fwd+bwd cho ảnh sinh. Bản "ensemble augment" tránh artefact lượn sóng ✅. **Làm trước.**
- **DIFT** (`dift_sd.py`, SD2.1 `t=261`, `up_ft_index=1`, 1/16 độ phân giải ✅; code gốc `no_grad` + detach, phải sửa): 3–6× VGG, +4–7 GB; rủi ro mù màu/over-smoothing ở 1/16. **Làm sau nếu tay chưa cải thiện.**

### 5.2 Hiện thực
- `greg/diffusion/latent_lpips.py`: `ELatentLPIPSLoss(nn.Module)`: ảnh 256 → [−1, 1] → VAE-enc SD1.5 (đóng băng, **có grad** qua ảnh sinh) → `elatentlpips` (ensemble augment bật). Khoá `loss.weights.elpips`, `loss.elpips: {augment: bg+cutout}`.
- `greg/diffusion/dift_loss.py` (giai đoạn 2): chép forward UNet cắt sau `up_blocks[1]` (kiểu DIFT), cùng ε cho pred/GT, bf16, `requires_grad=False` toàn bộ SD.
- Áp **toàn ảnh** và **crop tay 128²** (thay regional perceptual VGG trong ablation).
- Chi phí ⚠️: E-LatentLPIPS ≈ +0.1–0.2 s/bước (gốc 0.73 s); DIFT ≈ +0.8–1.2 s/bước.

### 5.3 Thí nghiệm
| # | Run |
|---|---|
| 1 | base + `elpips` toàn ảnh (λ 0.5 / 1 / 2) |
| 2 | base + `elpips` crop tay thay regional VGG |
| 3 | base, `adv=0` + `elpips` (critic diffusion **thay** GAN) |
| 4 | base + `dift` toàn ảnh (chỉ nếu 1–3 không động đến `hand_lpips`) |

Đọc: `hand_lpips`, `lowfreq_drift` (critic latent có thể kéo trôi màu), PSNR (kỳ vọng không đổi).

### 5.4 Đã hiện thực (nhánh `feat/diff-d3-diffusion-loss`)
- API thật của gói (đọc mã nguồn `elatentlpips`): `ELatentLPIPS(encoder="sd15", augment="bg")`, `forward(z0, z1, normalize=True, add_l1_loss=True)`; latent **chưa nhân** hệ số (gói tự nhân 0.18215). Trọng số latent-VGG tải ~1.1 GB lần đầu. Kiểm tra cục bộ: `elpips(a, a) = 0` ✅.
- `GRegLoss`: khoá `elpips`, `elpips_hand` (crop tay 64² → 256² → latent 32², chỉ tay nhìn thấy). Sweep `configs/sweeps/d3_diffusion_loss.yaml`: `base`, `elp1`, `elp1_hand`, `elp1_noadv`. DIFT chưa viết (chỉ khi 3 run này không cải thiện `hand_lpips`).

---

## 6. D4 — Tăng cường dữ liệu bằng diffusion, nhất quán theo clip (`feat/diff-d4-augment`)

### 6.1 Ý tưởng và ràng buộc
Pose-guided cần **nguồn và đích cùng ngoại hình** → mọi augmentation phải áp **giống hệt cho cả clip**. Hai phép, mỗi phép quyết định một lần theo clip:
1. **Nền mới sinh bằng SD** (text → ảnh nền studio/phòng, 256²), ghép bằng mask thân; mask thân lấy từ **matting** (RVM, nhẹ) chứ không từ mesh (mesh mỏng hơn áo/tóc → viền).
2. **Đổi màu quần áo/da nhất quán:** hue/sat/giá trị theo clip bên trong mask thân (không phải diffusion, nhưng rẻ và đúng mục tiêu "tổng quát hoá").
3. (Tùy chọn, có rủi ro) **Đổi áo bằng SD inpainting trên khung nguồn**, rồi lan sang khung đích bằng tương ứng mesh ở vùng V=1; vùng V=0 giữ gốc → **không nhất quán** → chỉ thử nếu 1–2 đã có hiệu quả.

### 6.2 Hiện thực
- `scripts/22_make_backgrounds.py`: 500 nền SD1.5 (1 h). `greg/data/augment.py`: `ClipConsistentAugment` (seed theo `(epoch, key)`), bật bằng `data.augment: {background_p: 0.3, color_p: 0.5}`. Matting mask cache: `matte/<key>/<row>.png` (RVM, ~1 h cho 430k khung ⚠️).
- Thí nghiệm: base vs base+aug, 15 epoch; đo **cross-person** (`08_eval_cross_person.py`) là chính, reconstruction là phụ (kỳ vọng ≈).

### 6.3 Đã hiện thực (nhánh `feat/diff-d4-augment`) — thay đổi so với 6.2
- **Phát hiện lỗi trong code GReg (ảnh hưởng cả run5):** `PairDataset.set_epoch()` chỉ đổi bản sao ở tiến trình chính; DataLoader dùng `persistent_workers=True` nên worker giữ epoch 0 → **mọi epoch dùng lại đúng 4 585 cặp (nguồn, đích)**, không phải cặp mới mỗi epoch như docstring nói. Đã kiểm chứng bằng test (`tests/test_d4_augment.py::test_shared_epoch_reaches_persistent_workers`). Sửa có tuỳ chọn: `data.fresh_pairs_per_epoch: true` (epoch đặt trong bộ nhớ chia sẻ). **Mặc định tắt** để baseline vẫn tái lập được; run `fresh` trong sweep D4 đo riêng tác dụng của bản sửa.
- Matting: **không** dùng RVM qua `torch.hub` (chạy mã tải từ GitHub); dùng DeepLabV3-ResNet101 của torchvision (lớp `person`) hợp với mặt nạ SMPL-X giãn 2 px, làm mềm 3×3 (`scripts/28_matte_cache.py`, `--check` đo độ phủ thân SMPL-X).
- Nền: 600 ảnh SD1.5 (`scripts/27_make_backgrounds.py`): 0–499 cho train, **500–599 giữ lại**.
- Đánh giá tổng quát hoá: `08_eval_cross_person.py` **không có ground truth** (chỉ xuất khung), nên thay bằng `scripts/29_eval_unseen_backgrounds.py`: tái tạo trên clip val được ghép vào nền **chưa thấy** (cùng nền cho nguồn và đích), đủ mọi chỉ số. Sweep: `base` / `fresh` / `aug` (`configs/sweeps/d4_augment.yaml`).

---

## 7. D5 — LDM từng khung: SD1.5 + ControlNet, điều kiện mesh, attention theo tương ứng (`feat/diff-d5-frame-ldm`)

### 7.0 Vị trí so với SignRefine / SignViP (`docs/notes_signrefine_signvip_signs_as_tokens.md`) ✅
- SignRefine: **không có code, checkpoint, hay dữ liệu NVSign công khai**; không trích dẫn GReg; không dùng SignAvatars. Điều kiện = keypoint RTMPose-X + render keypoint vùng 256² + CoordConv; loss vùng `w_h = w_f = 10` (Eq. 4) ✅; metric = MPJPE/LPIPS/SSIM theo vùng mặt/tay trái/tay phải. **Table 5:** thêm điều kiện dày (MANO + normal mặt) *không* giúp tay — nhưng với fit WiLoR nhiễu; SignAvatars có SMPL-X tối ưu theo thời gian (2.5 px). Luận văn phải trả lời đúng điểm này: "mesh giúp tay *khi fit đủ tốt*".
- SignViP: code + checkpoint RWTH công khai (SD1.5 + AnimateDiff + ReferenceNet, 2.78 B tham số, 1.24 s/khung, DDIM 50) → có thể chạy làm dòng tham chiếu nếu kịp; metric thêm **Hand SSIM**.
- Vì vậy dòng "keypoint 2D, cùng backbone" phải tự dựng: D5 (k) trên SD1.5 + ControlNet, và (nếu D6 chạy) Wan2.1-Fun-1.3B-Control công khai với video keypoint. Thêm vào §2.3: MPJPE/LPIPS/SSIM theo vùng mặt/tay T/P trên crop từ hộp GT (để đọc cạnh Table 2 của SignRefine) và Hand SSIM.

### 7.1 Vì sao đáng làm trước D6
Đây là nơi chạy **so sánh có kiểm soát** của luận văn với 1/10 chi phí Wan: cùng backbone, đổi điều kiện:
- (k) keypoint 2D vẽ thành skeleton (DWPose-style, lấy từ khớp SMPL-X chiếu 2D) — "kiểu SignRefine";
- (n) normal map `N_t`;
- (n+g) `N_t` + `I_st` warp + `V·C` + CCM — "kiểu mesh".
Nếu (n+g) > (n) > (k) ở `hand_*` thì giả thuyết trung tâm đứng vững trước khi tốn tiền cho D6.

### 7.2 Kiến trúc 🛠
- Backbone SD1.5 đóng băng + **ControlNet** train từ bản sao encoder (chuẩn ControlNet), điều kiện pixel `c ∈ {k, n, n+g}` ở 512² (cache 512² tạo lại bằng `02_preprocess --size 512`, ~4 h ⚠️).
- **Tham chiếu danh tính:** "reference-only" attention: K/V của self-attention mở rộng bằng đặc trưng của `I_s` (encode qua cùng UNet ở cùng t, không train thêm) + IP-Adapter ảnh (tuỳ chọn). Đây là cách Animate Anyone/MagicAnimate thu gọn, không cần ReferenceNet riêng.
- **Thời gian (K3, Generative Rendering):** lúc infer, pre-attn K/V = `[f_t, f_{I_s}, f_{t−1}]`, post-attn `F ← α·V·C·π(F_{t−1}) + (1−α·V·C)·F̂_t` ở self-attention độ phân giải cao nhất; **tắt** trong mask tay (bai_hoc §3.2). Lúc train: `L_temp` như D1 trên `x0_hat`.
- Lấy mẫu: DDIM 25 bước, CFG 2 trên điều kiện hình học (dropout điều kiện p = 0.1 lúc train, bai_hoc §2.3).

### 7.3 Chi phí ⚠️
ControlNet 512², batch 4, bf16, grad ckpt: ~1.1 s/bước trên 4090; 12k bước ≈ 4 h mỗi biến thể × 3 = 12 h. Infer: 25 bước × 0.08 s ≈ 2 s/khung → **tập con test cố định 200 clip × 16 khung = 3.2k khung ≈ 2 h** mỗi biến thể; GReg run5 và D1 cũng được đo lại trên đúng tập con này để so sánh công bằng.

### 7.4 Thí nghiệm
| # | Run | Câu hỏi |
|---|---|---|
| 1 | (k) keypoint 2D | baseline kiểu SignRefine, cùng backbone |
| 2 | (n) normal | hình học bề mặt |
| 3 | (n+g) normal + warp + V·C + CCM | hình học + tương ứng |
| 4 | (n+g) − CCM; (n+g) với (u,v) nguồn kiểu GenWarp thay CCM | phản biện GenWarp (bai_hoc §K2) |
| 5 | ± attention tương ứng lúc infer | `ccm_tc`, FVD |
| 6 | ± dán lại pixel warp ở V≈1 (K4) | danh tính |

---

## 8. D6 — Video DiT warp-rồi-diffuse (`feat/diff-d6-video-dit`)

Thiết kế đã có đầy đủ ở `docs/bai_hoc_diffusion_mesh_cho_greg.md` §2 (H1′) và ma trận ablation §6; không lặp lại. Tóm tắt quyết định:
- Wan2.1-Fun-V1.1-1.3B-InP, khe masked-video = `I_st` (warp mesh hoặc TPSMM — ablation 9), mask = `V·C` (4 kênh/latent frame, **`1 − mask`** theo quy ước DWM `pipeline_wan_fun_inpaint_hand_concat.py:357-366` ✅ — sai dấu là hỏng), kênh mới `N_t`, CCM qua patch-embed mở rộng zero-init (`wan_transformer3d_with_conditions.py:161-167` ✅), LoRA r16–32, loss vùng trên `x0_hat`.
- Dữ liệu: cache 480² mới (`--size 480`), cửa sổ 49 khung stride {1, 2}, 25 fps giữ nguyên; cache **moments** VAE.
- Máy: 1× A100/H100 80 GB (vast có A100_SXM4 ~1.3–1.8 USD/h ✅ ngày 2026-11-02). Thử pipeline ở 320² trên 4090 trước (ablation 0 zero-shot + ablation 1).
- Code: port `WanTransformer3DModelWithConcat` (DWM, Apache-2.0, ~270 dòng) sang 1.3B; dataloader riêng; **spawn agent đọc lại toàn bộ `training/wan/` của DWM khi bắt đầu nhánh** (note hiện có ở `tong_hop_bai_bao_diffusion_mesh.md` §Bài 4).
- Thứ tự ablation: 0 (zero-shot) → 1 (LoRA) → 2 (mask toàn 1 vs V·C) → 3 (+N_t) → 4 (+CCM) → 9 (TPSMM vs mesh warp). Chỉ chạy 5–8, 10–11 nếu còn ngân sách.
- Đo trên **cùng tập con 200 clip** của D5.

---

## 9. Lịch chạy và máy

| Bước | Nhánh | Máy | Thời gian GPU ⚠️ | Chi phí ⚠️ |
|---|---|---|---|---|
| 0 | `feat/diff-common`: setup, preprocess 256² + meshbuf, nén cache lên Drive, export run5 (ep40 train, ep100 val/test), metric mới + `hkd_real` trần dưới | 4090 #1 (giữ sống) | 8 h | 4 |
| 1 | D1 run 0 (training-free) → run 1–3 → 4–6 | 4090 #1 | 25 h | 12 |
| 2 | D2 giai đoạn RE → sweep 2 run | 4090 #1 | 12 h | 6 |
| 3 | D3 run 1–3 | 4090 #1 | 25 h | 11 |
| 4 | D4 nền + matte + 1 run + cross-person | 4090 #1 | 7 h | 3 |
| 5 | Final 100 epoch cho run thắng của D2/D3/D4 (+ test đầy đủ, FVD, AKD) | 4090 #1 | 25 h × 1–2 | 12–24 |
| 6 | D5: cache 512², 3 ControlNet, infer tập con, ablation 4–6 | 4090 #2 (song song với 1–5) | 70 h | 32 |
| 7 | D6: cache 480², port DWM, ablation 0–4, 9 | A100 80 GB | 60–120 h | 110–200 |

Lưu đĩa khi stop: ~0.03 USD/h/máy. Tổng: **≈ 230–330 USD**. Nạp 100 USD cho bước 0–5; quyết định D5/D6 sau khi có bảng D1–D4.

---

## 10. Bảng kết quả đích (điền dần)

| Dòng | Backbone | Điều kiện | GPU-h train | s/khung | L1 | LPIPS | hand_l1@large | hand_lpips | disocc_l1 | hkd | ccm_tc | FVD | AKD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GReg run5 (tái hiện) | GAN | N_t | ~22 | ~0.02 | 0.0174 | 0.0484 | | | | | | 43.8 | 0.0102 |
| + D2 RE diffusion | GAN | N_t | | | | | | | | | | | |
| + D3 loss diffusion | GAN | N_t | | | | | | | | | | | |
| + D4 augment (cross-person) | GAN | N_t | | | | | | | | | | | |
| + D1 refiner | GAN + SD-Turbo 1 bước | N_t, V·C, M | | | | | | | | | | | |
| D5 (k) keypoint 2D | SD1.5 + ControlNet | skeleton | | | | | | | | | | | |
| D5 (n) normal | SD1.5 + ControlNet | N_t | | | | | | | | | | | |
| D5 (n+g) mesh | SD1.5 + ControlNet | N_t, I_st, V·C, CCM | | | | | | | | | | | |
| D6 Wan + V·C + N_t (+CCM) | Wan-1.3B-InP + LoRA | như trên, video | | | | | | | | | | | |
| SignRefine / SignViP (số công bố) | Wan-Fun-Control / – | keypoint 2D | 4× GH200 | | | | | | | | | | |
