# Báo cáo chương trình diffusion D1–D6 cho GReg

Thư mục này có **một báo cáo cho mỗi mô hình** của chương trình "sáu hướng đưa diffusion vào GReg". Kế hoạch gốc nằm ở `docs/diffusion_6_huong_ke_hoach.md`. Các thí nghiệm chạy từ 02/11 đến 05/11/2026 trên một RTX 4090 thuê ở vast.ai.

Mỗi báo cáo có cùng cấu trúc:
- §0 Tóm tắt
- §1 Bối cảnh
- §2 Ý tưởng và điểm mới
- §3 Kiến trúc chi tiết
- §4 Giao thức đánh giá
- §5 Kết quả
- §6 Phân tích nguyên nhân
- §7 Hạn chế và việc tiếp
- §8 Trạng thái dữ liệu

Ký hiệu: ✅ đã đo hoặc kiểm chứng; ⚠️ giả thuyết hoặc ước lượng.

| Báo cáo | Diffusion nằm ở đâu | Kết luận |
|---|---|---|
| [D1: Refiner một bước SD-Turbo](D1_refiner_sd_turbo.md) | Sau GReg, chỉ trong cổng tay, mặt, vùng bị che | ✅ Cải tiến nhỏ: LPIPS −4.7 %, hand LPIPS −7 %, mặt +5 % |
| [D2: Regional Expert bằng flow matching](D2_regional_expert_diffusion.md) | Trong GReg, thay khâu tạo prior tay và mặt | ❌ LPIPS +4 % đến +21 % |
| [D3: E-LatentLPIPS làm loss](D3_diffusion_loss.md) | Chỉ trong loss khi train | ❌ Không đổi; bỏ GAN thì +14 % |
| [D4: Tăng cường nền và màu bằng SD](D4_augment_diffusion.md) | Trước khi train (offline) | ❌ Thắng ở 15 epoch, mất lợi thế ở 100 epoch. Kèm phát hiện lỗi "cặp cố định" |
| [D5: SD1.5 + ControlNet theo khung](D5_sd15_controlnet.md) | Thay hẳn bộ sinh | ❌ Kém GReg khoảng 8 lần |
| [D6: Video DiT Wan2.1-Fun-1.3B](D6_wan_video_dit.md) | Thay hẳn bộ sinh, sinh cả video | Tay −22 %, toàn khung +11 % |
| [**GReg + tay D6**](GReg_plus_D6_hands.md) | GReg cho toàn khung, D6 chỉ vẽ tay | ⭐ Tốt hơn GReg ở mọi chỉ số: LPIPS −9 %, hand LPIPS −22 %, mặt giữ nguyên |

## Các tập đánh giá

Chỉ so số **trong cùng một tập**.

| Tập | Mô tả | LPIPS của GReg run5 |
|---|---|---|
| test-full | Toàn bộ test, 1 146 clip, mọi khung | 0.0485 |
| u16 | Toàn bộ test, 16 khung lấy đều mỗi clip | 0.0487 |
| val-full | Toàn bộ val, 14 198 khung | 0.0484 |
| s200 | 200 clip test × 16 khung; tập chung của D5, D6 và bản ghép | 0.0476 |
| sweep15 | Train 15 epoch, val 100 clip × 16 khung; vòng lọc của D2, D3, D4 | 0.0617 |

## Bài học chung

1. **Diffusion thắng khi được giao đúng vùng nó giỏi** (bàn tay, chi tiết tần số cao) và **thua khi phải giữ danh tính, màu và nền**. Ở những vùng đó, bộ hồi quy GReg chép pixel từ ảnh nguồn nên chính xác hơn. Đây là đánh đổi giữa chất lượng cảm nhận và độ sai lệch pixel. VAE của SD nén 8× cũng làm mất chi tiết ở 256 px.
2. **Diffusion ở tầng điều kiện (D2) và ở tầng loss (D3) đều không giúp.** Giá trị của nó nằm ở đầu ra.
3. **Kết quả ở 15 epoch có thể không còn khi train đủ 100 epoch** (D4).
4. **Điều kiện mesh** (warp, V·C, CCM) giúp giữ mặt và màu trong D6. Ở tay, bốn biến thể D6 gần như bằng nhau, nên lợi ích ở tay đến từ prior video và loss vùng, không đến từ mesh.
5. Khoảng cách giữa run5 (LPIPS 0.0485) và bài báo GReg (0.0359) **có từ trước khi thêm diffusion**. Không hướng nào nhắm vào nó.

## Mã và dữ liệu

- Code của từng hướng nằm trên các nhánh `feat/diff-d1-refiner` … `feat/diff-d6-video-dit`.
- Một số bản sửa lỗi đã chạy trên máy chỉ có ở các nhánh local **chưa push**: `fix-d3-tmp`, `fix-d5-tmp`, `fix-d6-tmp`.
- Mã ghép tay và bootstrap được chép vào `code/` (`compose_hands.py`, `bootstrap_ci.py`, `post_d6.sh`).
- Máy vast đã xoá ngày 05/11/2026. Google Drive đầy từ 02:43 UTC ngày 04/11, nên **mất toàn bộ** các thứ sau:
  - checkpoint, ảnh và file điểm từng khung của D5 và D6;
  - checkpoint D4 100 epoch;
  - kết quả của bản ghép tay.
- Số liệu trong các báo cáo lấy từ log và kết quả đã báo trong quá trình chạy. Vòng lọc D1–D4 và checkpoint refiner D1 run3 vẫn còn trên Drive (`GReg_runs/diff_common`).
