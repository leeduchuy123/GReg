# Proposal: Sign Language Video Generation (SLVG) và GReg (10 slide)

> Ghi chú cho người trình bày: số liệu GReg lấy từ bài báo (ICCV 2025). Trạng thái code công khai được tra ngày 2026-10-18 qua tìm kiếm web. Mục "chưa tìm thấy" nghĩa là **không thấy link code trong kết quả tìm kiếm**, chưa chắc là không tồn tại. Nên mở lại arXiv/GitHub của từng bài để xác nhận trước khi trình bày.

---

## Slide 1: Ba hướng nghiên cứu của SLVG

Sau khi tìm hiểu, em thấy SLVG chia làm 3 hướng, khác nhau ở đầu vào và cách điều khiển.

| | A. Ảnh tham chiếu + pose | B. Text-to-sign-video | C. Dùng foundation model |
|---|---|---|---|
| Ý tưởng | Cho 1 ảnh người + chuỗi pose/video ký hiệu, sinh video người đó ra ký hiệu theo chuỗi pose (cùng họ với Animate Anyone) | Từ câu văn bản, sinh thẳng video ký hiệu | Dùng video diffusion pretrained, điều khiển bằng pose/keypoint |
| Đầu vào | Ảnh + pose/video | Văn bản | Ảnh + pose (thường), có thể thêm điều kiện khác |
| Đại diện | PGMM (ECCV 2024), GReg (ICCV 2025) | SignGen (ECCV 2024) | SignRefine (2026) |

(Hướng B có đầu vào/ra khác A, nên không so sánh trực tiếp được với PGMM.)

---

## Slide 2: Ưu/nhược điểm và SOTA mới nhất của từng hướng

Ưu/nhược dưới đây là đánh giá của em dựa trên các bài đã đọc.

| | A. Ảnh tham chiếu + pose | B. Text-to-sign | C. Foundation model |
|---|---|---|---|
| **Ưu** | Kiểm soát chuyển động rõ ràng, giữ được danh tính người trong ảnh. Kiến trúc nhỏ, huấn luyện được trên 1 GPU (PGMM/GReg báo cáo RTX 3090) | Đầu-cuối từ văn bản, không cần pose đầu vào | Tận dụng prior hình ảnh của model pretrained nên ngoại hình chân thực, tổng quát hóa tốt |
| **Nhược** | Cần chuỗi pose đã có sẵn (không tự hiểu ngôn ngữ). Bàn tay và mặt nhỏ nên khó vẽ đúng. Khó giữ nhất quán theo thời gian | Khó đúng ngữ nghĩa và ngữ pháp ký hiệu. Diffusion tốn tài nguyên, độ chính xác ngón tay hạn chế | Tốn tài nguyên (SignRefine báo cáo 4 GPU GH200). Model gốc hay lỗi tay/mặt, phải thêm adapter để sửa |
| **SOTA mới nhất em tìm được** | GReg (ICCV 2025). Cùng nhóm hướng: SignViP (NeurIPS 2025, diffusion + ảnh tham chiếu) | SignMoD (TPAMI 2026), FlashSign (ICME 2026 Oral, không cần pose), Text2Sign (arXiv 7/2026, đăng IEEE Access 2026, baseline 1 GPU) | SignRefine (arXiv 9/2026, poster ECCV 2026): giữ cố định video DiT Wan-Fun-Control-1.3B, thêm adapter vùng tay/mặt |
| **Code public?** (kiểm tra 2026-10-18) | PGMM: chưa tìm thấy. GReg: chưa tìm thấy. SignViP: **có**, code huấn luyện/đánh giá, checkpoint RWTH-T trên Hugging Face, How2Sign "coming soon" (github.com/umnooob/signvip, 25 sao) | SignGen: **có một phần** (github.com/mingtiannihao/SignGen, MIT, 23 sao). README ghi code huấn luyện "coming soon", chưa có checkpoint riêng. SignMoD (repo tên SignMoDP): repo tồn tại nhưng chưa thấy file nguồn hay weights. FlashSign: repo tồn tại nhưng README gần như trống (2 commit, chưa thấy code). Text2Sign: **có** code huấn luyện (github.com/xiaruize0911/text2sign, MIT), chưa thấy weights | SignRefine: arXiv 2609.08496 (8/2026), poster ECCV 2026. Trang dự án có nhãn "Code" nhưng **không có URL**, chưa nêu việc phát hành NVSign. Chưa thấy repo |

Kết luận: ở hướng A, bài mạnh nhất (GReg) và PGMM đều **chưa có code công khai**, nên muốn làm tiếp phải tự cài đặt lại.

---

## Slide 3: Lựa chọn hướng và câu hỏi nghiên cứu

Do hạn chế phần cứng, em tập trung vào hướng A (PGMM/GReg).

**Câu hỏi 1.** GReg dùng hình học 3D (SMPL-X). Liệu đặc trưng dạng đồ thị biểu diễn động học bàn tay trong không gian và thời gian (3D mesh) có cải thiện độ chính xác ngón tay và tính nhất quán theo thời gian trong SLVG có điều kiện ảnh tham chiếu không?
→ Việc cần làm: tìm toàn bộ bài báo liên quan.

**Câu hỏi 2.** GReg và PGMM cải tiến dùng U-Net làm "chuyên gia vùng". Có thể thay bằng phương pháp mới hơn U-Net mà vẫn cải thiện được không?

Để hiểu sâu PGMM và trả lời hai câu hỏi trên, bước đầu tiên em chọn là **cài đặt lại GReg**.

---

## Slide 4: GReg giải quyết vấn đề gì, vì sao cần 3D

- Đầu vào: 1 ảnh người + chuỗi tư thế. Đầu ra: video người đó ra ký hiệu.
- Khó nhất là bàn tay và mặt: nhỏ, nhiều chi tiết, mang phần lớn nghĩa. Ngón tay dính nhau hoặc mặt nhòe là hiểu sai.
- Bộ xương que và đốm sáng ở khớp chỉ cho biết khớp ở đâu, không cho biết độ dày hay ngón nào che ngón nào.
- GReg dùng **SMPL-X** (mô hình người 3D, 10.475 điểm, đủ năm ngón) nên có hình khối thật.
- Hai biểu diễn được dùng:
  - **Bản đồ hướng mặt (normal map)**: màu từng điểm theo hướng bề mặt.
  - **Bản đồ đường viền (gradient map)**: đổi normal map sang xám, rồi mỗi điểm lấy độ sáng trừ điểm bên trái và điểm bên trên (đạo hàm rời rạc). Hiệu càng lớn càng nằm trên biên.
  - Công thức: G = √(Iₓ² + I_y² + ε).

---

## Slide 5: Luồng tổng thể (7 bước)

1. **Đầu vào**: ảnh nguồn, normal map tư thế nguồn và tư thế đích.
2. **Đoán chuyển động**: ra bản đồ luồng F và mặt nạ O (TPSMM, stage 1).
3. **Kéo ảnh nguồn** theo F → ảnh đã kéo.
4. **Chuyên gia vùng** dự đoán đường viền mặt/tay → khung gợi ý R.
5. **Lấy đường viền** từ normal map đích, giữ quanh mặt và tay → mã hóa thành X_g → gửi tới 3 khối GRM.
6. **Vẽ ảnh**: generator nhận [ảnh đã kéo, normal map, R], GRM tinh chỉnh bằng X_g.
7. **Trộn** theo mặt nạ O → khung hình cuối.

Bước 2, 4, 5 độc lập nhau. Bước 6 cần kết quả của 3, 4, 5.

---

## Slide 6: Đoán chuyển động, kéo ảnh và mặt nạ O

- Dùng lại **TPSMM**: 50 điểm mốc chia 10 nhóm × 5 điểm, mỗi nhóm là một phép biến dạng TPS. Trộn lại ra bản đồ luồng F.
- Biến dạng này trơn nên kéo tốt thân và tay áo, **nhưng không theo kịp từng ngón tay**. Ảnh đã kéo thường méo ở bàn tay.
- **Mặt nạ O** (0 đến 1) trả lời "pixel này lấy từ ảnh đã kéo có đáng tin không?". O≈1 dùng lại ảnh cũ, O≈0 phải vẽ mới.
- Bước trộn cuối: **Î = O·(ảnh đã kéo) + (1−O)·(ảnh generator vẽ)**.
  - Ảnh đã kéo giữ chi tiết và danh tính (áo, tóc, nền).
  - Generator chỉ cần vẽ phần bị che hoặc mới lộ, vẽ tốt hơn là vẽ lại cả khung.

---

## Slide 7: Chuyên gia vùng (Regional Expert) và khung gợi ý R

- Mặt và hai tay chỉ chiếm phần nhỏ khung hình, một bộ vẽ chung dễ bỏ qua. GReg cắt ba vùng (mặt, tay trái, tay phải) và giao mỗi vùng cho một mạng riêng.
- Mỗi chuyên gia là một **U-Net nhỏ** dùng conv điều biến theo style (kiểu StyleGAN2). Đầu vào là crop normal map + mã style của ảnh nguồn. Hai tay dùng chung một chuyên gia.
- Đầu ra là bản đồ đường viền của vùng (1 kênh), dán về đúng vị trí trên nền đen → **khung gợi ý R**.
- Huấn luyện bằng **L_struct**: so với đường viền của vùng trong ảnh thật.
- Vì sao cần: lúc suy luận chưa có ảnh đích, nên phải dự đoán đường viền của ảnh RGB thật từ normal map (vốn nhẵn, không có kẽ ngón hay viền môi).

---

## Slide 8: Đường viền, bộ mã hóa viền và khối GRM

- Từ normal map đích tính đường viền, nhân với 3 "quả chuông Gaussian" đặt ở mặt và hai tay (σ) → chỉ còn viền quanh mặt và tay (G_R).
- **Bộ mã hóa viền** (CNN nén dần) biến G_R thành đặc trưng X_g nhiều kênh ở độ phân giải 32/64/128, để khớp không gian đặc trưng của generator.
- **GRM** (cross-attention) nằm trong decoder của generator:
  - X_out = X_in + softmax(Q(X_g)·K(X_in)ᵀ/√C)·V(X_in)
  - Q lấy từ viền, K/V lấy từ đặc trưng generator. Kết quả **cộng** vào X_in nên nếu chưa học được gì thì giữ nguyên.
- Generator là U-Net (skip cộng), 3 GRM ở cỡ 32, 64, 128.
- Lưu ý: ở bước này không có chuyển 3D→2D. Normal map vốn là ảnh 2D chứa thông tin hướng bề mặt.

---

## Slide 9: Cách GReg học

L = λ_per·L_per + λ_r-per·L_r-per + λ_struct·L_struct + λ_adv·L_adv

| Loss | So gì | Để làm gì |
|---|---|---|
| L_per | Cả khung vẽ ra vs thật (qua VGG-19) | Nội dung và kết cấu tổng thể |
| L_r-per | Riêng mặt + hai tay vs thật (VGG-19) | Ép chú ý chi tiết nhỏ |
| L_struct | Viền chuyên gia vùng dự đoán vs viền thật | Dạy chuyên gia vùng |
| L_adv | 3 discriminator (PatchGAN), mỗi cái một vùng, nhìn vùng ảnh kèm viền | Mặt/tay sắc và thật |

Cài đặt trong bài báo: SignAvatars (HamNoSys ~5.800 video + How2Sign), chia 8:2, ảnh 256×256, 100 epoch, Adam 2e-4 (giảm 10 lần ở epoch 70 và 90), 1 GPU RTX 3090. Mỗi video lấy ngẫu nhiên 1 khung nguồn và 1 khung đích.

---

## Slide 10: Kết quả trong bài báo, phần bài báo bỏ trống, kế hoạch cài đặt lại

**Kết quả FVD trên HamNoSys (càng thấp càng tốt):**

| So sánh với phương pháp khác | FVD |
|---|---|
| GReg | 391,3 |
| PGMM | 454,2 |
| NTED | 554,8 |
| SignGAN | 582,4 |

| Ablation | FVD |
|---|---|
| GReg đầy đủ | 391,3 |
| Dùng đốm sáng | 412,3 |
| Dùng xương que | 422,1 |
| Bỏ discriminator | 434,6 |
| Bỏ GRM | 441,4 |
| Bỏ chuyên gia vùng | 458,3 |

Bỏ chuyên gia vùng làm FVD tăng nhiều nhất. Cả 3D (normal map) lẫn từng module đều có đóng góp.

**Bài báo không nêu (phải tự chọn khi cài lại):** kích thước vùng cắt H_r×W_r, σ, bốn trọng số λ, chuyên gia vùng dự đoán chính xác đại lượng nào, kiến trúc chi tiết của RE/bộ mã hóa viền/discriminator, vị trí 3 GRM, và flow có huấn luyện trước hay cùng lúc. Em đã chọn giá trị cho từng mục và ghi trong `docs/decisions.md`.

**Trạng thái cài đặt lại:** code đã hoàn chỉnh, đã chạy thông end-to-end trên CPU với 5 clip thật và 20 test pass. **Chưa huấn luyện đầy đủ trên GPU, chưa chạy AKD và FVD.**

**Bước tiếp theo:** huấn luyện stage 1 (flow) và stage 2 (GReg) trên GPU thuê, chạy sweep ngắn cho các giá trị bài báo bỏ trống, rồi dùng làm nền cho hai câu hỏi nghiên cứu ở Slide 3.

---

*Nguồn tham khảo: bài báo GReg (ICCV 2025, Shi et al.), PGMM (ECCV 2024, arXiv 2409.16709), SignGen (ECCV 2024), SignViP (arXiv 2506.15980), FlashSign (arXiv 2603.27915), SignRefine (arXiv 2609.08496), Text2Sign (arXiv 2607.13164), SignMoD (TPAMI 2026).*
