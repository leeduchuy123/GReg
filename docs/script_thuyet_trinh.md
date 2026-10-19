# Script thuyết trình proposal: SLVG và GReg (10 phút)

Bài nói dài khoảng 1.700 từ, tức khoảng 8 phút 30 giây nếu nói 200 từ mỗi phút. Khoảng 1 phút 20 giây còn lại dành cho thao tác trên slide (rê quả cầu, kéo thanh σ, bấm ô GRM, chờ sơ đồ chạy), nên tổng khoảng 9 phút 50 giây. Phần trong `[ngoặc vuông]` là chỉ dẫn thao tác, không đọc ra.

| Slide | Nội dung | Thời gian |
|---|---|---|
| 1 | Giới thiệu đề tài | 0:30 |
| 2 | Ba hướng nghiên cứu | 1:00 |
| 3 | Ưu nhược điểm, tình trạng code | 1:00 |
| 4 | Vì sao cần hình học 3D | 1:15 |
| 5 | Luồng chạy từ đầu đến cuối | 1:45 |
| 6 | TPSMM và chuyên gia vùng | 0:50 |
| 7 | Gaussian và GRM | 1:00 |
| 8 | Cách GReg học | 0:30 |
| 9 | Hai câu hỏi nghiên cứu | 1:00 |
| 10 | Kết quả và bước tiếp theo | 1:00 |
| | **Tổng** | **9:50** |

Quy ước dùng từ (giữ nhất quán suốt bài nói):
- **Bản đồ hướng mặt** (normal map): ảnh mà màu của mỗi điểm cho biết bề mặt ở điểm đó quay về hướng nào.
- **Đường viền**: chỗ màu thay đổi đột ngột trên bản đồ hướng mặt.
- **Chuyên gia vùng**: mạng nhỏ vẽ riêng mặt hoặc riêng tay.
- **GRM**: khối dùng đường viền để làm sắc đặc trưng của bộ vẽ ảnh.
- **Gaussian**: mặt nạ mềm, lớn nhất ở tâm và giảm dần về 0 khi ra xa.

---

## Slide 1: Giới thiệu (0:30)

Em chào thầy và các bạn. Hôm nay em trình bày đề xuất nghiên cứu về sinh video ngôn ngữ ký hiệu, tiếng Anh gọi là Sign Language Video Generation, viết tắt là SLVG.

Bài toán là: cho một ảnh của một người và chuỗi chuyển động, ta tạo ra video người đó thực hiện ký hiệu.

Đề xuất của em có ba bước: tái lập một baseline, kiểm chứng baseline, rồi mới mở rộng bằng hai câu hỏi nghiên cứu. Baseline em chọn là GReg, một bài của hội nghị ICCV 2025.

[Chuyển slide]

---

## Slide 2: Ba hướng nghiên cứu (1:00)

Sau khi đọc tài liệu, em chia SLVG thành ba hướng. Điểm khác nhau là đầu vào và cách tạo chuyển động.

Hướng A: ảnh tham chiếu cộng pose. Đầu vào là một ảnh người và chuỗi pose cho sẵn. Mô hình giữ ngoại hình của người trong ảnh và vẽ từng khung hình theo pose. Đại diện là PGMM và GReg.

Hướng B: từ văn bản ra video. Đầu vào chỉ là một câu văn bản, không có pose. Hệ thống phải tự biến câu đó thành chuyển động ký hiệu. Đại diện là SignGen và FlashSign.

Hướng C: dùng foundation model, tức mô hình video diffusion đã huấn luyện sẵn. Người ta thêm điều kiện pose và các module nhỏ để sửa những vùng khó như bàn tay. Đại diện là SignRefine.

Em lưu ý một điều: ba hướng này có thể chồng lấn. Hướng C là cách xây bộ sinh, không phải một loại đầu vào riêng.

[Chuyển slide]

---

## Slide 3: Ưu nhược điểm và tình trạng code (1:00)

Slide này so sánh ba hướng. Phần ưu nhược là đánh giá của em dựa trên các bài đã đọc.

Hướng A: ưu điểm là chuyển động được điều khiển rõ, giữ được người trong ảnh, và huấn luyện được trên một GPU. Nhược điểm là cần pose có sẵn, bàn tay và mặt nhỏ nên khó vẽ đúng.

Hướng B: ưu điểm là chỉ cần văn bản. Nhược điểm là khó đúng ngữ nghĩa ký hiệu và tốn tài nguyên.

Hướng C: ưu điểm là hình ảnh chân thực nhờ mô hình có sẵn. Nhược điểm là tốn tài nguyên, bài SignRefine báo cáo dùng bốn GPU.

Dòng cuối là điều quan trọng nhất: tình trạng code. Em kiểm tra ngày 18 tháng 10. Ở hướng A, PGMM và GReg em chưa thấy link code công khai. Em lưu ý là "chưa thấy" khác với "không tồn tại". Vì vậy, muốn làm tiếp hướng A, em phải tự cài đặt lại.

[Chuyển slide]

---

## Slide 4: Vì sao cần hình học 3D (1:15)

Từ đây em đi vào GReg. Câu hỏi đầu tiên: vì sao GReg dùng hình học 3D?

[Chỉ vào ba ảnh bàn tay] Đây là cùng một bàn tay, mô tả theo ba cách. Cách một là bộ xương que: chỉ biết khớp nằm ở đâu, không biết bàn tay dày hay mỏng. Cách hai là đốm sáng ở khớp: các ngón bị hòa vào nhau. Cách ba là bản đồ hướng mặt từ mô hình 3D SMPL-X: thấy được hình khối, ngón nào nằm trước ngón nào.

Vậy bản đồ hướng mặt là gì? [Rê chuột trên quả cầu] Mỗi điểm trên bề mặt quay về một hướng, và hướng đó được tô thành màu. Em rê chuột trên quả cầu: sang phải thì đỏ hơn, lên trên thì xanh lá hơn, hướng thẳng về phía mình thì tím xanh.

Phía dưới là bản đồ đường viền. Đổi bản đồ hướng mặt sang ảnh xám, rồi với mỗi điểm lấy độ sáng trừ độ sáng của điểm bên trái và điểm bên trên. Hiệu càng lớn thì điểm đó càng nằm trên đường viền.

Một lưu ý: không có phép chiếu 3D sang 2D ở đây. Bản đồ hướng mặt đã là ảnh 2D.

[Chuyển slide]

---

## Slide 5: Luồng chạy từ đầu đến cuối (1:45)

[Để sơ đồ tự chạy; chỉ vào từng phần khi nó sáng lên]

Đây là toàn bộ GReg. Mỗi chấm sáng là một gói dữ liệu, màu của chấm cho biết loại dữ liệu.

Bước 1: đầu vào có ba thứ. Ảnh nguồn của người cần vẽ, bản đồ hướng mặt của tư thế cũ, và bản đồ hướng mặt của tư thế mới.

Bước 2: so hai bản đồ hướng mặt để đoán chuyển động. Kết quả là bản đồ luồng F, cho biết mỗi điểm đi từ đâu tới đâu, và mặt nạ O, cho biết chỗ nào của ảnh cũ còn dùng được.

Bước 3: kéo ảnh nguồn theo luồng F. Ảnh kéo giữ đúng màu da và màu áo, nhưng sai ở bàn tay.

Bước 4: chuyên gia vùng. GReg cắt ba vùng là mặt, tay trái, tay phải, và mỗi vùng đi qua một mạng nhỏ để tạo khung gợi ý R.

Bước 5: tính đường viền của bản đồ hướng mặt mới, chỉ giữ phần quanh mặt và tay, rồi mã hóa thành đặc trưng.

Bước 6: bộ vẽ ảnh nhận ảnh kéo, bản đồ hướng mặt và khung gợi ý R. Ba khối GRM trong bộ vẽ ảnh dùng đường viền để làm sắc mặt và tay.

Bước 7: trộn. Chỗ nào O nói ảnh cũ dùng được thì giữ ảnh kéo. Chỗ nào bị che thì dùng ảnh vừa vẽ. Làm lại cho mọi tư thế là có video.

[Chuyển slide]

---

## Slide 6: TPSMM và chuyên gia vùng (0:50)

Hai phần này là bước 2, 3 và 4 vừa rồi, em nói kỹ hơn.

Phần đoán chuyển động dùng lại mô hình TPSMM. Nó đặt 50 điểm mốc, chia 10 nhóm, mỗi nhóm một kiểu uốn cong ảnh. Kiểu uốn này trơn và mềm, nên kéo tốt thân áo và cánh tay, nhưng không theo kịp từng ngón tay. Vì vậy ảnh kéo sai ở bàn tay, và các phần sau phải sửa.

[Chỉ vào hình bên dưới] Mặt nạ O đen quanh bàn tay, nghĩa là ở đó phải vẽ mới.

Phần sửa là chuyên gia vùng. Mặt và tay chiếm một phần nhỏ của khung hình, nên một bộ vẽ chung dễ bỏ qua. GReg giao mỗi vùng cho một mạng U-Net nhỏ. Mạng nhận vùng cắt từ bản đồ hướng mặt và một mã tóm tắt ngoại hình từ ảnh nguồn. Đầu ra là đường viền dự đoán của vùng đó, và đây chính là mạng mà câu hỏi nghiên cứu thứ hai nhắm tới.

[Chuyển slide]

---

## Slide 7: Gaussian và GRM (1:00)

Slide này nói về đường viền đi vào bộ vẽ ảnh.

Đường viền tính từ bản đồ hướng mặt có viền của mọi thứ: vai, áo, nền, thân người. Viền mặt và tay chỉ là phần nhỏ, nên tín hiệu bị loãng. Vì vậy GReg đặt một Gaussian ở tâm mỗi vùng, lớn nhất ở tâm và giảm dần về 0 khi ra xa. Nhân đường viền với Gaussian cho ra ba ảnh, mỗi ảnh chỉ còn rõ viền ở một vùng.

Độ rộng σ của Gaussian thì bài báo không cho. [Kéo thanh trượt σ] σ nhỏ thì chỉ còn lòng bàn tay, σ lớn thì viền vai và áo lọt vào. Con số này em phải tự chọn bằng thí nghiệm.

Sau đó là khối GRM. Mỗi điểm viền đặt một câu hỏi: những chỗ nào trong bản phác của bộ vẽ ảnh liên quan tới tôi? Rồi GRM lấy thông tin từ các chỗ đó cộng vào bản phác. [Bấm một ô trên bàn tay] Ô trên viền bàn tay chú ý tới các ô viền khác của cùng bàn tay. Phép cộng đảm bảo nếu GRM chưa học được gì thì bản phác giữ nguyên.

[Chuyển slide]

---

## Slide 8: Cách GReg học (0:30)

Khi huấn luyện, GReg vẽ thử một khung hình rồi so với khung hình thật. Có bốn phép so, cộng lại theo bốn trọng số.

Thứ nhất, so cả khung hình qua mạng VGG-19, để ảnh tổng thể đúng nội dung và kết cấu. Thứ hai, so riêng mặt và hai tay, để mô hình chú ý chi tiết nhỏ. Thứ ba, so đường viền của chuyên gia vùng với đường viền thật, để dạy chuyên gia vùng. Thứ tư, ba mạng chấm thật hay giả cho mặt và hai tay, để ảnh sắc và trông thật.

Bài báo không cho giá trị bốn trọng số này. Em cũng phải chọn bằng thí nghiệm.

[Chuyển slide]

---

## Slide 9: Hai câu hỏi nghiên cứu (1:00)

Sau khi đã thấy GReg hoạt động thế nào, em xin nêu hai câu hỏi nghiên cứu. Do hạn chế phần cứng, em chọn hướng A, tập trung vào PGMM và GReg.

Câu hỏi thứ nhất: nếu biểu diễn bàn tay bằng mesh hoặc đồ thị 3D theo không gian và thời gian, thì độ chính xác của ngón tay và tính nhất quán của video có tốt hơn không?

Câu hỏi thứ hai: GReg dùng mạng U-Net làm chuyên gia vùng. Nếu thay U-Net bằng một kiến trúc mới hơn, chi tiết bàn tay và mặt có tốt hơn không?

Để trả lời được, em đi theo ba bước: tái lập GReg, kiểm tra các độ đo, rồi làm ablation có kiểm soát. Ablation nghĩa là bỏ hoặc thay từng phần để xem phần đó đóng góp bao nhiêu.

Em đặt một nguyên tắc: chưa tái lập được baseline thì chưa tuyên bố là cải thiện. Mọi so sánh phải dùng cùng dữ liệu, cùng độ phân giải, cùng seed và cùng cách đánh giá.

[Chuyển slide]

---

## Slide 10: Kết quả và bước tiếp theo (1:00)

Đây là kết quả bài báo báo cáo, đo bằng FVD. FVD càng thấp càng tốt. Trên bộ HamNoSys, GReg đạt 391,3, PGMM đạt 454,2. Số của GReg thấp hơn các phương pháp trước. Em nhấn mạnh đây là số liệu từ bài báo, cần đối chiếu với bảng gốc.

Bảng bên phải cho thấy từng phần đóng góp bao nhiêu. Bỏ chuyên gia vùng làm FVD tăng nhiều nhất, từ 391,3 lên 458,3. Điều này giải thích vì sao câu hỏi thứ hai của em nhắm vào chuyên gia vùng.

Về tiến độ: em đã cài đặt xong GReg. Toàn bộ quy trình chạy được từ đầu đến cuối trên CPU với 5 clip thật, và 20 bài kiểm tra đều đạt. Nhưng em chưa huấn luyện đầy đủ trên GPU, và chưa chạy AKD và FVD. Vì vậy em chưa có số liệu tái lập để báo cáo.

Bước tiếp theo: huấn luyện giai đoạn một về chuyển động, rồi giai đoạn hai là GReg. Sau đó thử các giá trị bài báo bỏ trống như σ và các trọng số, rồi đánh giá bằng AKD và FVD. Cuối cùng dùng baseline này để kiểm tra hai câu hỏi nghiên cứu.

Em xin cảm ơn thầy và các bạn. Em sẵn sàng nhận câu hỏi.

---

## Câu hỏi có thể gặp (chuẩn bị sẵn)

**Vì sao chưa có kết quả huấn luyện?** Máy của em không có GPU. Em dự định thuê GPU trên vast.ai. Với mức chi phí em ước tính, tổng khoảng 16 đến 22 đô la cho cả quy trình, nhưng đây là ước tính thô, em sẽ đo tốc độ thực tế trước.

**Vì sao không làm hướng B hoặc C?** Hướng B cần giải thêm bài toán dịch văn bản sang ký hiệu, và hướng C đòi hỏi tài nguyên lớn hơn nhiều so với phần cứng hiện có.

**Em có chắc chuyên gia vùng dự đoán đường viền của ảnh thật không?** Bài báo không nói rõ. Đó là cách đọc của em, được ghi trong tài liệu quyết định. Em sẽ kiểm tra bằng ablation, vì code đã có sẵn tùy chọn so sánh.

**SignViP có giúp thay U-Net không?** Không. SignViP cũng dùng U-Net. Em chỉ lấy ý tưởng đưa ngoại hình vào bằng attention và thêm lớp thời gian sau.
