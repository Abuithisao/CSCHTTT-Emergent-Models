# Kết quả thực nghiệm CEM1D trên CartPole

## 1. Các kịch bản tự thực nghiệm

Notebook [test.ipynb](./test.ipynb) ghi lại hai kịch bản: chạy mô hình CEM1D trên CartPole và thử thêm một bộ điều khiển tuyến tính làm đối chứng.

| Kịch bản | Thiết lập và mục đích |
|---|---|
| **A. CEM1D — tái lập gần với tác giả** | Dùng chuỗi 1D gồm 19 ô; cập nhật 30 vi bước mỗi bước CartPole; huấn luyện 70 thế hệ với quần thể 120 cá thể. Ba hệ số của quy tắc cập nhật được tiến hóa; trạng thái nội tại được giữ lại và kế thừa. Môi trường, cách mã hóa bốn biến quan sát, lực tác động, ngưỡng kết thúc và hàm fitness nhìn chung theo cấu hình bài báo. |
| **B. Bộ điều khiển tuyến tính — mở rộng của tự thực nghiệm** | Dùng bốn trọng số trên cùng bốn biến quan sát đã chuẩn hóa để chọn trái/phải. Bộ điều khiển không có chuỗi ô hay trạng thái nội tại được kế thừa. Giữ 70 thế hệ, quần thể 120, 100 episode mỗi thế hệ và 200 episode đánh giá để làm đối chứng. Đây không phải mô hình CEM1D của tác giả. |

### Điểm giống và khác so với tác giả

**Điểm giống trong kịch bản A:** cấu trúc CEM1D, kích thước chuỗi và số vi bước, cơ chế giữ trạng thái, môi trường CartPole, ngưỡng dừng, số thế hệ và quy mô quần thể. Cấu hình được mô tả trong [Emergent_Models_VN.pdf](./Emergent_Models_VN.pdf) và phần tóm tắt kết quả của [README.md](./README.md).

**Điểm khác cần lưu ý:** notebook dùng **100 rollout mỗi thế hệ và 200 rollout đánh giá**, trong khi cấu hình bài báo ghi **250 rollout huấn luyện và 1.000 rollout đánh giá**. Vì vậy, đây là tái lập cùng phương pháp với ngân sách rollout thấp hơn, không phải bản sao hoàn toàn điều kiện đánh giá. Bài báo/README cũng nêu đây là một lần chạy đại diện, chưa có thống kê nhiều seed.

## 2. Kết quả tự thực nghiệm và đánh giá

### Kịch bản A: CEM1D

Trong 70 thế hệ, return trung bình của cá thể được chọn tăng từ khoảng **331** ở thế hệ đầu lên **781,79 ± 40,86** ở thế hệ 70. Return trung bình toàn quần thể ở thế hệ cuối khoảng **563,67**. Fitness của cá thể tốt nhất dao động đáng kể, nhưng nhìn chung các thế hệ sau đạt hiệu quả cao hơn giai đoạn đầu.

Khi đánh giá cá thể thế hệ 70 trên 200 rollout, kết quả là:

| Chỉ số | Kết quả |
|---|---:|
| Return trung bình | **749,32 / 800** |
| Độ lệch chuẩn return | **137,21** |
| Return cao nhất / thấp nhất | **800 / 54** |
| Tỷ lệ rollout có trạng thái nội tại hợp lệ | **100%** |
| Tỷ lệ hành động ngẫu nhiên trung bình | **22,66%** |

CEM1D đã học được chính sách có thể giữ cột cân bằng lâu trong nhiều lượt chạy, nhưng kết quả chưa ổn định tuyệt đối: độ lệch chuẩn còn lớn và có rollout kết thúc sớm. Return đánh giá cũng thấp hơn return của cá thể được chọn trong một số rollout huấn luyện cuối. Đây là lý do nên thận trọng khi nhận xét khả năng tổng quát hóa từ một lần chạy.

Số liệu chi tiết nằm trong [experiment_record.json](./runs/em1d_20261126_100557/experiment_record.json) và diễn biến theo thế hệ trong [history.csv](./runs/em1d_20261126_100557/history.csv). Đồ thị huấn luyện: [paper_training_fitness.png](./runs/em1d_20261126_100557/paper_training_fitness.png) và [paper_training_return.png](./runs/em1d_20261126_100557/paper_training_return.png).

### Kịch bản B: Bộ điều khiển tuyến tính

Đối chứng tuyến tính đạt **800/800 ở cả 200 rollout đánh giá**, với độ lệch chuẩn bằng 0. Trong lịch sử huấn luyện, cá thể được chọn đạt 800 từ khoảng thế hệ 20 và duy trì đến thế hệ 70.

Trong lần chạy này, bộ điều khiển tuyến tính đơn giản đạt điểm cao và ổn định hơn CEM1D theo thiết lập đánh giá đã dùng. Số liệu nằm trong [experiment_record.json](./runs/linear4_no_retention_20261126_140458/experiment_record.json), còn đồ thị nằm tại [training_curves.png](./runs/linear4_no_retention_20261126_140458/training_curves.png).

### Đối chiếu với kết quả bài báo

README ghi kết quả CEM1D của bài báo là **737,32 ± 104,63** trong huấn luyện và **734,93 ± 107,00** khi đánh giá trên 1.000 rollout. Return đánh giá của lần chạy trong notebook (**749,32 ± 137,21**, 200 rollout) gần với mức trung bình được báo cáo, nhưng độ phân tán cao hơn. Số rollout đánh giá khác nhau và đây chỉ là một lần chạy, nên chưa thể kết luận chênh lệch có ý nghĩa thống kê.

## 3. Kết luận

Tự thực nghiệm đã tái lập được kết quả CEM1D ở mức hiệu năng cao và bổ sung một đối chứng tuyến tính. Trong lần chạy này, đối chứng tuyến tính đạt điểm tối đa ổn định hơn. Kết quả cho thấy CEM1D học được chính sách hữu ích, nhưng chưa chứng minh được rằng nó vượt trội bộ điều khiển đơn giản trên CartPole theo thiết lập đang dùng.

Để so sánh chắc chắn hơn, nên đánh giá cả hai phương pháp trên cùng số rollout lớn hơn và lặp lại nhiều seed. Điều này giúp giảm ảnh hưởng của ngẫu nhiên trong quá trình huấn luyện và đánh giá.
