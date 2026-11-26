# Mô tả quá trình tự thực nghiệm

Phần này tổng hợp quá trình chạy trong [`test.ipynb`](./test.ipynb), đối chiếu với lịch sử tại [`runs/run_20261126_174520`](./runs/run_20261126_174520), và phân biệt kết quả tự thực nghiệm với kết quả tác giả trình bày trong [`README.md`](./README.md) và mục 3.6.1 của [`Emergent_Models_VN.pdf`](./Emergent_Models_VN.pdf).

## 1. Các kịch bản tự thực nghiệm

Notebook ghi lại một lần chạy chính gồm: khởi tạo cấu hình và giao diện GoL, chạy các kiểm tra tự động, huấn luyện bộ điều khiển CartPole, đánh giá genome tốt nhất và tạo video minh họa. Các thiết lập huấn luyện chính trùng với cấu hình tác giả nêu trong README: quần thể 100 genome, 10 thế hệ, 5 rollout huấn luyện, 20 rollout đánh giá, ngân sách GoL tối đa 300 bước mỗi lần điều khiển và giới hạn 500 bước CartPole mỗi episode.

| Kịch bản | Nội dung và đối chiếu với tác giả |
|---|---|
| Huấn luyện bộ điều khiển GoL–CartPole | **Giống tác giả.** Dùng cùng thiết kế mô hình, cách mã hóa trạng thái và giải mã lực điều khiển, thuật toán di truyền chỉ dùng đột biến, cùng các tham số huấn luyện chính. Lịch sử chạy ghi đủ 10 thế hệ. |
| Kiểm tra mô phỏng và giao diện | Notebook chạy `run_self_tests` và đầu ra ghi “All self-tests passed.” Đây là kiểm tra độ nhất quán của các thành phần, không phải một kịch bản điều khiển CartPole riêng. |
| Đánh giá genome tốt nhất trên 20 rollout | Notebook gọi hàm đánh giá với cấu hình 20 rollout, tương tự cách tác giả báo cáo kết quả đánh giá. Tuy nhiên, đầu ra của `eval_report` không được lưu/hiển thị trong notebook hiện có nên không thể xác nhận thống kê của lần chạy này. |
| Tạo video một episode | Notebook bật `MAKE_VIDEOS=True` và tạo video GoL cùng CartPole. Đây là bước minh họa bổ sung. Episode được dùng để tạo video dài 176 bước. |
| Các kịch bản mở rộng | Không thấy bằng chứng trong các tệp được cung cấp về baseline ngẫu nhiên do chính lần chạy này, nhiều lần chạy với seed khác nhau, hay thí nghiệm giữ trạng thái GoL giữa các bước. README nhắc rằng tác giả đã thử giữ trạng thái và bộ điều khiển khi đó mất ổn định; notebook hiện tại dùng cách reset bàn cờ ở mỗi bước điều khiển. |

## 2. Kết quả thực nghiệm và đánh giá

### 2.1. Kết quả huấn luyện

Theo [`history.csv`](./runs/run_20261126_174520/history.csv), điểm CartPole tốt nhất tăng từ **128,2** ở thế hệ 1 lên **417,2** ở thế hệ 10. Điểm CartPole trung bình của quần thể ở thế hệ 10 là **171,36**. Điểm thô tốt nhất tại thế hệ 10 là **345,32**, còn trung bình là **144,09**. Trung bình quần thể nhìn chung tăng qua các thế hệ, cho thấy quá trình tiến hóa tìm được các bộ điều khiển có kết quả tốt hơn; tuy nhiên, điểm tốt nhất theo từng thế hệ dao động và không tăng đều.

Các số tổng kết ở thế hệ 10 trong CSV trùng với các số tác giả nêu trong README: điểm tốt nhất 417,20 và trung bình 171,36. Sự trùng khớp này cho thấy kết quả lưu trong thư mục chạy khớp với kết quả tham chiếu ở các trường được ghi. Tuy nhiên, một lần chạy không đủ để đánh giá độ ổn định khi thay đổi seed hoặc lặp lại thí nghiệm độc lập.

### 2.2. Tỷ lệ GoL không hội tụ và episode video

Ở thế hệ 10, tỷ lệ GoL không hội tụ của genome tốt nhất là khoảng **57,5%**; tỷ lệ trung bình của quần thể là khoảng **38,3%**. Như vậy, bộ điều khiển đạt điểm CartPole khá cao nhưng nhiều lần GoL vẫn chưa dừng ở trạng thái ổn định trong ngân sách đã đặt. Đây cũng là hạn chế được README của tác giả đề cập.

Đầu ra của bước tạo video ghi một episode dài **176 bước**, điểm thô **144,5**. Đây là một rollout riêng dùng để dựng video, không phải kết quả trung bình 20 episode. Vì vậy không nên so trực tiếp số này với kết quả đánh giá của tác giả là **316,90 ± 121,52 trên 20 episode**. Notebook có gọi hàm đánh giá 20 rollout, nhưng báo cáo đó không xuất hiện trong đầu ra hiện có; cần lưu hoặc in `eval_report` để đối chiếu trực tiếp.

### 2.3. Nhận xét chung

Lần tự thực nghiệm bám sát kịch bản và tham số chính của tác giả. Lịch sử huấn luyện khớp với các số tổng kết tác giả công bố trong README và mục 3.6.1 của PDF. Kết quả cho thấy bộ điều khiển học được cách giữ thăng bằng lâu hơn baseline ngẫu nhiên theo mô tả của tác giả, nhưng tỷ lệ GoL không hội tụ cao và episode video chỉ đạt 176 bước cho thấy hiệu quả còn biến động. Do chưa có báo cáo đánh giá 20 episode được lưu và chưa có nhiều lần chạy với seed khác nhau, chưa thể kết luận độ ổn định của kết quả tự thực nghiệm.
