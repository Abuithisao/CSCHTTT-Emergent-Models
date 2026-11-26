# Báo cáo tự thực nghiệm GoL-EM

## 1. Mục tiêu và cách tiến hành

Thí nghiệm xem liệu Game of Life (GoL) với luật Conway cố định có thể thực hiện các phép cộng, trừ số nguyên sau khi chỉ tìm kiếm trạng thái ban đầu hay không. Chương trình được biểu diễn bằng vị trí và trạng thái bật/tắt của các glider. Input được mã hóa thành khối sống 2×2 trên đường chéo; mô phỏng dừng khi đạt trạng thái ổn định, sau đó bộ giải mã tìm khối 2×2 hợp lệ ngoài cùng bên trái để đọc output.

Hai notebook dùng cùng cấu hình `g2`: quần thể `P=5000`, hai glider, `Lp=16`, `Ls=6`, `Le=40`, stride `s=2`, giới hạn mô phỏng train `T_max=600`, và tìm kiếm ngẫu nhiên. `Lp=16` là miền tọa độ neo glider trong mã; README mô tả footprint chương trình tương ứng có cạnh khoảng 18 ô.

### Kịch bản 1: huấn luyện một hàm `x + 2`

Trong [test.ipynb](./test.ipynb), tập train là `x=0…9`, với mục tiêu `y=x+2`. Mỗi vòng lấy mẫu 5.000 genome và notebook chạy đủ 8 vòng, tức **40.000 genome được lấy mẫu**. Sau huấn luyện, genome tốt nhất được đánh giá bằng bộ chạy CPU tự mở rộng bàn cờ trên `x=0…102`. Kết quả được ghi trong [runs/](./runs), gồm genome tốt nhất và log random search.

### Kịch bản 2: huấn luyện nhiều hàm riêng biệt

Trong [run_multitask.ipynb](./run_multitask.ipynb), notebook chạy lần lượt sáu bài toán: `x+1`, `x−1`, `x+2`, `x−2`, `x+5`, `x−5`. Mỗi hàm được huấn luyện bằng một lần tìm kiếm riêng, dùng seed 0, tối đa 8 vòng và dừng sớm nếu đạt exact 100% trên train. Mỗi tập train có 10 giá trị; chẳng hạn `x+1` dùng `x=0…9`, còn `x−5` dùng `x=5…14` để output không âm. Các genome và bảng tổng hợp được lưu trong [results_multitask/](./results_multitask).

Trong Python, `np.arange(lo, hi)` không bao gồm `hi`. Vì vậy các khoảng trong mã thực tế có đầu trên loại trừ, dù bảng của bài báo ghi khoảng bằng dấu ngoặc vuông.

## 2. Kịch bản nào giống hoặc khác tác giả?

| Nội dung | Tự thực nghiệm | Bài báo |
|---|---|---|
| Substrate và chương trình | Luật GoL cố định; tìm vị trí hai glider | Cùng cách tiếp cận |
| Tìm kiếm | Random search | Random search được báo cáo; tác giả nhận thấy kết quả gần tương đương genetic algorithm |
| Bài toán chung | `x+2` trong notebook đơn; năm hàm `x±1`, `x±2`, `x+5` trong notebook nhiều bài | Năm hàm `x−2`, `x−1`, `x+1`, `x+2`, `x+5` |
| Bài toán mở rộng | Thêm `x−5` | Không có `x−5` trong bảng kết quả |
| Bài toán chưa thử | Không chạy hàm phụ thuộc chẵn/lẻ | Có thử `x+1` nếu chẵn, `x−1` nếu lẻ |
| Ngân sách tìm kiếm | Notebook đơn: 40.000 genome cho một hàm. Notebook nhiều bài: thực tế 5.000 genome mỗi hàm, 30.000 genome cho sáu hàm | 40.000 genome cho từng hàm cộng/trừ trong bảng kết quả |
| Đánh giá ngoài train | `x=0…102` cho `x+2`; `x=0…59` cho một genome `x+5` | Test được báo cáo đến 200 |

Do đó, notebook đơn gần với lần chạy `x+2` của tác giả về cấu hình và tổng ngân sách genome. Notebook nhiều bài tái hiện năm bài toán cộng/trừ và mở rộng thêm `x−5`, nhưng ngân sách mỗi bài nhỏ hơn. Cả hai chưa đánh giá đầy đủ đến 200 như bài báo.

## 3. Kết quả tự thực nghiệm

### 3.1. Kết quả `x + 2`

Theo [random_search_log.csv](./runs/random_search_log.csv), nghiệm tốt nhất xuất hiện ở vòng 1 với loss khoảng `−0,700`, MAE bằng 0, exact `100%`, invalid `0`, interact `100%` và không có output block thừa. Các vòng còn lại không cải thiện điểm tốt nhất. Genome được lưu tại [best_random_genome.npz](./runs/best_random_genome.npz).

Trên dải `x=0…102`, kết quả đã lưu là **103/103 hợp lệ và chính xác**, exact rate `100%`, valid rate `100%`, MAE bằng 0. Dải này gồm 10 điểm train (`0…9`) và 93 điểm ngoài train (`10…102`); vì toàn bộ 103 điểm đúng nên các điểm ngoài train cũng đều đúng.

### 3.2. Kết quả sáu hàm

Theo [summary.csv](./results_multitask/summary.csv), cả sáu hàm đều đạt exact `100%`, MAE bằng 0 và invalid `0` ngay trong vòng đầu:

| Hàm | Vòng tìm thấy nghiệm | Exact trên train | MAE | Loss |
|---|---:|---:|---:|---:|
| `x+1` | 1 | 100% | 0 | −0,700 |
| `x−1` | 1 | 100% | 0 | −0,700 |
| `x+2` | 1 | 100% | 0 | −0,700 |
| `x−2` | 1 | 100% | 0 | −0,700 |
| `x+5` | 1 | 100% | 0 | −0,700 |
| `x−5` | 1 | 100% | 0 | −0,691 |

Loss của `x−5` cao hơn một chút vì genome có thêm output block và bị phạt ở thành phần `multi`; nó vẫn đúng trên toàn bộ tập train. Đây là sáu lần huấn luyện độc lập, không phải một genome duy nhất học đồng thời sáu hàm.

Ô đánh giá tùy chọn chỉ nạp genome `x_plus_5__g2.npz`. Trên `x=0…59`, genome này đạt 60/60 hợp lệ và chính xác, MAE bằng 0. Trong đó `x=10…59` nằm ngoài train và cũng chính xác. Chưa có output đánh giá dải lớn tương tự cho năm genome còn lại.

## 4. Đối chiếu và đánh giá

Trong mục 3.3.2, bài báo báo cáo năm hàm `x−2`, `x−1`, `x+1`, `x+2`, `x+5` đạt train exact `100%` và test exact `100%`; mỗi bài dùng 40.000 genome và hai glider. Bài báo cũng thử hàm có điều kiện `x+1` nếu chẵn, `x−1` nếu lẻ, đạt 60% train và 50% test sau ngân sách 1 triệu genome. README trong dự án tóm tắt cùng kết luận: các phép `x+k` tổng quát hóa tốt, còn bài toán có điều kiện chẵn/lẻ khó học. [Trang nghiên cứu của tác giả](https://emergentcomputing.org/research/emergent-models) cũng nêu các kết quả này.

Kết quả tự thực nghiệm phù hợp với kết luận về các phép cộng/trừ đơn giản: các genome tìm được dự đoán chính xác trên tập train và các dải kiểm tra đã chạy. Đặc biệt, `x−5` là một mở rộng thành công trong lần chạy này. Tuy nhiên, đây chỉ là các kết quả từ một lần chạy đã lưu: notebook đơn không đặt seed cố định, notebook nhiều bài dùng một seed (`0`), và dải kiểm tra ngắn hơn dải của tác giả. Vì vậy chưa thể kết luận độ ổn định qua nhiều seed hoặc rằng cả sáu hàm đều đạt chính xác đến 200. Bài toán chẵn/lẻ của tác giả cũng chưa được kiểm tra trong hai notebook này.

## Tài liệu và tệp kết quả

- [README.md](./README.md)
- [Emergent_Models_VN.pdf](./Emergent_Models_VN.pdf), mục 3.3.2
- [test.ipynb](./test.ipynb) và [run_multitask.ipynb](./run_multitask.ipynb)
- [runs/](./runs) và [results_multitask/](./results_multitask)
