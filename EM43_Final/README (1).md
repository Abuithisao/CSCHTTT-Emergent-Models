# EM43_Final — Bộ thực nghiệm EM43

## 1. Mục đích

Thư mục này lưu notebook huấn luyện/đánh giá, các genome `.npz`, kết quả CSV và hình trực quan hóa cho sáu bài toán EM43. Các kết quả trong `Results/` được lấy từ các CSV có sẵn trong bộ ZIP người dùng gửi; `EM43_run_summary.csv` là bảng tổng hợp các CSV đó.

## 2. Thứ tự bài toán

| STT | Alias | Hàm mục tiêu | Số run | Số thế hệ cấu hình | Lp | Tổng tham số (60 LUT + Lp) |
|---:|---|---|---:|---:|---:|---:|
| 01 | `plus1` | `x + 1` | 8 | 20 | 10 | 70 |
| 02 | `times2` | `2x` | 8 | 90 | 10 | 70 |
| 03 | `round_div3` | `round(x/3)` | 8 | 400 | 10 | 70 |
| 04 | `mod4` | `x mod 4` | 8 | 150 | 10 | 70 |
| 05 | `parity_scaled` | `x * (1 + x mod 2)` | 2 | 500 | 10 | 70 |
| 06 | `multiply_ab` | `a * b` | 2 | 800 | 50 | 110 |

Các giá trị số thế hệ và Lp ở bảng lấy từ notebook huấn luyện trong bundle. Số tham số được tính theo 60 mục LUT có thể học + Lp ô chương trình.

## 3. Cấu trúc thư mục

- `Colab Notebooks/trainer/`: notebook huấn luyện 6 bài toán.
- `Colab Notebooks/evaluator/`: notebook đánh giá; `evaluator_plus1.ipynb` được bổ sung trong gói này.
- `Models/<task_alias>/`: genome `.npz` và CSV log/trajectory được lưu từ quá trình train. CSV log có thể được chuyển sang `Results/Training_Logs/` trong lần dọn thư mục tiếp theo; không xóa chúng.
- `Results/`: CSV extrapolation, runtime và probe.
- `Figures/`: các hình space-time, accuracy, runtime và input extension.
- `EM43_run_summary.csv`: tổng quan sáu bài toán, lấy từ các file extrapolation result hiện có.

## 4. Kết quả đánh giá hiện có

Năm bài toán một đầu vào (`plus1`, `times2`, `round_div3`, `mod4`, `parity_scaled`) có kết quả exact-match 100% trên các CSV hiện có. Bốn bài toán đầu có 8 run mỗi bài; `parity_scaled` có 2 run. Tập extrapolation một đầu vào là `x = 31..1000`, tương ứng 970 mẫu cho mỗi run.

`multiply_ab` hiện có 2 run và CSV đánh giá dùng **7 cặp probe đại diện**. Cả hai run đạt 0/7 exact-match trên probe đó. Đây là một tập probe nhỏ, không phải bằng chứng đánh giá exhaustively trên toàn bộ miền; không nên trình bày 0% này như accuracy trên một test grid đầy đủ hoặc như kết quả tái lập trực tiếp Bảng 1 của bài báo.

## 5. Cách chạy evaluator plus1

1. Mở `Colab Notebooks/evaluator/evaluator_plus1.ipynb` bằng Google Colab.
2. Đặt thư mục dự án trên Google Drive với cấu trúc `EM43_Final/Models/plus1/` và chứa đủ 8 file `best_em43_plus1_run_001.npz` đến `...008.npz`.
3. Ở ô Setup, xác nhận `PROJECT_ROOT = /content/drive/MyDrive/EM43_Final` đúng với vị trí thư mục của bạn. Nếu tên/vị trí khác, sửa dòng này.
4. Chạy các cell từ trên xuống. Notebook chỉ load model và đánh giá trên `x = 31..1000`; không huấn luyện lại.
5. Kết quả được lưu vào `Results/plus1_extrapolation_results.csv`; hình được lưu vào `Figures/`.

## 6. Lưu ý về khả năng tái lập

- Không xóa các file `.npz`, CSV run log hoặc trajectory nếu chưa lưu bản sao.
- Không thay đổi tên alias của thư mục model nếu notebook còn tham chiếu alias đó.
- Các notebook huấn luyện có thể đang ghi file vào đường dẫn Google Drive cũ `/content/drive/MyDrive/EM43`; khi chuyển cấu trúc thư mục, cần kiểm tra lại `DRIVE_ROOT` trước khi chạy train.
- Gói này bổ sung README, bảng tổng hợp và evaluator plus1; notebook evaluator plus1 mới được tạo để hoàn thiện luồng đánh giá, chưa có execution output trong notebook. CSV `plus1_extrapolation_results.csv` đã tồn tại trong bộ dữ liệu ban đầu và được giữ nguyên.
