# BÀI TẬP THỰC HÀNH 2

**MSSV:** 24520126\
**Họ và tên:** Phan Tuấn Anh

## 1. Giới thiệu

Xây dựng mô hình phân loại ảnh chữ số viết tay MNIST:

- **Bài 1:** Logistic Regression bằng NumPy, phân loại hai lớp `0` và `1`.
- **Bài 2:** Softmax Regression bằng NumPy, phân loại 10 chữ số `0`–`9`.
- **Bài 3:** Logistic Regression và Softmax Regression bằng Scikit-learn.

Mỗi ảnh có kích thước `28 × 28` pixel, được chuyển thành vector `784` đặc trưng và chuẩn hóa pixel về `[0, 1]`.

## 2. Phương pháp

### Logistic Regression bằng NumPy

Mô hình sử dụng hàm sigmoid để tính xác suất:

```text
P(Y = 1 | X) = sigmoid(XW + b)
```

Quy tắc dự đoán: xác suất lớn hơn `0.5` thì thuộc lớp `1`, ngược lại thuộc lớp `0`.

Quá trình huấn luyện sử dụng:

- Binary Cross-Entropy Loss.
- Gradient Descent.
- Learning rate: `0.1`.
- Số epoch: `100`.

### Softmax Regression bằng NumPy

Mô hình tính điểm cho 10 lớp rồi chuyển thành xác suất:

```text
Z = XW + b
P(Y = c | X) = exp(Z_c) / sum_j(exp(Z_j))
```

Chọn lớp có xác suất lớn nhất làm kết quả dự đoán. Trừ điểm lớn nhất của từng ảnh trước khi tính hàm mũ để tránh tràn số.

Quá trình huấn luyện sử dụng:

- Multiclass Cross-Entropy Loss.
- Gradient Descent trên toàn bộ tập train.
- Learning rate mặc định: `0.5`.
- Số epoch mặc định: `200`.

Mô hình được xây dựng bằng NumPy; Scikit-learn chỉ dùng để in báo cáo đánh giá. Loss được lưu trong `loss_history` và in mỗi 10 epoch. Phiên bản hiện tại không vẽ hoặc lưu biểu đồ.

### Sử dụng Scikit-learn

Sử dụng `sklearn.linear_model.LogisticRegression` với solver `lbfgs` và `max_iter=1000`:

- Logistic Regression nhị phân cho hai lớp `0`, `1`.
- Softmax Regression với multinomial cross-entropy cho 10 lớp `0`–`9`.

## 3. Cài đặt môi trường

**Yêu cầu:** Python 3.13, VS Code và Python Extension.

Tạo môi trường ảo:

```powershell
python -m venv .venv
```

Cài đặt thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy idx2numpy matplotlib scikit-learn
```

Trong VS Code, chọn Python Interpreter tại `.venv\Scripts\python.exe`.

Thư mục `data` cần chứa bốn file:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

## 4. Cách chạy chương trình

Chạy Logistic Regression bằng NumPy (`assignment.py` sử dụng mô hình trong `logistic_regression.py`):

```powershell
.\.venv\Scripts\python.exe assignment.py
```

Chạy Softmax Regression bằng NumPy:

```powershell
.\.venv\Scripts\python.exe softmax_regression.py
```

Có thể điều chỉnh tham số:

```powershell
.\.venv\Scripts\python.exe softmax_regression.py --epochs 100 --lr 0.1
```

Chạy hai mô hình bằng Scikit-learn:

```powershell
.\.venv\Scripts\python.exe sklearn_assignment.py
```

File Scikit-learn in hai classification report theo thứ tự: Logistic Regression cho `0`, `1`, sau đó Softmax Regression cho `0`–`9`.

Các chương trình đọc MNIST từ thư mục `data`, chuẩn hóa pixel, huấn luyện trên tập train và đánh giá trên tập test. Hai file mới hỗ trợ `--train-limit` để giới hạn số mẫu train khi chạy thử; mặc định sử dụng toàn bộ dữ liệu.

## 5. Đánh giá kết quả

Sử dụng các chỉ số Accuracy, Precision, Recall và F1-score. Classification report hiển thị kết quả từng lớp, macro average và weighted average với bốn chữ số thập phân.

**Accuracy mẫu của Logistic Regression NumPy cho `0`, `1`: `0.9991` (99,91%).** Kết quả thực tế có thể thay đổi tùy cấu hình và cách triển khai mô hình.

## 6. Lưu ý

- Đặt đầy đủ bốn file dữ liệu MNIST đúng tên trong thư mục `data` trước khi chạy chương trình.
- Sử dụng Python trong môi trường `.venv` và cài thư viện bằng chính interpreter đó.
- Logistic Regression sử dụng hai lớp `0`, `1`; Softmax Regression sử dụng đủ 10 lớp từ `0` đến `9`.
- Chỉ sử dụng tập train để huấn luyện; các chỉ số trong classification report được tính trên tập test.
- Tham số `--train-limit` chỉ dùng để chạy thử nhanh. Khi đánh giá kết quả chính thức, bỏ tham số này để huấn luyện trên toàn bộ tập train.
- Nếu Scikit-learn báo chưa hội tụ, tăng số vòng lặp bằng tham số `--max-iter` khi chạy `sklearn_assignment.py`.
