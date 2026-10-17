# BÀI TẬP THỰC HÀNH 2

**MSSV:** 24520126  
**Họ và tên:** Phan Tuấn Anh

## 1. Giới thiệu

Xây dựng mô hình **Logistic Regression nhị phân** để phân loại ảnh chữ số viết tay MNIST, chỉ sử dụng hai lớp `0` và `1`. Mỗi ảnh có kích thước `28 × 28` pixel, được chuyển thành vector `784` đặc trưng.

## 2. Phương pháp

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

## 3. Cài đặt môi trường

**Yêu cầu:** Python 3.13, VS Code và Python Extension.

Tạo môi trường ảo:

```powershell
python -m venv .venv
```

Cài đặt thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy idx2numpy matplotlib opencv-python seaborn scikit-learn
```

Trong VS Code, chọn Python Interpreter tại `.venv\Scripts\python.exe`.

## 4. Cách chạy chương trình

Chạy mô hình Logistic Regression:

```powershell
.\.venv\Scripts\python.exe assignment.py
```


Chương trình đọc dữ liệu MNIST từ thư mục `data`, lọc hai lớp `0` và `1`, chuẩn hóa pixel về `[0, 1]`, huấn luyện mô hình và đánh giá trên tập test.

## 5. Đánh giá kết quả

Sử dụng các chỉ số Accuracy, Precision, Recall và F1-score.

**Accuracy mẫu: `0.9991` (99,91%).** Kết quả thực tế có thể thay đổi tùy cấu hình và cách triển khai mô hình.

## 6. Lưu ý

Khi phân loại hai chữ số khác, ví dụ `3` và `5`, cần ánh xạ nhãn về `0` và `1` trước khi huấn luyện, sau đó ánh xạ ngược kết quả dự đoán.

Một số lỗi thường gặp:
- `FileNotFoundError`: kiểm tra tên và vị trí các file dữ liệu MNIST.
- `ModuleNotFoundError`: kiểm tra Python Interpreter và cài đặt thư viện trong đúng môi trường ảo.
- Không hiển thị ảnh: chạy chương trình trong môi trường hỗ trợ giao diện đồ họa.
