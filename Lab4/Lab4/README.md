# BÀI THỰC HÀNH 4: SO SÁNH SỰ TƯƠNG ĐỒNG CỦA CÁC HÌNH ẢNH SỬ DỤNG WAVELET HASH

**Môn học:** Thị giác máy tính (Computer Vision)
**Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên
**Tài liệu tham chiếu:** Slide Bài thực hành chương 3 (Trang 16 - 18)

---

## 1. Giới thiệu bài thực hành
Bài thực hành số 4 tập trung vào kỹ thuật **Băm hình ảnh wavelet (Wavelet Hash)** để tạo mã băm nhỏ gọn đại diện cho cấu trúc của hình ảnh. Hệ thống kết hợp với **Luật Tác tử (Agent Rule)** để tự động đánh giá sự tương đồng, hỗ trợ phát hiện vi phạm bản quyền hoặc nhận diện ảnh gốc bị làm nhiễu.

---

## 2. Kiến trúc và Nguyên lý hoạt động

Quy trình xử lý cốt lõi bao gồm 4 giai đoạn chính:

### 2.1. Tiền xử lý và Biến đổi Wavelet (Wavelet Transform)
- Hình ảnh đầu vào được chuyển sang mức xám (Grayscale) và chuẩn hóa kích thước (256x256) nhằm loại bỏ yếu tố màu sắc và đồng nhất độ dài vector đặc trưng.
- Thuật toán áp dụng phân tích ảnh thành các thành phần tần số thông qua thư viện `PyWavelets` (họ `db4`, mức phân giải `level=3`).
- Dải xấp xỉ tần số thấp (LL - Low-Low) được trích xuất để làm mảng đặc trưng, vì dải LL chứa các cấu trúc ổn định nhất và có khả năng kháng nhiễu (Robustness) tốt nhất trước các thay đổi cục bộ.

### 2.2. Lượng tử hóa hệ số (Quantization)
- Trị số trung vị (Median) của mảng LL được sử dụng làm ngưỡng lượng tử hóa thay vì giá trị trung bình, giúp hệ thống ít bị ảnh hưởng bởi các nhiễu biên độ lớn (outliers).
- Các hệ số lớn hơn trung vị được gán giá trị `1`, ngược lại gán giá trị `0`, tạo ra một chuỗi nhị phân (Hash code) phẳng.

### 2.3. Khoảng cách Hamming (Hamming Distance)
Mức độ khác biệt giữa hai ảnh được đo lường bằng khoảng cách Hamming giữa hai mã băm:
- Số lượng các bit khác biệt càng nhỏ, hai hình ảnh càng có cấu trúc tương đồng.

### 2.4. Luật Tác tử (Agent Rule)
Tác tử tự động đối chiếu khoảng cách Hamming với một ngưỡng phân định (Threshold = 150):
- Nếu $\text{Distance} \le 150$: Tác tử gán nhãn `1` (Matched) -> Cảnh báo phát hiện tương đồng.
- Nếu $\text{Distance} > 150$: Tác tử gán nhãn `0` (Unknown) -> Khác biệt.

---

## 3. Cấu trúc thư mục Lab 4

```text
Lab4/
├── dataset/                     # Thư mục chứa dữ liệu test mang tính challenge
│   ├── anh1.jpg                 # Ảnh gốc 1
│   ├── anh1_challenge.jpg       # Ảnh 1 đã bị thêm nhiễu/vẽ bậy
│   ├── anh2.jpg                 # Ảnh gốc 2
│   └── anh2_challenge.jpg       # Ảnh 2 đã bị thêm nhiễu/vẽ bậy
├── Lab4.py                      # Mã nguồn chính chạy Wavelet Hash & Agent Rule
└── README.md                    # Tài liệu báo cáo
```

---

## 4. Giải quyết Bài tập nâng cao
### 4.1. Khảo sát và so sánh các phương pháp băm Wavelet
Trong quá trình phát triển, hệ thống đã chạy thử nghiệm các họ Wavelet khác nhau:
- **Haar Wavelet:** Tốc độ tính toán nhanh nhưng khả năng kháng nhiễu xoay (Rotation invariance) kém, khoảng cách Hamming dao động mạnh khi ảnh bị xoay 90 độ hoặc 180 độ.
- **Daubechies (db4):** Cho sự cân bằng tối ưu. Dải xấp xỉ LL kháng mờ (Gaussian Blur) và nhiễu cực tốt, giữ khoảng cách Hamming ổn định dưới ngưỡng 150 đối với cả ảnh gốc và ảnh challenge. Quyết định chọn `db4` và `level=3` làm core cho Agent.

### 4.2. Ứng dụng tìm kiếm hình ảnh và Data Augmentation
- Đã xây dựng script `tao_du_lieu.py` tự động sinh ảnh xoay góc và làm mờ để tăng cường dataset.
- Đã xây dựng module `app_tim_kiem.py`. Ứng dụng sẽ băm ảnh đầu vào, quét toàn bộ thư mục `dataset` để tính khoảng cách Hamming chéo, sau đó trích xuất ra top 3 bức ảnh có độ tương đồng cao nhất trong cơ sở dữ liệu.
