# BÀI THỰC HÀNH 4: SO SÁNH SỰ TƯƠNG ĐỒNG CỦA CÁC HÌNH ẢNH SỬ DỤNG WAVELET HASH

**Môn học:** Thị giác máy tính (Computer Vision)
**Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên
**Tài liệu tham chiếu:** Slide Bài thực hành chương 3 (Trang 16 - 18)

---

## Tổng quan

Bài tập thực hành tập trung vào việc hiện thực hóa thuật toán **Wavelet Hash (Băm hình ảnh bằng phép biến đổi Wavelet)** nhằm tạo ra một chuỗi nhị phân ngắn gọn đại diện cho đặc trưng cấu trúc của hình ảnh. Hệ thống được thiết kế tích hợp với **Luật Tác tử (Agent Rule)** để tự động ra quyết định về mức độ tương đồng giữa hai ảnh, hỗ trợ ứng dụng trong tìm kiếm ảnh (Image Retrieval) và phát hiện vi phạm bản quyền (Copyright Detection).

---

## Cài đặt thư viện

Để chạy được chương trình, cần cài đặt các thư viện toán học và xử lý ảnh cốt lõi:

    pip install opencv-python numpy PyWavelets matplotlib scikit-learn

---

## Cấu trúc các hàm chính

### **Phần I: Băm hình ảnh (Wavelet Hash Extraction)**

1. `bam_wavelet(duong_dan_anh, wavelet='db4', level=3)`:
   - **Đọc và Tiền xử lý (Preprocessing)**:
     - Sử dụng OpenCV (`cv2.imread`) đọc ảnh thẳng vào mảng mức xám (`cv2.IMREAD_GRAYSCALE`).
     - Chuẩn hóa kích thước (`cv2.resize` về `256x256`): Đảm bảo thuật toán luôn sinh ra mảng Hash có cùng chiều dài bất kể kích thước ảnh gốc, từ đó mới có thể tính được khoảng cách Hamming.
   - **Phân tích Tần số (Wavelet Transform)**:
     - Ứng dụng hàm `pywt.wavedec2` từ thư viện `PyWavelets` để phân rã ảnh theo không gian - tần số.
     - _Nguyên lý_: Hệ thống chỉ trích xuất **dải xấp xỉ tần số thấp (LL - Low-Low)**. Đây là dải giữ lại phần năng lượng và cấu trúc tổng quan nhất của bức ảnh, triệt tiêu đi các nhiễu chi tiết (tần số cao).
   - **Lượng tử hóa Phi tuyến (Quantization)**:
     - Dùng hàm `np.median(LL)` tính giá trị Trung vị.
     - _Tại sao dùng Trung vị thay vì Trung bình (Mean)?_ Trung vị có khả năng kháng các nhiễu biên độ lớn (outliers) cực kỳ hiệu quả (ví dụ như khi ảnh bị vẽ bậy bằng các vệt màu đen đặc trong các file `_challenge.jpg`).
     - Tiến hành băm thành chuỗi nhị phân (Hash code): Gán `1` cho các hệ số lớn hơn trung vị, ngược lại gán `0`.

---

### **Phần II: Đo lường và Ra quyết định (Distance & Agent Rule)**

1. `tinh_khoang_cach_hamming(hash1, hash2)`:
   - **Khoảng cách Hamming (Hamming Distance)**:
     - Tính tổng số bit sai biệt giữa hai mảng nhị phân phẳng bằng biểu thức logic (`np.sum(hash1 != hash2)`).
     - Số bit sai biệt càng nhỏ, cấu trúc không gian của hai ảnh càng giống nhau.

2. `agent_rule(distance, threshold)`:
   - **Luật Tác tử (Agent Gating Decision)**:
     - Đóng vai trò như một màng lọc cứng (Hard thresholding). Hệ thống sử dụng cấu hình ngưỡng tĩnh `threshold = 150`.
     - Nếu khoảng cách <= 150: Tác tử kết luận `1` (Tương đồng - Matched).
     - Nếu khoảng cách > 150: Tác tử kết luận `0` (Khác biệt - Unknown).

---

### **Phần III: Bài tập nâng cao (Benchmarking & Data Augmentation)**

1. `khao_sat_cac_wavelet(data_test, nguong=150)`:
   - **Khảo sát Đa phương pháp**:
     - Hàm tự động quét qua 4 họ Wavelet: `haar`, `db4`, `sym4`, `coif1` trên toàn bộ tập dữ liệu mở rộng (`12 cặp test`).
     - Áp dụng `sklearn.metrics` để tính toán 3 chỉ số hiệu năng chuyên sâu:
       - **Độ chính xác (Accuracy):** Tỷ lệ Tác tử ra quyết định đúng trên toàn cục.
       - **Độ nhạy (Sensitivity):** Tỷ lệ True Positive (nhận diện đúng các ảnh giống nhau dù bị mờ/xoay).
       - **Độ đặc hiệu (Specificity):** Tỷ lệ True Negative (loại trừ đúng các ảnh khác biệt).
     - _Kết luận thực nghiệm_: Daubechies 4 (`db4`) cho ra sự cân bằng tốt nhất giữa khả năng lưu giữ thông tin và kháng nhiễu hình học.

2. Phân tích ROC (Receiver Operating Characteristic):
   - Sử dụng `roc_curve` và `auc` vẽ đường cong ROC để trực quan hóa hiệu suất phân loại của hệ thống tại mọi điểm ngưỡng. Chỉ số AUC (Area Under Curve) được hiển thị trên góc đồ thị.

3. Module Tìm kiếm Hình ảnh (`app_tim_kiem.py`):
   - Xây dựng ứng dụng hoạt động độc lập. Người dùng nhập tên ảnh đầu vào, hệ thống băm ảnh và tính toán khoảng cách chéo với toàn bộ Cơ sở dữ liệu để lọc ra Top 3 bức ảnh giống nhất. Độ phức tạp tìm kiếm tuyến tính O(N).

---

## Cấu trúc thư mục

    Lab4/
    ├── dataset/                     # Thư mục chứa 12 cặp dữ liệu thử nghiệm (ảnh gốc, làm mờ, xoay, vẽ bậy challenge)
    ├── agent_rule.md                # Tài liệu quy tắc Tác tử & Workflow
    ├── Lab4.py                      # Mã nguồn trung tâm: Pipeline Wavelet Hash, Khảo sát đa mô hình & Vẽ ROC
    ├── app_tim_kiem.py              # Ứng dụng tìm kiếm ảnh (Bài tập nâng cao 2)
    ├── tao_du_lieu.py               # Script tự động tăng cường dữ liệu (Data Augmentation)
    └── README.md                    # Báo cáo kỹ thuật tổng hợp

---

## Hướng dẫn chạy chương trình

1. **Khảo sát thuật toán & Vẽ ROC (Bài chính):**

        python Lab4/Lab4.py

   Chương trình sẽ tự động in bảng so sánh hiệu suất lên Terminal và mở cửa sổ biểu đồ ROC. Đóng cửa sổ biểu đồ để kết thúc chương trình.

2. **Chạy ứng dụng tìm kiếm ảnh (Nâng cao):**

        python Lab4/app_tim_kiem.py

   Chương trình sẽ yêu cầu nhập tên file ảnh muốn tìm (ví dụ: `anh1.jpg`), sau đó tự động quét trong thư mục `dataset` và trả về kết quả Top 3 ảnh giống nhất kèm khoảng cách Hamming tương ứng.