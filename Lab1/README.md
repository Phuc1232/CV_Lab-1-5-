# Lab 1: Các thao tác cơ bản với hình ảnh (Image Basics in OpenCV & Pillow)

## Tổng quan

Bài tập thực hành tìm hiểu và hiện thực hóa các kỹ thuật nền tảng trong xử lý ảnh số, sử dụng hai thư viện cốt lõi là **OpenCV** (`cv2`) và **Pillow** (`PIL`) trên nền tảng Python.

---

## Cài đặt thư viện

Để chạy được chương trình, cần cài đặt các thư viện phụ thuộc:

```bash
pip install opencv-python pillow
```

---

## Cấu trúc các hàm chính

### **Phần I: Đọc, hiển thị và lưu ảnh**

1. `doc_va_hien_thi_anh(duong_dan_anh)`:
   - **Đọc ảnh**:
     - **OpenCV (`cv2.imread`)**: Đọc ảnh vào mảng nhiều chiều NumPy (`numpy.ndarray`). Mặc định thứ tự các kênh màu là **BGR** (Blue - Green - Red).
     - **Pillow (`Image.open`)**: Đọc ảnh thành đối tượng ảnh Pillow (`PIL.Image`). Mặc định thứ tự kênh màu là **RGB** (Red - Green - Blue).
   - **Kiểm tra thông số ảnh**:
     - Chiều cao (`height`), chiều rộng (`width`), số kênh màu (`channels`).
     - Đảm bảo an toàn khi kiểm tra số kênh màu (`len(shape) > 2`) để tránh lỗi `IndexError` đối với ảnh đơn sắc (grayscale).
   - **Lưu ảnh sang định dạng khác**:
     - Chuyển đổi và lưu định dạng ảnh gốc sang `.png` bằng OpenCV và `.jpg` bằng Pillow.
     - Sử dụng đường dẫn tương đối động (`os.path.dirname(__file__)`) để đảm bảo ảnh được lưu chính xác trong thư mục bài làm trên mọi hệ điều hành.

---

### **Phần II: Chuyển đổi không gian màu**

1. `chuyen_doi_khong_gian_mau(anh)`:
   - **1. Grayscale (Mức xám)**:
     - Hàm sử dụng: `cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)`.
     - _Nguyên lý_: Mắt người nhạy cảm với màu xanh lá nhất và ít nhạy với xanh dương nhất. Công thức chuẩn quốc tế (ITU-R BT.601) được áp dụng:
       $$Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$$
     - Do ảnh OpenCV là BGR, việc dùng cờ `COLOR_BGR2GRAY` sẽ tự động gán đúng trọng số $0.114$ cho Blue và $0.299$ cho Red mà không cần đổi sang RGB trung gian.
   - **2. Không gian màu HSV (Hue - Saturation - Value)**:
     - Hàm sử dụng: `cv2.cvtColor(anh, cv2.COLOR_BGR2HSV)`.
     - _Bản chất 3 kênh_:
       - **H (Hue - Sắc thái màu)**: Góc màu trên vòng tròn màu ($0^\circ \rightarrow 360^\circ$). Trong kiểu dữ liệu 8-bit (`uint8`), OpenCV chia đôi dải này thành $[0, 179]$.
       - **S (Saturation - Độ no màu/bão hòa)**: Trực quan hóa độ đậm nhạt của màu, nhận giá trị trong đoạn $[0, 255]$.
       - **V (Value - Cường độ sáng)**: Thể hiện độ sáng của điểm ảnh trong đoạn $[0, 255]$.
     - _Ứng dụng_: Rất quan trọng trong các bài toán phân đoạn vật thể theo màu sắc (Color Segmentation) hoặc bám theo màu (Color Tracking), vì màu sắc ($H$) không bị phụ thuộc vào điều kiện ánh sáng môi trường ($V$).
     - _Lưu ý khi hiển thị_: Do `cv2.imshow` mặc định coi ma trận pixel là BGR, nên việc hiển thị ảnh HSV trực tiếp sẽ xuất hiện bảng màu biến dị (pseudo-color).
   - **3. Không gian màu LAB (CIE L*a*b*)**:
     - Hàm sử dụng: `cv2.cvtColor(anh, cv2.COLOR_BGR2LAB)`.
     - _Cấu trúc_: Kênh **L** (Độ sáng - Lightness), kênh **A** (Trục màu Lục - Đỏ), kênh **B** (Trục màu Lam - Vàng). Không gian này được thiết kế xấp xỉ thị giác con người và độc lập với thiết bị hiển thị.

---

### **Phần III: Cắt xén và Thay đổi kích thước theo tỷ lệ**

1. `cat_xen_va_thay_doi_kich_thuoc(anh, ti_le=0.5)`:
   - **Cắt xén ảnh (Cropping / Slicing)**:
     - Sử dụng kỹ thuật mảng cắt lát (Array Slicing) trong NumPy: `anh[y1:y2, x1:x2]`.
     - Cắt vùng trung tâm 50% ảnh gốc một cách linh hoạt theo kích thước thực tế của bức ảnh.
   - **Thay đổi kích thước theo tỷ lệ (Scale Factor Resize)**:
     - Hàm sử dụng: `cv2.resize(anh, None, fx=ti_le, fy=ti_le, interpolation=cv2.INTER_AREA)`.
     - _Tại sao nên chọn resize theo tỷ lệ thay vì ép kích thước cố định (ví dụ 500x400)?_
       - **Bảo toàn tỷ lệ khung hình (Aspect Ratio)**: Ép cứng cả chiều rộng và chiều cao sẽ làm méo hình nếu ảnh gốc không trùng tỷ lệ.
       - **Tránh vỡ ảnh khi ảnh nhỏ**: Nếu ảnh đầu vào nhỏ hơn kích thước cố định, việc phóng to sẽ làm vỡ nét (pixelation). Ngược lại, resize theo tỷ lệ đảm bảo ảnh luôn giữ đúng hình dạng tự nhiên của vật thể.
     - _Kỹ thuật nội suy (Interpolation)_: Sử dụng `cv2.INTER_AREA` khi thu nhỏ ảnh để đạt chất lượng khử răng cưa tốt nhất.

---

### **Phần IV: Vẽ hình cơ bản và Thêm văn bản**

1. `ve_hinh_va_them_van_ban(anh)`:
   - **Tọa độ tương đối linh hoạt (Relative Coordinates)**:
     - Thay vì gán cứng các tọa độ pixel cố định (dễ bị văng ra ngoài khung hình nếu ảnh nhỏ), toàn bộ tọa độ và bán kính được tính theo tỷ lệ chiều cao (`cao`) và chiều rộng (`rong`) của bức ảnh.
     - Độ dày nét vẽ (`thickness`) và kích cỡ phông chữ (`font_scale`) được co giãn thích ứng (adaptive) theo độ phân giải ảnh.
   - **Các đối tượng vẽ**:
     - **Đường thẳng (`cv2.line`)**: Kẻ đường ngang ở 10% chiều cao bức ảnh (Màu xanh dương: `(255, 0, 0)`).
     - **Hình chữ nhật (`cv2.rectangle`)**: Đóng khung vùng trung tâm ảnh (Màu xanh lá: `(0, 255, 0)`).
     - **Hình tròn (`cv2.circle`)**: Vẽ hình tròn tại tâm ảnh (Màu đỏ: `(0, 0, 255)`).
     - **Văn bản (`cv2.putText`)**: Viết chữ `"OpenCV & Python"` ở mép dưới ảnh (Màu trắng: `(255, 255, 255)` với phông chữ `FONT_HERSHEY_SIMPLEX` và khử răng cưa `cv2.LINE_AA`).

---

## Hướng dẫn chạy chương trình

1. Chạy trực tiếp từ dòng lệnh:
   ```bash
   python Lab1/Lab1.py
   ```
2. Khi chương trình hỏi đường dẫn ảnh:
   - **Cách 1**: Nhấn phím `Enter` để chương trình tự động sử dụng ảnh mẫu có sẵn (`anh.jpg`).
   - **Cách 2**: Nhập đường dẫn bất kỳ đến tệp ảnh trên máy tính của bạn.
3. Nhấn phím bất kỳ trên cửa sổ ảnh để xem tuần tự kết quả từng phần (Phần 1 $\rightarrow$ Phần 2 $\rightarrow$ Phần 3 $\rightarrow$ Phần 4).
