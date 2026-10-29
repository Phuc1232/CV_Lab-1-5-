# BÀI THỰC HÀNH LAB 5: NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC VỚI FACENET & MTCNN TRÊN WEBCAM

**Môn học:** Thị giác máy tính (Computer Vision)  
**Đơn vị:** Viện Công nghệ Thông tin, Điện, Điện tử – Trường Đại học Giao thông Vận tải TP.HCM  
**Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)  

---

## 1. Giới thiệu bài thực hành

Bài thực hành tập trung vào việc xây dựng hệ thống **nhận diện khuôn mặt thời gian thực (Real-time Face Recognition)** từ luồng video Webcam, kết hợp giữa hai mô hình học sâu hiện đại:
1. **MTCNN (Multi-task Cascaded Convolutional Networks):** Phát hiện vị trí khuôn mặt (Face Detection) và các điểm đặc trưng (Facial Landmarks).
2. **FaceNet (Inception-ResNet-v1):** Trích xuất vector đặc trưng khuôn mặt (512-dimensional Embedding) trong không gian vector Euclidean.
3. **Cosine Similarity Matching:** Đối sánh vector đặc trưng với cơ sở dữ liệu khuôn mặt mẫu đã đăng ký theo tiêu chí:
   - Nếu $\text{similarity} > 0.7$: Hiển thị nhãn **`Matched: <Tên người>`** kèm khung viền màu **Xanh lá cây (Green)**.
   - Nếu $\text{similarity} \le 0.7$ (hoặc $< 0.7$): Hiển thị nhãn **`Unknown`** kèm khung viền màu **Đỏ (Red)**.

---

## 2. Kiến trúc và Nguyên lý hoạt động

```
                         LUỒNG XỬ LÝ NHẬN DIỆN KHUÔN MẶT
                         
   [ Webcam (OpenCV) ]
           │
           ▼ (Frame BGR -> RGB)
   [ MTCNN Detector ]
     ├── P-Net: Tạo vùng đề xuất khuôn mặt (Candidate Bounding Boxes)
     ├── R-Net: Tinh chỉnh và loại bỏ hộp bao nhiễu (Box Refinement)
     └── O-Net: Xuất tọa độ hộp bao cuối cùng + 5 điểm Landmarks
           │
           ▼ (Crop & Align 160x160)
   [ FaceNet (Inception-ResNet-v1) ]
     └── Huấn luyện với Triplet Loss trên tập VGGFace2
     └── Xuất vector đặc trưng 512 chiều: e ∈ R^512
     └── Chuẩn hóa L2-norm: ||e||_2 = 1.0
           │
           ▼ (Cosine Similarity)
   [ Bộ so sánh cơ sở dữ liệu (Database Matcher) ]
     ├── Tính sim(u, v) = u . v đối với từng khuôn mặt mẫu
     └── Tìm độ tương đồng cực đại: max_similarity
           │
     ┌─────┴─────────────────────────────────┐
     ▼                                       ▼
  max_similarity > 0.7                 max_similarity <= 0.7
  ┌────────────────────────┐           ┌────────────────────────┐
  │ Trạng thái: "Matched"  │           │ Trạng thái: "Unknown"  │
  │ Nhãn: Matched: <Tên>   │           │ Nhãn: Unknown (<sim>)  │
  │ Khung: Xanh lá cây     │           │ Khung: Đỏ              │
  └────────────────────────┘           └────────────────────────┘
```

### 2.1. MTCNN (Multi-task Cascaded Convolutional Networks)
MTCNN giải quyết đồng thời hai tác vụ: phát hiện khuôn mặt và định vị 5 điểm mốc chính (mắt trái, mắt phải, mũi, hai khóe miệng) thông qua kiến trúc phân tầng 3 giai đoạn:
- **Proposal Network (P-Net):** Quét ảnh ở nhiều tỷ lệ (Image Pyramid) bằng mạng tích chập hoàn toàn (FCN) để sinh các vùng ứng viên ban đầu.
- **Refine Network (R-Net):** Lọc bỏ phần lớn các vùng âm tính giả (false positives) và áp dụng Non-Maximum Suppression (NMS).
- **Output Network (O-Net):** Xác định chính xác vị trí hộp bao (bounding box) và xuất tọa độ 5 điểm mốc khuôn mặt.

### 2.2. FaceNet & Triplet Loss
FaceNet ánh xạ trực tiếp ảnh khuôn mặt thành một vector đặc trưng 512 chiều. Mạng được tối ưu bằng hàm mất mát **Triplet Loss**:
$$\mathcal{L} = \max\left(0, \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha\right)$$
- $x^a$ (Anchor): Khuôn mặt gốc.
- $x^p$ (Positive): Khuôn mặt cùng một người với Anchor.
- $x^n$ (Negative): Khuôn mặt của người khác.
- $\alpha$: Khoảng lề (margin).

Nhờ đó, khoảng cách giữa các vector của cùng một người được thu nhỏ tối đa, trong khi khoảng cách giữa người khác nhau bị đẩy ra xa.

### 2.3. Độ đo tương đồng Cosine Similarity
Do vector đặc trưng $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{512}$ đã được chuẩn hóa L2 ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$), độ tương đồng Cosine được tính đơn giản bằng tích vô hướng:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{k=1}^{512} u_k v_k$$
- Giá trị nằm trong khoảng $[-1.0, 1.0]$.
- Càng gần $1.0$, độ tương đồng giữa hai khuôn mặt càng cao.

---

## 3. Cấu trúc thư mục

```
Lab5/
├── AGENTS.md            # Quy định kiến trúc Multi-Agent & quy trình kiểm thử
├── CHEATSHEET.md        # Cẩm nang tra cứu toàn bộ thông số, cụm code & giải thích toán học
├── Lab5.py              # Ứng dụng nhận diện thời gian thực qua Webcam
├── register_face.py     # Công cụ đăng ký khuôn mặt mới (Webcam / Ảnh)
├── test_lab5.py         # Bộ kiểm thử tự động (Unit Tests với pytest)
├── known_faces/         # Cơ sở dữ liệu ảnh khuôn mặt mẫu đã biết
│   ├── Mau_Kiem_Thu.jpg # Ảnh mẫu kiểm thử ban đầu
│   └── download.png     # Ảnh mẫu bổ sung
└── README.md            # Tài liệu hướng dẫn chi tiết
```

---

## 4. Yêu cầu cài đặt môi trường

Môi trường khuyến nghị: **Python 3.10 – 3.13**.

Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision facenet-pytorch opencv-python pillow numpy pytest
```

Kiểm tra thư viện đã sẵn sàng:
```bash
python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; import cv2; print('Môi trường đã sẵn sàng!')"
```

---

## 5. Hướng dẫn sử dụng

### 5.1. Chạy nhận diện khuôn mặt qua Webcam thời gian thực
Chạy chương trình chính bằng lệnh:
```bash
python Lab5/Lab5.py
```

Các tham số tùy chọn:
- `--camera 0`: Chỉ định ID webcam (mặc định là `0`).
- `--threshold 0.7`: Ngưỡng nhận diện (mặc định là `0.7` theo yêu cầu bài thực hành).
- `--frame-skip 2`: Xử lý MTCNN sau mỗi 2 frames để tối ưu tốc độ 30+ FPS trên CPU.
- `--device cpu` hoặc `--device cuda`: Chỉ định thiết bị chạy mô hình.

Ví dụ:
```bash
python Lab5/Lab5.py --camera 0 --threshold 0.7 --frame-skip 2
```

### 5.2. Các phím tắt điều khiển trong khi chạy Webcam
| Phím tắt | Chức năng |
|:---:|:---|
| **`S`** hoặc **`C`** | **Đăng ký nhanh:** Chụp và nạp khuôn mặt đang đứng trước webcam vào database ngay lập tức. |
| **`R`** | **Tải lại database:** Quét lại thư mục `known_faces/` mà không cần khởi động lại ứng dụng. |
| **`L`** | **Điểm mốc (Landmarks):** Bật/tắt hiển thị 5 điểm đặc trưng (mắt, mũi, miệng). |
| **`F`** | **Tối ưu hóa CPU:** Bật/tắt cơ chế Frame Skipping giữa các khung hình. |
| **`Q`** hoặc **`ESC`** | **Thoát:** Đóng cửa sổ và giải phóng camera an toàn. |

---

### 5.3. Đăng ký khuôn mặt mới vào Cơ sở dữ liệu

#### Cách 1: Đăng ký trực tiếp bằng webcam
```bash
python Lab5/register_face.py --name "NguyenVanA"
```
Cửa sổ camera sẽ mở ra, người dùng canh góc mặt và nhấn phím **SPACE** (hoặc phím **C**) để chụp và lưu vào hệ thống.

#### Cách 2: Đăng ký từ file ảnh có sẵn
```bash
python Lab5/register_face.py --name "TranThiB" --image "C:/duong_dan/anh_chan_dung.jpg"
```
Hệ thống sẽ dùng MTCNN tự động phát hiện, cắt viền khuôn mặt chuẩn xác và lưu vào `known_faces/`.

---

### 5.4. Chạy kiểm thử tự động (Automated Unit Tests)
Bộ kiểm thử `test_lab5.py` được thiết kế độc lập, không yêu cầu phần cứng webcam vật lý, đảm bảo kiểm chứng toàn bộ thuật toán trong môi trường CI/Headless:

```bash
pytest Lab5/test_lab5.py -v
```

Kết quả kiểm thử 6 test cases:
1. `test_models_initialization`: Kiểm tra khởi tạo MTCNN và FaceNet.
2. `test_embedding_dimension_and_l2_normalization`: Kiểm tra vector đặc trưng 512 chiều và L2-norm = 1.0.
3. `test_cosine_similarity_threshold_decision`: Kiểm tra điều kiện so sánh tương đồng.
4. `test_strict_boundary_condition`: Kiểm tra nghiêm ngặt giá trị biên xung quanh ngưỡng 0.7.
5. `test_process_frame_without_faces`: Kiểm tra khả năng xử lý khung hình không có khuôn mặt (kháng lỗi).
6. `test_open_webcam_backend_anti_green_screen`: Kiểm tra cơ chế tự động chọn backend DirectShow chống màn hình xanh.

---

## 6. Xử lý sự cố & Lưu ý kỹ thuật (Troubleshooting)

### 6.1. Lỗi màn hình camera toàn màu xanh lá cây (Solid Green `[0, 135, 0]`) trên Windows
- **Nguyên nhân:** Khi gọi `cv2.VideoCapture(0)` mặc định trên Windows, OpenCV kích hoạt backend **MSMF (Microsoft Media Foundation)**. Một số driver webcam hoặc thiết bị tích hợp không handshake chuẩn định dạng giải mã YUV sang BGR, khiến bộ đệm buffer trả về ma trận pixel đồng nhất `[B=0, G=135, R=0]` (màu xanh lá cây).
- **Giải pháp xử lý:** Hàm `open_webcam()` đã tích hợp cơ chế tự động nhận diện và chuyển đổi sang backend **DirectShow (`cv2.CAP_DSHOW`)**, đồng thời tự động kiểm tra phương sai `std` của khung hình để phát hiện và khắc phục ngay lập tức hiện tượng màn hình xanh.

---

## 7. Kết quả thực nghiệm và Minh chứng

- **Tỷ lệ nhận diện đúng:** Đạt độ tương đồng $> 0.95$ với khuôn mặt chính diện, đủ ánh sáng.
- **Khả năng phân biệt người lạ:** Các vector ngẫu nhiên hoặc khuôn mặt người ngoài cơ sở dữ liệu có độ tương đồng Cosine dao động từ $-0.1$ đến $0.4$, thấp hơn nhiều so với ngưỡng $0.7$, hệ thống phân loại chính xác nhãn **`Unknown`**.
- **Tốc độ khung hình (FPS):**
  - CPU (Intel Core): Đạt ~15 – 25 FPS khi bật chế độ Frame Skipping tối ưu (`frame_skip=2`).
  - GPU (NVIDIA CUDA): Đạt 35 – 60 FPS xử lý thời gian thực mượt mà.
