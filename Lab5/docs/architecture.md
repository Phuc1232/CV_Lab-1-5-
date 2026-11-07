# Kiến trúc nhận diện khuôn mặt Lab 5

Xem [README](../README.md) để cài đặt và chạy chương trình. Xem [cheatsheet](cheatsheet.md) để tra thông số và các đoạn mã minh họa.

## Luồng xử lý

Ngưỡng mặc định là 0.7. MTCNN trả về hộp bao và các điểm mốc để hiển thị. Ứng dụng cắt ảnh theo hộp bao trước khi tính embedding.

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
           ▼ (Cắt và đổi kích thước thành 160x160)
   [ FaceNet (Inception-ResNet-v1) ]
     └── Dùng trọng số pretrained trên VGGFace2
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
  max_similarity > ngưỡng              max_similarity <= ngưỡng
  ┌────────────────────────┐           ┌────────────────────────┐
  │ Trạng thái: "Matched"  │           │ Trạng thái: "Unknown"  │
  │ Nhãn: Matched: <Tên>   │           │ Nhãn: Unknown (<sim>)  │
  │ Khung: Xanh lá cây     │           │ Khung: Đỏ              │
  └────────────────────────┘           └────────────────────────┘
```

### MTCNN (Multi-task Cascaded Convolutional Networks)
MTCNN phát hiện khuôn mặt và định vị 5 điểm mốc chính (hai mắt, mũi và hai khóe miệng) qua ba giai đoạn.
- **Proposal Network (P-Net)** quét ảnh ở nhiều tỷ lệ (Image Pyramid) bằng mạng tích chập hoàn toàn (FCN) để sinh các vùng ứng viên ban đầu.
- **Refine Network (R-Net)** lọc các vùng dương tính giả (false positives) và áp dụng Non-Maximum Suppression (NMS).
- **Output Network (O-Net)** tinh chỉnh vị trí hộp bao và xuất tọa độ 5 điểm mốc khuôn mặt.

### FaceNet & Triplet Loss
FaceNet ánh xạ trực tiếp ảnh khuôn mặt thành một vector đặc trưng 512 chiều. Trong phương pháp FaceNet, **Triplet Loss** giúp tách biệt embedding giữa các danh tính. Lab 5 chỉ suy luận bằng mô hình có sẵn.
$$\mathcal{L} = \max\left(0, \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha\right)$$
- $x^a$ (Anchor) là khuôn mặt gốc.
- $x^p$ (Positive) là khuôn mặt cùng một người với Anchor.
- $x^n$ (Negative) là khuôn mặt của người khác.
- $\alpha$ là khoảng lề (margin).

Nhờ đó, khoảng cách giữa các vector của cùng một người được thu nhỏ tối đa, trong khi khoảng cách giữa người khác nhau bị đẩy ra xa.

### Độ đo tương đồng Cosine Similarity
Do vector đặc trưng $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{512}$ đã được chuẩn hóa L2 ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1.0$), tích vô hướng bằng độ tương đồng cosine.
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{k=1}^{512} u_k v_k$$
- Giá trị nằm trong khoảng $[-1.0, 1.0]$.
- Càng gần $1.0$, độ tương đồng giữa hai khuôn mặt càng cao.

## Xử lý sự cố webcam trên Windows

Một số webcam có thể trả về hình xanh đồng nhất khi dùng Media Foundation (MSMF). Hàm `open_webcam()` thử lần lượt DirectShow, `CAP_ANY` và MSMF. Với mỗi backend mở được camera, hàm gọi `read()` năm lần và kiểm tra khung hình hợp lệ cuối cùng. Hàm thử backend tiếp theo khi thiếu khung hình hợp lệ hoặc khi phép kiểm tra báo hình xanh. Nếu tất cả lần thử đều thất bại, hàm trả về kết quả của `cv2.VideoCapture(camera_id)` với cấu hình mặc định.

Phép kiểm tra hiện dùng độ lệch chuẩn trên cả ba kênh BGR. Ngay cả ảnh có mọi pixel bằng `[0, 135, 0]` vẫn có độ lệch chuẩn khác 0 giữa các kênh, nên điều kiện `std < 3` có thể bỏ sót ảnh lỗi. Nếu webcam không mở, hãy đóng ứng dụng đang dùng camera và thử lại với chỉ số `--camera` phù hợp.

Xem [kết quả đã ghi nhận](../results/README.md#kết-quả-được-ghi-nhận-trước-đây) để tham khảo độ tương đồng và tốc độ khung hình.
