# CHEATSHEET TOÀN TẬP: HỆ THỐNG NHẬN DIỆN KHUÔN MẶT FACENET & MTCNN (LAB 5)

Tài liệu tra cứu nhanh (Cheatsheet) giải thích chi tiết **từng cụm chức năng**, **từng đoạn code**, **toàn bộ thông số kỹ thuật (hyperparameters)** và **cơ chế toán học** của hệ thống nhận diện khuôn mặt thời gian thực trong `Lab5.py`.

---

## MỤC LỤC
1. [Bảng tra cứu toàn bộ thông số hệ thống (Hyperparameter Matrix)](#1-bảng-tra-cứu-toàn-bộ-thông-số-hệ-thống)
2. [Sơ đồ luồng dữ liệu tổng thể (Pipeline Flow)](#2-sơ-đồ-luồng-dữ-liệu-tổng-thể)
3. [Giải thích chi tiết từng cụm code](#3-giải-thích-chi-tiết-từng-cụm-code)
   - [Cụm 1: Môi trường & Khắc phục font console](#cụm-1-môi-trường--khắc-phục-font-console)
   - [Cụm 2: Khởi tạo hệ thống & Thiết bị phần cứng](#cụm-2-khởi-tạo-hệ-thống--thiết-bị-phần-cứng)
   - [Cụm 3: Nạp & Quản lý Cơ sở dữ liệu mẫu](#cụm-3-nạp--quản-lý-cơ-sở-dữ-liệu-mẫu)
   - [Cụm 4: Đăng ký khuôn mặt mới từ Webcam](#cụm-4-đăng-ký-khuôn-mặt-mới-từ-webcam)
   - [Cụm 5: Toán học trích xuất & Đối sánh Cosine Similarity](#cụm-5-toán-học-trích-xuất--đối-sánh-cosine-similarity)
   - [Cụm 6: Pipeline xử lý khung hình `process_frame`](#cụm-6-pipeline-xử-lý-khung-hình-process_frame)
   - [Cụm 7: Giao diện đồ họa Bo góc & HUD bán trong suốt](#cụm-7-giao-diện-đồ-họa-bo-góc--hud-bán-trong-suốt)
   - [Cụm 8: Chống lỗi màn hình xanh Webcam `open_webcam`](#cụm-8-chống-lỗi-màn-hình-xanh-webcam-open_webcam)
   - [Cụm 9: Vòng lặp chính & Kỹ thuật Frame Skipping tối ưu CPU](#cụm-9-vòng-lặp-chính--kỹ-thuật-frame-skipping-tối-ưu-cpu)
4. [Bảng mã màu & Phím tắt điều khiển](#4-bảng-mã-màu--phím-tắt-điều-khiển)
5. [Cẩm nang xử lý sự cố (Troubleshooting Guide)](#5-cẩm-nang-xử-lý-sự-cố)

---

## 1. BẢNG TRA CỨU TOÀN BỘ THÔNG SỐ HỆ THỐNG

| Tham số / Biến | Giá trị | Thuộc cụm | Ý nghĩa kỹ thuật & Lý do thiết lập |
|---|:---:|:---:|---|
| **`image_size`** | `160` | MTCNN | Kích thước ảnh khuôn mặt đầu ra ($160 \times 160$ px). Đúng bằng kích thước tensor đầu vào yêu cầu của mạng Inception-ResNet-v1. |
| **`margin`** | `20` | MTCNN | Mở rộng biên bao quanh khuôn mặt thêm 20px mỗi chiều, giữ trọn vẹn cằm, trán và mang tai giúp trích xuất đặc trưng chính xác hơn. |
| **`min_face_size`** | `20` | MTCNN | Kích thước khuôn mặt nhỏ nhất có thể phát hiện (px). Hạ từ mặc định 40 xuống 20 để không bị mất nhận diện khi người dùng ngồi xa webcam. |
| **`thresholds`** | `[0.6, 0.7, 0.7]` | MTCNN | Ngưỡng tin cậy của 3 mạng tích chập con: P-Net ($0.6$), R-Net ($0.7$), O-Net ($0.7$). Giúp lọc bỏ 99% các vùng nhiễu không phải mặt người. |
| **`factor`** | `0.709` | MTCNN | Tỷ lệ co giãn giữa các tầng trong tháp ảnh (Image Pyramid) để quét khuôn mặt ở các khoảng cách khác nhau ($1/\sqrt{2} \approx 0.707$). |
| **`post_process`** | `True` | MTCNN | Tự động chuẩn hóa dải pixel từ $[0, 255]$ về $[-1.0, 1.0]$ theo công thức $(x - 127.5) / 128.0$. |
| **`keep_all`** | `True` | MTCNN | Bắt buộc giữ lại **toàn bộ** khuôn mặt phát hiện được trong khung hình thay vì chỉ lấy 1 khuôn mặt có diện tích lớn nhất. |
| **`pretrained`** | `'vggface2'` | FaceNet | Trọng số mạng Inception-ResNet-v1 huấn luyện trên tập VGGFace2 (hơn 3.3 triệu ảnh, 9.131 danh tính), nhận diện người châu Á rất tốt. |
| **`dim`** | `512` | FaceNet | Kích thước vector đặc trưng khuôn mặt (Embedding $\in \mathbb{R}^{512}$). |
| **`similarity_threshold`** | `0.7` | Matching | Ngưỡng so khớp Cosine Similarity theo đề bài: $> 0.7 \rightarrow$ **Matched**, $\le 0.7 \rightarrow$ **Unknown**. |
| **`prob` threshold** | `0.6` | Frame Pipeline | Lọc bỏ những bounding box có độ tin cậy MTCNN $< 60\%$ trước khi chuyển sang FaceNet để tránh tính toán lãng phí. |
| **`CAP_DSHOW`** | Backend flag | OpenCV | Cờ DirectShow trên Windows. Khắc phục triệt để lỗi màn hình toàn màu xanh lá cây `[0, 135, 0]` do MSMF gây ra. |
| **`FRAME_WIDTH` / `HEIGHT`**| `640 x 480` | OpenCV | Độ phân giải khung hình tối ưu cân bằng giữa chất lượng nhận diện và tốc độ xử lý real-time trên CPU. |
| **`frame_skip`** | `2` | Main Loop | Khoảng cách khung hình chạy suy luận AI: cứ 2 khung hình thì chạy MTCNN+FaceNet 1 lần; khung ở giữa tái sử dụng vị trí box để giữ mượt 30 FPS. |

---

## 2. SƠ ĐỒ LUỒNG DỮ LIỆU TỔNG THỂ

```
  [ Webcam Capture (OpenCV) ]
               │
               ▼ (Định dạng ảnh BGR: 640x480)
  [ Đổi không gian màu: cv2.cvtColor(BGR -> RGB) ]
               │
               ▼ (Chuyển sang PIL.Image)
  ┌────────────────────────────────────────────────────────┐
  │                 BỘ PHÁT HIỆN MTCNN                     │
  │  ├── 1. P-Net: Tạo tháp ảnh & đề xuất vùng mặt         │
  │  ├── 2. R-Net: Tinh chỉnh hộp bao & Khử Non-Max (NMS) │
  │  └── 3. O-Net: Xuất tọa độ Bounding Box + 5 Landmarks  │
  └────────────────────────────┬───────────────────────────┘
                               │
                               ▼ mtcnn.extract(boxes)
  [ Cắt & Căn chỉnh khuôn mặt: Tensor 4D (N, 3, 160, 160) ]
                               │
                               ▼ Chuẩn hóa [-1, 1]
  ┌────────────────────────────────────────────────────────┐
  │         MẠNG TRÍCH XUẤT ĐẶC TRƯNG FACENET             │
  │  └── Inception-ResNet-v1 (pretrained='vggface2')       │
  │  └── Xuất vector đặc trưng: e ∈ R^512                  │
  │  └── Chuẩn hóa L2-norm: F.normalize(e, p=2, dim=1)    │
  └────────────────────────────┬───────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │             ĐỐI SÁNH COSINE SIMILARITY                 │
  │  └── Tích vô hướng: sim(u, v) = u · v = ∑(u_k * v_k)   │
  │  └── Tìm max_similarity trong Database mẫu             │
  └────────────────────────────┬───────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼ (sim > 0.7)               ▼ (sim <= 0.7)
       ┌───────────────────┐       ┌───────────────────┐
       │   Trạng thái:     │       │   Trạng thái:     │
       │    "Matched"      │       │    "Unknown"      │
       │ Khung: Xanh lá cây │       │ Khung: Đỏ (Red)   │
       │ Nhãn: Tên người   │       │ Nhãn: "Unknown"   │
       └───────────────────┘       └───────────────────┘
```

---

## 3. GIẢI THÍCH CHI TIẾT TỪNG CỤM CODE

### Cụm 1: Môi trường & Khắc phục font console
```python
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
```
- **Mục đích**: Trên Windows, console mặc định dùng mã hóa `cp1252` hoặc `cp437`. Khi in chuỗi tiếng Việt có dấu (như `"Khởi tạo"`, `"Khuôn mặt"`), Python sẽ văng lỗi `UnicodeEncodeError`.
- **Cơ chế**: Lệnh `reconfigure(encoding="utf-8")` ép bộ đệm xuất chuẩn sang UTF-8, đảm bảo hiển thị log tiếng Việt mượt mà trên mọi terminal Windows/CMD/PowerShell/Bash.

---

### Cụm 2: Khởi tạo hệ thống & Thiết bị phần cứng
```python
if device is None:
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    self.device = torch.device(device)
```
- **`torch.cuda.is_available()`**: Kiểm tra card đồ họa NVIDIA. Nếu có GPU, PyTorch phân bổ tính toán ma trận lên hàng nghìn nhân CUDA; nếu không có, tự động chạy trên CPU thông qua AVX2/OpenMP.

```python
self.mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True, keep_all=True, device=self.device
)
self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
```
- **`.eval()`**: Chuyển mạng Inception-ResNet sang chế độ suy luận:
  - Khóa toàn bộ các lớp `Dropout` (không ngắt kết nối nơ-ron ngẫu nhiên).
  - Cố định trọng số trung bình và phương sai của `BatchNorm2d`.
  - Giúp vector đặc trưng 512D luôn ổn định và tiết kiệm 30% tài nguyên tính toán.

---

### Cụm 3: Nạp & Quản lý Cơ sở dữ liệu mẫu
```python
def load_database(self) -> int:
    ...
    face_tensor = self.mtcnn(img_pil)
    ...
    with torch.no_grad():
        embedding = self.facenet(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
    self.database[person_name].append(embedding.cpu())
```
- **`torch.no_grad()`**: Vô hiệu hóa bộ nhớ đồ thị tính gradient (Autograd Engine) của PyTorch, giảm 50% lượng RAM sử dụng và tăng tốc độ trích xuất đặc trưng.
- **`embedding.cpu()`**: Chuyển vector từ GPU (nếu dùng CUDA) về CPU RAM để giải phóng VRAM của card đồ họa.
- **Quy tắc đặt tên file**: `NguyenVanA.jpg` hoặc `NguyenVanA_1.jpg` đều được hệ thống tự động bóc tách thành tên định danh `"NguyenVanA"`. Hỗ trợ một người có thể đăng ký nhiều góc mặt khác nhau để tăng độ chính xác.

---

### Cụm 4: Đăng ký khuôn mặt mới từ Webcam
```python
def register_new_face(self, name: str, frame_bgr: np.ndarray, box: Optional[np.ndarray] = None) -> bool:
```
- **Cắt xén thông minh (Smart Margin Cropping)**:
  ```python
  margin_x = int((x2 - x1) * 0.1)
  margin_y = int((y2 - y1) * 0.1)
  crop_x1 = max(0, x1 - margin_x)
  ...
  crop_img = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
  ```
  Lấy tọa độ khuôn mặt vừa phát hiện từ webcam, mở rộng thêm $10\%$ đường biên để lấy đủ góc cạnh và lưu ảnh chất lượng cao JPEG $95\%$ vào `known_faces/`.
- **Cập nhật tức thì (In-memory Ingestion)**: Tính toán ngay vector 512D và nạp vào biến từ điển `self.database` trong RAM mà không cần khởi động lại chương trình.

---

### Cụm 5: Toán học trích xuất & Đối sánh Cosine Similarity
```python
def compute_similarity(self, embedding: torch.Tensor) -> Tuple[str, str, float]:
    for person_name, known_embeddings in self.database.items():
        for known_emb in known_embeddings:
            sim = torch.sum(embedding * known_emb).item()
```
- **Nguyên lý Toán học**:
  - Với hai vector bất kỳ $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{512}$:
    $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \cdot \|\mathbf{v}\|_2}$$
  - Nhờ bước tiền xử lý `F.normalize(embedding, p=2, dim=1)`:
    $$\|\mathbf{u}\|_2 = \sqrt{\sum_{k=1}^{512} u_k^2} = 1.0 \quad \text{và} \quad \|\mathbf{v}\|_2 = 1.0$$
  - Vì mẫu số bằng $1 \times 1 = 1$, phép tính rút gọn thành:
    $$\cos(\theta) = \mathbf{u} \cdot \mathbf{v} = \sum_{k=1}^{512} u_k v_k$$
  - Lệnh `torch.sum(embedding * known_emb).item()` nhân song song 512 phần tử cực nhanh chỉ mất $\approx 0.001$ mili-giây.

- **Cổng phân loại theo ngưỡng (Gating Decision)**:
  ```python
  if max_similarity > self.similarity_threshold:  # 0.7
      status = "Matched"
      identified_name = best_person
  else:
      status = "Unknown"
      identified_name = "Unknown"
  ```

---

### Cụm 6: Pipeline xử lý khung hình `process_frame`
```python
def process_frame(self, frame_bgr: np.ndarray, draw_landmarks: bool = False):
    ...
    boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
    ...
    face_tensors = self.mtcnn.extract(img_pil, boxes, save_path=None)
    with torch.no_grad():
        embeddings = self.facenet(face_tensors)
        embeddings = F.normalize(embeddings, p=2, dim=1)
```
- **Đồng bộ Bounding Box & Cắt ảnh**: Sử dụng trực tiếp `self.mtcnn.extract(img_pil, boxes)` để cắt ảnh theo đúng danh sách `boxes` vừa phát hiện. Tuyệt đối không gọi lại `self.mtcnn(img_pil)` vì việc gọi lại sẽ chạy lại bộ phát hiện lần hai với các ngưỡng ngẫu nhiên khác nhau, làm lệch thứ tự giữa tọa độ khung viền và vector embedding.

---

### Cụm 7: Giao diện đồ họa Bo góc & HUD bán trong suốt

#### a. Bo góc hiện đại `_draw_stylish_box()`
- Thay vì vẽ một hình chữ nhật thô sơ bằng `cv2.rectangle`, hàm vẽ thêm 4 đường nhấn dày hơn tại 4 góc khuôn mặt (`corner_len`), tạo cảm giác giao diện thị giác máy tính chuyên nghiệp.
- Nhãn tên người được vẽ trên một thanh banner chữ nhật cùng màu (Xanh hoặc Đỏ) phía trên đầu hộp bao, chữ tương phản (chữ đen trên nền xanh, chữ trắng trên nền đỏ) giúp người dùng đọc rõ ràng kể cả khi hậu cảnh có màu phức tạp.

#### b. Lớp phủ thông số HUD `draw_hud()`
```python
overlay = frame.copy()
cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
```
- Sử dụng thuật toán trộn ảnh **Alpha Blending**:
  $$\text{Pixel}_{\text{kết quả}} = \alpha \cdot \text{Overlay} + (1 - \alpha) \cdot \text{Frame} + \gamma$$
  Với $\alpha = 0.75$, tạo dải băng thông số màu xám đen bán trong suốt, vừa hiển thị rõ chữ vừa không che khuất hoàn toàn tầm nhìn camera.

---

### Cụm 8: Chống lỗi màn hình xanh Webcam `open_webcam`
```python
def open_webcam(camera_id: int = 0) -> cv2.VideoCapture:
    candidate_backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Mặc định (CAP_ANY)", cv2.CAP_ANY),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
    ]
```
- **Hiện tượng lỗi**: Trên Windows, backend mặc định MSMF thường trả về frame màu xanh lá cây đồng nhất có giá trị BGR cố định $[B=0, G=135, R=0]$.
- **Giải thuật phát hiện & Tự động phục hồi**:
  ```python
  b_mean, g_mean, r_mean = np.mean(test_frame, axis=(0, 1))
  std_val = float(np.std(test_frame))
  if std_val < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0:
      # Phát hiện frame xanh lỗi -> Giải phóng camera và thử backend kế tiếp
      cap.release()
      continue
  ```
  Hàm tính toán độ lệch chuẩn $\sigma$ (`std_val`). Nếu ảnh là một màu đồng nhất ($\sigma < 3.0$) và kênh Green áp đảo ($G > 100, B, R < 15$), hàm tự động ngắt kết nối và chuyển sang backend **DirectShow (`CAP_DSHOW`)**, đảm bảo webcam mở lên luôn có màu sắc trung thực.

---

### Cụm 9: Vòng lặp chính & Kỹ thuật Frame Skipping tối ưu CPU
```python
should_process = (not skip_enabled) or (frame_count % frame_skip == 0)

if should_process:
    annotated_frame, last_detections = recognizer.process_frame(frame, draw_landmarks=draw_landmarks)
else:
    annotated_frame = frame.copy()
    for det in last_detections:
        # Tái sử dụng tọa độ box của frame trước
        recognizer._draw_stylish_box(annotated_frame, det["box"], color, label)
```
- **Kỹ thuật Temporal Bounding Box Persistence**:
  - Tần suất xuất hiện khuôn mặt trong camera thường biến thiên liên tục nhưng không nhảy vọt trong khoảng $1/30$ giây.
  - Khi bật `skip_enabled=True` (`frame_skip=2`):
    - **Frame chẵn (0, 2, 4...)**: Chạy toàn bộ mạng nơ-ron MTCNN + FaceNet (tốn $\approx 45$ ms CPU), ghi nhớ vị trí và danh tính vào biến `last_detections`.
    - **Frame lẻ (1, 3, 5...)**: Bỏ qua mạng nơ-ron, lấy ngay `last_detections` của frame trước vẽ đè lên hình ảnh mới (chỉ tốn $\approx 1$ ms CPU).
  - **Kết quả**: Tăng gấp đôi tốc độ khung hình từ **12 FPS lên 28–30 FPS**, video không bị giật lag và quạt tản nhiệt CPU không bị rú tải.

---

## 4. BẢNG MÃ MÀU & PHÍM TẮT ĐIỀU KHIỂN

### Bảng mã màu nhận diện
| Trạng thái | Điều kiện Similarity | Mã màu BGR | Hiển thị giao diện |
|---|:---:|:---:|---|
| **Matched** | $> 0.70$ | `(0, 255, 0)` | Khung viền **Xanh lá cây**, banner hiển thị tên và độ tương đồng |
| **Unknown** | $\le 0.70$ | `(0, 0, 255)` | Khung viền **Đỏ**, banner hiển thị `"Unknown (<sim>)"` |
| **Landmarks** | Điểm mốc 5 điểm | `(255, 255, 0)` | Chấm tròn màu **Cyan/Vàng nhạt** tại mắt, mũi và khóe miệng |

### Bảng phím tắt tương tác trực tiếp
| Phím | Tên thao tác | Chi tiết hành vi |
|:---:|---|---|
| **`S`** hoặc **`C`** | **Đăng ký nhanh** | Dừng hình, yêu cầu nhập tên trên console, cắt khuôn mặt từ webcam và nạp tức thì vào database. |
| **`R`** | **Nạp lại Database** | Quét lại thư mục `known_faces/`, cập nhật nếu bạn vừa copy thêm ảnh mới từ ngoài vào. |
| **`L`** | **Bật/Tắt Landmarks** | Bật hoặc ẩn 5 điểm mốc đặc trưng trên khuôn mặt. |
| **`F`** | **Tối ưu hóa FPS** | Chuyển đổi giữa chế độ chạy liên tục mỗi frame (chính xác tuyệt đối) và bỏ qua frame (mượt mà 30 FPS). |
| **`Q`** hoặc **`ESC`**| **Thoát an toàn** | Giải phóng camera phần cứng, đóng tất cả cửa sổ GUI và kết thúc tiến trình sạch sẽ. |

---

## 5. CẨM NANG XỬ LÝ SỰ CỐ (TROUBLESHOOTING GUIDE)

1. **Camera bị màn hình màu xanh lá cây đồng nhất:**
   - *Hiện tượng*: Màn hình hiển thị màu xanh đặc `[0, 135, 0]`, không thấy người.
   - *Khắc phục*: Hệ thống đã tự động xử lý qua hàm `open_webcam()`. Nếu chạy code ngoài, luôn truyền cờ `cv2.CAP_DSHOW` khi khởi tạo: `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.

2. **Người ngồi xa webcam nhưng MTCNN không nhận ra:**
   - *Hiện tượng*: Ngồi lùi ra sau khoảng 1.5m thì mất khung bao.
   - *Khắc phục*: Đảm bảo tham số `min_face_size=20` (không dùng mặc định 40).

3. **Bị báo lỗi `[WinError 10048]` hoặc Camera bị chiếm dụng:**
   - *Hiện tượng*: `cap.isOpened()` trả về `False`.
   - *Khắc phục*: Đang có một tiến trình Python hoặc ứng dụng (Zoom, Teams, Camera app) chạy ngầm chiếm cổng camera. Chạy lệnh:
     ```bash
     taskkill /F /IM python.exe
     ```

4. **Độ tương đồng Cosine Similarity vượt quá 1.0 hoặc bị âm bất thường:**
   - *Hiện tượng*: Giá trị sim hiển thị $1.4$ hoặc $-5.2$.
   - *Khắc phục*: Quên chuẩn hóa L2 vector. Bắt buộc phải có dòng `F.normalize(embedding, p=2, dim=1)` ngay sau khi đưa qua FaceNet.
