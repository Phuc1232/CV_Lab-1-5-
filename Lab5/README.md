# Lab 5: Nhận diện khuôn mặt với MTCNN + FaceNet

Bài nộp chính: [Lab5.ipynb](Lab5.ipynb), chạy pipeline trên ảnh tĩnh và lưu hình cho report. App webcam [Lab5.py](Lab5.py) dùng để demo trực tiếp. Ảnh vào: `du_lieu/known_faces/` (cơ sở dữ liệu) và `du_lieu/kiem_tra/` (ảnh cần nhận diện, không bắt buộc). Ảnh ra: `ket_qua/`.

## 1. Mục tiêu

1. Lấy hình từ webcam bằng OpenCV.
2. Phát hiện khuôn mặt bằng MTCNN.
3. Trích xuất đặc trưng bằng FaceNet (Inception-ResNet-v1, pretrained VGGFace2) và so sánh với cơ sở dữ liệu.
4. Similarity > 0.7 thì hiện **Matched** (khung xanh), ≤ 0.7 thì hiện **Unknown** (khung đỏ).

## 2. Lý thuyết

```
Ảnh / khung hình webcam (BGR → RGB)
  → MTCNN: P-Net (đề xuất vùng) → R-Net (lọc vùng sai, NMS) → O-Net (hộp bao + 5 điểm mốc)
  → cắt và căn khuôn mặt 160×160
  → FaceNet: vector e ∈ R^512, chuẩn hóa L2 (‖e‖ = 1)
  → cosine similarity với từng mẫu trong cơ sở dữ liệu, lấy giá trị lớn nhất
  → > 0.7: Matched   |   ≤ 0.7: Unknown
```

- **MTCNN** gồm 3 mạng CNN nối tiếp chạy trên ảnh nhiều tỉ lệ (image pyramid), cho ra hộp bao và 5 điểm mốc (2 mắt, mũi, 2 khóe miệng).
- **FaceNet** được huấn luyện bằng Triplet Loss `L = max(0, ‖f(a) − f(p)‖² − ‖f(a) − f(n)‖² + α)`: kéo ảnh cùng người lại gần nhau, đẩy ảnh người khác ra xa.
- **Cosine similarity:** `cos(u, v) = u·v / (‖u‖‖v‖)`. Vì vector đã chuẩn hóa L2 nên công thức chỉ còn `Σ u_k v_k`, giá trị nằm trong [−1, 1].

## 3. Cách làm

| File | Việc làm |
|---|---|
| `Lab5.py` | Lớp `FaceRecognitionSystem`: `load_database`, `compute_similarity`, `process_frame`, `register_new_face`. Hàm `run_webcam` là vòng lặp webcam. |
| `register_face.py` | Đăng ký khuôn mặt mới từ webcam hoặc từ file ảnh vào `du_lieu/known_faces/` |
| `test_lab5.py` | 6 unit test (pytest): khởi tạo mô hình, embedding 512 chiều và L2, quyết định theo ngưỡng 0.7, giá trị biên, khung hình không có mặt, mở webcam |
| `Lab5.ipynb` | Gọi lại `FaceRecognitionSystem` trên ảnh tĩnh: phát hiện → embedding → so khớp, rồi lưu hình |

Phím tắt của app webcam: `S`/`C` đăng ký mặt, `R` nạp lại cơ sở dữ liệu, `L` bật/tắt điểm mốc, `F` bật/tắt bỏ khung để tăng tốc, `Q`/`ESC` thoát.

## 4. Kết quả

| Hình (`ket_qua/`) | Nội dung |
|---|---|
| `bai1_mtcnn_phat_hien.png` | Hộp bao và 5 điểm mốc MTCNN trên ảnh trong cơ sở dữ liệu |
| `bai2_facenet_embedding.png` | Khuôn mặt 160×160 sau MTCNN và vector embedding 512 chiều |
| `bai3_nhan_dien_anh_tinh.png` | Kết quả `process_frame` trên các ảnh kiểm tra |
| `bai3_similarity_nguong.png` | Biểu đồ similarity so với ngưỡng 0.7 |

Kết quả lần chạy hiện tại (ảnh mẫu `download.png`):

| Ảnh kiểm tra | Kết quả | Similarity |
|---|---|---|
| Ảnh mẫu gốc | Matched | 1.000 |
| Lật ngang | Matched | 0.987 |
| Giảm sáng ×0.5 | Matched | 0.979 |
| Làm mờ Gaussian 15×15 | Matched | 0.988 |

Unit test: **6/6 passed** (`pytest test_lab5.py`).

## 5. Nhận xét

- Embedding của FaceNet ổn định với lật ngang, giảm sáng và làm mờ (similarity đều > 0.97).
- **MTCNN phát hiện nhầm:** `Mau_Kiem_Thu.jpg` là tranh hoạt hình, không có mặt người, nhưng MTCNN vẫn báo có 1 khuôn mặt (prob 0.927) ở miếng dưa hấu. Ảnh này vì thế bị nạp vào cơ sở dữ liệu như một "người".
- **Hạn chế của dữ liệu:** cơ sở dữ liệu chỉ có 1 ảnh người thật; các ảnh kiểm tra đều tạo từ chính ảnh đó. Vì vậy chưa có ca **Unknown** và chưa có ảnh thật khác của cùng người. Muốn có hai ca này, bỏ ảnh vào `du_lieu/kiem_tra/` (ví dụ một ảnh khác của cùng người, một ảnh người lạ) rồi chạy lại notebook.

## Môi trường và cách chạy

Dùng chung `.venv` ở `CV_Labs/` (Python 3.14). Thêm thư viện:

```bash
.venv/bin/pip install torch torchvision pytest tqdm requests
.venv/bin/pip install --no-deps facenet-pytorch   # bản 2.6.0 ghim torch<2.3 và numpy<2; cài --no-deps vẫn chạy được, đã kiểm tra bằng pytest
```

```bash
cd Lab5
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab5.ipynb
../.venv/bin/python -m pytest test_lab5.py -v
../.venv/bin/python Lab5.py                                   # webcam (macOS sẽ hỏi quyền camera)
../.venv/bin/python register_face.py --name TenNguoi --image duong_dan_anh.jpg
```

Lần chạy đầu sẽ tải trọng số FaceNet (khoảng 100 MB) vào thư mục cache của torch.

Tài liệu làm việc (không phải bài nộp): `ghi_chu/AGENTS.md`, `ghi_chu/CHEATSHEET.md`.
