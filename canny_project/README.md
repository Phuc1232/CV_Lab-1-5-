# canny_project: Canny Edge Detector từ toán học đến thực nghiệm

Mở rộng [Lab3](../Lab3/README.md) (bài thực hành chương 2 phần 2, slide trang 26) thành một dự án nghiên cứu thuật toán. Canny được **tự cài đặt bằng NumPy và tái tạo từng pixel** `cv2.Canny` lẫn `skimage.feature.canny`. Dự án kèm hệ thống dữ liệu có ground truth, Experiment Engine tái lập được, Evaluation Engine và 10 thực nghiệm trả lời toàn bộ câu hỏi của đề. Không có GUI.

## Kết quả chính
| | Kết quả | Nguồn |
|---|---|---|
| Khớp thư viện (P1) | **0 pixel lệch** với cv2 và skimage: 6 ảnh thật × 12 cấu hình | X1 |
| Tham số (P2) | Đỉnh gradient ∝ 1/σ: ngưỡng cố định thì σ lớn mất cạnh (Recall 0.95 → 0.81); σ cần thiết tăng theo nhiễu | X2 |
| Ảnh khó (P3) | Tham số mặc định thất bại (F1 < 0.1 với nhiễu); ngưỡng theo nhiễu / phân vị đạt 0.95–0.99; texture vẫn khó (≤ 0.79) | X3 |
| So với Sobel, LoG (T1b) | F1 một-một: Canny 0.92, LoG 0.54–0.92, Sobel 0.66–0.69 | X4 |
| Phân đoạn + nhận dạng (P4) | 60/60 vật, 100% nhận dạng trên ảnh sạch; σ lớn bền nhiễu nhưng bo góc | X5 |
| Cải thiện (E2) | Ngưỡng phổ quát √(2 ln N): F1 0.12 → 0.99 với nhiễu Gauss; median 3×3 cho muối tiêu (0.995) | X6 |
| Ảnh màu (E3) | Chuyển xám: F1 = 0 với cạnh đẳng sáng; OR / max_channel / Di Zenzo: 0.999 | X7 |
| Video (E4) | Làm trơn thời gian + ngưỡng theo nhiễu: F1 0.989 (tĩnh); gây nhòe khi chuyển động | X8 |
| Hiệu năng | Mọi bước O(N) (k ≈ 1); hysteresis label nhanh hơn BFS khoảng 30 lần; cv2 nhanh hơn bản tự cài khoảng 5.6 lần | X9 |

Chi tiết: [docs/experiments/README.md](docs/experiments/README.md). Trả lời lý thuyết: [docs/theory/README.md](docs/theory/README.md).

## Cài đặt và chạy
Dùng chung `.venv` ở `CV_Labs/` (Python 3.14):
```bash
cd CV_Labs
.venv/bin/pip install -e canny_project
.venv/bin/python -m pytest canny_project/tests -q            # 223 test (thêm -m "not slow" để chạy nhanh)
.venv/bin/python -m cannylab list                             # 10 thuật toán đã đăng ký
.venv/bin/python -m cannylab verify                           # so khớp với cv2 / skimage
.venv/bin/python -m cannylab run canny_project/configs/x3_anh_kho.toml
.venv/bin/python -m cannylab dataset /tmp/ds --family shapes --count 8 --degradations '[{"noise_sigma":0.1}]'
```

Dùng như thư viện:
```python
from cannylab.core import canny, OPENCV
r = canny(img, sigma=2.0, threshold_strategy="noise")      # hồ sơ skimage, ngưỡng theo nhiễu
r.smoothed, r.gx, r.magnitude, r.nms, r.strong, r.weak, r.edges   # mọi bước trung gian
canny(img_uint8, None, 100, 200, profile=OPENCV).edges     # ≡ cv2.Canny(img, 100, 200) > 0
```

## Cấu trúc
```
canny_project/
├── AGENTS.md, rules/        # quy tắc chung + 8 quy tắc chuyên biệt
├── configs/                 # 10 thực nghiệm X1–X9 (TOML)
├── src/cannylab/
│   ├── imaging/             # kiểm tra, dtype, RGB/BGR, đọc/ghi, suy giảm
│   ├── core/                # TỰ CÀI ĐẶT (NumPy, cấm cv2/skimage)
│   ├── analysis/            # phân đoạn, nhận dạng hình
│   ├── backends.py          # bản Optimized (OpenCV / skimage / SciPy)
│   ├── registry.py          # tên -> thuật toán
│   ├── datasets/            # ảnh tổng hợp có ground truth, Dataset
│   ├── evaluation/          # chỉ số cạnh / vùng / phân loại / hiệu năng
│   ├── experiments/         # cấu hình, runner, báo cáo
│   └── cli.py
├── tests/                   # unit, validation, numerical, edge cases, integration, regression, performance
├── results/                 # kết quả của lần chạy cuối (summary, records, hình)
└── docs/                    # xem docs/README.md
```

## Tài liệu
[docs/README.md](docs/README.md): lý thuyết, toán học, thuật toán, kiến trúc, dữ liệu, thực nghiệm, đánh giá, tài liệu tham khảo, nhật ký phát triển (10 phase).
