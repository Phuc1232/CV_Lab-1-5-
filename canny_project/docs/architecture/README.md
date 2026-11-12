# Kiến trúc hệ thống (Phase 4)

## 1. Sơ đồ phụ thuộc

```
                 cli.py  (python -m cannylab run|list|verify)
                    │
          experiments/  (config TOML → lưới → runner → lưu kết quả → báo cáo)
           │       │        │
     datasets/  registry  evaluation/
           │       │        │
           │    algorithms (core/, analysis/, backends.py)
           │       │
           └── imaging/  (io, validate, convert, noise)   ← không phụ thuộc gì trong gói
```

**Quy tắc phụ thuộc (một chiều, có test kiểm tra):**
- `imaging` không import gì khác trong gói.
- `core`, `analysis` chỉ import `imaging` cùng NumPy/SciPy. `core` **không** import `cv2` hay `skimage`: đây là phần tự cài đặt.
- `backends.py` là nơi duy nhất bọc thư viện ngoài (OpenCV, skimage) cho phiên bản Optimized.
- `evaluation` không biết gì về thuật toán; chỉ nhận mảng `bool`.
- `experiments` chỉ gọi thuật toán qua `registry` (theo tên), nên thêm thuật toán mới không phải sửa runner.

## 2. Cấu trúc thư mục

```
canny_project/
├── pyproject.toml
├── README.md · AGENTS.md · rules/*.md
├── configs/                  # thực nghiệm X1–X9 (TOML)
├── src/cannylab/
│   ├── imaging/              # io.py, convert.py, validate.py, noise.py
│   ├── core/                 # TỰ CÀI ĐẶT (NumPy)
│   │   ├── convolution.py    # đệm biên, tương quan 1D/2D, tách được
│   │   ├── gaussian.py       # kernel, làm trơn, chuẩn hóa bleed-over
│   │   ├── gradient.py       # Sobel, độ lớn L1/L2, hướng
│   │   ├── nms.py            # lượng tử hóa (OpenCV) / nội suy (skimage)
│   │   ├── hysteresis.py     # ngưỡng kép, BFS (manual), label (optimized)
│   │   ├── canny.py          # pipeline + hồ sơ + kết quả trung gian
│   │   ├── thresholds.py     # chiến lược chọn ngưỡng (E2)
│   │   ├── color.py          # gradient màu: xám, max kênh, Di Zenzo (E3)
│   │   ├── baselines.py      # Sobel-ngưỡng, LoG zero-crossing (T1b)
│   │   └── morphology.py     # dilate/erode/closing, tô lỗ (P4a)
│   ├── analysis/
│   │   ├── segmentation.py   # cạnh → vùng (P4a)
│   │   └── shapes.py         # Douglas–Peucker, đặc trưng, phân loại (P4b)
│   ├── backends.py           # bản Optimized: OpenCV / skimage / SciPy
│   ├── registry.py           # tên → hàm phát hiện cạnh
│   ├── datasets/             # synthetic.py, dataset.py
│   ├── evaluation/           # edge_metrics, region_metrics, classification, performance
│   ├── experiments/          # config.py, runner.py, reporting.py, video.py
│   ├── utils.py              # môi trường, seed, đường dẫn
│   └── cli.py
├── tests/                    # unit / integration / validation / numerical / performance / regression
├── results/                  # đầu ra thực nghiệm (mỗi lần chạy một thư mục)
└── docs/                     # theory, algorithms, mathematics, architecture, datasets,
                              # experiments, evaluation, references, development
```

## 3. Hợp đồng dữ liệu (data contract)

| Khái niệm | Kiểu | Quy ước |
|---|---|---|
| Ảnh xám đầu vào | `np.ndarray` 2D, `uint8` hoặc `float` | `float` phải nằm trong [0, 1]; kiểm tra bằng `imaging.validate` |
| Ảnh màu | 3D `(H, W, 3)`, **RGB** | Mọi chỗ đọc bằng OpenCV phải đổi BGR→RGB ngay ở `imaging.io` |
| Bản đồ cạnh | 2D `bool` | Mọi backend chuẩn hóa về `bool` trước khi trả về |
| Hàm phát hiện cạnh | `fn(image, **params) -> bool[H, W]` | Đăng ký bằng `@register("tên")` |
| Kết quả trung gian | `CannyResult` (dataclass) | `smoothed, gx, gy, magnitude, nms, strong, weak, edges` |
| Mẫu dữ liệu | `Sample` (dataclass) | `image, edges_gt?, mask_gt?, labels_gt?, meta` |

## 4. Hồ sơ (profile) của Canny tự cài đặt
Một `CannyProfile` là dataclass bất biến, gồm: `smoothing` (none / gaussian), `gaussian_truncate`, `bleedover`, `scale` (255 / 1), `sobel_border`, `norm` (L1 / L2), `nms` (quantized / bilinear), `strict` (`>` hay `>=`), `drop_border`. Có hai hằng số `OPENCV` và `SKIMAGE`. Tạo biến thể mới chỉ cần `dataclasses.replace`.

## 5. Luồng một thực nghiệm
```
configs/x2_tham_so.toml
  → ExperimentConfig (kiểm tra khóa, mở rộng lưới tham số)
  → Dataset (sinh hoặc nạp; seed cố định)
  → for sample × run_config: edges = registry[alg](img, **params); metrics = evaluate(edges, gt); time, mem
  → results/<tên>_<thời gian>/ : config.toml (bản sao), environment.json, records.jsonl, summary.csv, figures/
```
Tái lập: chạy lại cùng file cấu hình thì ra cùng `records.jsonl` (trừ các cột thời gian). Có test kiểm tra điều này.

## 6. Quyết định kiến trúc
| Quyết định | Phương án khác | Lý do chọn |
|---|---|---|
| src layout + `pyproject.toml` | Script rời | Import sạch, test cài gói được |
| TOML cho cấu hình | YAML | `tomllib` có sẵn trong stdlib, không thêm phụ thuộc |
| JSONL + CSV cho kết quả | pandas / parquet | Không có pandas; JSONL đọc được bằng tay và nối thêm được |
| Registry hàm (không dùng lớp trừu tượng) | ABC `EdgeDetector` | Thuật toán là hàm thuần; registry đủ để mở rộng mà không cần thừa kế |
| Hồ sơ là dữ liệu | Hai lớp Canny riêng | Một pipeline, khác nhau ở lựa chọn; tránh lặp code |
| `core` cấm import cv2/skimage | Trộn lẫn | Bảo đảm phần Manual thực sự tự cài; có test kiểm tra |
