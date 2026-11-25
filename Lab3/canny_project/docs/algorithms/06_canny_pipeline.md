# 06. Canny: pipeline và hồ sơ

**WHAT.** `canny(image, sigma, low, high, profile=…)` trả `CannyResult` gồm mọi bước trung gian: `smoothed, gx, gy, magnitude, nms, strong, candidate (weak = candidate & ~strong), edges, low, high`.

**WHY hai hồ sơ.** Đề (P1) yêu cầu dùng hai thư viện có quy ước khác nhau ([../mathematics/05](../mathematics/05_khac_biet_opencv_skimage.md)). Hồ sơ là **dữ liệu** (`CannyProfile`, frozen dataclass), không phải lớp con: cùng một pipeline, khác nhau ở các lựa chọn.

| Trường | OPENCV | SKIMAGE |
|---|---|---|
| scale | `uint8` (số nguyên) | `unit` ([0,1]) |
| smoothing / default_sigma | không / None | có / 1.0 |
| Gaussian | (xấp xỉ, nếu truyền sigma) truncate 3, reflect101 | truncate 4, constant + bleed-over |
| sobel_mode | replicate | reflect |
| norm | L1 | L2 |
| nms | quantized | bilinear |
| strict | `>` | `>=` |
| drop_border | không | có |
| ngưỡng mặc định | bắt buộc truyền | 0.1 / 0.2 (thang [0,1]) |

**Quy ước ngưỡng.** Với hồ sơ SKIMAGE, ngưỡng truyền theo **đơn vị dtype đầu vào** (ảnh `uint8`: 25.5 ≡ 0.1). Với OPENCV, ngưỡng so trực tiếp với độ lớn Sobel trên thang 0..255; khi dùng L2 thì so bình phương nguyên với $\lfloor T^2\rfloor$, đúng như OpenCV.

**PSEUDOCODE.**
```
smoothed  = prepare(image, profile, sigma)           # đổi thang + Gaussian
gx, gy    = sobel(smoothed, profile.sobel_mode)
M, M_cmp  = magnitude theo profile.norm
local_max = NMS(gx, gy, M_cmp)                       # một lần, không phụ thuộc ngưỡng
T_l, T_h  = cố định | quantile | otsu | noise | median
strong, candidate = double_threshold(M_cmp tại local_max, T_l, T_h, profile.strict)
edges     = hysteresis(strong, candidate)            # BFS (manual) hoặc label (fast)
```

**IMPLEMENTATION.** [`core/canny.py`](../../src/cannylab/core/canny.py); bọc trong registry thành `canny_manual`, `canny_manual_fast`. Tham số `low_ratio`, `prefilter*` và tham số chiến lược ngưỡng được tách ở [`registry.py`](../../src/cannylab/registry.py).

**COMPLEXITY.** $O(N\sigma)$ cho Gaussian, $O(N)$ cho các bước còn lại. Hệ số mũ đo được ở X9.

**LIMITATIONS.**
- Hồ sơ OPENCV **có sigma** chỉ xấp xỉ `cv2.GaussianBlur`+`cv2.Canny`: đo được IoU = 0.897 trên `camera` (σ = 2, 30/90), vì OpenCV làm mờ `uint8` bằng số học điểm cố định. Không có sigma thì trùng tuyệt đối.
- Chỉ hỗ trợ Sobel 3×3.

**APPLICATIONS.** Tiền xử lý cho phân đoạn (P4a), nhận dạng hình (P4b), đo đạc công nghiệp, nhận dạng làn đường, ảnh y tế (xem [../theory/README.md](../theory/README.md)).

**KẾT QUẢ (kiểm chứng).**
- `cannylab verify`: 4 ảnh × 4 cấu hình, 0 px lệch.
- X1: 6 ảnh thật × 12 cấu hình, n_diff = 0, IoU = 1.
- `test_validation_libraries.py`: 6 bộ ngưỡng OpenCV (kể cả ngưỡng hoán đổi, ngưỡng lẻ, L2) × 4 ảnh; 5 cấu hình skimage × 4 ảnh; đầu vào float và uint16; chiến lược quantile ≡ `use_quantiles=True`. Tất cả trùng từng pixel.
