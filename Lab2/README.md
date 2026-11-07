# Lab 2: Toán tử điểm ảnh và các bộ lọc không gian

Bài nộp chính: [Lab2.ipynb](Lab2.ipynb). Script chạy độc lập: [Lab2.py](Lab2.py). Ảnh vào: `du_lieu/images.jpg`. Ảnh ra: `ket_qua/` (notebook) và `ket_qua/script/` (script).

## 1. Mục tiêu

- **Phần I – Toán tử điểm ảnh:** thay đổi độ sáng và độ tương phản, tạo ảnh âm bản, cắt ngưỡng nhị phân.
- **Phần II – Lọc tuyến tính:** lọc trung bình, lọc Gaussian, làm sắc nét.
- **Phần III – Nâng cao:** phát hiện cạnh bằng Sobel/Prewitt, tự thiết kế kernel, so sánh các bộ lọc (kể cả lọc phi tuyến Median và Bilateral) trên ảnh nhiễu.

## 2. Lý thuyết

**Toán tử điểm ảnh:** mỗi pixel được biến đổi độc lập với các pixel lân cận.
- Độ sáng / độ tương phản: `s = contrast · r + light`. Phép tính làm trên `float32` rồi `np.clip` về [0, 255] để tránh tràn số `uint8`.
- Âm bản: `s = 255 − r`.
- Cắt ngưỡng: `s = 255` nếu `r ≥ T`, ngược lại `s = 0`. Chỉ áp dụng cho ảnh xám.

**Lọc không gian:** trượt kernel qua ảnh để tính tích chập.
- **Trung bình:** mọi hệ số bằng `1/(k·k)`. Kernel càng lớn thì ảnh càng mờ và cạnh càng nhòe.
- **Gaussian:** trọng số theo `G(x,y) = 1/(2πσ²) · e^{−(x²+y²)/(2σ²)}`, σ càng lớn càng mờ. Khi `ksize = 0`, OpenCV tự tính kích thước kernel theo σ.
- **Làm nét:** tâm dương lớn, xung quanh âm, tổng hệ số = 1 để giữ độ sáng nền.
- **Quy tắc tổng hệ số:** làm mờ và làm nét thì tổng = 1, phát hiện cạnh thì tổng = 0.

**Phát hiện cạnh:** Prewitt và Sobel xấp xỉ đạo hàm bậc nhất theo x và y, độ lớn `G = √(Gx² + Gy²)`. Sobel có trọng số 2 ở tâm nên chống nhiễu tốt hơn Prewitt.

**Lọc phi tuyến:**
- **Median:** lấy trung vị trong cửa sổ, nên các điểm nhiễu 0/255 bị đẩy ra hai đầu dãy và bị loại.
- **Bilateral:** kết hợp trọng số không gian (`sigmaSpace`) với trọng số chênh lệch màu (`sigmaColor`), nhờ đó làm mịn vùng phẳng mà vẫn giữ được biên.

## 3. Cách làm

| Hàm | Phần | Thư viện / công thức |
|---|---|---|
| `lightAndContrast` | I.1–2 | NumPy, `np.clip` |
| `imgNative` | I.3 | `255 - img` |
| `imgBinary` | I.4 | boolean indexing, T = 64 / 128 / 192 |
| `filterMean` | II.1 | `cv2.blur`, k = 5 / 9, 4 kiểu xử lý biên |
| `filterGaussian` | II.2 | `cv2.GaussianBlur`, σ = 0.25 → 1.25 |
| `filterSharpen`, `diffHistogram` | II.3 | `cv2.filter2D` + histogram chênh lệch `D = I_nét − I_gốc` |
| `detectEdges` | III.1 | `cv2.Sobel`, kernel Prewitt tự viết + `cv2.filter2D` |
| `designKernelAndFilter` | III.2 | 4 kernel: Gauss 3×3, mờ chuyển động chéo, làm nét, làm nét mạnh |
| `compareFilter` | III.3–4 | Mean / Gaussian / Median / Bilateral trên nhiễu muối tiêu và nhiễu Gauss |

Notebook cố định `np.random.seed(0)` để ảnh nhiễu giống nhau mỗi lần chạy.

## 4. Kết quả

| Hình (`ket_qua/`) | Nội dung |
|---|---|
| `bai1_sang_tuong_phan.png` | Tăng/giảm sáng ±50, tương phản ×1.5 / ×0.95, kèm histogram |
| `bai1_am_ban.png` | Ảnh âm bản |
| `bai1_cat_nguong.png` | Cắt ngưỡng T = 64 / 128 / 192 |
| `bai2_loc_trung_binh.png` | Lọc trung bình k = 9 / 5 và các kiểu biên |
| `bai2_loc_gaussian.png` | Lọc Gaussian với σ khác nhau |
| `bai2_lam_net.png`, `bai2_lam_net_histogram_chenh_lech.png` | Làm nét và histogram chênh lệch |
| `bai3_bien_sobel_prewitt.png` | Cạnh Prewitt so với Sobel |
| `bai3_kernel_tu_thiet_ke.png` | 4 kernel tự thiết kế |
| `bai3_so_sanh_loc_salt.png`, `bai3_so_sanh_loc_gaussian.png` | So sánh 4 bộ lọc trên 2 loại nhiễu |

## 5. Nhận xét

- **Histogram:** tăng sáng làm histogram dịch sang phải, giảm sáng dịch sang trái. Tăng tương phản làm histogram giãn rộng, giảm tương phản làm nó co cụm lại.
- **Lọc trung bình:** kernel càng lớn thì ảnh càng nhòe vì mỗi pixel bị san phẳng theo vùng lân cận rộng hơn.
- **Gaussian:** σ < 1 gần như không làm mờ; σ lớn hơn thì mờ mịn và tự nhiên hơn lọc trung bình.
- **Làm nét:** histogram chênh lệch có cột giữa cao (vùng phẳng giữ nguyên) và hai đuôi (phía sáng và phía tối của biên).
- **Nhiễu muối tiêu:** Median khử gần sạch và giữ được hình khối. Mean và Gaussian chỉ làm nhòe các đốm nhiễu ra xung quanh. Bilateral giữ biên tốt nhưng không xử lý được điểm nhiễu cực đoan.
- **Sobel và Prewitt:** Sobel cho biên liền và ít nhiễu hơn Prewitt.

| Tiêu chí | Mean | Gaussian | Median | Bilateral |
|---|---|---|---|---|
| Chống nhiễu muối tiêu | Kém | Kém | Rất tốt | Trung bình |
| Bảo toàn cạnh | Kém | Khá | Tốt | Rất tốt |
| Tốc độ | Rất nhanh | Rất nhanh | Chậm hơn (phải sắp xếp) | Chậm nhất |

## Chạy lại

```bash
cd Lab2 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab2.ipynb
../.venv/bin/python Lab2.py        # Enter để dùng du_lieu/images.jpg; hình lưu vào ket_qua/script/
```

Lưu ý: tham số trong `Lab2.py` khác notebook ở vài chỗ (nhiễu muối tiêu 1% so với 20%, Sobel `ksize=3` so với `ksize=-5`). Vì vậy script lưu hình riêng vào `ket_qua/script/`; hình cho report lấy từ notebook.
