# Tài liệu thuật toán

Mỗi file theo khuôn **WHAT · WHY · HOW · MATHEMATICS · PSEUDOCODE · IMPLEMENTATION · PARAMETERS · COMPLEXITY · LIMITATIONS · APPLICATIONS · KẾT QUẢ** (rules/documentation-rules.md). Phần chứng minh và phân tích toán nằm ở [../mathematics/](../mathematics/README.md); file ở đây tập trung vào **cài đặt và kiểm chứng**.

| # | Thuật toán | Manual (`core/`) | Optimized (`backends.py`) | Kiểm chứng |
|---|---|---|---|---|
| [01](01_tich_chap_va_bien.md) | Tương quan 1D/2D, đệm biên | `convolution.py` | `scipy.ndimage.correlate*` | sai số ≤ 1e-12 |
| [02](02_gaussian.md) | Làm trơn Gaussian (+ bleed-over) | `gaussian.py` | `ndi.gaussian_filter`, `cv2.GaussianBlur` | ≤ 1e-12 |
| [03](03_sobel_gradient.md) | Sobel, độ lớn L1/L2, hướng | `gradient.py` | `cv2.Sobel` | trùng tuyệt đối (int) |
| [04](04_nms.md) | NMS lượng tử hóa / nội suy | `nms.py` | (bên trong cv2 / skimage) | qua pipeline: 0 px |
| [05](05_nguong_kep_hysteresis.md) | Ngưỡng kép + hysteresis BFS | `hysteresis.py` | `ndi.label` | trùng tuyệt đối |
| [06](06_canny_pipeline.md) | Canny, hồ sơ OPENCV / SKIMAGE | `canny.py` | `cv2.Canny`, `skimage.feature.canny` | 0 px lệch |
| [07](07_phan_doan_nhan_dang.md) | Hình thái học, phân đoạn, nhận dạng hình | `morphology.py`, `analysis/` | cv2 / SciPy | trùng tuyệt đối (hình thái học) |
| [08](08_canny_mau.md) | Canny ảnh màu (4 cách) | `color.py` | `cv2.Canny` 3 kênh | 0 px (max_channel) |
| [09](09_nguong_tu_dong_va_tien_loc.md) | Ngưỡng tự động, median, bilateral | `thresholds.py`, `denoise.py` | `cv2.medianBlur`, `cv2.bilateralFilter` | median trùng; bilateral sai TB < 0.01 |
| [10](10_doi_chung_sobel_log.md) | Sobel-ngưỡng, LoG zero-crossing | `baselines.py` | `cv2.Sobel`, `ndi.gaussian_laplace` | Sobel trùng; LoG IoU > 0.85 |
