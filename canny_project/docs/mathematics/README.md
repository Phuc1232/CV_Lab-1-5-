# Cơ sở toán học: Canny Edge Detector

Kết quả của Phase 2. Mỗi file theo khung **WHAT · WHY · HOW · MATHEMATICS · PARAMETERS · COMPLEXITY · LIMITATIONS · WHICH**. Mọi con số trong tài liệu đều lấy từ lần chạy thật của [kiem_chung/kiem_chung_phase2.py](kiem_chung/kiem_chung_phase2.py). Log: [ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), số liệu: [ket_qua_phase2.json](kiem_chung/ket_qua_phase2.json).

| File | Nội dung | Câu hỏi của đề |
|---|---|---|
| [01_lam_tron_gaussian.md](01_lam_tron_gaussian.md) | Gaussian, kernel rời rạc, nhiễu ∝ 1/σ, đỉnh gradient ∝ 1/σ, SNR ∝ σ | T1a, T2a |
| [02_gradient_va_laplacian.md](02_gradient_va_laplacian.md) | Sobel, L1/L2, khuếch đại nhiễu, Laplacian/LoG | T1a, T1b, T3a |
| [03_non_maximum_suppression.md](03_non_maximum_suppression.md) | NMS lượng tử hóa / nội suy, xử lý điểm hòa | T1a |
| [04_nguong_kep_va_hysteresis.md](04_nguong_kep_va_hysteresis.md) | Ngưỡng kép, hysteresis = thành phần liên thông, tính đơn điệu, chọn ngưỡng | T1a, T2b, E2 |
| [05_khac_biet_opencv_skimage.md](05_khac_biet_opencv_skimage.md) | Khác biệt chính xác giữa hai thư viện, quy đổi ngưỡng | P1, P2 |
| [06_chi_so_danh_gia.md](06_chi_so_danh_gia.md) | P/R/F có dung sai, FOM, chamfer, IoU/Dice, hiệu năng; chỉ số không dùng | E1, P3, P4 |
| [07_canny_anh_mau_va_video.md](07_canny_anh_mau_va_video.md) | Cạnh đẳng sáng, Di Zenzo, cách OpenCV xử lý ảnh 3 kênh, nhấp nháy trong video | E3, E4 |

## Tóm tắt kết quả kiểm chứng

| # | Mệnh đề | Kết quả |
|---|---|---|
| 1 | std nhiễu sau Gaussian = σₙ / (2√π σ) | Sai số < 0.6% với σ = 1..5 |
| 2 | Sobel: hệ số khuếch đại 8; đỉnh trên bậc thang 4h | 8.000 / 4.000 |
| 3 | Đỉnh gradient sau Gaussian ≈ 3.19h/σ | Sai số < 6% khi σ ≥ 2; σ < 1 thì bị chặn bởi 4h |
| 4 | SNR gradient ≈ 2hσ/σₙ | 4.11 / 6.03 / 9.80 so với 4 / 6 / 10 |
| 5 | Nhiễu sau Sobel = √12 σₙ; sau Laplacian = √20 σₙ | 3.453 / 4.473 |
| 6 | L2 ≤ L1 ≤ √2 L2 | min 1.000, max 1.414 |
| 7 | `ndi.sobel` ≡ `cv2.Sobel` ở phần trong ảnh | Lệch 0.0 |
| 8 | `cv2.Canny(ảnh)` ≡ `cv2.Canny(Sobel_replicate)` | 0 px lệch (L1 và L2) |
| 9 | OpenCV so ngưỡng bằng `>`, skimage bằng `>=` | Xác nhận |
| 10 | Điểm hòa trên bậc thang: skimage giữ 2 px, OpenCV giữ 1 px | Xác nhận |
| 11 | Quy đổi ngưỡng skimage → OpenCV (L2, gradient float) | IoU 0.67 → 0.86 (mặc định) |
| 12 | Cạnh đơn điệu theo ngưỡng | 112 cặp, 0 vi phạm |
| 13 | Cạnh **không** đơn điệu theo σ | 2562 px mới xuất hiện khi σ: 1 → 2 |
| 14 | `cv2.Canny` 3 kênh = chọn kênh có max L1 | 0 px lệch |
| 15 | Kích thước kernel tự động của `GaussianBlur` | `round(6σ+1)\|1` (uint8), `round(8σ+1)\|1` (float32) |

Chạy lại (ở `CV_Labs/`):
```bash
.venv/bin/python canny_project/docs/mathematics/kiem_chung/kiem_chung_phase2.py
```
Môi trường đã chạy: Python 3.14.7, NumPy 2.5.4, OpenCV 5.0.0, SciPy 1.18.1, scikit-image 0.26.0.

## Tài liệu tham khảo
- J. Canny, "A Computational Approach to Edge Detection," *IEEE TPAMI*, 8(6), 1986.
- D. Marr, E. Hildreth, "Theory of Edge Detection," *Proc. R. Soc. Lond. B*, 207, 1980.
- S. Di Zenzo, "A Note on the Gradient of a Multi-Image," *CVGIP*, 33(1), 1986.
- W. K. Pratt, *Digital Image Processing*, Wiley, 1978 (Figure of Merit).
- D. Martin, C. Fowlkes, J. Malik, "Learning to Detect Natural Image Boundaries…," *IEEE TPAMI*, 26(5), 2004 (đánh giá kiểu BSDS, khớp một-một).
- Mã nguồn `skimage/feature/_canny.py` (v0.26) và tài liệu `cv2.Canny` (v5.0).
