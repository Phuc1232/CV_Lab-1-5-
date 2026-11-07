# 01. Tương quan và xử lý biên

**WHAT.** $\text{out}[i]=\sum_k w[k]\,x[i+k-r]$ (tương quan), cùng bốn chế độ đệm biên: `constant`, `replicate`, `reflect`, `reflect101`.

**WHY.** Đây là phép toán nền của Gaussian, Sobel, Laplacian và bộ ước lượng nhiễu. Chế độ biên quyết định kết quả ở viền ảnh: OpenCV Canny dùng `replicate`, `skimage` dùng `reflect` cho Sobel và `constant` + bleed-over cho Gaussian. Chọn sai là lệch 83–99 px ở viền ([../mathematics/05](../mathematics/05_khac_biet_opencv_skimage.md)).

**HOW / PSEUDOCODE.**
```
padded = pad(image, r, mode)
out = 0
for k, w in enumerate(kernel):          # vòng lặp theo tap, mỗi tap là phép toán vector trên cả ảnh
    out += w * padded[k : k + n]
```
Ảnh nguyên × kernel nguyên thì tích lũy bằng `int64`, nên Sobel trên `uint8` chính xác tuyệt đối và không tràn số.

**IMPLEMENTATION.** [`core/convolution.py`](../../src/cannylab/core/convolution.py): `pad`, `correlate1d`, `correlate_separable`, `correlate2d`.

| Tên dự án | `numpy.pad` | SciPy | OpenCV |
|---|---|---|---|
| constant | constant | constant | BORDER_CONSTANT |
| replicate | edge | nearest | BORDER_REPLICATE |
| reflect | symmetric | reflect | BORDER_REFLECT |
| reflect101 | reflect | mirror | BORDER_REFLECT_101 (mặc định của cv2) |

**COMPLEXITY.** 1D: $O(Nk)$; tách được: $O(2Nk)$; 2D tổng quát: $O(Nk_hk_w)$. Bộ nhớ $O(N)$.

**LIMITATIONS.** Kernel rất lớn (k > ~50) thì FFT nhanh hơn. Dự án không cần trường hợp này (Phase 3).

**KẾT QUẢ.** `test_correlate1d_matches_scipy` (4 chế độ × 2 trục) và `test_correlate2d_matches_scipy`: sai số ≤ 1e-12.
