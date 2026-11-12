# 03. Gradient Sobel

**WHAT / WHY / MATHEMATICS.** Xem [../mathematics/02](../mathematics/02_gradient_va_laplacian.md). $K_x=[1,2,1]^\top\otimes[-1,0,1]$; $M_{L1}=|G_x|+|G_y|$, $M_{L2}=\sqrt{G_x^2+G_y^2}$; $\theta=\operatorname{atan2}(G_y,G_x)$.

**PSEUDOCODE.**
```
gx = correlate_separable(I, [1,2,1], [-1,0,1], mode)   # làm trơn theo y, đạo hàm theo x
gy = correlate_separable(I, [-1,0,1], [1,2,1], mode)
```

**IMPLEMENTATION.** [`core/gradient.py`](../../src/cannylab/core/gradient.py): `sobel`, `magnitude`, `direction`. Ảnh `uint8` được đưa vào dạng `int64` nên kết quả nguyên và trùng `cv2.Sobel(CV_16S)`.

**PARAMETERS.** `mode` biên (`replicate` cho hồ sơ OpenCV, `reflect` cho skimage); `norm` là L1 hoặc L2.

**COMPLEXITY.** $O(N)$, khoảng 12 phép nhân-cộng mỗi pixel cho cả hai hướng.

**LIMITATIONS.** Kernel 3×3 cố định (không hỗ trợ `apertureSize` 5/7 của OpenCV). L1 không đẳng hướng (√2 lần với cạnh chéo).

**KẾT QUẢ.** `test_sobel_integer_exact_vs_opencv`: trùng tuyệt đối trên `camera`. `test_sobel_gain_is_eight_on_ramp`: hệ số khuếch đại đúng bằng 8.
