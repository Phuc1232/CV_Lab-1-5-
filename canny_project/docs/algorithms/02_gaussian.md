# 02. Làm trơn Gaussian

**WHAT / WHY / MATHEMATICS.** Xem [../mathematics/01](../mathematics/01_lam_tron_gaussian.md). Kernel $g[x]\propto e^{-x^2/2\sigma^2}$ trên $[-r,r]$, với $r=\lfloor4\sigma+0.5\rfloor$, chuẩn hóa tổng bằng 1, áp dụng tách được.

**Chuẩn hóa bleed-over** (tái tạo `skimage.feature.canny`, `mode='constant'`):
$$I_s=\frac{G_\sigma*I}{G_\sigma*\mathbf 1+\varepsilon}$$
Ở biên ảnh, phép chia này cho trung bình có trọng số chỉ trên phần ảnh có thật, thay vì kéo giá trị về 0 như đệm 0 thông thường.

**PSEUDOCODE.**
```
k = exp(-0.5 x² / σ²), x = -r..r ;  k /= sum(k)
I_s = correlate_separable(I, k, k, mode)
if bleedover: I_s /= correlate_separable(ones, k, k, 'constant') + eps
```

**IMPLEMENTATION.** [`core/gaussian.py`](../../src/cannylab/core/gaussian.py): `gaussian_kernel1d`, `gaussian_blur`, `opencv_auto_ksize`.

**PARAMETERS.** `sigma > 0` (ValueError nếu ≤ 0), `truncate` (mặc định 4), `mode`, `bleedover`.

**COMPLEXITY.** $O(2N(2r+1))=O(N\sigma)$.

**LIMITATIONS.** Đẳng hướng, làm mờ cả biên; không loại được nhiễu xung. Bản OpenCV trên `uint8` dùng số học điểm cố định nên bản Manual không tái tạo từng bit được (xem [06](06_canny_pipeline.md)).

**APPLICATIONS.** Bước 1 của Canny, tiền xử lý LoG, không gian tỉ lệ (scale-space).

**KẾT QUẢ.** `test_gaussian_kernel_normalized_and_blur_matches_scipy` (σ = 0.5–4): sai số ≤ 1e-12 so với `ndi.gaussian_filter`. Qua pipeline Canny hồ sơ SKIMAGE: 0 px lệch ([06](06_canny_pipeline.md)).
