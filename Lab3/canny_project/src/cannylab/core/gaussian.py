"""Làm trơn Gaussian tự cài đặt (bước 1 của Canny). Toán học: docs/mathematics/01_lam_tron_gaussian.md.

    g[x] = exp(-x^2 / (2 sigma^2)) / sum_u exp(-u^2 / (2 sigma^2)),   x in [-r, r],
    r = floor(truncate * sigma + 0.5)       (cùng quy ước với scipy.ndimage)

Tách được: G = g ⊗ g nên chi phí O(2 N k) thay vì O(N k^2).
Chuẩn hóa bleed-over (skimage canny, mode='constant'): I_s = (G*I) / (G*1 + eps), nghĩa là
trung bình có trọng số chỉ trên phần ảnh có thật ở gần biên.
"""

from __future__ import annotations

import numpy as np

from .convolution import correlate_separable


def gaussian_radius(sigma: float, truncate: float = 4.0) -> int:
    """Bán kính kernel: floor(truncate*sigma + 0.5)."""
    return int(truncate * float(sigma) + 0.5)


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Kernel Gauss 1D rời rạc đã chuẩn hóa tổng = 1."""
    if sigma <= 0:
        raise ValueError(f"sigma phải > 0, nhận {sigma}")
    if truncate <= 0:
        raise ValueError("truncate phải > 0")
    r = gaussian_radius(sigma, truncate)
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    return w / w.sum()


def opencv_auto_ksize(sigma: float, is_uint8: bool) -> int:
    """Kích thước kernel mà cv2.GaussianBlur tự chọn khi ksize=(0,0): round(sigma*(3|4)*2+1) | 1 (đã kiểm chứng)."""
    return int(round(sigma * (3 if is_uint8 else 4) * 2 + 1)) | 1


def gaussian_blur(image: np.ndarray, sigma: float, *, truncate: float = 4.0, mode: str = "reflect",
                  cval: float = 0.0, bleedover: bool = False) -> np.ndarray:
    """Làm trơn ảnh 2D float bằng Gaussian tách được. ``bleedover=True`` (chỉ với mode 'constant')
    chia cho Gaussian của mặt nạ toàn 1 cộng eps, tái tạo đúng ``skimage.feature.canny``."""
    if image.ndim != 2:
        raise ValueError("gaussian_blur cần ảnh 2D")
    img = image.astype(np.float64, copy=False)
    k = gaussian_kernel1d(sigma, truncate)
    smoothed = correlate_separable(img, k, k, mode, cval)
    if bleedover:
        if mode != "constant":
            raise ValueError("bleedover chỉ có nghĩa với mode='constant'")
        weight = correlate_separable(np.ones_like(img), k, k, "constant", 0.0) + np.finfo(np.float64).eps
        smoothed = smoothed / weight
    return smoothed
