"""Gradient Sobel tự cài đặt (bước 2 của Canny). Toán học: docs/mathematics/02_gradient_va_laplacian.md.

    K_x = [1, 2, 1]^T ⊗ [-1, 0, 1],   K_y = K_x^T
    M_L2 = sqrt(Gx^2 + Gy^2),  M_L1 = |Gx| + |Gy|,  theta = atan2(Gy, Gx)

Ảnh nguyên (uint8) được tích lũy bằng int64 nên kết quả trùng tuyệt đối với cv2.Sobel(CV_16S).
"""

from __future__ import annotations

import numpy as np

from .convolution import correlate_separable

SOBEL_DERIVATIVE = np.array([-1, 0, 1], dtype=np.int64)
SOBEL_SMOOTHING = np.array([1, 2, 1], dtype=np.int64)
NORMS = ("L1", "L2")


def sobel(image: np.ndarray, mode: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    """Trả ``(gx, gy)``: gx đạo hàm theo cột (trục 1), gy theo hàng (trục 0)."""
    if image.ndim != 2:
        raise ValueError("sobel cần ảnh 2D")
    gx = correlate_separable(image, SOBEL_SMOOTHING, SOBEL_DERIVATIVE, mode)
    gy = correlate_separable(image, SOBEL_DERIVATIVE, SOBEL_SMOOTHING, mode)
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray, norm: str = "L2") -> np.ndarray:
    """Độ lớn gradient theo chuẩn L1 hoặc L2. Với L1, đầu vào nguyên cho đầu ra nguyên (int64)."""
    if norm == "L2":
        return np.hypot(gx.astype(np.float64), gy.astype(np.float64))
    if norm == "L1":
        return np.abs(gx) + np.abs(gy)
    raise ValueError(f"norm phải là một trong {NORMS}, nhận {norm!r}")


def direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Hướng gradient theta = atan2(gy, gx) trong (-pi, pi]; trục y hướng xuống."""
    return np.arctan2(gy.astype(np.float64), gx.astype(np.float64))
