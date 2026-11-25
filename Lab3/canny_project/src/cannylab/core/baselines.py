"""Bộ phát hiện cạnh đối chứng cho câu T1b: Sobel + một ngưỡng, và LoG + điểm qua 0 (Marr–Hildreth).

Toán học: docs/mathematics/02_gradient_va_laplacian.md. Cả hai làm trên ảnh float [0, 1].
"""

from __future__ import annotations

import numpy as np

from ..imaging.convert import to_float
from ..imaging.validate import check_gray
from .convolution import correlate2d
from .gaussian import gaussian_blur
from .gradient import magnitude, sobel

LAPLACE_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def sobel_threshold(image: np.ndarray, threshold: float = 0.3, sigma: float = 0.0) -> np.ndarray:
    """Cạnh = {M_L2 > threshold}, M tính trên ảnh [0,1] (thang của hồ sơ skimage). Không NMS, không hysteresis."""
    img = to_float(check_gray(image))
    if sigma > 0:
        img = gaussian_blur(img, sigma, mode="reflect")
    gx, gy = sobel(img, mode="reflect")
    return magnitude(gx, gy, "L2") > threshold


def laplacian_of_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """LoG rời rạc: Gaussian tách được rồi Laplacian 4 lân cận (ảnh [0,1], biên reflect)."""
    img = to_float(check_gray(image))
    if sigma > 0:
        img = gaussian_blur(img, sigma, mode="reflect")
    return correlate2d(img, LAPLACE_4, mode="reflect")


def zero_crossings(response: np.ndarray, threshold: float) -> np.ndarray:
    """Điểm qua 0: cặp lân cận (phải, dưới) trái dấu và chênh lệch > threshold; đánh dấu pixel có |giá trị| nhỏ hơn."""
    out = np.zeros(response.shape, dtype=bool)
    for axis in (0, 1):
        a = response[:-1, :] if axis == 0 else response[:, :-1]
        b = response[1:, :] if axis == 0 else response[:, 1:]
        cross = (a * b < 0) & (np.abs(a - b) > threshold)
        mark_a = cross & (np.abs(a) <= np.abs(b))
        mark_b = cross & ~mark_a
        if axis == 0:
            out[:-1, :] |= mark_a
            out[1:, :] |= mark_b
        else:
            out[:, :-1] |= mark_a
            out[:, 1:] |= mark_b
    return out


def log_zero_crossing(image: np.ndarray, sigma: float = 2.0, threshold_rel: float = 0.05) -> np.ndarray:
    """Marr–Hildreth: cạnh = điểm qua 0 của LoG có độ dốc > threshold_rel * max|LoG| (loại cạnh ảo)."""
    if threshold_rel < 0:
        raise ValueError("threshold_rel phải >= 0")
    response = laplacian_of_gaussian(image, sigma)
    return zero_crossings(response, threshold_rel * float(np.abs(response).max()))
