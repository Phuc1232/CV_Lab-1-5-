"""Tiền lọc thay cho Gaussian (câu E2): median và bilateral tự cài đặt. Làm trên float [0, 1].

median   : trung vị trong cửa sổ k x k. Loại nhiễu xung (muối tiêu) vì giá trị cực trị bị đẩy ra
           hai đầu dãy sắp xếp. O(N k^2) bộ nhớ khi vector hóa bằng sliding_window_view.
bilateral: I'(p) = sum_q w_s(p,q) w_r(p,q) I(q) / sum_q w_s w_r,
           w_s = exp(-|p-q|^2 / (2 sigma_space^2)), w_r = exp(-(I(p)-I(q))^2 / (2 sigma_color^2)),
           q trong hình tròn bán kính d//2 (như OpenCV). Làm trơn vùng phẳng nhưng giữ biên. O(N d^2).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..imaging.convert import to_float
from ..imaging.validate import check_gray
from .convolution import pad


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("size phải là số lẻ >= 1")
    img = to_float(check_gray(image))
    r = size // 2
    windows = sliding_window_view(pad(img, r, r, "reflect101"), (size, size))
    return np.median(windows, axis=(-2, -1))


def bilateral_filter(image: np.ndarray, diameter: int = 5, sigma_color: float = 0.1,
                     sigma_space: float = 2.0) -> np.ndarray:
    if diameter < 1 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("diameter >= 1, sigma_color > 0, sigma_space > 0")
    img = to_float(check_gray(image))
    r = diameter // 2
    padded = pad(img, r, r, "reflect101")
    h, w = img.shape
    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist2 = dy * dy + dx * dx
            if dist2 > r * r:
                continue
            q = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            weight = np.exp(-dist2 / (2 * sigma_space ** 2)) * np.exp(-((q - img) ** 2) / (2 * sigma_color ** 2))
            num += weight * q
            den += weight
    return num / den
