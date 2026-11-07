"""Suy giảm ảnh có kiểm soát để tạo ca thử P3 (nhiễu, tương phản thấp, mờ). Mọi hàm nhận/trả float [0,1]."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .convert import to_float


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Cộng nhiễu Gauss N(0, sigma^2) (sigma theo thang [0,1]) rồi clip về [0,1]."""
    if sigma < 0:
        raise ValueError("sigma nhiễu phải >= 0")
    img = to_float(image)
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0.0, 1.0)


def add_salt_pepper(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Nhiễu muối tiêu: tỉ lệ ``amount`` pixel bị đặt về 0 hoặc 1 (mỗi loại một nửa)."""
    if not 0 <= amount <= 1:
        raise ValueError("amount phải trong [0, 1]")
    img = to_float(image).copy()
    u = rng.random(img.shape[:2])
    img[u < amount / 2] = 0.0
    img[(u >= amount / 2) & (u < amount)] = 1.0
    return img


def reduce_contrast(image: np.ndarray, factor: float, center: float = 0.5) -> np.ndarray:
    """Co miền giá trị quanh ``center``: I' = center + factor (I - center), 0 < factor <= 1."""
    if not 0 < factor <= 1:
        raise ValueError("factor phải trong (0, 1]")
    return np.clip(center + factor * (to_float(image) - center), 0.0, 1.0)


def optical_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Mờ quang học mô phỏng bằng Gaussian (SciPy; đây là bước tạo dữ liệu, không phải thuật toán đang xét)."""
    img = to_float(image)
    if sigma <= 0:
        return img
    sig = (sigma, sigma, 0) if img.ndim == 3 else sigma
    return ndi.gaussian_filter(img, sig, mode="nearest")


def correlated_texture(shape: tuple[int, int], scale: float, amplitude: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Texture ngẫu nhiên: nhiễu trắng lọc Gaussian (độ hạt ``scale`` px), chuẩn hóa std = ``amplitude``."""
    field = ndi.gaussian_filter(rng.normal(size=shape), scale, mode="wrap")
    return amplitude * field / (field.std() + 1e-12)
