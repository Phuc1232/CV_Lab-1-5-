"""Kiểm tra ảnh đầu vào (quy tắc image-data-rules.md mục 1, 4, 5)."""

from __future__ import annotations

import numpy as np

SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)
FLOAT_RANGE_TOL = 1e-6  # sai số làm tròn cho phép quanh [0, 1] của ảnh float
MIN_SIZE = 3  # Sobel 3x3 cần ít nhất 3 pixel mỗi chiều


class InvalidImageError(ValueError):
    """Ảnh không hợp lệ: None, rỗng, sai số chiều, dtype không hỗ trợ, NaN/Inf, sai miền giá trị."""


def check_image(image: object, *, allow_color: bool = True, name: str = "image") -> np.ndarray:
    """Kiểm tra và trả về ảnh dưới dạng ``np.ndarray`` (không sao chép).

    Ảnh hợp lệ: 2D ``(H, W)`` hoặc 3D ``(H, W, 3)`` khi ``allow_color``; dtype thuộc
    ``SUPPORTED_DTYPES``; ảnh float hữu hạn và nằm trong [0, 1].
    """
    if image is None:
        raise InvalidImageError(f"{name} là None (đọc file thất bại hoặc chưa truyền ảnh)")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"{name} phải là numpy.ndarray, nhận {type(image).__name__}")
    if image.size == 0:
        raise InvalidImageError(f"{name} rỗng, shape={image.shape}")
    if image.ndim == 3 and allow_color:
        if image.shape[2] != 3:
            raise InvalidImageError(f"{name} màu phải có 3 kênh, nhận {image.shape[2]}")
    elif image.ndim != 2:
        expected = "2D hoặc (H, W, 3)" if allow_color else "2D (ảnh xám)"
        raise InvalidImageError(f"{name} phải là {expected}, nhận shape={image.shape}")
    if image.shape[0] < MIN_SIZE or image.shape[1] < MIN_SIZE:
        raise InvalidImageError(f"{name} quá nhỏ ({image.shape[:2]}), cần tối thiểu {MIN_SIZE}x{MIN_SIZE}")
    if image.dtype.type not in SUPPORTED_DTYPES:
        raise InvalidImageError(
            f"{name} có dtype {image.dtype} không được hỗ trợ; dùng uint8, uint16, float32 hoặc float64")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise InvalidImageError(f"{name} chứa NaN hoặc Inf")
        lo, hi = float(image.min()), float(image.max())
        if lo < -FLOAT_RANGE_TOL or hi > 1 + FLOAT_RANGE_TOL:
            raise InvalidImageError(
                f"{name} float phải nằm trong [0, 1], nhận [{lo:.4g}, {hi:.4g}]; "
                "chia cho 255 nếu đây là ảnh 8 bit đã ép sang float")
    return image


def check_gray(image: object, name: str = "image") -> np.ndarray:
    """Như :func:`check_image` nhưng bắt buộc ảnh xám 2D."""
    return check_image(image, allow_color=False, name=name)


def dtype_max(image: np.ndarray) -> float:
    """Giá trị lớn nhất danh nghĩa của dtype: 255 (uint8), 65535 (uint16), 1.0 (float)."""
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    return 1.0
