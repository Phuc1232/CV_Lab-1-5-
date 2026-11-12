"""Chuyển dtype và không gian màu có kiểm soát (image-data-rules.md mục 2, 3)."""

from __future__ import annotations

import numpy as np

from .validate import check_image, dtype_max

# ITU-R BT.601: cùng hệ số với cv2.COLOR_RGB2GRAY và Lab1
GRAY_WEIGHTS_RGB = np.array([0.299, 0.587, 0.114])


def to_float(image: np.ndarray) -> np.ndarray:
    """Đưa ảnh về float64 trong [0, 1]. uint8 chia 255, uint16 chia 65535; float được clip sai số."""
    image = check_image(image)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float64) / dtype_max(image)
    return np.clip(image.astype(np.float64), 0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Đưa ảnh về uint8 [0, 255] bằng làm tròn rồi clip (không ép kiểu trực tiếp, tránh quấn số)."""
    image = check_image(image)
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return np.clip(np.rint(image / 257.0), 0, 255).astype(np.uint8)
    return np.clip(np.rint(image.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) sang xám: Y = 0.299R + 0.587G + 0.114B. Trả float64 [0,1] nếu vào là float, uint8 nếu vào uint8.

    Với uint8, làm tròn về số nguyên gần nhất giống ``cv2.cvtColor`` (sai lệch tối đa 1 mức xám
    do OpenCV dùng số học điểm cố định).
    """
    image = check_image(image)
    if image.ndim != 3:
        raise ValueError("rgb_to_gray cần ảnh (H, W, 3)")
    gray = image.astype(np.float64) @ GRAY_WEIGHTS_RGB
    if image.dtype == np.uint8:
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    if image.dtype == np.uint16:
        return np.clip(np.rint(gray), 0, 65535).astype(np.uint16)
    return np.clip(gray, 0.0, 1.0)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Trả ảnh xám: giữ nguyên ảnh 2D, chuyển ảnh RGB bằng :func:`rgb_to_gray`."""
    image = check_image(image)
    return image if image.ndim == 2 else rgb_to_gray(image)
