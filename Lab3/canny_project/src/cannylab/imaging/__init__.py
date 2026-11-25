"""Tầng ảnh: kiểm tra hợp lệ, chuyển dtype/màu, đọc/ghi file, suy giảm có kiểm soát."""

from .convert import ensure_gray, rgb_to_gray, to_float, to_uint8
from .validate import InvalidImageError, check_gray, check_image

__all__ = ["InvalidImageError", "check_image", "check_gray", "to_float", "to_uint8", "rgb_to_gray", "ensure_gray"]
