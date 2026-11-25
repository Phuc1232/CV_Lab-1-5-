"""Thuật toán TỰ CÀI ĐẶT bằng NumPy. Gói này KHÔNG được import cv2 hay skimage (có test kiểm tra).

Tài liệu toán học tương ứng: docs/mathematics/01..07.
"""

from .canny import OPENCV, SKIMAGE, CannyProfile, CannyResult, canny

__all__ = ["canny", "CannyProfile", "CannyResult", "OPENCV", "SKIMAGE"]
