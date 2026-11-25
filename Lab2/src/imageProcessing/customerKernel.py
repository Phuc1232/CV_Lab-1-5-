"""
Custom Kernels Module.

Cung cấp chức năng áp dụng kernel tùy chỉnh lên ảnh:

- Áp dụng bốn kernel mẫu có sẵn.
- Áp dụng kernel do người dùng thiết kế.
"""

import cv2
import numpy as np

from numpy.typing import NDArray

from src.utils import (
    validate_image,
    validate_kernel_matrix,
    validate_border_type
)


ImageArray = NDArray[np.uint8]


# ==========================================================
# KERNEL MẪU
# ==========================================================

_PRESET_KERNELS = [
    np.ones((3, 3), dtype=np.float32) / 9.0,

    np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    ),

    np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    ),

    np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ],
        dtype=np.float32
    )
]


# ==========================================================
# 1. ÁP DỤNG BỐN KERNEL MẪU
# ==========================================================

def apply_preset_kernels(
    image: ImageArray,
    border_type: int = cv2.BORDER_DEFAULT
) -> list[ImageArray]:
    """
    Áp dụng bốn kernel mẫu lên ảnh đầu vào.

    Args:
        image: Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh,
            kiểu uint8.
        border_type: Phương pháp xử lý pixel tại biên ảnh.
            Mặc định là cv2.BORDER_DEFAULT.

    Returns:
        Danh sách bốn ảnh kết quả theo thứ tự:
            1. Box Blur.
            2. Sharpen.
            3. Laplacian.
            4. Emboss.

    Raises:
        TypeError: Nếu ảnh đầu vào không đúng kiểu dữ liệu.
        ValueError: Nếu ảnh hoặc border_type không hợp lệ.
    """

    validate_image(image)
    border_type = validate_border_type(border_type)

    results = []

    for kernel in _PRESET_KERNELS:

        result = cv2.filter2D(
            image,
            ddepth=-1,
            kernel=kernel,
            borderType=border_type
        )

        results.append(result)

    return results


# ==========================================================
# 2. ÁP DỤNG KERNEL DO NGƯỜI DÙNG THIẾT KẾ
# ==========================================================

def apply_custom_kernel(
    image: ImageArray,
    kernel: np.ndarray,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Kiểm tra và áp dụng kernel do người dùng thiết kế lên ảnh.

    Args:
        image: Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh,
            kiểu uint8.
        kernel: Ma trận kernel do người dùng cung cấp.
            Kernel phải thỏa mãn các điều kiện hợp lệ.
        border_type: Phương pháp xử lý pixel tại biên ảnh.
            Mặc định là cv2.BORDER_DEFAULT.

    Returns:
        Ảnh sau khi lọc, có cùng kích thước, số kênh
        và kiểu dữ liệu uint8 với ảnh đầu vào.

    Raises:
        TypeError: Nếu ảnh hoặc kernel không đúng kiểu dữ liệu.
        ValueError: Nếu ảnh, kernel hoặc border_type
            không hợp lệ.
    """

    # Kiểm tra ảnh đầu vào
    validate_image(image)

    # Kiểm tra và chuẩn hóa kernel
    kernel = validate_kernel_matrix(kernel)

    # Kiểm tra phương pháp xử lý biên
    border_type = validate_border_type(border_type)

    # Áp dụng kernel
    result = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=border_type
    )

    return result
