
"""
Edge Detection Module.

Cung cấp các thuật toán phát hiện cạnh trong ảnh:
- Sobel Operator.
- Prewitt Operator.
"""

import cv2
import numpy as np

from numpy.typing import NDArray
from src.utils import validate_image, validate_kernel_size, validate_border_type


ImageArray = NDArray[np.uint8]



def _calculate_gradient_magnitude(
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32]
) -> ImageArray:
    """
    Tính độ lớn gradient và chuẩn hóa thành ảnh uint8.

    Parameters
    ----------
    gradient_x : np.ndarray
        Ma trận gradient theo hướng X.
    gradient_y : np.ndarray
        Ma trận gradient theo hướng Y.

    Returns
    -------
    np.ndarray
        Ảnh độ lớn gradient đã chuẩn hóa về [0, 255],
        có kiểu dữ liệu uint8.
    """

    # Tính độ lớn gradient
    magnitude = cv2.magnitude(gradient_x, gradient_y)

    min_value = float(np.min(magnitude))
    max_value = float(np.max(magnitude))

    # Xử lý trường hợp gradient không thay đổi
    if np.isclose(max_value, min_value):
        if max_value <= 0:
            return np.zeros(magnitude.shape, dtype=np.uint8)

        return np.full(magnitude.shape, 255, dtype=np.uint8)

    # Chuẩn hóa về khoảng [0, 255]
    normalized = cv2.normalize(
        magnitude,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    return np.clip(
        np.rint(normalized),
        0,
        255
    ).astype(np.uint8)


def sobel_edge_detection(
    image: ImageArray,
    kernel_size: int = 3,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Phát hiện cạnh bằng toán tử Sobel (Sobel Operator).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào bắt buộc là grayscale 2D, kiểu uint8.
    kernel_size : int, default=3
        Kích thước kernel Sobel, là số lẻ trong khoảng [3, 31].
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh biên sau khi tính độ lớn gradient và chuẩn hóa
        về khoảng [0, 255], kiểu uint8.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc tham số không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không phải grayscale hoặc tham số không hợp lệ.

    Notes
    -----
    Sobel sử dụng hai kernel để tính gradient theo hướng X và Y.

    Kernel X:

        [-1   0   1]
        [-2   0   2]
        [-1   0   1]

    Kernel Y:

        [-1  -2  -1]
        [ 0   0   0]
        [ 1   2   1]

    Độ lớn gradient:

        G(x, y) = sqrt(Gx² + Gy²)

    Với kernel_size lớn hơn 3, OpenCV tự xây dựng
    kernel Sobel tương ứng với kích thước được chỉ định.
    """

    # Kiểm tra ảnh grayscale
    validate_image(image, grayscale_only=True)

    # Kiểm tra tham số
    kernel_size = validate_kernel_size(kernel_size)
    border_type = validate_border_type(border_type)

    # Tính gradient theo hướng X
    gradient_x = cv2.Sobel(
        image,
        ddepth=cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=kernel_size,
        borderType=border_type
    )

    # Tính gradient theo hướng Y
    gradient_y = cv2.Sobel(
        image,
        ddepth=cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=kernel_size,
        borderType=border_type
    )

    # Tính độ lớn gradient và chuẩn hóa
    return _calculate_gradient_magnitude(
        gradient_x,
        gradient_y
    )


def prewitt_edge_detection(
    image: ImageArray,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Phát hiện cạnh bằng toán tử Prewitt (Prewitt Operator).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào bắt buộc là grayscale 2D, kiểu uint8.
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh biên sau khi tính độ lớn gradient và chuẩn hóa
        về khoảng [0, 255], kiểu uint8.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc border_type không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không phải grayscale hoặc border_type không hợp lệ.

    Notes
    -----
    Prewitt sử dụng hai kernel 3x3 để tính gradient.

    Kernel X:

        [-1   0   1]
        [-1   0   1]
        [-1   0   1]

    Kernel Y:

        [-1  -1  -1]
        [ 0   0   0]
        [ 1   1   1]

    Độ lớn gradient:

        G(x, y) = sqrt(Gx² + Gy²)

    Phép tính gradient được thực hiện bằng cv2.filter2D()
    với kiểu đầu ra CV_32F để giữ được giá trị âm và dương.
    """

    # Kiểm tra ảnh grayscale
    validate_image(image, grayscale_only=True)

    # Kiểm tra tham số
    border_type = validate_border_type(border_type)

    # Kernel Prewitt theo hướng X
    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        dtype=np.float32
    )

    # Kernel Prewitt theo hướng Y
    kernel_y = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        dtype=np.float32
    )

    # Tính gradient theo hướng X
    gradient_x = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel_x,
        borderType=border_type
    )

    # Tính gradient theo hướng Y
    gradient_y = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel_y,
        borderType=border_type
    )

    # Tính độ lớn gradient và chuẩn hóa
    return _calculate_gradient_magnitude(
        gradient_x,
        gradient_y
    )
