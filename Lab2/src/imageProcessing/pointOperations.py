
"""
Point Operations Module.

Cung cấp các phép biến đổi điểm ảnh:
- Điều chỉnh độ sáng.
- Điều chỉnh độ tương phản.
- Biến đổi ảnh âm bản.
- Phân ngưỡng ảnh nhị phân.
"""

import numpy as np

from src.utils import validate_image
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]



def adjust_brightness_contrast(
    image: ImageArray,
    alpha: float = 1.0,
    beta: float = 0.0
) -> ImageArray:
    """
    Điều chỉnh đồng thời độ sáng và độ tương phản của ảnh.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    alpha : float, default=1.0
        Hệ số điều chỉnh độ tương phản, phải lớn hơn 0.
    beta : float, default=0.0
        Độ lệch sáng, nằm trong khoảng [-255, 255].

    Returns
    -------
    np.ndarray
        Ảnh sau biến đổi, có cùng kích thước và kiểu uint8.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc tham số không đúng kiểu dữ liệu.
    ValueError
        Nếu alpha hoặc beta nằm ngoài phạm vi cho phép.

    Notes
    -----
    Công thức biến đổi tuyến tính:

        g(x, y) = clip(alpha * f(x, y) + beta, 0, 255)

    Giá trị sau biến đổi được làm tròn về số nguyên gần nhất.
    """

    validate_image(image)

    if alpha <= 0:
        raise ValueError("alpha phải lớn hơn 0.")

    if not -255 <= beta <= 255:
        raise ValueError("beta phải nằm trong [-255, 255].")

    result = (
        alpha * image.astype(np.float32) + beta
    )

    return np.rint(
        np.clip(result, 0, 255)
    ).astype(np.uint8)

def negative_image(image: ImageArray) -> ImageArray:
    """
    Tạo ảnh âm bản bằng cách đảo ngược giá trị từng pixel.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.

    Returns
    -------
    np.ndarray
        Ảnh âm bản có cùng kích thước và kiểu uint8.

    Raises
    ------
    TypeError
        Nếu ảnh không phải NumPy array kiểu uint8.
    ValueError
        Nếu cấu trúc ảnh không hợp lệ.

    Notes
    -----
    Công thức: g(x, y) = 255 - f(x, y).
    Phép biến đổi được thực hiện độc lập trên từng kênh màu.
    """

    validate_image(image)

    return 255 - image


def threshold_image(
    image: ImageArray,
    threshold: int = 128
) -> ImageArray:
    """
    Chuyển ảnh grayscale thành ảnh nhị phân bằng phương pháp cắt ngưỡng.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào bắt buộc là grayscale 2D, kiểu uint8.
    threshold : int, default=128
        Ngưỡng phân chia cường độ sáng, nằm trong [0, 255].

    Returns
    -------
    np.ndarray
        Ảnh nhị phân có cùng kích thước, chỉ chứa giá trị 0 và 255.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc threshold không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không phải grayscale hoặc ngưỡng ngoài phạm vi.

    Notes
    -----
    Quy tắc phân ngưỡng:

        g(x, y) = 255 nếu f(x, y) >= threshold
        g(x, y) = 0   nếu f(x, y) < threshold
    """

    validate_image(image, grayscale_only=True)

    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, np.integer)
    ):
        raise TypeError("threshold phải là số nguyên.")

    if not 0 <= threshold <= 255:
        raise ValueError(
            "threshold phải nằm trong khoảng [0, 255]."
        )

    result = np.where(
        image >= threshold,
        255,
        0
    )

    return result.astype(np.uint8)
