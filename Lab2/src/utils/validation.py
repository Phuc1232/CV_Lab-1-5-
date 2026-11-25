
"""
Image Validation Module.

Cung cấp các chức năng kiểm tra tính hợp lệ
của dữ liệu ảnh NumPy.
"""

import numpy as np


def validate_image(
    image: np.ndarray,
    grayscale_only: bool = False,
    allowed_channels: tuple[int, ...] = (3,)
) -> None:
    """
    Kiểm tra cấu trúc và kiểu dữ liệu ảnh đầu vào.

    Parameters
    ----------
    image : np.ndarray
        Dữ liệu ảnh cần kiểm tra.
    grayscale_only : bool, default=False
        Chỉ chấp nhận ảnh grayscale 2D nếu True.
    allowed_channels : tuple[int, ...], default=(3,)
        Các số kênh màu được chấp nhận đối với ảnh 3D.

    Raises
    ------
    TypeError
        Nếu dữ liệu không phải NumPy array hoặc không phải uint8.
    ValueError
        Nếu ảnh rỗng hoặc cấu trúc ảnh không hợp lệ.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Ảnh phải là NumPy array.")

    if image.dtype != np.uint8:
        raise TypeError("Ảnh phải có kiểu dữ liệu uint8.")

    if image.size == 0:
        raise ValueError("Ảnh không được rỗng.")

    # Ảnh Grayscale
    if image.ndim == 2:
        return

    # Yêu cầu ảnh grayscale nhưng đầu vào là ảnh màu
    if grayscale_only:
        raise ValueError("Thuật toán yêu cầu ảnh Grayscale 2D.")

    # Kiểm tra ảnh màu
    if image.ndim == 3:
        channels = image.shape[2]

        if channels not in allowed_channels:
            raise ValueError(
                f"Số kênh ảnh không hợp lệ: {channels}."
            )

        return

    raise ValueError(
        f"Số chiều ảnh không hợp lệ: {image.ndim}."
    )
