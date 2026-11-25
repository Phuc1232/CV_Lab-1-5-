
"""
Nonlinear Filters Module.

Cung cấp các bộ lọc phi tuyến tính trong xử lý ảnh:
- Median Filter: Khử nhiễu bằng giá trị trung vị.
- Bilateral Filter: Làm mượt ảnh đồng thời bảo toàn đường biên.
"""

import cv2
import numpy as np

from numpy.typing import NDArray
from src.utils import validate_image, validate_kernel_size, validate_border_type, validate_non_negative_int


ImageArray = NDArray[np.uint8]


def median_filter(
    image: ImageArray,
    kernel_size: int = 3
) -> ImageArray:
    """
    Khử nhiễu ảnh bằng bộ lọc trung vị (Median Filter).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    kernel_size : int, default=3
        Kích thước vùng lân cận hình vuông, phải là số nguyên
        lẻ lớn hơn 1, ví dụ: 3, 5, 7.

    Returns
    -------
    np.ndarray
        Ảnh sau khi lọc trung vị, có cùng kích thước
        và kiểu dữ liệu với ảnh đầu vào.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc kernel_size không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không hợp lệ hoặc kernel_size không thỏa mãn
        điều kiện kích thước.

    Notes
    -----
    Mỗi pixel đầu ra được thay thế bằng giá trị trung vị
    của các pixel trong vùng lân cận.

        g(x, y) = median{f(i, j) | (i, j) thuộc W}

    Trong đó W là vùng lân cận có kích thước
    kernel_size x kernel_size.

    Bộ lọc đặc biệt phù hợp để khử nhiễu muối tiêu
    (Salt-and-Pepper Noise).

    Đối với ảnh màu, OpenCV thực hiện lọc riêng trên từng kênh.
    Phương pháp xử lý biên được OpenCV cố định bên trong hàm.
    """

    # Kiểm tra dữ liệu ảnh
    validate_image(image)

    # Kiểm tra kích thước vùng lọc
    kernel_size = validate_kernel_size(kernel_size)

    # Thực hiện lọc trung vị
    return cv2.medianBlur(
        image,
        ksize=kernel_size
    )


def bilateral_filter(
    image: ImageArray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Làm mượt ảnh bằng bộ lọc Bilateral (Bilateral Filter).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    diameter : int, default=9
        Đường kính vùng lân cận dùng để lọc.
        Nếu bằng 0, OpenCV tự xác định kích thước từ sigma_space.
    sigma_color : float, default=75.0
        Độ lệch chuẩn trong không gian cường độ màu.
        Giá trị lớn cho phép các pixel có mức chênh lệch
        màu sắc lớn hơn ảnh hưởng đến pixel trung tâm.
    sigma_space : float, default=75.0
        Độ lệch chuẩn trong không gian tọa độ.
        Giá trị lớn cho phép các pixel ở xa hơn
        tham gia vào quá trình lọc.
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh sau khi lọc Bilateral, có cùng kích thước
        và kiểu dữ liệu với ảnh đầu vào.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc các tham số không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không hợp lệ, diameter âm,
        sigma không hữu hạn hoặc không lớn hơn 0,
        hoặc border_type không được hỗ trợ.

    Notes
    -----
    Bilateral Filter kết hợp hai thành phần trọng số:

    1. Trọng số không gian: Phụ thuộc khoảng cách vị trí.
    2. Trọng số cường độ: Phụ thuộc mức chênh lệch màu sắc.

    Nhờ đó, các pixel gần nhau và có cường độ tương đồng
    sẽ có ảnh hưởng lớn hơn đến giá trị pixel đầu ra.

    Bộ lọc giúp giảm nhiễu trong khi hạn chế làm mờ đường biên.
    Tuy nhiên, chi phí tính toán thường cao hơn các bộ lọc
    làm mượt thông thường.
    """

    # Kiểm tra dữ liệu ảnh
    validate_image(image)

    # Kiểm tra diameter
    if isinstance(diameter, (bool, np.bool_)) or not isinstance(
        diameter, (int, np.integer)
    ):
        raise TypeError("diameter phải là số nguyên.")

    diameter = int(diameter)

    if diameter < 0:
        raise ValueError(
            "diameter phải lớn hơn hoặc bằng 0."
        )

    

    # Kiểm tra phương pháp xử lý biên
    border_type = validate_border_type(border_type)

    # Thực hiện lọc Bilateral
    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=border_type
    )
