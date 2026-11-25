
"""
Linear Filters Module.

Cung cấp các bộ lọc tuyến tính trong xử lý ảnh:
- Mean Filter: Làm mờ ảnh bằng phép tính trung bình.
- Gaussian Filter: Làm mờ ảnh bằng phân phối Gaussian.
- Sharpen Filter: Làm sắc nét ảnh bằng kernel tích chập.
"""

import cv2
import numpy as np
import warnings

from numpy.typing import NDArray
from src.utils import validate_image, validate_kernel_size,  validate_border_type


ImageArray = NDArray[np.uint8]
KernelArray = NDArray[np.float32]


def mean_filter(
    image: ImageArray,
    kernel_size: int = 3,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Làm mờ ảnh bằng bộ lọc trung bình (Mean Filter).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    kernel_size : int, default=3
        Kích thước kernel hình vuông, phải là số nguyên dương lẻ.
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh sau khi lọc trung bình, có cùng kích thước
        và kiểu dữ liệu với ảnh đầu vào.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc tham số không đúng kiểu dữ liệu.
    ValueError
        Nếu ảnh không hợp lệ, kernel_size hoặc border_type
        không thỏa mãn điều kiện.

    Notes
    -----
    Mỗi pixel đầu ra được tính bằng trung bình cộng
    các pixel trong vùng lân cận kernel_size x kernel_size.

        g(x, y) = (1 / k²) * sum(f(i, j))

    Trong đó k là kích thước kernel.
    """

    # Kiểm tra dữ liệu đầu vào
    validate_image(image)

    kernel_size = validate_kernel_size(kernel_size)
    border_type = validate_border_type(border_type)

    # Thực hiện lọc trung bình
    return cv2.blur(
        image,
        (kernel_size, kernel_size),
        borderType=border_type
    )


def gaussian_filter(
    image: ImageArray,
    kernel_size: int = 3,
    sigma: float = 0.0,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Làm mờ ảnh bằng bộ lọc Gaussian (Gaussian Filter).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    kernel_size : int, default=3
        Kích thước kernel hình vuông, phải là số nguyên dương lẻ.
        Có thể bằng 0 để OpenCV tự xác định từ sigma.
    sigma : float, default=0.0
        Độ lệch chuẩn của phân phối Gaussian.
        Phải không âm; bằng 0 cho phép OpenCV tự tính từ kernel_size.
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh sau khi lọc Gaussian, có cùng kích thước
        và kiểu dữ liệu với ảnh đầu vào.

    Raises
    ------
    TypeError
        Nếu ảnh hoặc tham số không đúng kiểu dữ liệu.
    ValueError
        Nếu kernel_size, sigma hoặc border_type không hợp lệ.

    Notes
    -----
    Kernel Gaussian hai chiều được xác định bởi:

        G(x, y) = (1 / (2*pi*sigma²))
                  * exp(-(x²+y²)/(2*sigma²))

    Khi sigma bằng 0 và kernel_size lớn hơn 0,
    OpenCV tự xác định sigma phù hợp với kích thước kernel.
    """

    # Kiểm tra dữ liệu ảnh
    validate_image(image)

    kernel_size = validate_kernel_size(
        kernel_size,
        allow_zero=True
    )

    border_type = validate_border_type(border_type)

    # Kiểm tra sigma
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(
        sigma, (int, float, np.integer, np.floating)
    ):
        raise TypeError("sigma phải là một giá trị số.")

    sigma = float(sigma)

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma phải là số hữu hạn không âm.")

    if kernel_size == 0 and sigma == 0:
        raise ValueError(
            "Khi kernel_size bằng 0, sigma phải lớn hơn 0."
        )

    # Thực hiện lọc Gaussian
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=border_type
    )


def sharpen_filter(
    image: ImageArray,
    kernel: np.ndarray | None = None,
    border_type: int = cv2.BORDER_DEFAULT
) -> ImageArray:
    """
    Làm sắc nét ảnh bằng bộ lọc tích chập (Sharpen Filter).

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
    kernel : np.ndarray, optional
        Ma trận kernel tùy chỉnh, phải là ma trận số 2D
        có kích thước lẻ và ít nhất 3x3.
        Mặc định sử dụng kernel làm sắc nét 3x3.
    border_type : int, default=cv2.BORDER_DEFAULT
        Phương pháp xử lý pixel tại biên ảnh.

    Returns
    -------
    np.ndarray
        Ảnh sau khi làm sắc nét, có cùng kích thước
        và kiểu dữ liệu với ảnh đầu vào.

    Raises
    ------
    TypeError
        Nếu ảnh, kernel hoặc border_type không đúng kiểu dữ liệu.
    ValueError
        Nếu kernel rỗng, không phải ma trận 2D,
        có kích thước không hợp lệ hoặc chứa giá trị không hữu hạn.

    Notes
    -----
    Phép lọc được thực hiện bằng phép tích chập:

        g(x, y) = f(x, y) * h(x, y)

    Trong đó:
        f: Ảnh đầu vào.
        h: Kernel làm sắc nét.

    Kernel có tổng trọng số gần bằng 1 thường được sử dụng
    để bảo toàn độ sáng trung bình của ảnh.
    """

    # Kiểm tra dữ liệu ảnh
    validate_image(image)

    border_type = validate_border_type(border_type)

    # Kernel mặc định dùng để làm sắc nét
    if kernel is None:
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            dtype=np.float32
        )

    else:
        # Kiểm tra kiểu dữ liệu kernel
        if not isinstance(kernel, np.ndarray):
            raise TypeError("kernel phải là numpy.ndarray.")

        # Kiểm tra số chiều
        if kernel.ndim != 2:
            raise ValueError("kernel phải là ma trận 2 chiều.")

        # Kiểm tra kích thước
        rows, cols = kernel.shape

        if rows < 3 or cols < 3:
            raise ValueError(
                "Kernel làm sắc nét phải có kích thước "
                "ít nhất 3x3."
            )

        if rows % 2 == 0 or cols % 2 == 0:
            raise ValueError(
                "Số hàng và số cột của kernel phải là số lẻ."
            )

        # Kiểm tra kiểu dữ liệu số
        if (
            not np.issubdtype(kernel.dtype, np.number)
            or np.issubdtype(kernel.dtype, np.complexfloating)
        ):
            raise TypeError(
                "Các phần tử kernel phải có kiểu dữ liệu số thực."
            )

        # Chuyển kernel sang float32
        kernel = kernel.astype(np.float32)

        # Kiểm tra giá trị hữu hạn
        if not np.all(np.isfinite(kernel)):
            raise ValueError(
                "Kernel không được chứa NaN hoặc Infinity."
            )

        # Kiểm tra tổng trọng số kernel
        kernel_sum = float(np.sum(kernel, dtype=np.float64))

        if not np.isclose(kernel_sum, 1.0, atol=1e-5):
            warnings.warn(
                f"Tổng trọng số kernel = {kernel_sum:.4f}. "
                "Kernel có tổng gần bằng 1 thường giúp "
                "bảo toàn độ sáng trung bình của ảnh.",
                RuntimeWarning,
                stacklevel=2
            )

    # Thực hiện làm sắc nét
    return cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=border_type
    )
