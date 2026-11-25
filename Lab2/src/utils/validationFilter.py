
"""
Filter Validation Module.

Cung cấp các chức năng kiểm tra tham số
dùng chung cho các bộ lọc xử lý ảnh:
- Kernel size.
- Kernel matrix.
- Border type.
- Non-negative integer parameters.
"""

import cv2
import numpy as np

from numpy.typing import NDArray


KernelArray = NDArray[np.float32]


# Các phương pháp xử lý biên hỗ trợ chung
VALID_BORDER_TYPES = frozenset({
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT,
    cv2.BORDER_REFLECT_101
})


def validate_kernel_size(
    kernel_size: int,
    *,
    name: str = "kernel_size",
    min_size: int = 1,
    max_size: int | None = None,
    allow_zero: bool = False,
    require_odd: bool = True
) -> int:
    """
    Kiểm tra tính hợp lệ của kích thước kernel.

    Parameters
    ----------
    kernel_size : int
        Kích thước kernel cần kiểm tra.
    name : str, default="kernel_size"
        Tên tham số dùng trong thông báo lỗi.
    min_size : int, default=1
        Kích thước nhỏ nhất được chấp nhận.
    max_size : int, optional
        Kích thước lớn nhất được chấp nhận.
    allow_zero : bool, default=False
        Cho phép kernel_size bằng 0 nếu True.
    require_odd : bool, default=True
        Yêu cầu kích thước kernel phải là số lẻ.

    Returns
    -------
    int
        Kích thước kernel sau khi kiểm tra.

    Raises
    ------
    TypeError
        Nếu kernel_size không phải số nguyên.
    ValueError
        Nếu kích thước kernel nằm ngoài phạm vi
        hoặc không thỏa mãn điều kiện chẵn lẻ.
    """

    if isinstance(kernel_size, (bool, np.bool_)) or not isinstance(
        kernel_size, (int, np.integer)
    ):
        raise TypeError(f"{name} phải là số nguyên.")

    kernel_size = int(kernel_size)

    # Một số thuật toán cho phép bằng 0
    if allow_zero and kernel_size == 0:
        return kernel_size

    if kernel_size < min_size:
        raise ValueError(
            f"{name} phải lớn hơn hoặc bằng {min_size}."
        )

    if max_size is not None and kernel_size > max_size:
        raise ValueError(
            f"{name} không được lớn hơn {max_size}."
        )

    if require_odd and kernel_size % 2 == 0:
        raise ValueError(
            f"{name} phải là số lẻ."
        )

    return kernel_size


def validate_kernel_matrix(
    kernel: np.ndarray,
    min_size: int = 1
) -> KernelArray:
    """
    Kiểm tra tính hợp lệ của ma trận kernel tùy chỉnh.

    Parameters
    ----------
    kernel : np.ndarray
        Ma trận kernel cần kiểm tra.
    min_size : int, default=1
        Kích thước tối thiểu của mỗi chiều kernel.

    Returns
    -------
    np.ndarray
        Kernel sau khi chuyển sang float32.

    Raises
    ------
    TypeError
        Nếu kernel không phải NumPy array
        hoặc không chứa dữ liệu số.
    ValueError
        Nếu kernel rỗng, không phải ma trận 2D,
        có kích thước không hợp lệ hoặc chứa
        giá trị không hữu hạn.
    """

    if not isinstance(kernel, np.ndarray):
        raise TypeError("kernel phải là numpy.ndarray.")

    if kernel.size == 0:
        raise ValueError("kernel không được rỗng.")

    if kernel.ndim != 2:
        raise ValueError("kernel phải là ma trận 2 chiều.")

    if isinstance(min_size, bool) or not isinstance(
        min_size, (int, np.integer)
    ) or min_size < 1:
        raise ValueError("min_size phải là số nguyên dương.")

    rows, cols = kernel.shape

    if rows < min_size or cols < min_size:
        raise ValueError(
            f"Mỗi chiều kernel phải có kích thước "
            f"ít nhất {min_size}."
        )

    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError(
            "Số hàng và số cột của kernel phải là số lẻ."
        )

    if (
        not np.issubdtype(kernel.dtype, np.number)
        or np.issubdtype(kernel.dtype, np.complexfloating)
    ):
        raise TypeError(
            "Các phần tử kernel phải thuộc kiểu dữ liệu số thực."
        )

    # Chuyển đổi sang float32
    with np.errstate(over="ignore", invalid="ignore"):
        kernel = kernel.astype(np.float32)

    if not np.all(np.isfinite(kernel)):
        raise ValueError(
            "kernel không được chứa NaN hoặc Infinity."
        )

    return kernel


def validate_border_type(
    border_type: int,
    allowed_types: frozenset[int] | None = None
) -> int:
    """
    Kiểm tra kiểu xử lý pixel tại biên ảnh.

    Parameters
    ----------
    border_type : int
        Hằng số BORDER của OpenCV.
    allowed_types : frozenset[int], optional
        Danh sách các kiểu biên được phép.
        Mặc định sử dụng VALID_BORDER_TYPES.

    Returns
    -------
    int
        Giá trị border_type hợp lệ.

    Raises
    ------
    TypeError
        Nếu border_type không phải số nguyên.
    ValueError
        Nếu border_type không được hỗ trợ.
    """

    if isinstance(border_type, (bool, np.bool_)) or not isinstance(
        border_type, (int, np.integer)
    ):
        raise TypeError("border_type phải là số nguyên.")

    border_type = int(border_type)

    if allowed_types is None:
        allowed_types = VALID_BORDER_TYPES

    if border_type not in allowed_types:
        raise ValueError(
            f"border_type không được hỗ trợ: {border_type}."
        )

    return border_type


def validate_non_negative_int(
    value: int,
    name: str = "value"
) -> int:
    """
    Kiểm tra tham số số nguyên không âm.

    Parameters
    ----------
    value : int
        Giá trị cần kiểm tra.
    name : str, default="value"
        Tên tham số dùng trong thông báo lỗi.

    Returns
    -------
    int
        Giá trị sau khi kiểm tra.

    Raises
    ------
    TypeError
        Nếu value không phải số nguyên.
    ValueError
        Nếu value nhỏ hơn 0.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} phải là số nguyên.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} không được nhỏ hơn 0.")

    return value
