
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np


# Các định dạng ảnh hỗ trợ
SUPPORTED_READ_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

SUPPORTED_WRITE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


ImageMode = Literal["BGR", "RGB", "GRAY", "UNCHANGED"]
ColorOrder = Literal["AUTO", "BGR", "RGB", "GRAY", "BGRA", "RGBA"]


def read_image(
    path: str | Path,
    mode: ImageMode = "BGR"
) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn và chuyển đổi sang chế độ màu yêu cầu.

    Parameters:
        path: Đường dẫn đến file ảnh.
        mode: BGR, RGB, GRAY hoặc UNCHANGED.

    Returns:
        Ảnh dưới dạng NumPy array.

    Raises:
        FileNotFoundError: Không tìm thấy file.
        ValueError: Định dạng không hợp lệ hoặc không đọc được ảnh.
    """

    image_path = Path(path).expanduser()
    mode = mode.upper().strip()

    if mode == "GRAYSCALE":
        mode = "GRAY"

    # 1. Kiểm tra chế độ đọc
    valid_modes = {"BGR", "RGB", "GRAY", "UNCHANGED"}

    if mode not in valid_modes:
        raise ValueError(
            f"Chế độ đọc không hợp lệ: {mode}"
        )

    # 2. Kiểm tra đường dẫn
    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Đường dẫn không phải file: {image_path}"
        )

    # 3. Kiểm tra định dạng
    if image_path.suffix.lower() not in SUPPORTED_READ_FORMATS:
        raise ValueError(
            f"Định dạng ảnh không được hỗ trợ: {image_path.suffix}"
        )

    # 4. Thiết lập chế độ đọc
    flags = {
        "BGR": cv2.IMREAD_COLOR,
        "RGB": cv2.IMREAD_COLOR,
        "GRAY": cv2.IMREAD_GRAYSCALE,
        "UNCHANGED": cv2.IMREAD_UNCHANGED
    }

    # 5. Đọc dữ liệu ảnh
    try:
        image_data = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        if image_data.size == 0:
            raise ValueError("File ảnh rỗng.")

        # Nếu `mode` là "RGB", tại bước này ảnh vẫn đang được giải mã thành hệ màu BGR theo cơ chế mặc định của OpenCV
        img = cv2.imdecode(image_data, flags[mode])

    except OSError as error:
        raise OSError(
            f"Lỗi khi đọc file ảnh: {error}"
        ) from error

    # 6. Kiểm tra kết quả giải mã
    if img is None:
        raise ValueError(
            f"Không thể giải mã ảnh: {image_path}"
        )

    # 7. Chuyển đổi BGR sang RGB nếu cần
    if mode == "RGB":
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    return img


def write_image(
    path: str | Path,
    img: np.ndarray,
    color_order: ColorOrder = "AUTO",
    create_dirs: bool = True,
    params: Sequence[int] | None = None
) -> Path:
    """
    Lưu ảnh NumPy array xuống ổ đĩa.

    Parameters:
        path: Đường dẫn file đầu ra.
        img: Dữ liệu ảnh cần lưu.
        color_order: AUTO, BGR, RGB, GRAY, BGRA hoặc RGBA.
        create_dirs: Tự động tạo thư mục nếu chưa tồn tại.
        params: Các tham số mã hóa ảnh của OpenCV.

    Returns:
        Đường dẫn file đã lưu.

    Raises:
        ValueError: Dữ liệu hoặc định dạng ảnh không hợp lệ.
        TypeError: Kiểu dữ liệu ảnh không được hỗ trợ.
        RuntimeError: Không thể mã hóa ảnh.
    """

    image_path = Path(path).expanduser()
    color_order = color_order.upper().strip()

    # 1. Kiểm tra dữ liệu ảnh
    if not isinstance(img, np.ndarray):
        raise TypeError("Dữ liệu đầu vào phải là NumPy array.")

    if img.size == 0:
        raise ValueError("Ảnh đầu vào rỗng.")

    if img.dtype not in (np.uint8, np.uint16):
        raise TypeError(
            "Ảnh phải có kiểu dữ liệu uint8 hoặc uint16. "
            "Hãy chuyển đổi ảnh trước khi lưu."
        )

    # 2. Kiểm tra định dạng file đầu ra
    extension = image_path.suffix.lower()

    if extension not in SUPPORTED_WRITE_FORMATS:
        raise ValueError(
            f"Định dạng lưu ảnh không được hỗ trợ: {extension}"
        )

    # 3. Kiểm tra thứ tự kênh màu và chuyển đổi nếu cần
    if img.ndim == 2:
        if color_order not in {"AUTO", "GRAY"}:
            raise ValueError(
                "Ảnh grayscale phải sử dụng color_order='GRAY' hoặc 'AUTO'."
            )

        save_img = img

    elif img.ndim == 3 and img.shape[2] == 3:

        if color_order in {"AUTO", "BGR"}:
            save_img = img

        elif color_order == "RGB":
            save_img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )

        else:
            raise ValueError(
                "Ảnh 3 kênh phải sử dụng BGR hoặc RGB."
            )

    elif img.ndim == 3 and img.shape[2] == 4:

        if color_order in {"AUTO", "BGRA"}:
            save_img = img

        elif color_order == "RGBA":
            save_img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2BGRA
            )

        else:
            raise ValueError(
                "Ảnh 4 kênh phải sử dụng BGRA hoặc RGBA."
            )

    else:
        raise ValueError(
            f"Cấu trúc ảnh không hợp lệ: {img.shape}"
        )

    # 4. Tạo thư mục lưu nếu chưa tồn tại
    if create_dirs:
        image_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    elif not image_path.parent.is_dir():
        raise FileNotFoundError(
            f"Thư mục lưu không tồn tại: {image_path.parent}"
        )

    # 5. Mã hóa ảnh
    try:
        success, encoded_img = cv2.imencode(
            extension,
            save_img,
            list(params) if params is not None else []
        )

        if not success:
            raise RuntimeError("Không thể mã hóa ảnh.")

        # Hỗ trợ đường dẫn có ký tự Unicode trên Windows
        encoded_img.tofile(str(image_path))

    except cv2.error as error:
        raise RuntimeError(
            f"Lỗi OpenCV khi lưu ảnh: {error}"
        ) from error

    return image_path


def get_image_info(img: np.ndarray) -> dict:
    """
    Lấy thông tin cơ bản của ảnh.

    Returns:
        Dictionary gồm width, height, channels và dtype.
    """

    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Ảnh không hợp lệ.")

    height, width = img.shape[:2]

    channels = 1 if img.ndim == 2 else img.shape[2]

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(img.dtype)
    }
