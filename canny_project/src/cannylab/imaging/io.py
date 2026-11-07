"""Đọc/ghi ảnh. OpenCV đọc BGR nên đổi sang RGB NGAY tại đây (quy tắc 4)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .validate import InvalidImageError, check_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path, mode: str = "gray") -> np.ndarray:
    """Đọc ảnh thành uint8. ``mode='gray'`` trả (H, W); ``mode='rgb'`` trả (H, W, 3) theo thứ tự RGB.

    Dùng ``np.fromfile`` + ``cv2.imdecode`` để đọc được đường dẫn có ký tự Unicode (tiếng Việt).
    Ném :class:`InvalidImageError` nếu file không tồn tại hoặc không giải mã được.
    """
    path = Path(path)
    if mode not in ("gray", "rgb"):
        raise ValueError(f"mode phải là 'gray' hoặc 'rgb', nhận {mode!r}")
    if not path.is_file():
        raise InvalidImageError(f"Không tìm thấy file ảnh: {path}")
    buffer = np.fromfile(path, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if mode == "gray" else cv2.IMREAD_COLOR
    image = cv2.imdecode(buffer, flag) if buffer.size else None
    if image is None:
        raise InvalidImageError(f"Không giải mã được ảnh (file hỏng hoặc sai định dạng): {path}")
    if mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return check_image(image, name=str(path))


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Ghi ảnh (xám, RGB hoặc bản đồ cạnh bool) ra file; RGB được đổi về BGR trước khi ghi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    image = check_image(image, name=str(path))
    if image.dtype != np.uint8:
        from .convert import to_uint8
        image = to_uint8(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Không mã hóa được ảnh để ghi: {path}")
    encoded.tofile(path)
    return path
