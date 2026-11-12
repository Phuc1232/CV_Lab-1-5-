"""Tiện ích: thư mục gốc dự án, thông tin môi trường (để tái lập), dấu thời gian."""

from __future__ import annotations

import datetime as _dt
import os
import platform
from pathlib import Path


def project_root() -> Path:
    """Thư mục chứa pyproject.toml của canny_project (tìm ngược từ vị trí gói); không hardcode đường dẫn."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "cannylab").exists():
            return parent
    return Path.cwd()


def environment_info() -> dict:
    import cv2
    import numpy
    import scipy
    import skimage
    from . import __version__
    return {"cannylab": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "skimage": skimage.__version__}


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
