"""Open a webcam through the available OpenCV backends."""

import sys

import cv2
import numpy as np


def open_webcam(
    camera_id: int = 0,
    *,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    """Return the first validated webcam capture, or an unopened capture."""
    if sys.platform == "win32":
        candidate_backends = [
            ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
            ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
            ("Mặc định (CAP_ANY)", cv2.CAP_ANY),
        ]
    else:
        candidate_backends = [("Mặc định (CAP_ANY)", cv2.CAP_ANY)]

    for backend_name, backend_flag in candidate_backends:
        cap = None
        keep_capture = False
        try:
            cap = cv2.VideoCapture(camera_id, backend_flag)
            if not cap.isOpened():
                continue

            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            test_frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if (
                    ret
                    and isinstance(frame, np.ndarray)
                    and frame.ndim == 3
                    and frame.shape[2] == 3
                    and frame.size > 0
                    and frame.dtype == np.uint8
                ):
                    test_frame = frame

            if test_frame is None:
                continue

            means, spatial_stds = cv2.meanStdDev(test_frame)
            b_mean, g_mean, r_mean = means.ravel()
            if spatial_stds.max() < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0:
                print(f"  [!] Backend {backend_name} trả về khung hình lỗi màu xanh lá (solid green). Đang chuyển đổi backend khác...")
                continue

            print(f"[Webcam] Kết nối thành công Camera #{camera_id} sử dụng backend: {backend_name}")
            keep_capture = True
            return cap
        except Exception as error:
            print(f"  [!] Thử backend {backend_name} thất bại: {error}")
        finally:
            if cap is not None and not keep_capture:
                cap.release()

    print(f"[Webcam] Không tìm thấy luồng hình ảnh hợp lệ cho Camera #{camera_id}.")
    return cv2.VideoCapture()
