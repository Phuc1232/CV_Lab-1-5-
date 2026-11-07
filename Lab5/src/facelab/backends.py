"""Open and monitor a webcam through the available OpenCV backends."""

import sys

import cv2
import numpy as np


def _is_bgr_frame(frame: object) -> bool:
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.size > 0
        and frame.dtype == np.uint8
    )


def _is_solid_green(frame: np.ndarray) -> bool:
    means, spatial_stds = cv2.meanStdDev(frame)
    b_mean, g_mean, r_mean = means.ravel()
    return spatial_stds.max() < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0


class WebcamCapture:
    """Keep one validated capture open and advance through fallback modes once."""

    def __init__(self, camera_id: int, width: int | None, height: int | None):
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._capture = None
        self.generation = 0
        default = ("Mặc định (CAP_ANY)", cv2.CAP_ANY, None)
        if sys.platform == "win32":
            self._candidates = iter((
                ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW, None),
                ("DirectShow (CAP_DSHOW, MJPG)", cv2.CAP_DSHOW, "MJPG"),
                ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF, None),
                default,
            ))
        else:
            self._candidates = iter((default,))
        self._open_next()

    def _open_next(self):
        self.release()
        for name, backend, codec in self._candidates:
            cap = None
            try:
                cap = cv2.VideoCapture(self._camera_id, backend)
                if not cap.isOpened():
                    continue

                if self._width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                if self._height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                if codec is not None and not cap.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec)
                ):
                    continue

                test_frame = None
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and _is_bgr_frame(frame):
                        test_frame = frame

                if test_frame is None:
                    continue
                if _is_solid_green(test_frame):
                    print(f"  [!] {name} trả về khung hình xanh lá đồng nhất.")
                    continue

                print(f"[Webcam] Kết nối Camera #{self._camera_id} qua {name}.")
                self._capture = cap
                self.generation += 1
                return
            except Exception as error:
                print(f"  [!] Thử {name} thất bại: {error}")
            finally:
                if cap is not None and cap is not self._capture:
                    cap.release()

        print(f"[Webcam] Không tìm thấy luồng hình ảnh hợp lệ cho Camera #{self._camera_id}.")

    def isOpened(self) -> bool:
        return self._capture is not None

    def get(self, prop_id: int) -> float:
        return self._capture.get(prop_id) if self._capture is not None else 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        while self._capture is not None:
            for _ in range(3):
                try:
                    ret, frame = self._capture.read()
                except cv2.error as error:
                    print(f"  [!] Không đọc được khung hình: {error}")
                    break

                if ret and _is_bgr_frame(frame) and not _is_solid_green(frame):
                    return True, frame

            print("  [!] Luồng webcam mất hình hoặc bị xanh. Đang thử chế độ tiếp theo...")
            self._open_next()

        return False, None

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def open_webcam(
    camera_id: int = 0,
    *,
    width: int | None = None,
    height: int | None = None,
) -> WebcamCapture:
    """Return a webcam session that filters bad frames and recovers when possible."""
    return WebcamCapture(camera_id, width, height)
