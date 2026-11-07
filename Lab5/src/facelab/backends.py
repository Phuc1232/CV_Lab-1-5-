"""Open a webcam through the available OpenCV backends."""

import sys

import cv2
import numpy as np


def open_webcam(camera_id: int = 0) -> cv2.VideoCapture:
    """
    Mở camera với cơ chế tự động chọn backend tối ưu trên Windows (DirectShow)
    nhằm khắc phục triệt để lỗi khung hình toàn màu xanh lá cây (solid green [0, 135, 0])
    do backend MSMF (Media Foundation) của OpenCV gây ra.
    """
    if sys.platform == "win32":
        candidate_backends = [
            ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
            ("Mặc định (CAP_ANY)", cv2.CAP_ANY),
            ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ]
    else:
        candidate_backends = [("Mặc định (CAP_ANY)", cv2.CAP_ANY)]

    for backend_name, backend_flag in candidate_backends:
        try:
            cap = cv2.VideoCapture(camera_id, backend_flag)
            if cap.isOpened():
                valid_stream = False
                test_frame = None
                for _ in range(5):
                    ret, frame_read = cap.read()
                    if ret and frame_read is not None:
                        test_frame = frame_read
                        valid_stream = True

                if valid_stream and test_frame is not None:
                    b_mean, g_mean, r_mean = np.mean(test_frame, axis=(0, 1))
                    std_val = float(np.std(test_frame))

                    if std_val < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0:
                        print(f"  [!] Backend {backend_name} trả về khung hình lỗi màu xanh lá (solid green). Đang chuyển đổi backend khác...")
                        cap.release()
                        continue

                    print(f"[Webcam] Kết nối thành công Camera #{camera_id} sử dụng backend: {backend_name}")
                    return cap
                cap.release()
        except Exception as e:
            print(f"  [!] Thử backend {backend_name} thất bại: {e}")

    print("[Webcam] Đang mở camera với cấu hình mặc định cuối cùng...")
    return cv2.VideoCapture(camera_id)
