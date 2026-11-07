"""Draw labels, bounding boxes and the webcam HUD."""

from typing import Tuple

import cv2
import numpy as np


def draw_stylish_box(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str,
):
    """
    Vẽ hộp bao khuôn mặt góc vuông nổi bật và thanh thông tin nhãn rõ ràng.
    """
    x1, y1, x2, y2 = box
    line_thickness = 2
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, line_thickness + 1)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, line_thickness + 1)

    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, line_thickness + 1)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, line_thickness + 1)

    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, line_thickness + 1)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, line_thickness + 1)

    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, line_thickness + 1)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, line_thickness + 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    label_y1 = max(0, y1 - text_h - 10)
    label_y2 = y1
    label_x2 = min(img.shape[1], x1 + text_w + 12)

    cv2.rectangle(img, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
    text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
    cv2.putText(
        img,
        label,
        (x1 + 6, label_y2 - 5),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_faces: int,
    db_size: int,
    threshold: float,
    skip_enabled: bool,
):
    """
    Vẽ thanh thông số HUD (Heads-Up Display) hiển thị trạng thái hệ thống.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    hud_text = (
        f"FPS: {fps:.1f} | Khuon mat: {num_faces} | "
        f"Database: {db_size} | Nguong: >{threshold:.2f} (Matched) | "
        f"Toi uu CPU: {'BAT' if skip_enabled else 'TAT'}"
    )
    cv2.putText(frame, hud_text, (10, 23), font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 26), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    help_text = "[S/C] Dang ky khuon mat | [R] Nap lai DB | [L] Landmarks | [F] Toi uu FPS | [Q/ESC] Thoat"
    cv2.putText(frame, help_text, (10, h - 8), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
