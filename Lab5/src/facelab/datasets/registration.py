"""Register persistent face samples from image files or a webcam."""

from collections.abc import Iterable
import sqlite3

import cv2
from PIL import Image

from ..backends import open_webcam
from ..core.recognition import FaceRecognitionSystem
from ..paths import DEFAULT_DB_DIR
from .store import normalize_identity


def _first_detection(recognizer, image):
    boxes, probs = recognizer.mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None, None
    probability = probs[0] if probs is not None and len(probs) > 0 else None
    return boxes[0], probability


def _crop_face(image, box):
    x1, y1, x2, y2 = box
    width, height = image.size
    margin_x = (x2 - x1) * 0.15
    margin_y = (y2 - y1) * 0.15
    return image.crop((
        max(0, int(x1 - margin_x)),
        max(0, int(y1 - margin_y)),
        min(width, int(x2 + margin_x)),
        min(height, int(y2 + margin_y)),
    ))


def register_from_images(
    image_paths: Iterable[str],
    name: str,
    db_dir: str = DEFAULT_DB_DIR,
) -> int:
    """Register each usable image under one identity using a shared model."""
    try:
        identity = normalize_identity(name)
    except ValueError as error:
        print(f"[Đăng ký] {error}")
        return 0

    recognizer = None
    registered = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if recognizer is None:
                recognizer = FaceRecognitionSystem(db_dir=db_dir, device="cpu", load_existing=False)
            box, probability = _first_detection(recognizer, image)
            if box is None:
                print(f"[LỖI] MTCNN không phát hiện được khuôn mặt nào trong ảnh: {image_path}")
                continue
            if probability is not None and probability < 0.7:
                print(f"[CẢNH BÁO] Độ tin cậy khuôn mặt hơi thấp ({probability:.2f}).")
            if recognizer.save_face_sample(identity, _crop_face(image, box)) is not None:
                registered += 1
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            print(f"[LỖI] Không thể đăng ký ảnh {image_path}: {error}")
    return registered


def register_from_image(
    image_path: str,
    name: str,
    db_dir: str = DEFAULT_DB_DIR,
) -> bool:
    return register_from_images((image_path,), name, db_dir) == 1


def register_from_webcam(
    name: str,
    camera_id: int = 0,
    db_dir: str = DEFAULT_DB_DIR,
    *,
    samples: int = 1,
) -> bool:
    """Capture and store samples for one identity in a single camera session."""
    try:
        identity = normalize_identity(name)
        if type(samples) is not int or samples < 1:
            raise ValueError("Số mẫu phải là số nguyên dương.")
    except ValueError as error:
        print(f"[Đăng ký] {error}")
        return False

    print(f"\n[Webcam] Đang khởi động camera #{camera_id}...")
    cap = open_webcam(camera_id)
    window_open = False
    captured = 0

    try:
        if not cap.isOpened():
            print(f"[LỖI] Không thể mở webcam #{camera_id}")
            return False
        recognizer = FaceRecognitionSystem(db_dir=db_dir, device="cpu", load_existing=False)
        window_name = f"Dang ky khuon mat: {identity}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_open = True

        print(f"[Đăng ký] Cần chụp {samples} mẫu cho ID '{identity}'.")
        print("Nhấn Space/c để chụp. Thay đổi góc mặt giữa các mẫu. Nhấn q/Esc để dừng.")

        while captured < samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Webcam] Không nhận được tín hiệu hình ảnh từ camera.")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box, probability = _first_detection(recognizer, image)
            valid_face = box is not None and probability is not None and probability > 0.8
            if valid_face:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                message = f"Mau {captured + 1}/{samples} ({probability:.2f}) - Nhan SPACE"
                origin, color = (x1, max(20, y1 - 10)), (0, 255, 0)
            else:
                message = "Vui long dua khuon mat vao giua khung hinh..."
                origin, color = (30, 40), (0, 0, 255)
            cv2.putText(display_frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key in (ord(" "), ord("c")):
                if not valid_face:
                    print("[Cảnh báo] Chưa phát hiện được khuôn mặt rõ ràng, vui lòng thử lại.")
                elif recognizer.save_face_sample(identity, _crop_face(image, box)) is not None:
                    captured += 1
                    print(f"[Đăng ký] Đã lưu {captured}/{samples} mẫu.")

    except KeyboardInterrupt:
        print("\n[Ngắt] Đã dừng chụp mẫu.")
    except (OSError, RuntimeError, ValueError, sqlite3.Error, cv2.error) as error:
        print(f"[LỖI] Không thể tiếp tục đăng ký: {error}")
    finally:
        cap.release()
        if window_open:
            cv2.destroyAllWindows()

    return captured > 0
