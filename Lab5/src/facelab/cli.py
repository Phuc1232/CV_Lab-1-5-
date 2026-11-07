"""Run face recognition or register a new face sample."""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

import cv2

from .backends import open_webcam
from .config import load_recognition_config
from .core.recognition import FaceRecognitionSystem
from .datasets.registration import register_from_image, register_from_webcam
from .imaging.overlay import draw_hud
from .paths import DEFAULT_DB_DIR


def _configure_console():
    # Cấu hình console tiếng Việt trên Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def run_webcam(
    camera_id: int = 0,
    db_dir: str = DEFAULT_DB_DIR,
    similarity_threshold: float = 0.7,
    frame_skip: int = 2,
    device: Optional[str] = None,
):
    """
    Vòng lặp chính xử lý Webcam theo thời gian thực với MTCNN & FaceNet.
    """
    print("\n" + "=" * 70)
    print(" KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC (LAB 5)")
    print(" Thuật toán: MTCNN (Phát hiện) + FaceNet (Inception-ResNet 512D)")
    print(
        f" Tiêu chí: Similarity > {similarity_threshold:g} -> Matched"
        f" | Similarity <= {similarity_threshold:g} -> Unknown"
    )
    print("=" * 70)

    recognizer = FaceRecognitionSystem(
        db_dir=db_dir,
        similarity_threshold=similarity_threshold,
        device=device,
    )

    print(f"\n[Webcam] Đang kết nối tới Camera #{camera_id}...")
    cap = open_webcam(camera_id, width=640, height=480)

    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam với ID={camera_id}!")
        print("Vui lòng kiểm tra lại thiết bị camera hoặc quyền truy cập webcam trên máy.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Webcam] Kết nối thành công! Độ phân giải: {actual_w}x{actual_h}")
    print("[Hướng dẫn] Nhấn 'q' hoặc ESC trên cửa sổ hiển thị để thoát.\n")

    window_name = "Nhan dien khuon mat FaceNet & MTCNN (Lab 5)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    fps_counter = 0

    draw_landmarks = False
    skip_enabled = frame_skip > 1
    last_detections = []
    latest_face_box = None
    capture_generation = cap.generation

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Webcam] Không nhận được tín hiệu hình ảnh từ camera.")
                break

            frame = cv2.flip(frame, 1)

            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

            frame_count += 1
            capture_changed = cap.generation != capture_generation
            capture_generation = cap.generation
            should_process = capture_changed or not skip_enabled or frame_count % frame_skip == 0

            if should_process:
                annotated_frame, last_detections = recognizer.process_frame(
                    frame, draw_landmarks=draw_landmarks
                )
                if last_detections:
                    latest_face_box = last_detections[0]["box"]
                else:
                    latest_face_box = None
            else:
                annotated_frame = frame.copy()
                for det in last_detections:
                    x1, y1, x2, y2 = det["box"]
                    status = det["status"]
                    person_name = det["name"]
                    sim = det["similarity"]
                    color = (0, 255, 0) if status == "Matched" else (0, 0, 255)
                    label = (
                        f"Matched: {person_name} ({sim:.2f})"
                        if status == "Matched"
                        else f"Unknown ({sim:.2f})"
                    )
                    recognizer._draw_stylish_box(annotated_frame, (x1, y1, x2, y2), color, label)

            num_faces = len(last_detections)
            draw_hud(
                annotated_frame,
                fps,
                num_faces,
                len(recognizer.database),
                recognizer.similarity_threshold,
                skip_enabled,
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("\n[Thoát] Người dùng đã yêu cầu dừng ứng dụng.")
                break

            elif key == ord("s") or key == ord("c"):
                default_name = f"Nguoi_Dung_{int(time.time()) % 10000}"
                print(f"\n--- ĐĂNG KÝ KHUÔN MẶT MỚI ---")
                print("Nhập tên cho khuôn mặt đang đứng trước webcam:")
                print(f"(Nhấn Enter trực tiếp để dùng tên mặc định: '{default_name}')")
                try:
                    user_input = input(">> Tên người: ").strip()
                    reg_name = user_input if user_input else default_name
                except Exception:
                    reg_name = default_name

                success = recognizer.register_new_face(reg_name, frame, box=latest_face_box)
                if success:
                    print(f">> Đã cập nhật '{reg_name}' vào hệ thống! Tiếp tục nhận diện...\n")
                else:
                    print(">> Đăng ký thất bại. Vui lòng đứng đối diện camera và thử lại.\n")

            elif key == ord("r"):
                print("\n[Cơ sở dữ liệu] Đang nạp lại từ thư mục...")
                recognizer.load_database()

            elif key == ord("l"):
                draw_landmarks = not draw_landmarks
                print(f"[Hiển thị] Điểm mốc khuôn mặt (Landmarks): {'BẬT' if draw_landmarks else 'TẮT'}")

            elif key == ord("f"):
                skip_enabled = not skip_enabled
                print(f"[Hiệu năng] Tối ưu hóa CPU: {'BẬT' if skip_enabled else 'TẮT'}")

    except KeyboardInterrupt:
        print("\n[Ngắt] Nhận tín hiệu ngắt từ bàn phím (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Webcam] Đã giải phóng camera và đóng tất cả cửa sổ giao diện.")


def main(argv=None):
    _configure_console()
    parser = argparse.ArgumentParser(
        description="Nhận diện khuôn mặt thời gian thực với MTCNN & FaceNet trên Webcam (Lab 5)",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, default=None, help="Tệp cấu hình TOML tùy chọn")
    parser.add_argument(
        "--camera",
        dest="camera_id",
        type=int,
        help="Chỉ số ID camera (mặc định: 0)",
    )
    parser.add_argument(
        "--db-dir",
        help=f"Thư mục chứa ảnh khuôn mặt mẫu đã biết (mặc định: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--threshold",
        dest="similarity_threshold",
        type=float,
        help="Ngưỡng tương đồng Cosine Similarity (mặc định: 0.7 theo yêu cầu bài thực hành)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        help="Khoảng cách khung hình chạy phát hiện để tối ưu CPU (mặc định: 2)",
    )
    parser.add_argument(
        "--device",
        help="Thiết bị tính toán: 'cuda' hoặc 'cpu' (mặc định: tự động chọn)",
    )

    options = vars(parser.parse_args(argv))
    config_path = options.pop("config")
    settings = {}
    if config_path is not None:
        try:
            settings = load_recognition_config(config_path)
        except (OSError, ValueError, OverflowError) as error:
            parser.error(f"--config {config_path}: {error}")

    if "threshold" in settings:
        settings["similarity_threshold"] = settings.pop("threshold")
    if "camera" in settings:
        settings["camera_id"] = settings.pop("camera")
    settings.update(options)
    run_webcam(**settings)


def register_main(argv=None):
    _configure_console()
    parser = argparse.ArgumentParser(description="Đăng ký khuôn mặt mẫu vào cơ sở dữ liệu Lab 5")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Tên của người cần đăng ký (ví dụ: 'NguyenVanA' hoặc 'Alice')",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Đường dẫn file ảnh có sẵn trên đĩa (nếu không cung cấp, sẽ mở webcam để chụp)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="ID camera khi chụp từ webcam (mặc định: 0)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_DB_DIR,
        help=f"Thư mục lưu trữ cơ sở dữ liệu khuôn mặt (mặc định: {DEFAULT_DB_DIR})",
    )

    args = parser.parse_args(argv)

    if args.image:
        register_from_image(args.image, args.name, args.db_dir)
    else:
        register_from_webcam(args.name, args.camera, args.db_dir)


if __name__ == "__main__":
    main()
