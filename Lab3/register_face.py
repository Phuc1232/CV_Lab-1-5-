import os
import sys
import argparse
from typing import Optional
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from Lab3 import open_webcam
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = os.path.join(CURRENT_DIR, "known_faces")
def register_from_image(
    image_path: str,
    name: str,
    db_dir: str = DEFAULT_DB_DIR,
) -> bool:
    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy file ảnh: {image_path}")
        return False
    clean_name = name.strip().replace(" ", "_")
    os.makedirs(db_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=False, device="cpu")
    try:
        pil_img = Image.open(image_path).convert("RGB")
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            print(f"[LỖI] MTCNN không phát hiện được khuôn mặt nào trong ảnh: {image_path}")
            return False
        prob = probs[0]
        if prob < 0.7:
            print(f"[CẢNH BÁO] Độ tin cậy khuôn mặt hơi thấp ({prob:.2f}). Nên chọn ảnh rõ nét hơn.")
        x1, y1, x2, y2 = boxes[0]
        w, h = pil_img.size
        margin_x = (x2 - x1) * 0.15
        margin_y = (y2 - y1) * 0.15
        crop_box = (
            max(0, int(x1 - margin_x)),
            max(0, int(y1 - margin_y)),
            min(w, int(x2 + margin_x)),
            min(h, int(y2 + margin_y)),
        )
        face_crop = pil_img.crop(crop_box)
        target_path = os.path.join(db_dir, f"{clean_name}.jpg")
        idx = 1
        while os.path.exists(target_path):
            target_path = os.path.join(db_dir, f"{clean_name}_{idx}.jpg")
            idx += 1
        face_crop.save(target_path, format="JPEG", quality=95)
        print(f"[THÀNH CÔNG] Đã đăng ký khuôn mặt cho '{clean_name}': {target_path}")
        return True
    except Exception as e:
        print(f"[LỖI] Quá trình xử lý ảnh thất bại: {e}")
        return False
def register_from_webcam(
    name: str,
    camera_id: int = 0,
    db_dir: str = DEFAULT_DB_DIR,
) -> bool:
    clean_name = name.strip().replace(" ", "_")
    os.makedirs(db_dir, exist_ok=True)
    print(f"\n[Webcam] Đang khởi động camera 
    cap = open_webcam(camera_id)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam 
        return False
    mtcnn = MTCNN(keep_all=False, device="cpu")
    window_name = f"Dang ky khuon mat: {clean_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n--- HƯỚNG DẪN CHỤP ẢNH ---")
    print("  - Đứng thẳng, nhìn trực diện vào webcam, đảm bảo đủ ánh sáng.")
    print("  - Nhấn phím SPACE hoặc 'c' để CHỤP.")
    print("  - Nhấn 'q' hoặc ESC để HỦY BỎ.\n")
    captured = False
    target_path = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, probs = mtcnn.detect(pil_img)
            has_valid_face = False
            if boxes is not None and len(boxes) > 0 and probs is not None and len(probs) > 0 and probs[0] is not None and probs[0] > 0.8:
                has_valid_face = True
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                prob_val = float(probs[0])
                cv2.putText(
                    display_frame,
                    f"Khuon mat hop le ({prob_val:.2f}) - Nhan SPACE de chup",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "Vui long dua khuon mat vao giua khung hinh...",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[Hủy] Đã hủy đăng ký khuôn mặt.")
                break
            if key == ord(" ") or key == ord("c"):
                if not has_valid_face:
                    print("[Cảnh báo] Chưa nhận diện được khuôn mặt rõ ràng, vui lòng thử lại!")
                    continue
                x1, y1, x2, y2 = boxes[0]
                w, h = pil_img.size
                margin_x = (x2 - x1) * 0.15
                margin_y = (y2 - y1) * 0.15
                crop_box = (
                    max(0, int(x1 - margin_x)),
                    max(0, int(y1 - margin_y)),
                    min(w, int(x2 + margin_x)),
                    min(h, int(y2 + margin_y)),
                )
                face_crop = pil_img.crop(crop_box)
                target_path = os.path.join(db_dir, f"{clean_name}.jpg")
                idx = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(db_dir, f"{clean_name}_{idx}.jpg")
                    idx += 1
                face_crop.save(target_path, format="JPEG", quality=95)
                captured = True
                print(f"[THÀNH CÔNG] Đã lưu ảnh khuôn mặt cho '{clean_name}': {target_path}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return captured
def main():
    parser = argparse.ArgumentParser(description="Đăng ký khuôn mặt mẫu vào cơ sở dữ liệu Lab 3")
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
    args = parser.parse_args()
    if args.image:
        register_from_image(args.image, args.name, args.db_dir)
    else:
        register_from_webcam(args.name, args.camera, args.db_dir)
if __name__ == "__main__":
    main()