import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = os.path.join(CURRENT_DIR, "known_faces")
class FaceRecognitionSystem:
    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        similarity_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        self.db_dir = db_dir
        self.similarity_threshold = similarity_threshold
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[Khởi tạo] Sử dụng thiết bị tính toán: {self.device}")
        print(f"[Khởi tạo] Ngưỡng so khớp (Similarity Threshold): {self.similarity_threshold}")
        print("[Mô hình] Đang tải mô hình MTCNN (P-Net, R-Net, O-Net)...")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device,
        )
        print("[Mô hình] Đang tải mô hình FaceNet (InceptionResnetV1 Pretrained VGGFace2)...")
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.database: Dict[str, List[torch.Tensor]] = {}
        self.load_database()
    def load_database(self) -> int:
        self.database.clear()
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)
            print(f"[Cơ sở dữ liệu] Đã tạo thư mục lưu khuôn mặt: {self.db_dir}")
            return 0
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        image_files = [f for f in os.listdir(self.db_dir) if f.lower().endswith(valid_extensions)]
        print(f"\n--- ĐANG NẠP CƠ SỞ DỮ LIỆU KHUÔN MẶT MẪU ({len(image_files)} tệp tin) ---")
        loaded_count = 0
        for file_name in image_files:
            file_path = os.path.join(self.db_dir, file_name)
            raw_name = os.path.splitext(file_name)[0]
            person_name = raw_name.rsplit("_", 1)[0] if "_" in raw_name and raw_name.rsplit("_", 1)[1].isdigit() else raw_name
            try:
                img_pil = Image.open(file_path).convert("RGB")
                face_tensor = self.mtcnn(img_pil)
                if face_tensor is None:
                    print(f"  [!] Cảnh báo: Không tìm thấy khuôn mặt hợp lệ trong ảnh: {file_name}")
                    continue
                if face_tensor.dim() == 4:
                    face_tensor = face_tensor[0:1]
                else:
                    face_tensor = face_tensor.unsqueeze(0)
                face_tensor = face_tensor.to(self.device)
                with torch.no_grad():
                    embedding = self.facenet(face_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                if person_name not in self.database:
                    self.database[person_name] = []
                self.database[person_name].append(embedding.cpu())
                loaded_count += 1
                print(f"  [+] Đã nạp thành công: {person_name} (tệp: {file_name})")
            except Exception as e:
                print(f"  [Lỗi] Không thể xử lý ảnh {file_name}: {e}")
        print(f"[Cơ sở dữ liệu] Hoàn tất nạp {loaded_count} ảnh của {len(self.database)} người.\n")
        return loaded_count
    def register_new_face(self, name: str, frame_bgr: np.ndarray, box: Optional[np.ndarray] = None) -> bool:
        if not name or name.strip() == "":
            print("[Đăng ký] Lỗi: Tên không được để trống!")
            return False
        clean_name = name.strip().replace(" ", "_")
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame_bgr.shape[:2]
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)
            crop_img = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            crop_img = img_pil
        face_tensor = self.mtcnn(crop_img)
        if face_tensor is None:
            print(f"[Đăng ký] Thất bại: Không phát hiện được khuôn mặt rõ nét cho '{clean_name}'.")
            return False
        if face_tensor.dim() == 4:
            face_tensor = face_tensor[0:1]
        else:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        idx = 1
        save_path = os.path.join(self.db_dir, f"{clean_name}.jpg")
        while os.path.exists(save_path):
            save_path = os.path.join(self.db_dir, f"{clean_name}_{idx}.jpg")
            idx += 1
        crop_img.save(save_path, format="JPEG", quality=95)
        if clean_name not in self.database:
            self.database[clean_name] = []
        self.database[clean_name].append(embedding.cpu())
        print(f"[Đăng ký] Thành công! Đã đăng ký '{clean_name}' và lưu vào: {save_path}")
        return True
    def compute_similarity(self, embedding: torch.Tensor) -> Tuple[str, str, float]:
        if not self.database:
            return "Unknown", "Unknown", 0.0
        best_person = "Unknown"
        max_similarity = -1.0
        embedding = embedding.cpu()
        for person_name, known_embeddings in self.database.items():
            for known_emb in known_embeddings:
                sim = torch.sum(embedding * known_emb).item()
                if sim > max_similarity:
                    max_similarity = sim
                    best_person = person_name
        if max_similarity > self.similarity_threshold:
            status = "Matched"
            identified_name = best_person
        else:
            status = "Unknown"
            identified_name = "Unknown"
        return status, identified_name, max_similarity
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_landmarks: bool = False,
    ) -> Tuple[np.ndarray, List[dict]]:
        annotated_frame = frame_bgr.copy()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
        detections = []
        if boxes is None or len(boxes) == 0:
            return annotated_frame, detections
        try:
            face_tensors = self.mtcnn.extract(img_pil, boxes, save_path=None)
        except Exception:
            face_tensors = None
        if face_tensors is None:
            return annotated_frame, detections
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        face_tensors = face_tensors.to(self.device)
        with torch.no_grad():
            embeddings = self.facenet(face_tensors)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        for i, box in enumerate(boxes):
            prob = probs[i] if probs is not None else 0.0
            if prob < 0.6:
                continue
            emb = embeddings[i : i + 1]
            status, person_name, similarity = self.compute_similarity(emb)
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = annotated_frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            detections.append({
                "box": (x1, y1, x2, y2),
                "prob": float(prob),
                "status": status,
                "name": person_name,
                "similarity": float(similarity),
            })
            if status == "Matched":
                color = (0, 255, 0)      
                label = f"Matched: {person_name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)      
                label = f"Unknown ({similarity:.2f})"
            self._draw_stylish_box(annotated_frame, (x1, y1, x2, y2), color, label)
            if draw_landmarks and landmarks is not None and len(landmarks) > i:
                for pt in landmarks[i]:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(annotated_frame, (px, py), 2, (255, 255, 0), -1)
        return annotated_frame, detections
    @staticmethod
    def _draw_stylish_box(
        img: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        label: str,
    ):
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
def open_webcam(camera_id: int = 0) -> cv2.VideoCapture:
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
                    print(f"[Webcam] Kết nối thành công Camera")
                    return cap
                cap.release()
        except Exception as e:
            print(f"  [!] Thử backend {backend_name} thất bại: {e}")
    print("[Webcam] Đang mở camera với cấu hình mặc định cuối cùng...")
    return cv2.VideoCapture(camera_id)
def run_webcam(
    camera_id: int = 0,
    db_dir: str = DEFAULT_DB_DIR,
    similarity_threshold: float = 0.7,
    frame_skip: int = 2,
    device: Optional[str] = None,
):
    print("\n" + "=" * 70)
    print(" KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC")
    print(" Thuật toán: MTCNN (Phát hiện) + FaceNet (Inception-ResNet 512D)")
    print(" Tiêu chí: Similarity > 0.7 -> Matched | Similarity <= 0.7 -> Unknown")
    print("=" * 70)
    recognizer = FaceRecognitionSystem(
        db_dir=db_dir,
        similarity_threshold=similarity_threshold,
        device=device,
    )
    print(f"\n[Webcam] Đang kết nối tới Camera")
    cap = open_webcam(camera_id)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam với ID={camera_id}!")
        print("Vui lòng kiểm tra lại thiết bị camera hoặc quyền truy cập webcam trên máy.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Webcam] Kết nối thành công! Độ phân giải: {actual_w}x{actual_h}")
    print("[Hướng dẫn] Nhấn 'q' hoặc ESC trên cửa sổ hiển thị để thoát.\n")
    window_name = "Nhan dien khuon mat FaceNet & MTCNN (Lab 3)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    fps_counter = 0
    draw_landmarks = False
    skip_enabled = frame_skip > 1
    last_detections = []
    latest_face_box = None
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
            should_process = (not skip_enabled) or (frame_count % frame_skip == 0)
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
def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện khuôn mặt thời gian thực với MTCNN & FaceNet trên Webcam (Lab 3)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Chỉ số ID camera (mặc định: 0)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_DB_DIR,
        help=f"Thư mục chứa ảnh khuôn mặt mẫu đã biết (mặc định: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Ngưỡng tương đồng Cosine Similarity (mặc định: 0.7 theo yêu cầu bài thực hành)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Khoảng cách khung hình chạy phát hiện để tối ưu CPU (mặc định: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị tính toán: 'cuda' hoặc 'cpu' (mặc định: tự động chọn)",
    )
    args = parser.parse_args()
    run_webcam(
        camera_id=args.camera,
        db_dir=args.db_dir,
        similarity_threshold=args.threshold,
        frame_skip=args.frame_skip,
        device=args.device,
    )
if __name__ == "__main__":
    main()
