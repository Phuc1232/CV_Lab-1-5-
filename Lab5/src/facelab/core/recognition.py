"""MTCNN detection, FaceNet embeddings and face matching."""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

from ..imaging.overlay import draw_stylish_box
from ..paths import DEFAULT_DB_DIR


class FaceRecognitionSystem:
    """
    Hệ thống nhận diện khuôn mặt kết hợp MTCNN và FaceNet.
    - MTCNN: Phát hiện vị trí khuôn mặt (bounding boxes) và 5 điểm mốc (landmarks).
    - FaceNet (InceptionResnetV1): Trích xuất vector đặc trưng 512 chiều được chuẩn hóa L2.
    - So sánh: Sử dụng Cosine Similarity để đối sánh giữa khuôn mặt webcam và cơ sở dữ liệu.
    """

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        similarity_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        self.db_dir = db_dir
        self.similarity_threshold = similarity_threshold

        # Tự động lựa chọn phần cứng tính toán (GPU CUDA nếu có, ngược lại CPU)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Khởi tạo] Sử dụng thiết bị tính toán: {self.device}")
        print(f"[Khởi tạo] Ngưỡng so khớp (Similarity Threshold): {self.similarity_threshold}")

        # 1. Khởi tạo mô hình phát hiện khuôn mặt MTCNN
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

        # 2. Khởi tạo mạng FaceNet (Inception-ResNet-v1 pretrained VGGFace2)
        print("[Mô hình] Đang tải mô hình FaceNet (InceptionResnetV1 Pretrained VGGFace2)...")
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Cơ sở dữ liệu lưu trữ vector đặc trưng: { "Tên người": [vector_embedding_1, ...] }
        self.database: Dict[str, List[torch.Tensor]] = {}

        # Nạp cơ sở dữ liệu khuôn mặt mẫu đã biết
        self.load_database()

    def load_database(self) -> int:
        """
        Quét thư mục database, đọc từng ảnh và trích xuất vector đặc trưng bằng FaceNet.
        Hỗ trợ định dạng: .jpg, .jpeg, .png, .bmp, .webp.
        """
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
        """
        Đăng ký một khuôn mặt mới trực tiếp từ khung hình webcam và lưu vào disk/database.
        """
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
        """
        Tính toán Cosine Similarity giữa vector đặc trưng khuôn mặt đầu vào
        và các mẫu khuôn mặt trong cơ sở dữ liệu.

        Điều kiện so sánh:
          - Nếu similarity > 0.7: hiển thị "Matched"
          - Nếu similarity <= 0.7 (hoặc < 0.7): hiển thị "Unknown"
        """
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
        """
        Xử lý 1 khung hình từ OpenCV webcam:
          1. Chuyển sang ảnh RGB cho MTCNN.
          2. Phát hiện vị trí khuôn mặt (bounding boxes) và landmarks.
          3. Trích xuất khuôn mặt bằng MTCNN và tạo embedding 512D bằng FaceNet.
          4. So sánh embedding và vẽ kết quả lên khung hình.
        """
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
                color = (0, 255, 0)      # Xanh lá cây (Green)
                label = f"Matched: {person_name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)      # Đỏ (Red)
                label = f"Unknown ({similarity:.2f})"

            self._draw_stylish_box(annotated_frame, (x1, y1, x2, y2), color, label)

            if draw_landmarks and landmarks is not None and len(landmarks) > i:
                for pt in landmarks[i]:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(annotated_frame, (px, py), 2, (255, 255, 0), -1)

        return annotated_frame, detections

    _draw_stylish_box = staticmethod(draw_stylish_box)
