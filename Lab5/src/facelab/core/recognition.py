"""MTCNN detection, FaceNet embeddings and face matching."""

from io import BytesIO
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

from ..datasets.store import EmbeddingStore, normalize_identity
from ..imaging.overlay import draw_stylish_box
from ..paths import DEFAULT_DB_DIR


_MODEL_SIGNATURE = "mtcnn-160-margin20-first:v1|facenet-inceptionresnetv1-vggface2-l2-f32-512:v1"


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
        *,
        load_existing: bool = True,
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

        self.database = EmbeddingStore(db_dir, _MODEL_SIGNATURE)
        if load_existing:
            self.load_database()

    def load_database(self) -> int:
        """Reconcile the saved gallery with current images on disk."""
        loaded_count = self.database.reconcile(self._embedding_from_image_bytes)
        print(f"[Cơ sở dữ liệu] Đã lập chỉ mục {loaded_count} ảnh của {len(self.database)} ID.\n")
        return loaded_count

    def _embed_faces(self, face_tensors: torch.Tensor) -> torch.Tensor:
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.facenet(face_tensors.to(self.device))
            return F.normalize(embeddings, p=2, dim=1)

    def _embedding_from_image_bytes(self, image_bytes: bytes) -> np.ndarray | None:
        with Image.open(BytesIO(image_bytes)) as source:
            face_tensors = self.mtcnn(source.convert("RGB"))
        if face_tensors is None:
            return None
        if face_tensors.dim() == 4:
            face_tensors = face_tensors[:1]
        return self._embed_faces(face_tensors)[0].cpu().numpy()

    def save_face_sample(self, name: str, image: Image.Image) -> Path | None:
        """Persist a sample and the embedding of the exact JPEG written to disk."""
        try:
            identity = normalize_identity(name)
            with BytesIO() as encoded:
                image.save(encoded, format="JPEG", quality=95)
                jpeg_bytes = encoded.getvalue()
            embedding = self._embedding_from_image_bytes(jpeg_bytes)
            if embedding is None:
                print(f"[Đăng ký] Không phát hiện khuôn mặt hợp lệ trong mẫu cho ID '{identity}'.")
                return None
            path = self.database.save_jpeg(identity, jpeg_bytes, embedding)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            print(f"[Đăng ký] Không lưu được mẫu: {error}")
            return None
        print(f"[Đăng ký] Đã lưu ảnh và embedding cho ID '{path.parent.name}': {path}")
        return path

    def register_new_face(self, name: str, frame_bgr: np.ndarray, box: Optional[np.ndarray] = None) -> bool:
        """
        Đăng ký một khuôn mặt mới trực tiếp từ khung hình webcam và lưu vào disk/database.
        """
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

        return self.save_face_sample(name, crop_img) is not None

    def compute_similarity(self, embedding: torch.Tensor) -> Tuple[str, str, float]:
        """
        Tính toán Cosine Similarity giữa vector đặc trưng khuôn mặt đầu vào
        và các mẫu khuôn mặt trong cơ sở dữ liệu.

        Điều kiện so sánh:
          - Nếu similarity > 0.7: hiển thị "Matched"
          - Nếu similarity <= 0.7: hiển thị "Unknown"
        """
        best_person = None
        max_similarity = -1.0

        embedding = embedding.cpu().reshape(-1)
        for person_name, stored_embedding in self.database.iter_embeddings():
            known_embedding = torch.from_numpy(stored_embedding.astype(np.float32, copy=True))
            sim = torch.sum(embedding * known_embedding).item()
            if best_person is None or sim > max_similarity:
                max_similarity = sim
                best_person = person_name

        if best_person is None:
            return "Unknown", "Unknown", 0.0

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

        embeddings = self._embed_faces(face_tensors)

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
