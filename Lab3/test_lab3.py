"""
=============================================================================
UNIT TESTS FOR LAB 3: REAL-TIME FACE RECOGNITION (FACENET & MTCNN)
=============================================================================
Kiểm thử toàn diện các thành phần của hệ thống nhận diện khuôn mặt:
  1. Khởi tạo mô hình MTCNN và FaceNet (Inception-ResNet-v1)
  2. Kiểm tra kích thước và chuẩn hóa L2 của vector đặc trưng (512 chiều)
  3. Kiểm tra tính toán độ đo tương đồng Cosine Similarity
  4. Kiểm tra điều kiện ngưỡng đề bài (Threshold = 0.7):
     - similarity > 0.7 -> "Matched"
     - similarity <= 0.7 -> "Unknown"
  5. Kiểm tra xử lý khung hình không có khuôn mặt (kháng lỗi)
  6. Kiểm tra mở webcam chống lỗi màn hình xanh lá cây (solid green)
=============================================================================
"""

import os
import sys
import pytest
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Lab3 import FaceRecognitionSystem, open_webcam


@pytest.fixture(scope="module")
def face_system():
    """
    Fixture khởi tạo đối tượng FaceRecognitionSystem trên CPU để kiểm thử.
    """
    system = FaceRecognitionSystem(
        db_dir=os.path.join(CURRENT_DIR, "known_faces"),
        similarity_threshold=0.7,
        device="cpu",
    )
    return system


def test_models_initialization(face_system):
    """
    Kiểm tra khởi tạo thành công 2 mô hình MTCNN và InceptionResnetV1.
    """
    assert face_system.mtcnn is not None, "Mô hình MTCNN chưa được khởi tạo!"
    assert face_system.facenet is not None, "Mô hình FaceNet chưa được khởi tạo!"
    assert face_system.similarity_threshold == 0.7, "Ngưỡng mặc định phải là 0.7 theo đề bài!"


def test_embedding_dimension_and_l2_normalization(face_system):
    """
    Kiểm tra FaceNet trích xuất vector embedding đúng kích thước 512 chiều
    và đạt chuẩn hóa L2 (độ dài vector norm xấp xỉ 1.0).
    """
    dummy_face = torch.randn(1, 3, 160, 160, device=face_system.device)

    with torch.no_grad():
        embedding = face_system.facenet(dummy_face)
        embedding = F.normalize(embedding, p=2, dim=1)

    assert embedding.shape == (1, 512), f"Kích thước embedding không đúng: {embedding.shape}"
    norm = torch.norm(embedding, p=2, dim=1).item()
    assert abs(norm - 1.0) < 1e-4, f"Độ dài L2-norm không bằng 1.0: {norm}"


def test_cosine_similarity_threshold_decision(face_system):
    """
    Kiểm tra quyết định 'Matched' vs 'Unknown' theo đúng tiêu chí đề bài:
      - similarity > 0.7  => 'Matched'
      - similarity <= 0.7 => 'Unknown'
    """
    alice_vector = F.normalize(torch.randn(1, 512), p=2, dim=1)
    face_system.database["Alice"] = [alice_vector]

    # Trùng khớp hoàn toàn -> similarity = 1.0 > 0.7
    status_exact, name_exact, sim_exact = face_system.compute_similarity(alice_vector)
    assert status_exact == "Matched"
    assert name_exact == "Alice"
    assert abs(sim_exact - 1.0) < 1e-4

    # Vector trực giao -> similarity ~ 0.0 <= 0.7 -> 'Unknown'
    ortho_vector = torch.randn(1, 512)
    ortho_vector -= torch.sum(ortho_vector * alice_vector) * alice_vector
    ortho_vector = F.normalize(ortho_vector, p=2, dim=1)

    status_ortho, name_ortho, sim_ortho = face_system.compute_similarity(ortho_vector)
    assert status_ortho == "Unknown"
    assert name_ortho == "Unknown"
    assert sim_ortho <= 0.7


def test_strict_boundary_condition(face_system):
    """
    Kiểm tra nghiêm ngặt tại giá trị biên 0.7:
      - 0.75 > 0.7 => 'Matched'
      - 0.65 <= 0.7 => 'Unknown'
    """
    base_vec = F.normalize(torch.randn(1, 512), p=2, dim=1)
    face_system.database.clear()
    face_system.database["TestUser"] = [base_vec]

    ortho = torch.randn(1, 512)
    ortho -= torch.sum(ortho * base_vec) * base_vec
    ortho = F.normalize(ortho, p=2, dim=1)

    # 1. Test similarity = 0.75 (> 0.7)
    sim_target_high = 0.75
    b_high = np.sqrt(1.0 - sim_target_high**2)
    v_high = sim_target_high * base_vec + b_high * ortho
    v_high = F.normalize(v_high, p=2, dim=1)

    status_high, name_high, calc_sim_high = face_system.compute_similarity(v_high)
    assert calc_sim_high > 0.7
    assert status_high == "Matched"
    assert name_high == "TestUser"

    # 2. Test similarity = 0.65 (< 0.7)
    sim_target_low = 0.65
    b_low = np.sqrt(1.0 - sim_target_low**2)
    v_low = sim_target_low * base_vec + b_low * ortho
    v_low = F.normalize(v_low, p=2, dim=1)

    status_low, name_low, calc_sim_low = face_system.compute_similarity(v_low)
    assert calc_sim_low < 0.7
    assert status_low == "Unknown"
    assert name_low == "Unknown"


def test_process_frame_without_faces(face_system):
    """
    Kiểm tra hàm process_frame khi khung hình là nền đen (không có khuôn mặt).
    Hệ thống phải trả về ảnh vẽ và danh sách detections rỗng mà không bị crash.
    """
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, detections = face_system.process_frame(black_frame)

    assert annotated.shape == black_frame.shape
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_open_webcam_backend_anti_green_screen():
    """
    Kiểm tra hàm open_webcam tự động chuyển sang backend DirectShow (CAP_DSHOW)
    trên Windows, chống lại lỗi màn hình xanh lá cây (solid green) do MSMF gây ra.
    """
    cap = open_webcam(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            std_val = float(np.std(frame))
            b_mean, g_mean, r_mean = np.mean(frame, axis=(0, 1))
            is_solid_green = (std_val < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0)
            assert not is_solid_green, "Lỗi: Khung hình webcam bị hiện tượng màn hình xanh lá cây (solid green)!"
            assert std_val > 5.0, f"Độ biến thiên pixel quá thấp ({std_val}), nghi ngờ khung hình rỗng!"
