"""
=============================================================================
UNIT TESTS FOR LAB 5: REAL-TIME FACE RECOGNITION (FACENET & MTCNN)
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

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = str(Path(__file__).resolve().parents[1])
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.facelab import backends, cli
from src.facelab.core.recognition import FaceRecognitionSystem
from src.facelab.paths import DEFAULT_DB_DIR


@pytest.fixture(scope="module")
def face_system():
    """
    Fixture khởi tạo đối tượng FaceRecognitionSystem trên CPU để kiểm thử.
    """
    system = FaceRecognitionSystem(
        db_dir=DEFAULT_DB_DIR,
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


def test_open_webcam_skips_solid_green_and_keeps_textured_green(monkeypatch):
    solid_green = np.full((8, 8, 3), (0, 135, 0), dtype=np.uint8)
    textured_green = solid_green.copy()
    textured_green[:4, :, 1] = 110
    solid = Mock()
    solid.read.return_value = (True, solid_green)
    textured = Mock()
    textured.read.return_value = (True, textured_green)
    video_capture = Mock(side_effect=[solid, textured])
    monkeypatch.setattr(backends, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(backends.cv2, "VideoCapture", video_capture)

    cap = backends.open_webcam(2, width=640, height=480)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret and frame is textured_green
    assert video_capture.call_args_list == [
        call(2, backends.cv2.CAP_DSHOW),
        call(2, backends.cv2.CAP_DSHOW),
    ]
    settings = [
        call.isOpened(),
        call.set(backends.cv2.CAP_PROP_FRAME_WIDTH, 640),
        call.set(backends.cv2.CAP_PROP_FRAME_HEIGHT, 480),
    ]
    assert solid.method_calls[:4] == [*settings, call.read()]
    assert textured.method_calls[:5] == [
        *settings,
        call.set(backends.cv2.CAP_PROP_FOURCC, backends.cv2.VideoWriter_fourcc(*"MJPG")),
        call.read(),
    ]
    solid.release.assert_called_once_with()
    textured.release.assert_not_called()
    cap.release()
    textured.release.assert_called_once_with()


def test_open_webcam_releases_failed_candidates_without_unvalidated_fallback(monkeypatch):
    unavailable = Mock()
    unavailable.isOpened.return_value = False
    unsupported_codec = Mock()
    unsupported_codec.set.return_value = False
    read_error = Mock()
    read_error.read.side_effect = RuntimeError("camera disconnected")
    malformed = Mock()
    malformed.read.side_effect = [
        (True, None),
        (True, np.empty((0, 8, 3), dtype=np.uint8)),
        (True, np.zeros((8, 8), dtype=np.uint8)),
        (True, np.zeros((8, 8, 4), dtype=np.uint8)),
        (True, np.zeros((8, 8, 3), dtype=np.float32)),
    ]
    video_capture = Mock(side_effect=[unavailable, unsupported_codec, read_error, malformed])
    monkeypatch.setattr(backends, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(backends.cv2, "VideoCapture", video_capture)

    cap = backends.open_webcam(2)

    assert not cap.isOpened()
    assert cap.read() == (False, None)
    assert video_capture.call_args_list == [
        call(2, backends.cv2.CAP_DSHOW),
        call(2, backends.cv2.CAP_DSHOW),
        call(2, backends.cv2.CAP_MSMF),
        call(2, backends.cv2.CAP_ANY),
    ]
    unsupported_codec.read.assert_not_called()
    unsupported_codec.release.assert_called_once_with()
    for capture in (unavailable, read_error, malformed):
        capture.set.assert_not_called()
        capture.release.assert_called_once_with()


def test_webcam_filters_late_green_and_recovers_until_candidates_run_out(monkeypatch):
    healthy = np.full((8, 8, 3), (20, 40, 60), dtype=np.uint8)
    green = np.full((8, 8, 3), (0, 135, 0), dtype=np.uint8)
    recovered = np.full((8, 8, 3), (30, 50, 70), dtype=np.uint8)
    initial = Mock()
    initial.read.side_effect = [
        *([(True, healthy)] * 5),
        (True, green),
        (True, healthy),
        *([(True, green)] * 3),
    ]
    alternate = Mock()
    alternate.read.side_effect = [
        *([(True, recovered)] * 6),
        *([(True, green)] * 3),
    ]
    solid = Mock()
    solid.read.return_value = (True, green)
    unavailable = Mock()
    unavailable.isOpened.return_value = False
    video_capture = Mock(side_effect=[initial, alternate, solid, unavailable])
    monkeypatch.setattr(backends, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(backends.cv2, "VideoCapture", video_capture)

    cap = backends.open_webcam(2)
    ret, frame = cap.read()
    assert ret and frame is healthy
    assert cap.generation == 1
    video_capture.assert_called_once_with(2, backends.cv2.CAP_DSHOW)
    ret, frame = cap.read()
    assert ret and frame is recovered
    assert cap.generation == 2
    initial.release.assert_called_once_with()

    assert cap.read() == (False, None)
    assert cap.read() == (False, None)
    assert not cap.isOpened()
    cap.release()
    for capture in (alternate, solid, unavailable):
        capture.release.assert_called_once_with()
    assert video_capture.call_args_list == [
        call(2, backends.cv2.CAP_DSHOW),
        call(2, backends.cv2.CAP_DSHOW),
        call(2, backends.cv2.CAP_MSMF),
        call(2, backends.cv2.CAP_ANY),
    ]


def test_open_webcam_backend_anti_green_screen():
    cap = backends.open_webcam(0, width=640, height=480)
    try:
        if not cap.isOpened():
            pytest.skip("Không có webcam trả về hình ảnh hợp lệ")

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                assert frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0
                b_mean, g_mean, r_mean = np.mean(frame, axis=(0, 1))
                std_val = np.std(frame, axis=(0, 1)).max()
                is_solid_green = std_val < 3.0 and g_mean > 100.0 and b_mean < 15.0 and r_mean < 15.0
                assert not is_solid_green, "Khung hình webcam bị hiện tượng màn hình xanh lá cây"
                return
        pytest.fail("Webcam đã mở nhưng không đọc được hình ảnh")
    finally:
        cap.release()


def test_recognition_config_and_cli_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "recognition.toml"
    config_path.write_text(
        '[recognition]\ncamera = 3\nthreshold = 0.8\nframe_skip = 5\n'
        'db_dir = "faces"\ndevice = "cpu"\n',
        encoding="utf-8",
    )
    received = {}
    monkeypatch.setattr(cli, "run_webcam", lambda **settings: received.update(settings))

    cli.main(["--config", str(config_path), "--camera", "0", "--threshold", "0"])

    assert received == {
        "camera_id": 0,
        "similarity_threshold": 0.0,
        "frame_skip": 5,
        "db_dir": str(tmp_path / "faces"),
        "device": "cpu",
    }


def test_recognition_config_rejects_invalid_values(tmp_path, capsys):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[recognition]\nframe_skip = true\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config_path)])

    assert error.value.code == 2
    assert "--config" in capsys.readouterr().err
