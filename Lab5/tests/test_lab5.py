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

from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

CURRENT_DIR = str(Path(__file__).resolve().parents[1])
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.facelab import backends, cli
from src.facelab.core.recognition import FaceRecognitionSystem
from src.facelab.datasets import registration
from src.facelab.datasets.store import EmbeddingStore
from src.facelab.paths import DEFAULT_DB_DIR


@pytest.fixture(scope="module")
def face_system(tmp_path_factory):
    """
    Fixture khởi tạo đối tượng FaceRecognitionSystem trên CPU để kiểm thử.
    """
    system = FaceRecognitionSystem(
        db_dir=str(tmp_path_factory.mktemp("default-test-gallery")),
        similarity_threshold=0.7,
        device="cpu",
    )
    return system


@pytest.fixture
def isolated_face_system(face_system, tmp_path, monkeypatch):
    store = EmbeddingStore(tmp_path / "gallery", face_system.database.model_signature)
    monkeypatch.setattr(face_system, "database", store)
    return face_system


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


def test_cosine_similarity_threshold_decision(isolated_face_system):
    """
    Kiểm tra quyết định 'Matched' vs 'Unknown' theo đúng tiêu chí đề bài:
      - similarity > 0.7  => 'Matched'
      - similarity <= 0.7 => 'Unknown'
    """
    face_system = isolated_face_system
    alice_vector = F.normalize(torch.randn(1, 512), p=2, dim=1)
    face_system.database.upsert("Alice.jpg", "Alice", "example", alice_vector.numpy())

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


def test_strict_boundary_condition(isolated_face_system):
    """
    Kiểm tra nghiêm ngặt tại giá trị biên 0.7:
      - 0.75 > 0.7 => 'Matched'
      - 0.65 <= 0.7 => 'Unknown'
    """
    face_system = isolated_face_system
    base_vec = F.normalize(torch.randn(1, 512), p=2, dim=1)
    face_system.database.upsert("TestUser.jpg", "TestUser", "example", base_vec.numpy())

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


@pytest.mark.parametrize(
    ("entry_point", "capture_options"),
    [(cli.main, {"width": 640, "height": 480}), (cli.register_main, {})],
)
def test_camera_list_releases_probes_before_opening_next(
    entry_point, capture_options, monkeypatch, capsys,
):
    captures = [Mock() for _ in range(5)]
    for camera_id, capture in enumerate(captures):
        capture.isOpened.return_value = camera_id in (0, 3)

    def open_camera(camera_id, **kwargs):
        if camera_id > 0:
            captures[camera_id - 1].release.assert_called_once_with()
        return captures[camera_id]

    webcam = Mock(side_effect=open_camera)
    run = Mock()
    register = Mock()
    monkeypatch.setattr(cli, "open_webcam", webcam)
    monkeypatch.setattr(cli, "run_webcam", run)
    monkeypatch.setattr(cli, "register_from_webcam", register)

    entry_point(["--list-cameras"])

    assert webcam.call_args_list == [call(index, **capture_options) for index in range(5)]
    captures[-1].release.assert_called_once_with()
    assert "Camera khả dụng: #0, #3" in capsys.readouterr().out
    run.assert_not_called()
    register.assert_not_called()


def test_recognition_camera_selection_retries_and_overrides_config(tmp_path, monkeypatch):
    config_path = tmp_path / "recognition.toml"
    config_path.write_text("[recognition]\ncamera = 3\n", encoding="utf-8")
    prompt = Mock(side_effect=["không phải chỉ số", "2", "0"])
    run = Mock()
    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(cli, "_list_cameras", Mock(return_value=[0, 3]))
    monkeypatch.setattr(cli, "run_webcam", run)

    cli.main(["--config", str(config_path), "--select-camera"])

    assert prompt.call_count == 3
    run.assert_called_once_with(camera_id=0)


@pytest.mark.parametrize("cancel", [EOFError, KeyboardInterrupt])
def test_camera_selection_cancel_skips_recognition(cancel, monkeypatch):
    run = Mock()
    monkeypatch.setattr("builtins.input", Mock(side_effect=cancel))
    monkeypatch.setattr(cli, "_list_cameras", Mock(return_value=[0]))
    monkeypatch.setattr(cli, "run_webcam", run)

    cli.main(["--select-camera"])

    run.assert_not_called()


def test_registration_uses_interactive_camera_selection(monkeypatch):
    register = Mock()
    select = Mock(return_value=0)
    monkeypatch.setattr(cli, "_select_camera", select)
    monkeypatch.setattr(cli, "register_from_webcam", register)

    cli.register_main(["--name", "Alice", "--select-camera"])

    select.assert_called_once_with()
    register.assert_called_once_with("Alice", 0, DEFAULT_DB_DIR)


@pytest.mark.parametrize(
    "camera_options",
    [["--camera", "0", "--select-camera"], ["--image", "sample.jpg", "--select-camera"]],
)
def test_registration_rejects_conflicting_camera_options(camera_options, monkeypatch):
    select = Mock()
    monkeypatch.setattr(cli, "_select_camera", select)

    with pytest.raises(SystemExit) as error:
        cli.register_main(["--name", "Alice", *camera_options])

    assert error.value.code == 2
    select.assert_not_called()


def _axis_vector(index: int) -> np.ndarray:
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def test_gallery_reuses_cached_vectors_and_invalidates_changed_model(tmp_path):
    source = b"test sample"
    (tmp_path / "Lan.jpg").write_bytes(source)
    encode = Mock(return_value=_axis_vector(0))
    store = EmbeddingStore(tmp_path, "test-model-v1")
    assert store.reconcile(encode) == 1
    encode.assert_called_once_with(source)

    encode.reset_mock()
    reopened = EmbeddingStore(tmp_path, "test-model-v1")
    assert reopened.reconcile(encode) == 1
    encode.assert_not_called()
    identity, embedding = next(iter(reopened.iter_embeddings()))
    assert identity == "Lan"
    assert np.array_equal(embedding, _axis_vector(0))

    upgraded = EmbeddingStore(tmp_path, "test-model-v2")
    assert len(upgraded) == 0
    assert upgraded.reconcile(encode) == 1
    encode.assert_called_once_with(source)


def test_gallery_merges_legacy_and_id_folders_and_reconciles_files(tmp_path):
    (tmp_path / "Lan_2.jpg").write_bytes(b"front")
    deleted = tmp_path / "Minh_left.jpg"
    deleted.write_bytes(b"side")
    folder = tmp_path / "Lan"
    folder.mkdir()
    updated = folder / "side.jpg"
    updated.write_bytes(b"left")
    literal_id = tmp_path / "ID_4"
    literal_id.mkdir()
    invalidated = literal_id / "sample.png"
    invalidated.write_bytes(b"solo")
    embedding_by_source = {
        b"front": _axis_vector(0), b"side": _axis_vector(1), b"left": _axis_vector(2),
        b"solo": _axis_vector(3), b"back": _axis_vector(4), b"new": _axis_vector(5),
    }
    encode = Mock(side_effect=embedding_by_source.get)
    store = EmbeddingStore(tmp_path, "test-model")
    assert store.reconcile(encode) == 4
    assert sorted(identity for identity, _ in store.iter_embeddings()) == [
        "ID_4", "Lan", "Lan", "Minh_left",
    ]

    deleted.unlink()
    updated.write_bytes(b"back")
    invalidated.write_bytes(b"invalid")
    (folder / "another.jpg").write_bytes(b"new")
    encode.reset_mock()
    assert store.reconcile(encode) == 3
    assert {args[0] for args, _ in encode.call_args_list} == {b"back", b"invalid", b"new"}
    assert {(identity, int(vector.argmax())) for identity, vector in store.iter_embeddings()} == {
        ("Lan", 0), ("Lan", 4), ("Lan", 5),
    }


@pytest.mark.parametrize("replacement_indexed", [False, True])
def test_gallery_drops_stale_snapshot_if_source_changes_during_encoding(tmp_path, replacement_indexed):
    source = tmp_path / "Lan.jpg"
    source.write_bytes(b"first")
    store = EmbeddingStore(tmp_path, "test-model")
    assert store.reconcile(lambda _: _axis_vector(0)) == 1
    source.write_bytes(b"second")

    def encode_changed_sample(image_bytes):
        assert image_bytes == b"second"
        source.write_bytes(b"third")
        if replacement_indexed:
            concurrent_store = EmbeddingStore(tmp_path, "test-model")
            assert concurrent_store.reconcile(lambda _: _axis_vector(2)) == 1
        return _axis_vector(1)

    assert store.reconcile(encode_changed_sample) == int(replacement_indexed)
    assert [int(vector.argmax()) for _, vector in store.iter_embeddings()] == (
        [2] if replacement_indexed else []
    )


def test_multi_angle_matches_the_best_sample_and_counts_distinct_ids(isolated_face_system, monkeypatch):
    face_system = isolated_face_system
    query = torch.from_numpy(_axis_vector(1)).unsqueeze(0)
    assert face_system.compute_similarity(query) == ("Unknown", "Unknown", 0.0)
    store = face_system.database
    competitor = (_axis_vector(1) + _axis_vector(2)) / np.sqrt(np.float32(2))
    store.upsert("Lan/front.jpg", "Lan", "example", _axis_vector(0))
    store.upsert("Minh/front.jpg", "Minh", "example", competitor)
    assert face_system.compute_similarity(query)[:2] == ("Matched", "Minh")
    store.upsert("Lan/side.jpg", "Lan", "example", _axis_vector(1))
    assert len(store) == 2
    assert face_system.compute_similarity(query) == ("Matched", "Lan", 1.0)

    monkeypatch.setattr(face_system, "similarity_threshold", 1.0)
    assert face_system.compute_similarity(query) == ("Unknown", "Unknown", 1.0)


def test_quick_registration_embeds_the_saved_jpeg(isolated_face_system, monkeypatch):
    face_system = isolated_face_system
    encode = Mock(return_value=_axis_vector(0))
    monkeypatch.setattr(face_system, "_embedding_from_image_bytes", encode)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    assert face_system.register_new_face("  Lan 3 ", frame)
    root = face_system.database.root
    sample, = (root / "Lan_3").glob("*.jpg")
    jpeg_bytes = sample.read_bytes()
    assert jpeg_bytes.startswith(b"\xff\xd8")
    encode.assert_called_once_with(jpeg_bytes)
    with closing(sqlite3.connect(face_system.database.path)) as connection:
        row = connection.execute("SELECT image_path, identity, source_hash FROM samples").fetchone()
    assert row == (sample.relative_to(root).as_posix(), "Lan_3", sha256(jpeg_bytes).hexdigest())
    assert face_system.compute_similarity(torch.from_numpy(_axis_vector(0))) == ("Matched", "Lan_3", 1.0)
    assert face_system.load_database() == 1
    encode.assert_called_once()


@pytest.mark.parametrize("identity", [
    "../outside", "folder\\outside", "NUL", "embeddings.sqlite3",
    "embeddings.sqlite3-journal", "EMBEDDINGS.SQLITE3-WAL", "embeddings.sqlite3-shm",
])
def test_registration_rejects_unsafe_identity(isolated_face_system, identity):
    face_system = isolated_face_system
    assert not face_system.register_new_face(identity, np.zeros((24, 32, 3), dtype=np.uint8))
    assert len(face_system.database) == 0
    assert set(face_system.database.root.iterdir()) == {face_system.database.path}


def test_registration_rolls_back_the_image_if_database_write_fails(isolated_face_system, monkeypatch):
    face_system = isolated_face_system
    monkeypatch.setattr(face_system, "_embedding_from_image_bytes", Mock(return_value=_axis_vector(0)))
    monkeypatch.setattr(
        face_system.database, "upsert", Mock(side_effect=sqlite3.OperationalError("disk full")),
    )

    assert not face_system.register_new_face("Lan", np.zeros((24, 32, 3), dtype=np.uint8))
    assert len(face_system.database) == 0
    assert not list((face_system.database.root / "Lan").iterdir())


def test_cli_registers_multiple_images_for_one_id(isolated_face_system, tmp_path, monkeypatch):
    face_system = isolated_face_system
    detector = Mock()
    detector.detect.return_value = (np.array([[3, 3, 19, 19]]), np.array([0.95]))
    monkeypatch.setattr(face_system, "mtcnn", detector)
    encode = Mock(side_effect=[_axis_vector(0), _axis_vector(1)])
    monkeypatch.setattr(face_system, "_embedding_from_image_bytes", encode)
    create_system = Mock(return_value=face_system)
    monkeypatch.setattr(registration, "FaceRecognitionSystem", create_system)
    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    Image.new("RGB", (24, 24), "red").save(front)
    Image.new("RGB", (24, 24), "green").save(side)

    root = face_system.database.root
    cli.register_main([
        "--id", "SV_01", "--image", str(front), "--image", str(side), "--db-dir", str(root),
    ])

    create_system.assert_called_once_with(db_dir=str(root), device="cpu", load_existing=False)
    assert encode.call_count == 2
    assert len(list((root / "SV_01").glob("*.jpg"))) == 2
    assert [identity for identity, _ in face_system.database.iter_embeddings()] == ["SV_01", "SV_01"]


def test_webcam_registration_keeps_samples_when_stopped_early(isolated_face_system, monkeypatch):
    face_system = isolated_face_system
    detector = Mock()
    detector.detect.return_value = (np.array([[3, 3, 19, 19]]), np.array([0.95]))
    monkeypatch.setattr(face_system, "mtcnn", detector)
    monkeypatch.setattr(
        face_system, "_embedding_from_image_bytes", Mock(side_effect=[_axis_vector(0), _axis_vector(1)]),
    )
    monkeypatch.setattr(registration, "FaceRecognitionSystem", Mock(return_value=face_system))
    cap = Mock()
    cap.read.return_value = (True, np.zeros((24, 24, 3), dtype=np.uint8))
    open_webcam = Mock(return_value=cap)
    monkeypatch.setattr(registration, "open_webcam", open_webcam)
    for operation in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(registration.cv2, operation, Mock())
    monkeypatch.setattr(registration.cv2, "waitKey", Mock(side_effect=[ord(" "), ord("c"), ord("q")]))

    root = face_system.database.root
    cli.register_main(["--id", "Lan", "--camera", "2", "--samples", "3", "--db-dir", str(root)])

    open_webcam.assert_called_once_with(2)
    cap.release.assert_called_once_with()
    assert len(list((root / "Lan").glob("*.jpg"))) == 2
    assert [identity for identity, _ in face_system.database.iter_embeddings()] == ["Lan", "Lan"]
