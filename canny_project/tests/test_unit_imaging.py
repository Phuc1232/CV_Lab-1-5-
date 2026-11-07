"""Tầng ảnh: kiểm tra hợp lệ, dtype, màu, đọc/ghi (image-data-rules.md)."""

import cv2
import numpy as np
import pytest

from cannylab.imaging import InvalidImageError, check_gray, check_image, rgb_to_gray, to_float, to_uint8
from cannylab.imaging.io import load_image, save_image


@pytest.mark.parametrize("bad, msg", [
    (None, "None"),
    (np.zeros((0, 5), np.uint8), "rỗng"),
    (np.zeros((1, 1), np.uint8), "quá nhỏ"),
    (np.zeros((5, 5), np.int64), "dtype"),
    (np.zeros((5, 5), bool), "dtype"),
    (np.full((5, 5), np.nan), "NaN"),
    (np.full((5, 5), 2.0), "\\[0, 1\\]"),
    (np.zeros((5, 5, 4), np.uint8), "3 kênh"),
    (np.zeros((5, 5, 5, 3), np.uint8), "shape"),
    ([[1, 2], [3, 4]], "ndarray"),
])
def test_check_image_rejects_invalid(bad, msg):
    with pytest.raises(InvalidImageError, match=msg):
        check_image(bad)


def test_check_gray_rejects_color():
    with pytest.raises(InvalidImageError):
        check_gray(np.zeros((5, 5, 3), np.uint8))


def test_to_float_and_back_roundtrip():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    f = to_float(img)
    assert f.dtype == np.float64 and f.min() == 0 and f.max() == 1
    assert np.array_equal(to_uint8(f), img)


def test_to_uint8_rounds_and_clips_without_wraparound():
    f = np.array([[0.0, 0.5, 1.0], [0.998, 0.002, 1.0 + 1e-7], [0.25, 0.75, 0.1]])
    out = to_uint8(f)
    assert out.dtype == np.uint8
    assert out[0, 1] == 128 and out[1, 0] == 254 and out[1, 2] == 255  # không quấn về 0


def test_rgb_to_gray_matches_opencv_within_one_level():
    rgb = np.random.default_rng(0).integers(0, 256, (64, 64, 3)).astype(np.uint8)
    diff = np.abs(rgb_to_gray(rgb).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(int))
    assert diff.max() <= 1


def test_io_roundtrip_keeps_rgb_order_and_unicode_path(tmp_path):
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 200  # kênh đỏ
    path = save_image(tmp_path / "ảnh đỏ.png", rgb)
    back = load_image(path, "rgb")
    assert np.array_equal(back, rgb)
    assert load_image(path, "gray").shape == (10, 12)


def test_load_corrupted_or_missing_file(tmp_path):
    bad = tmp_path / "hong.png"
    bad.write_bytes(b"khong phai anh")
    with pytest.raises(InvalidImageError, match="giải mã"):
        load_image(bad)
    with pytest.raises(InvalidImageError, match="Không tìm thấy"):
        load_image(tmp_path / "khong_co.png")
