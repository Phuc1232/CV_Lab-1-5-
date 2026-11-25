"""Ca biên bắt buộc (testing-rules.md mục 2) cho mọi bộ phát hiện cạnh ảnh xám đã đăng ký."""

import numpy as np
import pytest

from cannylab import registry
from cannylab.imaging import InvalidImageError

GRAY_ALGOS = [s.name for s in registry.available() if not s.color]
DEFAULT_PARAMS = {"canny_manual": {}, "canny_manual_fast": {}, "canny_opencv": {}, "canny_skimage": {},
                  "sobel_threshold_manual": {}, "sobel_threshold_opencv": {}, "log_manual": {}, "log_scipy": {}}


@pytest.mark.parametrize("name", GRAY_ALGOS)
@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_constant_images_have_no_edges(name, value):
    img = np.full((40, 50), value)
    assert registry.run(name, img, **DEFAULT_PARAMS[name]).sum() == 0


@pytest.mark.parametrize("name", GRAY_ALGOS)
def test_noise_and_low_contrast_run_and_return_bool(name, rng):
    noisy = rng.random((64, 64))
    low = 0.5 + 0.02 * (np.arange(64)[None, :] > 32) * np.ones((64, 1))
    for img in (noisy, low, (noisy * 255).astype(np.uint8)):
        out = registry.run(name, img, **DEFAULT_PARAMS[name])
        assert out.dtype == bool and out.shape == img.shape


@pytest.mark.parametrize("name", GRAY_ALGOS)
@pytest.mark.parametrize("bad", [None, np.zeros((0, 0)), np.zeros((2, 2)), np.full((8, 8), np.nan),
                                 np.zeros((8, 8), np.int64), np.full((8, 8), 3.0)])
def test_invalid_inputs_raise(name, bad):
    with pytest.raises(InvalidImageError):
        registry.run(name, bad, **DEFAULT_PARAMS[name])


@pytest.mark.parametrize("name", GRAY_ALGOS)
def test_multichannel_rejected_by_gray_algorithms(name):
    with pytest.raises(InvalidImageError):
        registry.run(name, np.zeros((16, 16, 3), np.uint8), **DEFAULT_PARAMS[name])


def test_color_algorithms_accept_rgb(rng):
    rgb = rng.random((32, 32, 3))
    for m in ("gray", "or", "max_channel", "di_zenzo"):
        assert registry.run("canny_color_manual", rgb, method=m).shape == (32, 32)
    with pytest.raises(ValueError):
        registry.run("canny_color_manual", rgb, method="xyz")


def test_bad_parameters():
    img = np.random.default_rng(0).random((20, 20))
    with pytest.raises(ValueError):
        registry.run("canny_manual", img, sigma=-1)
    with pytest.raises(ValueError):
        registry.run("canny_manual", img, low_threshold=0.3, high_threshold=0.1)
    with pytest.raises(ValueError):
        registry.run("canny_manual", img, profile="opencv")  # OpenCV bắt buộc có ngưỡng
    with pytest.raises(ValueError):
        registry.run("canny_manual", img, threshold_strategy="abc")
    with pytest.raises(KeyError, match="Gợi ý"):
        registry.get("canny_manul")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["canny_manual_fast", "canny_opencv", "canny_skimage"])
def test_large_image(name, rng):
    img = rng.random((2048, 2048))
    assert registry.run(name, img).shape == (2048, 2048)
