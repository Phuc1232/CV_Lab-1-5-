"""Kiểm chứng Manual so với thư viện chuẩn (algorithm-rules.md mục 2): trùng TỪNG PIXEL."""

import cv2
import numpy as np
import pytest
from skimage import data, feature

from cannylab import backends, registry
from cannylab.core import OPENCV, SKIMAGE, canny
from cannylab.core.color import canny_color


@pytest.mark.parametrize("low, high, l2", [(100, 200, False), (50, 150, True), (20, 40, False), (0, 10, True),
                                          (200, 100, False), (30.5, 90.7, True)])
def test_opencv_profile_pixel_exact(real_images, low, high, l2):
    for name, img in real_images.items():
        ours = canny(img, None, low, high, profile=OPENCV, norm="L2" if l2 else "L1").edges
        ref = cv2.Canny(img, low, high, L2gradient=l2) > 0
        assert np.array_equal(ours, ref), f"{name}: lệch {(ours ^ ref).sum()} px"


@pytest.mark.parametrize("sigma, low, high", [(1.0, None, None), (0.5, 30, 60), (2.0, 15, 40), (3.0, 10, 30),
                                              (1.5, 5, 5)])
def test_skimage_profile_pixel_exact(real_images, sigma, low, high):
    for name, img in real_images.items():
        ours = canny(img, sigma, low, high, profile=SKIMAGE).edges
        ref = feature.canny(img, sigma=sigma, low_threshold=low, high_threshold=high)
        assert np.array_equal(ours, ref), f"{name}: lệch {(ours ^ ref).sum()} px"


def test_skimage_profile_float_and_uint16_inputs(camera):
    f = camera / 255.0
    assert np.array_equal(canny(f, 1.0, 0.05, 0.1).edges, feature.canny(f, 1.0, 0.05, 0.1))
    u16 = camera.astype(np.uint16) * 257
    assert np.array_equal(canny(u16, 1.0, 3000, 8000).edges, feature.canny(u16, 1.0, 3000, 8000))


def test_quantile_strategy_matches_skimage_use_quantiles(camera):
    ours = canny(camera, 1.5, threshold_strategy="quantile", strategy_params={"low_q": 0.8, "high_q": 0.9}).edges
    ref = feature.canny(camera, 1.5, 0.8, 0.9, use_quantiles=True)
    assert np.array_equal(ours, ref)


def test_noise_image_exact():
    noise = np.random.default_rng(0).integers(0, 256, (97, 131)).astype(np.uint8)
    assert np.array_equal(canny(noise, None, 50, 150, profile=OPENCV).edges, cv2.Canny(noise, 50, 150) > 0)
    assert np.array_equal(canny(noise, 2.0).edges, feature.canny(noise, 2.0))


def test_color_max_channel_matches_opencv_3channel():
    rgb = data.astronaut()
    for l2 in (False, True):
        ours = canny_color(rgb, "max_channel", None, 100, 200, profile=OPENCV, norm="L2" if l2 else "L1").edges
        assert np.array_equal(ours, cv2.Canny(rgb, 100, 200, L2gradient=l2) > 0)


def test_registry_wrappers_agree(camera):
    a = registry.run("canny_manual", camera, profile="opencv", low_threshold=50, high_threshold=120)
    b = registry.run("canny_opencv", camera, low_threshold=50, high_threshold=120)
    c = registry.run("canny_manual_fast", camera, profile="opencv", low_threshold=50, high_threshold=120)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_baselines_manual_vs_optimized(camera):
    img = camera / 255.0
    log_a = registry.run("log_manual", img, sigma=2.0, threshold_rel=0.05)
    log_b = registry.run("log_scipy", img, sigma=2.0, threshold_rel=0.05)
    # LoG: Manual = Gaussian rồi Laplacian 4-lân cận; SciPy = đạo hàm bậc 2 của Gaussian -> gần giống
    assert (log_a & log_b).sum() / max(1, (log_a | log_b).sum()) > 0.85
    s_a = registry.run("sobel_threshold_manual", img, threshold=0.3)
    s_b = registry.run("sobel_threshold_opencv", img, threshold=0.3)
    assert np.array_equal(s_a, s_b)


def test_opencv_with_sigma_is_approximate_but_close(camera):
    ours = canny(camera, 2.0, 30, 90, profile=OPENCV).edges
    ref = backends.canny_opencv(camera, 30, 90, sigma=2.0)
    iou = (ours & ref).sum() / (ours | ref).sum()
    # cv2.GaussianBlur uint8 dùng số học điểm cố định nên bản Manual chỉ xấp xỉ: đo được IoU = 0.897
    # (docs/algorithms/06_canny_pipeline.md). Ngưỡng 0.85 để phát hiện hồi quy, không phải mục tiêu khớp.
    assert iou > 0.85
