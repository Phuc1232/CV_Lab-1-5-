"""Mệnh đề toán học (docs/mathematics) và tính chất dùng làm property test."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from cannylab.core import canny
from cannylab.core.color import canny_color, structure_tensor_gradient
from cannylab.core.gradient import sobel
from cannylab.core.thresholds import estimate_noise_sigma, gradient_noise_gain


def test_sobel_gain_is_eight_on_ramp():
    ramp = np.tile(np.arange(32, dtype=np.float64), (32, 1))
    gx, _ = sobel(ramp, mode="replicate")
    assert np.allclose(gx[2:-2, 2:-2], 8.0)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
def test_smoothed_noise_std_formula(sigma, rng):
    w = rng.normal(size=(512, 512))
    measured = ndi.gaussian_filter(w, sigma)[20:-20, 20:-20].std()
    assert abs(measured - 1 / (2 * np.sqrt(np.pi) * sigma)) / measured < 0.03


@pytest.mark.parametrize("sigma", [None, 1.0, 2.0, 3.0])
def test_gradient_noise_gain_is_exact(sigma, rng):
    w = rng.normal(size=(600, 600))
    sm = w if not sigma else ndi.gaussian_filter(w, sigma, truncate=4.0)
    gx, _ = sobel(sm, mode="reflect")
    assert abs(gx[30:-30, 30:-30].std() - gradient_noise_gain(sigma)) / gradient_noise_gain(sigma) < 0.03


@pytest.mark.parametrize("sigma_n", [0.01, 0.05, 0.1])
def test_immerkaer_noise_estimate(sigma_n, scene, rng):
    noisy = scene["image"] + rng.normal(0, sigma_n, scene["image"].shape)
    est = estimate_noise_sigma(noisy)
    assert abs(est - sigma_n) / sigma_n < 0.15  # sai số tương đối 15%: còn sót pixel sát cạnh


def test_immerkaer_pure_noise_unbiased(rng):
    for s in (0.02, 0.1):
        assert abs(estimate_noise_sigma(rng.normal(0.5, s, (256, 256))) - s) / s < 0.05


def test_edges_monotone_in_thresholds(camera):
    grid = [10, 20, 35, 50, 80]
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            e = canny(camera, 1.5, lo, hi).edges
            for hi2 in [h for h in grid if h > hi]:
                assert not (canny(camera, 1.5, lo, hi2).edges & ~e).any()
            for lo2 in [g for g in grid if lo < g <= hi]:
                assert not (canny(camera, 1.5, lo2, hi).edges & ~e).any()


def test_di_zenzo_single_channel_reduces_to_l2(rng):
    gx, gy = rng.normal(size=(20, 20)), rng.normal(size=(20, 20))
    ex, ey, mag = structure_tensor_gradient([gx], [gy])
    assert np.allclose(mag, np.hypot(gx, gy))
    assert np.allclose(np.abs(ex), np.abs(gx)) and np.allclose(np.abs(ey), np.abs(gy))


def test_di_zenzo_gray_image_equals_gray_canny(camera):
    rgb = np.stack([camera] * 3, -1)
    # 3 kênh giống nhau: lambda = 3|g|^2 nên độ lớn gấp sqrt(3); nhân ngưỡng tương ứng
    s3 = np.sqrt(3)
    a = canny_color(rgb, "di_zenzo", 1.5, 20 * s3, 40 * s3).edges
    b = canny(camera, 1.5, 20, 40).edges
    assert (a ^ b).sum() <= 0.001 * b.sum()  # chỉ lệch ở điểm hòa do làm tròn cos/sin


def test_isoluminant_edges_lost_in_gray_but_found_in_color():
    from cannylab.datasets.synthetic import isoluminant_scene
    sc = isoluminant_scene(96, np.random.default_rng(0))
    assert canny_color(sc["image"], "gray", 1.0, 0.05, 0.15).edges.sum() == 0
    assert canny_color(sc["image"], "di_zenzo", 1.0, 0.05, 0.15).edges.sum() > 100
