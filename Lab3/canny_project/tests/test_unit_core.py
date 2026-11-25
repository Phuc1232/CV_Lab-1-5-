"""Từng bước thuật toán tự cài so với tham chiếu (SciPy/OpenCV) và các tính chất cơ bản."""

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from cannylab import backends
from cannylab.core import morphology as morph
from cannylab.core.convolution import correlate1d, correlate2d, pad
from cannylab.core.denoise import bilateral_filter, median_filter
from cannylab.core.gaussian import gaussian_blur, gaussian_kernel1d
from cannylab.core.gradient import magnitude, sobel
from cannylab.core.hysteresis import double_threshold, hysteresis_bfs
from cannylab.core.nms import nms_bilinear, nms_quantized

SCIPY_MODE = {"constant": "constant", "replicate": "nearest", "reflect": "reflect", "reflect101": "mirror"}


@pytest.mark.parametrize("mode", list(SCIPY_MODE))
def test_correlate1d_matches_scipy(mode, rng):
    img = rng.random((20, 17))
    k = rng.random(5)
    for axis in (0, 1):
        ref = ndi.correlate1d(img, k, axis=axis, mode=SCIPY_MODE[mode])
        # 1e-12: sai số cộng dồn 5 tap trên float64
        assert np.allclose(correlate1d(img, k, axis, mode), ref, atol=1e-12)


def test_correlate2d_matches_scipy(rng):
    img, k = rng.random((15, 15)), rng.random((3, 5))
    assert np.allclose(correlate2d(img, k, "reflect"), ndi.correlate(img, k, mode="reflect"), atol=1e-12)


def test_pad_rejects_unknown_mode():
    with pytest.raises(ValueError):
        pad(np.zeros((3, 3)), 1, 1, "wrap")


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5, 4.0])
def test_gaussian_kernel_normalized_and_blur_matches_scipy(sigma, rng):
    k = gaussian_kernel1d(sigma)
    assert abs(k.sum() - 1) < 1e-15 and np.allclose(k, k[::-1])
    img = rng.random((40, 50))
    assert np.allclose(gaussian_blur(img, sigma, mode="reflect"), ndi.gaussian_filter(img, sigma, mode="reflect"),
                       atol=1e-12)


def test_gaussian_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gaussian_kernel1d(0)


def test_sobel_integer_exact_vs_opencv(camera):
    gx, gy = sobel(camera.astype(np.int64), mode="replicate")
    rx, ry = backends.sobel_opencv(camera)
    assert np.array_equal(gx, rx) and np.array_equal(gy, ry)


def test_magnitude_norm_bounds(rng):
    gx, gy = rng.normal(size=(30, 30)), rng.normal(size=(30, 30))
    l1, l2 = magnitude(gx, gy, "L1"), magnitude(gx, gy, "L2")
    assert np.all(l2 <= l1 + 1e-12) and np.all(l1 <= np.sqrt(2) * l2 + 1e-12)
    with pytest.raises(ValueError):
        magnitude(gx, gy, "L3")


def test_nms_tie_rules_on_ideal_step():
    img = np.zeros((12, 12))
    img[:, 6:] = 1.0
    gx, gy = sobel(img, mode="reflect")
    m = np.hypot(gx, gy)
    assert set(np.nonzero(nms_bilinear(gx, gy, m, 0.0)[5])[0]) == {5, 6}  # skimage giữ cả hai
    gxi, gyi = sobel((img * 255).astype(np.int64), mode="replicate")
    keep = nms_quantized(gxi, gyi, np.abs(gxi) + np.abs(gyi), -1)
    assert set(np.nonzero(keep[5])[0]) == {5}  # OpenCV giữ một


def test_nms_invariant_to_magnitude_scaling(camera):
    gx, gy = sobel(camera.astype(np.float64) / 255, mode="reflect")
    m = np.hypot(gx, gy)
    a = nms_bilinear(gx, gy, m, 0.0) > 0
    b = nms_bilinear(gx * 4, gy * 4, m * 4, 0.0) > 0  # 4 = lũy thừa của 2: nhân chính xác trong float
    assert np.array_equal(a, b)


def test_double_threshold_strict_vs_nonstrict():
    v = np.array([[0.0, 1.0, 2.0, 3.0]])
    s1, c1 = double_threshold(v, 1.0, 2.0, strict=True)
    s2, c2 = double_threshold(v, 1.0, 2.0, strict=False)
    assert c1.tolist() == [[False, False, True, True]] and s1.tolist() == [[False, False, False, True]]
    assert c2.tolist() == [[False, True, True, True]] and s2.tolist() == [[False, False, True, True]]
    with pytest.raises(ValueError):
        double_threshold(v, 3, 2, strict=True)


def test_hysteresis_bfs_equals_label(rng):
    for _ in range(20):
        cand = rng.random((60, 70)) < 0.45
        strong = cand & (rng.random((60, 70)) < 0.05)
        assert np.array_equal(hysteresis_bfs(strong, cand), backends.hysteresis_label(strong, cand))


@pytest.mark.parametrize("shape", morph.SHAPES)
@pytest.mark.parametrize("size", [1, 3, 4, 5, 7, 9, 11, 15])
def test_structuring_element_matches_opencv(shape, size):
    assert np.array_equal(morph.structuring_element(shape, size), backends.structuring_element_opencv(shape, size))


@pytest.mark.parametrize("op", ["dilate", "erode", "closing"])
def test_morphology_matches_opencv(op, rng):
    mask = rng.random((50, 60)) < 0.3
    se = morph.structuring_element("ellipse", 5)
    se8 = se.astype(np.uint8)
    m8 = mask.astype(np.uint8)
    ref = {"dilate": cv2.dilate(m8, se8), "erode": cv2.erode(m8, se8),
           "closing": cv2.morphologyEx(m8, cv2.MORPH_CLOSE, se8)}[op] > 0
    assert np.array_equal(getattr(morph, op)(mask, se), ref)


def test_fill_holes_and_labels_match_scipy(rng):
    mask = morph.closing(rng.random((80, 80)) < 0.35, morph.structuring_element("rect", 3))
    assert np.array_equal(morph.fill_holes(mask), ndi.binary_fill_holes(mask))
    lab, n = morph.label_components(mask, 8)
    ref, nref = ndi.label(mask, np.ones((3, 3), bool))
    assert n == nref
    assert np.array_equal(lab > 0, ref > 0)
    # cùng một phân hoạch: mỗi nhãn của ta ứng với đúng một nhãn tham chiếu
    pairs = set(zip(lab[mask].tolist(), ref[mask].tolist()))
    assert len(pairs) == n


def test_remove_small_relabels():
    lab = np.array([[1, 1, 0, 2], [1, 0, 0, 0], [0, 3, 3, 3]], dtype=np.int32)
    out, n = morph.remove_small(lab, 3)
    assert n == 2 and set(np.unique(out)) == {0, 1, 2}


def test_median_matches_opencv_interior(camera):
    ours = median_filter(camera, 3)
    ref = backends.median_opencv(camera, 3)
    assert np.allclose(ours[1:-1, 1:-1], ref[1:-1, 1:-1], atol=1e-12)


def test_median_removes_salt_pepper(rng):
    img = np.full((40, 40), 0.5)
    noisy = img.copy()
    idx = rng.random(img.shape) < 0.05
    noisy[idx] = rng.choice([0.0, 1.0], idx.sum())
    assert np.abs(median_filter(noisy, 3) - img).mean() < 0.01 < np.abs(noisy - img).mean()


def test_bilateral_close_to_opencv(camera):
    img = camera[100:200, 100:200]
    diff = np.abs(bilateral_filter(img, 5, 0.1, 2.0) - backends.bilateral_opencv(img, 5, 0.1, 2.0))
    assert diff.mean() < 0.01  # OpenCV dùng bảng tra và float32: chỉ yêu cầu gần đúng
