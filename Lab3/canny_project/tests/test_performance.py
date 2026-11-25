"""Hiệu năng (đánh dấu slow): độ phức tạp tuyến tính và bản Optimized nhanh hơn bản Manual."""

import time

import numpy as np
import pytest

from cannylab import registry
from cannylab.evaluation.performance import fit_power_law


def _t(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@pytest.mark.slow
def test_manual_canny_scales_linearly(rng):
    sizes, times = [], []
    for n in (256, 512, 1024):
        img = rng.random((n, n))
        sizes.append(n * n)
        times.append(_t(lambda: registry.run("canny_manual_fast", img, sigma=1.5, low_threshold=0.05,
                                             high_threshold=0.15)))
    assert fit_power_law(sizes, times)["exponent"] < 1.3


@pytest.mark.slow
def test_label_hysteresis_faster_than_bfs(rng):
    img = rng.random((512, 512))
    kw = dict(sigma=1.0, low_threshold=0.02, high_threshold=0.1)
    assert _t(lambda: registry.run("canny_manual_fast", img, **kw)) < _t(lambda: registry.run("canny_manual", img, **kw))


@pytest.mark.slow
def test_opencv_faster_than_manual(camera):
    kw = dict(low_threshold=50, high_threshold=150)
    assert _t(lambda: registry.run("canny_opencv", camera, **kw)) < \
        _t(lambda: registry.run("canny_manual", camera, profile="opencv", **kw))
