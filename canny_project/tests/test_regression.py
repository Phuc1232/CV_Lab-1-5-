"""Hồi quy: so với số liệu đã chốt trong tests/data/regression.json (testing-rules.md mục 5)."""

import json
from pathlib import Path

import numpy as np
import pytest
from skimage import data

from cannylab import registry
from cannylab.core import OPENCV, canny
from cannylab.core.thresholds import estimate_noise_sigma
from cannylab.datasets import Dataset
from cannylab.evaluation.edge_metrics import edge_metrics

REF = json.loads((Path(__file__).parent / "data" / "regression.json").read_text())


@pytest.fixture(scope="module")
def cam():
    return data.camera()


def test_edge_counts(cam):
    assert int(canny(cam, None, 100, 200, profile=OPENCV).edges.sum()) == REF["camera_opencv_100_200"]
    assert int(canny(cam).edges.sum()) == REF["camera_skimage_default"]
    assert int(canny(cam, 2.0, threshold_strategy="noise").edges.sum()) == REF["camera_noise_strategy_sigma2"]
    assert int(canny(cam, 2.0, threshold_strategy="otsu").edges.sum()) == REF["camera_otsu_sigma2"]


def test_noise_estimate(cam):
    assert estimate_noise_sigma(cam.astype(float)) == pytest.approx(REF["camera_noise_sigma_estimate"], abs=1e-6)


def test_synthetic_f1():
    ds = Dataset.synthetic("shapes", 3, 128, seed=42, degradations=[{"noise_sigma": 0.1}])
    f1 = [edge_metrics(registry.run("canny_manual_fast", s.image, sigma=2.0, low_threshold=0.03,
                                    high_threshold=0.08), s.edges_gt)["f1"] for s in ds]
    assert np.allclose(f1, REF["shapes_seed42_noise0.1_f1"], atol=1e-6)
