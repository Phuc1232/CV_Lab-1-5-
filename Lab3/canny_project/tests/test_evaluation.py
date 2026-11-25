"""Evaluation Engine: định nghĩa, ca biên, giá trị đã biết."""

import numpy as np
import pytest

from cannylab.evaluation.classification import classification_report
from cannylab.evaluation.edge_metrics import agreement, continuity, edge_metrics
from cannylab.evaluation.image_quality import psnr
from cannylab.evaluation.performance import fit_power_law, measure
from cannylab.evaluation.region_metrics import dice, iou, match_objects


def _line(col: int, shape=(20, 20)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[2:18, col] = True
    return m


def test_identical_is_perfect():
    g = _line(10)
    r = edge_metrics(g, g)
    assert r["precision"] == r["recall"] == r["f1"] == r["f1_1to1"] == r["fom"] == 1.0 and r["chamfer"] == 0


def test_tolerance_controls_shift_acceptance():
    g, p = _line(10), _line(11)
    assert edge_metrics(p, g, 1.5)["f1"] == 1.0
    assert edge_metrics(p, g, 0.5)["f1"] == 0.0
    assert edge_metrics(p, g, 1.5)["fom"] == pytest.approx(1 / (1 + 1 / 9))


def test_double_edge_penalized_only_one_to_one():
    g = _line(10)
    p = _line(10) | _line(11)
    r = edge_metrics(p, g, 1.5)
    assert r["precision"] == 1.0 and r["precision_1to1"] == pytest.approx(0.5)


@pytest.mark.parametrize("p, g, exp", [
    (np.zeros((5, 5), bool), np.zeros((5, 5), bool), (1.0, 1.0, 1.0)),
    (np.zeros((5, 5), bool), np.eye(5, dtype=bool), (1.0, 0.0, 0.0)),
    (np.eye(5, dtype=bool), np.zeros((5, 5), bool), (0.0, 1.0, 0.0)),
])
def test_edge_cases(p, g, exp):
    r = edge_metrics(p, g)
    assert (r["precision"], r["recall"], r["f1"]) == exp


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        edge_metrics(np.zeros((3, 3), bool), np.zeros((4, 4), bool))


def test_continuity_counts_components():
    m = _line(3) | _line(10)
    c = continuity(m)
    assert c["components"] == 2 and c["mean_length"] == 16


def test_agreement():
    a = _line(3)
    assert agreement(a, a)["identical"] and agreement(a, _line(4))["n_diff"] == 32


def test_iou_dice_relation(rng):
    a, b = rng.random((30, 30)) < 0.5, rng.random((30, 30)) < 0.5
    j = iou(a, b)
    assert dice(a, b) == pytest.approx(2 * j / (1 + j))


def test_match_objects():
    gt = np.zeros((20, 20), np.int32)
    gt[2:8, 2:8] = 1
    gt[12:18, 12:18] = 2
    pred = np.zeros_like(gt)
    pred[2:8, 2:9] = 1      # khớp vật 1 (IoU 36/42)
    pred[0:2, 15:20] = 2    # vùng thừa
    r = match_objects(pred, gt)
    assert r["object_precision"] == 0.5 and r["object_recall"] == 0.5 and r["count_error"] == 0
    assert r["pairs"][0][:2] == (1, 1)


def test_classification_report():
    r = classification_report(["a", "a", "b", "c"], ["a", "b", "b", "c"])
    assert r["accuracy"] == 0.75 and r["confusion"][0] == [1, 1, 0]


def test_psnr_known_value():
    a = np.zeros((10, 10))
    assert psnr(a, a + 0.1) == pytest.approx(20.0)
    assert psnr(a, a) == float("inf")


def test_measure_and_power_law():
    r = measure(lambda: sum(range(1000)), repeats=3)
    assert r["time_median_s"] > 0 and "mem_peak_tracemalloc_mb" in r
    fit = fit_power_law([1, 10, 100], [2, 20, 200])
    assert fit["exponent"] == pytest.approx(1.0)
