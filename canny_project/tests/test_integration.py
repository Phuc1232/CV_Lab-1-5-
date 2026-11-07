"""Tích hợp: dataset, pipeline P4, runner tái lập được, kiến trúc."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cannylab import registry
from cannylab.analysis.segmentation import segment_by_edges
from cannylab.analysis.shapes import douglas_peucker, recognize_shapes, trace_boundary
from cannylab.datasets import Dataset
from cannylab.experiments.config import parse_config
from cannylab.experiments.runner import run_experiment
from cannylab.imaging.io import save_image

SRC = Path(__file__).resolve().parents[1] / "src" / "cannylab"

TINY = """
name = "tiny"
question = "test"
seed = 3
[dataset]
source = "synthetic"
family = "shapes"
count = 2
size = 96
degradations = [{}, { noise_sigma = 0.05 }]
[[runs]]
label = "a"
algorithm = "canny_manual_fast"
params = { profile = "skimage", sigma = [1.0, 2.0], low_threshold = 0.04, high_threshold = 0.1 }
[[runs]]
label = "b"
algorithm = "canny_opencv"
params = { low_threshold = 30, high_threshold = 90, sigma = 1.5 }
[evaluation]
metrics = ["edge", "continuity", "segmentation", "shapes", "timing"]
timing = { repeats = 1, memory = false }
"""


def _cfg(text=TINY):
    import tomllib
    return parse_config(tomllib.loads(text), text)


def test_core_does_not_import_opencv_or_skimage():
    for path in (SRC / "core").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                assert not n.split(".")[0] in {"cv2", "skimage"}, f"{path.name} import {n}"


def test_dependency_direction_imaging_is_leaf():
    for path in (SRC / "imaging").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                pytest.fail(f"{path.name} import ngược lên gói cha")


def test_runner_outputs_and_reproducibility(tmp_path):
    out1 = run_experiment(_cfg(), tmp_path / "r1")
    out2 = run_experiment(_cfg(), tmp_path / "r2")
    for f in ["config.toml", "environment.json", "records.jsonl", "summary.csv", "summary.md", "extra.json",
              "figures/examples.png"]:
        assert (out1 / f).exists(), f

    def strip(path):
        rows = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
        return [{k: v for k, v in r.items() if not k.startswith(("time_", "mem_"))} for r in rows]

    assert strip(out1) == strip(out2)
    assert len(strip(out1)) == 2 * 2 * 2 + 2 * 2  # (2 cấu hình run a + 1 run b) x 4 mẫu


def test_config_validation():
    with pytest.raises(KeyError):
        _cfg(TINY.replace('"canny_opencv"', '"khong_co"'))
    with pytest.raises(ValueError):
        _cfg(TINY.replace('"timing"]', '"abc"]'))
    with pytest.raises(ValueError):
        _cfg(TINY.replace('label = "b"', 'label = "a"'))


def test_grid_expansion():
    cfg = _cfg()
    assert len(cfg.runs[0].expand()) == 2 and cfg.runs[0].varying == ["sigma"]


def test_dataset_save_load_split_and_stats(tmp_path):
    ds = Dataset.synthetic("shapes", 3, 64, seed=0, degradations=[{}, {"noise_sigma": 0.1}])
    assert len(ds) == 6 and ds.stats()["with_ground_truth"] == 6
    back = Dataset.load(ds.save(tmp_path / "ds"))
    assert len(back) == 6 and np.array_equal(back.samples[0].edges_gt, ds.samples[0].edges_gt)
    train, test = ds.split(0.34, seed=1)
    assert len(train) + len(test) == 6 and not {s.name for s in train} & {s.name for s in test}
    assert [len(b) for b in ds.batches(4)] == [4, 2]


def test_folder_loader_skips_corrupted(tmp_path):
    save_image(tmp_path / "ok.png", np.zeros((10, 10), np.uint8))
    (tmp_path / "hong.jpg").write_bytes(b"xx")
    ds = Dataset.from_folder(tmp_path)
    assert len(ds) == 1 and len(ds.skipped) == 1


def test_synthetic_is_deterministic():
    a = Dataset.synthetic("shapes", 2, 64, seed=5, degradations=[{"noise_sigma": 0.1}])
    b = Dataset.synthetic("shapes", 2, 64, seed=5, degradations=[{"noise_sigma": 0.1}])
    assert all(np.array_equal(x.image, y.image) for x, y in zip(a, b))


@pytest.mark.parametrize("backend", ["manual", "opencv"])
def test_p4_pipeline_on_clean_shapes(backend):
    ds = Dataset.synthetic("shapes", 6, 256, seed=11)
    correct = total = 0
    for s in ds:
        e = registry.run("canny_manual_fast", s.image, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
        seg = segment_by_edges(e, backend=backend)
        assert seg.count == len(s.classes_gt)
        for r in recognize_shapes(seg.labels, seg.count, backend=backend):
            g = int(np.bincount(s.labels_gt[seg.labels == r["label"]]).argmax())
            correct += s.classes_gt[g] == r["shape"]
            total += 1
    assert correct / total >= 0.95


def test_trace_boundary_and_dp_on_square():
    m = np.zeros((20, 20), bool)
    m[5:15, 4:12] = True
    b = trace_boundary(m)
    assert len(b) == 2 * (10 - 1) + 2 * (8 - 1)
    assert len(douglas_peucker(b, 1.0)) == 4
