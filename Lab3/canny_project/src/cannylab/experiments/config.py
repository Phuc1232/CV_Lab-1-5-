"""Đọc và kiểm tra cấu hình thực nghiệm (experiment-rules.md mục 1–2).

Tham số có giá trị là list được mở rộng thành tích Descartes (lưới). Ví dụ
``params = {sigma = [1, 2], low_threshold = 0.05}`` sinh 2 cấu hình.
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .. import registry

KNOWN_METRICS = {"edge", "continuity", "agreement", "segmentation", "shapes", "temporal", "timing", "psnr"}
DATASET_SOURCES = {"synthetic", "skimage", "folder"}


@dataclass
class RunSpec:
    label: str
    algorithm: str
    params: dict
    reference: dict | None = None
    temporal_sigma: float = 0.0

    def expand(self) -> list[dict]:
        keys = [k for k, v in self.params.items() if isinstance(v, list)]
        fixed = {k: v for k, v in self.params.items() if k not in keys}
        if not keys:
            return [dict(fixed)]
        return [{**fixed, **dict(zip(keys, combo))} for combo in itertools.product(*(self.params[k] for k in keys))]

    @property
    def varying(self) -> list[str]:
        return [k for k, v in self.params.items() if isinstance(v, list)]


@dataclass
class ExperimentConfig:
    name: str
    question: str
    seed: int
    dataset: dict
    runs: list[RunSpec]
    evaluation: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    description: str = ""
    source_text: str = ""
    base_dir: Path = Path(".")

    @property
    def metrics(self) -> list[str]:
        return list(self.evaluation.get("metrics", ["edge", "continuity"]))


def parse_config(data: dict, source_text: str = "", base_dir: Path = Path(".")) -> ExperimentConfig:
    missing = {"name", "question", "dataset", "runs"} - set(data)
    if missing:
        raise ValueError(f"Cấu hình thiếu khóa bắt buộc: {sorted(missing)}")
    ds = data["dataset"]
    if ds.get("source", "synthetic") not in DATASET_SOURCES:
        raise ValueError(f"dataset.source phải thuộc {sorted(DATASET_SOURCES)}")
    runs = []
    labels = set()
    for r in data["runs"]:
        if "algorithm" not in r or "label" not in r:
            raise ValueError("Mỗi [[runs]] cần 'label' và 'algorithm'")
        registry.get(r["algorithm"])  # KeyError có gợi ý nếu sai tên
        if r.get("reference"):
            registry.get(r["reference"]["algorithm"])
        if r["label"] in labels:
            raise ValueError(f"label trùng: {r['label']}")
        labels.add(r["label"])
        runs.append(RunSpec(r["label"], r["algorithm"], dict(r.get("params", {})), r.get("reference"),
                            float(r.get("temporal_sigma", 0.0))))
    ev = dict(data.get("evaluation", {}))
    unknown = set(ev.get("metrics", [])) - KNOWN_METRICS
    if unknown:
        raise ValueError(f"metrics không hỗ trợ: {sorted(unknown)}; chọn {sorted(KNOWN_METRICS)}")
    return ExperimentConfig(name=data["name"], question=data["question"], seed=int(data.get("seed", 0)),
                            dataset=ds, runs=runs, evaluation=ev, figures=dict(data.get("figures", {})),
                            description=data.get("description", ""), source_text=source_text, base_dir=base_dir)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_config(tomllib.loads(text), text, path.parent)
