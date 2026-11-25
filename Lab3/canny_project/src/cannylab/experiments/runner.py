"""Chạy thực nghiệm: dataset × run × lưới tham số -> chỉ số -> thư mục kết quả tái lập được.

Thư mục kết quả: config.toml (bản sao), environment.json, records.jsonl, summary.csv, extra.json, figures/.
Khóa bắt đầu bằng 'time_' / 'mem_' là phần duy nhất được phép khác giữa hai lần chạy cùng cấu hình.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .. import registry
from ..analysis.segmentation import segment_by_edges
from ..analysis.shapes import recognize_shapes
from ..datasets import Dataset, Sample
from ..evaluation.edge_metrics import agreement, continuity, edge_metrics
from ..evaluation.classification import classification_report
from ..evaluation.image_quality import psnr
from ..evaluation.performance import measure
from ..evaluation.region_metrics import match_objects
from ..utils import environment_info, project_root, timestamp
from .config import ExperimentConfig

log = logging.getLogger(__name__)


def build_dataset(cfg: ExperimentConfig) -> Dataset:
    ds = cfg.dataset
    source = ds.get("source", "synthetic")
    if source == "skimage":
        return Dataset.from_skimage(ds.get("names", ["camera"]), ds.get("mode", "gray"))
    if source == "folder":
        return Dataset.from_folder(cfg.base_dir / ds["path"], ds.get("mode", "gray"))
    sizes = ds.get("sizes", [ds.get("size", 256)])
    samples = []
    for size in sizes:
        part = Dataset.synthetic(ds.get("family", "shapes"), int(ds.get("count", 4)), int(size), cfg.seed,
                                 ds.get("degradations"), **ds.get("family_kwargs", {}))
        for s in part:
            s.meta["size"] = int(size)
        samples.extend(part.samples)
    return Dataset(f"synthetic_{ds.get('family', 'shapes')}", samples)


def _temporal_stack(samples: list[Sample], sigma: float) -> dict[str, np.ndarray]:
    """Làm trơn theo thời gian (Gaussian 1D trục khung hình) cho từng video; trả {tên mẫu: ảnh mới}."""
    groups: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.meta.get("video_id", -1)].append(s)
    out = {}
    for frames in groups.values():
        frames.sort(key=lambda s: s.meta.get("frame", 0))
        stack = np.stack([f.image for f in frames])
        smooth = ndi.gaussian_filter1d(stack, sigma, axis=0, mode="nearest")
        out.update({f.name: smooth[i] for i, f in enumerate(frames)})
    return out


def reference_params(reference: dict, params: dict) -> dict:
    """Tham số cho thuật toán tham chiếu = tham số của run, bỏ các khóa trong ``drop``, ghi đè bằng ``params``."""
    drop = set(reference.get("drop", []))
    return {**{k: v for k, v in params.items() if k not in drop}, **reference.get("params", {})}


def _scalar_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool)) or k == "degradation"}


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path | None = None, dataset: Dataset | None = None) -> Path:
    np.random.seed(cfg.seed)
    dataset = dataset or build_dataset(cfg)
    if len(dataset) == 0:
        raise ValueError("Dataset rỗng")
    out = Path(out_dir) if out_dir else project_root() / "results" / f"{cfg.name}_{timestamp()}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(cfg.source_text or "", encoding="utf-8")
    ev = cfg.evaluation
    timing_cfg = dict(ev.get("timing", {}))
    if "opencv_threads" in timing_cfg:
        # Đặt trước khi ghi environment.json để bản ghi đúng. Lưu ý: OpenCV build với GCD (macOS) bỏ qua
        # setNumThreads(1); chỉ 0 mới chạy tuần tự (đã kiểm tra với OpenCV 5.0: get trả 8 sau set(1), 1 sau set(0)).
        import cv2
        cv2.setNumThreads(int(timing_cfg["opencv_threads"]))
    (out / "environment.json").write_text(json.dumps(environment_info(), indent=2))
    (out / "dataset.json").write_text(json.dumps({"stats": dataset.stats(), "skipped": dataset.skipped},
                                                 ensure_ascii=False, indent=2, default=str))

    metrics = set(cfg.metrics)
    ev = cfg.evaluation
    tol = float(ev.get("tolerance", 1.5))
    seg_cfg = dict(ev.get("segmentation", {}))
    records: list[dict] = []
    extra: dict = {"shapes": {}, "temporal": {}}
    examples: dict[str, np.ndarray] = {}

    for run in cfg.runs:
        spec = registry.get(run.algorithm)
        images = _temporal_stack(dataset.samples, run.temporal_sigma) if run.temporal_sigma > 0 else {}
        for params in run.expand():
            key = json.dumps(params, sort_keys=True)
            y_true: dict[str, list] = defaultdict(list)
            y_pred: dict[str, list] = defaultdict(list)
            per_video: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
            for sample in dataset:
                image = images.get(sample.name, sample.image)
                if image.ndim == 3 and not spec.color:
                    from ..imaging.convert import rgb_to_gray
                    image = rgb_to_gray(image)
                edges = registry.run(run.algorithm, image, **params)
                rec = {"run": run.label, "algorithm": run.algorithm, "params": params, "params_key": key,
                       "sample": sample.name, **_scalar_meta(sample.meta)}
                if "edge" in metrics and sample.edges_gt is not None:
                    rec.update(edge_metrics(edges, sample.edges_gt, tol))
                if "continuity" in metrics:
                    rec.update(continuity(edges))
                if "agreement" in metrics and run.reference:
                    ref = registry.run(run.reference["algorithm"], image, **reference_params(run.reference, params))
                    rec.update(agreement(edges, ref))
                if ("segmentation" in metrics or "shapes" in metrics) and sample.labels_gt is not None:
                    seg = segment_by_edges(edges, **seg_cfg)
                    obj = match_objects(seg.labels, sample.labels_gt)
                    pairs = obj.pop("pairs")
                    rec.update(obj)
                    if "shapes" in metrics:
                        shapes = {r["label"]: r["shape"] for r in
                                  recognize_shapes(seg.labels, seg.count, backend=seg_cfg.get("backend", "manual"))}
                        dkey = json.dumps(sample.meta.get("degradation", {}), sort_keys=True)
                        matched = {pp[0] for pp in pairs}
                        for g, p, _ in pairs:
                            y_true[dkey].append(sample.classes_gt[g])
                            y_pred[dkey].append(shapes.get(p, "unknown"))
                        for g in sample.classes_gt:  # vật không tìm thấy tính là sai
                            if g not in matched:
                                y_true[dkey].append(sample.classes_gt[g])
                                y_pred[dkey].append("missed")
                if "timing" in metrics:
                    rec.update(measure(lambda: registry.run(run.algorithm, image, **params),
                                       repeats=int(timing_cfg.get("repeats", 3)),
                                       memory=bool(timing_cfg.get("memory", True))))
                    rec["pixels"] = int(image.shape[0] * image.shape[1])
                if "temporal" in metrics and "video_id" in sample.meta:
                    per_video[sample.meta["video_id"]].append((sample.meta["frame"], edges))
                records.append(rec)
                if sample is dataset.samples[0] and len(examples) < 16:
                    examples[f"{run.label} {_short(params)}"] = edges
            for dkey in y_true:
                extra["shapes"][f"{run.label} {_label(params)} @ {dkey}"] = classification_report(y_true[dkey], y_pred[dkey])
            if per_video:
                scores = []
                for frames in per_video.values():
                    frames.sort(key=lambda x: x[0])
                    scores += [edge_metrics(b, a, tol)["f1"] for (_, a), (_, b) in zip(frames, frames[1:])]
                extra["temporal"][f"{run.label} {_label(params)}"] = {"temporal_f1_mean": float(np.mean(scores)),
                                                          "temporal_f1_std": float(np.std(scores)),
                                                          "pairs": len(scores)}
            log.info("xong %s %s", run.label, key)

    if "psnr" in metrics:
        extra["input_psnr"] = dataset_psnr(cfg)
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
    (out / "extra.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=_json_default))
    from .reporting import write_reports
    write_reports(cfg, records, extra, out, dataset, examples)
    return out


def dataset_psnr(cfg: ExperimentConfig) -> dict:
    """PSNR trung bình của ảnh suy giảm so với ảnh sạch, theo từng bộ suy giảm (mô tả đầu vào)."""
    ds = cfg.dataset
    if ds.get("source", "synthetic") != "synthetic" or not ds.get("degradations"):
        return {}
    clean = Dataset.synthetic(ds.get("family", "shapes"), int(ds.get("count", 4)), int(ds.get("size", 256)),
                              cfg.seed, [{}], **ds.get("family_kwargs", {}))
    degraded = Dataset.synthetic(ds.get("family", "shapes"), int(ds.get("count", 4)), int(ds.get("size", 256)),
                                 cfg.seed, ds["degradations"], **ds.get("family_kwargs", {}))
    by_deg: dict[str, list[float]] = defaultdict(list)
    for s in degraded:
        ref = clean.samples[s.meta["index"]].image
        by_deg[json.dumps(s.meta["degradation"], sort_keys=True)].append(psnr(ref, s.image))
    return {k: float(np.mean([v for v in vals if np.isfinite(v)])) if any(np.isfinite(vals)) else float("inf")
            for k, vals in by_deg.items()}


def _label(params: dict) -> str:
    """Nhãn đầy đủ (không cắt) để làm khóa: tránh hai cấu hình trùng khóa."""
    return ",".join(f"{k}={v}" for k, v in params.items() if k != "profile")


def _short(params: dict) -> str:
    """Nhãn rút gọn CHỈ dùng làm tiêu đề hình (không dùng làm khóa)."""
    return ",".join(f"{k}={v}" for k, v in params.items() if k not in ("profile",))[:40]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
