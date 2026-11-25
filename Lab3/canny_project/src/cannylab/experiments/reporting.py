"""Báo cáo thực nghiệm: summary.csv (trung bình, độ lệch chuẩn, n theo nhóm), summary.md, hình.

Nhóm = (run, params, degradation [, size]). Hình tự động:
  - line_<run>_<metric>.png : chỉ số theo tham số đang quét (mỗi đường một mức suy giảm)
  - examples.png            : bản đồ cạnh của mẫu đầu tiên cho từng cấu hình
  - timing.png              : thời gian theo số pixel (log–log) nếu có đo thời gian
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..evaluation.performance import fit_power_law  # noqa: E402

GROUP_KEYS = ("run", "params_key", "degradation", "size")
SKIP = {"run", "algorithm", "params", "params_key", "sample", "degradation", "family", "index", "seed",
        "source", "video_id", "frame", "size", "identical"}


def _group_key(r: dict) -> tuple:
    return (r["run"], r["params_key"], json.dumps(r.get("degradation", {}), sort_keys=True), r.get("size", ""))


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[_group_key(r)].append(r)
    rows = []
    for (run, pkey, deg, size), recs in groups.items():
        row = {"run": run, "params": pkey, "degradation": deg, "size": size, "n": len(recs)}
        numeric = sorted({k for r in recs for k, v in r.items()
                          if k not in SKIP and isinstance(v, (int, float)) and not isinstance(v, bool)})
        for k in numeric:
            vals = np.array([r[k] for r in recs if k in r], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{k}_std"] = float(finite.std()) if finite.size else float("nan")
        if any("identical" in r for r in recs):
            row["identical_all"] = all(r.get("identical", False) for r in recs)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _markdown(cfg, rows: list[dict], extra: dict, columns: list[str]) -> str:
    lines = [f"# {cfg.name}", "", f"Câu hỏi: **{cfg.question}**", "", cfg.description.strip(), "",
             "| run | params | suy giảm | n | " + " | ".join(columns) + " |",
             "|---|---|---|---|" + "---|" * len(columns)]
    for r in rows:
        cells = []
        for c in columns:
            m, s = r.get(f"{c}_mean"), r.get(f"{c}_std")
            cells.append("" if m is None else f"{m:.4g} ± {s:.2g}")
        params = json.loads(r["params"])
        p = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"| {r['run']} | {p} | {r['degradation']} {r['size']} | {r['n']} | " + " | ".join(cells) + " |")
    best = best_per_run(rows, columns[0] if columns else "f1") if cfg.figures.get("best", True) else []
    if best:
        lines += ["", f"## Tốt nhất theo lưới tham số (cực đại {columns[0]} trung bình, kiểu ODS)", "",
                  "| run | suy giảm | params tốt nhất | " + columns[0] + " |", "|---|---|---|---|"]
        lines += [f"| {b['run']} | {b['degradation']} {b['size']} | {b['params']} | {b['value']:.4f} |" for b in best]
    if extra.get("shapes"):
        lines += ["", "## Nhận dạng hình", "", "| cấu hình | accuracy | macro-F1 | n |", "|---|---|---|---|"]
        for k, rep in extra["shapes"].items():
            lines.append(f"| {k} | {rep['accuracy']:.4f} | {rep['macro_f1']:.4f} | {rep['n']} |")
    if extra.get("temporal"):
        lines += ["", "## Độ ổn định theo thời gian (F1 giữa khung liên tiếp)", "", "| cấu hình | F1 TB | std | số cặp |",
                  "|---|---|---|---|"]
        for k, t in extra["temporal"].items():
            lines.append(f"| {k} | {t['temporal_f1_mean']:.4f} | {t['temporal_f1_std']:.3f} | {t['pairs']} |")
    if extra.get("input_psnr"):
        lines += ["", "## PSNR ảnh đầu vào (mô tả mức suy giảm)", ""]
        lines += [f"- {k}: {v:.2f} dB" for k, v in extra["input_psnr"].items()]
    if extra.get("timing_fit"):
        lines += ["", "## Hệ số mũ thời gian t ∝ N^k", ""]
        lines += [f"- {k}: k = {v['exponent']:.3f}" for k, v in extra["timing_fit"].items()]
    return "\n".join(lines) + "\n"


def best_per_run(rows: list[dict], metric: str) -> list[dict]:
    """Với mỗi (run, suy giảm): cấu hình tham số cho ``metric`` trung bình cao nhất (chỉ khi run có lưới)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if f"{metric}_mean" in r:
            groups[(r["run"], r["degradation"], r["size"])].append(r)
    out = []
    for (run, deg, size), rs in groups.items():
        if len(rs) < 2:
            continue
        b = max(rs, key=lambda r: r[f"{metric}_mean"])
        out.append({"run": run, "degradation": deg, "size": size, "params": b["params"], "value": b[f"{metric}_mean"]})
    return out


def _line_plots(cfg, rows: list[dict], out: Path, metrics: list[str]) -> None:
    for run in cfg.runs:
        if len(run.varying) != 1:
            continue
        x_key = run.varying[0]
        mine = [r for r in rows if r["run"] == run.label]
        for metric in metrics:
            if not any(f"{metric}_mean" in r for r in mine):
                continue
            fig, ax = plt.subplots(figsize=(5, 3.4))
            by_deg: dict[str, list] = defaultdict(list)
            for r in mine:
                x = json.loads(r["params"])[x_key]
                if isinstance(x, (int, float)):
                    by_deg[f"{r['degradation']} {r['size']}".strip()].append((x, r[f"{metric}_mean"], r[f"{metric}_std"]))
            for deg, pts in by_deg.items():
                pts.sort()
                xs, ys, ss = map(np.array, zip(*pts))
                ax.errorbar(xs, ys, yerr=ss, marker="o", capsize=3, label=deg if len(by_deg) > 1 else None)
            ax.set_xlabel(x_key)
            ax.set_ylabel(metric)
            ax.set_title(f"{cfg.name}: {run.label}")
            ax.grid(alpha=0.3)
            if len(by_deg) > 1:
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / f"line_{run.label}_{metric}.png", dpi=130)
            plt.close(fig)


def _examples(examples: dict, image: np.ndarray, out: Path) -> None:
    if not examples:
        return
    n = len(examples) + 1
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(image, cmap=None if image.ndim == 3 else "gray", vmin=0, vmax=1 if image.dtype != np.uint8 else 255)
    axes[0].set_title("ảnh vào", fontsize=8)
    for ax, (name, e) in zip(axes[1:], examples.items()):
        ax.imshow(e, cmap="gray")
        ax.set_title(name, fontsize=7)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=120)
    plt.close(fig)


def _timing(records: list[dict], out: Path, extra: dict) -> None:
    timed = [r for r in records if "time_median_s" in r and "pixels" in r]
    if not timed:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    by_run: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in timed:
        by_run[r["run"]][r["pixels"]].append(r["time_median_s"])
    extra["timing_fit"] = {}
    for run, d in by_run.items():
        px = sorted(d)
        t = [float(np.median(d[p])) for p in px]
        ax.loglog(px, t, marker="o", label=run)
        if len(px) >= 2:
            extra["timing_fit"][run] = fit_power_law(px, t)
    ax.set_xlabel("số pixel N")
    ax.set_ylabel("thời gian (s, trung vị)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=130)
    plt.close(fig)


def write_reports(cfg, records: list[dict], extra: dict, out: Path, dataset, examples: dict) -> None:
    rows = summarize(records)
    write_csv(rows, out / "summary.csv")
    figs = out / "figures"
    metrics = list(cfg.figures.get("metrics", ["f1", "fom", "f1_1to1", "fragmentation"]))
    _line_plots(cfg, rows, figs, metrics)
    _examples(examples, dataset.samples[0].image, figs)
    _timing(records, figs, extra)
    (out / "extra.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=str))
    columns = list(cfg.figures.get("table", ["f1", "precision", "recall", "fom"]))
    (out / "summary.md").write_text(_markdown(cfg, rows, extra, columns), encoding="utf-8")
