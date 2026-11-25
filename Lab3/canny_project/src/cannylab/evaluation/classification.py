"""Chỉ số phân loại (P4b): accuracy, ma trận nhầm lẫn, macro-F1 (các lớp có thể lệch số lượng)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    classes = classes or sorted(set(y_true) | set(y_pred))
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, classes


def classification_report(y_true: list[str], y_pred: list[str], classes: list[str] | None = None) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải cùng độ dài")
    cm, classes = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(cm).astype(np.float64)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "confusion": cm.tolist(), "classes": classes, "n": int(cm.sum())}
