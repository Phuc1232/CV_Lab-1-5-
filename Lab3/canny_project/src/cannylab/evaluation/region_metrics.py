"""Chỉ số vùng (P4a): IoU, Dice và ghép đối tượng một-một.

IoU = |A∩B| / |A∪B|; Dice = 2|A∩B| / (|A|+|B|) = 2 IoU / (1 + IoU) (hàm đơn điệu của nhau: chỉ báo cáo IoU).
Ghép đối tượng: thuật toán Hungarian trên ma trận IoU, cặp có IoU >= 0.5 mới tính là tìm thấy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 1.0


def iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Ma trận IoU (n_gt x n_pred) tính bằng bảng đếm chung (O(N + n_gt n_pred))."""
    n_g, n_p = int(gt_labels.max()), int(pred_labels.max())
    joint = np.bincount(gt_labels.ravel().astype(np.int64) * (n_p + 1) + pred_labels.ravel(),
                        minlength=(n_g + 1) * (n_p + 1)).reshape(n_g + 1, n_p + 1)
    inter = joint[1:, 1:].astype(np.float64)
    area_g = joint[1:, :].sum(1, keepdims=True)
    area_p = joint[:, 1:].sum(0, keepdims=True)
    union = area_g + area_p - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_objects(pred_labels: np.ndarray, gt_labels: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Ghép đối tượng; trả precision/recall/F1 mức đối tượng, IoU trung bình các cặp, sai số đếm, các cặp."""
    m = iou_matrix(pred_labels, gt_labels)
    n_g, n_p = m.shape
    pairs = []
    if n_g and n_p:
        rows, cols = linear_sum_assignment(-m)
        pairs = [(int(r) + 1, int(c) + 1, float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= iou_threshold]
    tp = len(pairs)
    precision = tp / n_p if n_p else (1.0 if n_g == 0 else 0.0)
    recall = tp / n_g if n_g else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"object_precision": precision, "object_recall": recall, "object_f1": f1,
            "mean_iou_matched": float(np.mean([p[2] for p in pairs])) if pairs else 0.0,
            "n_gt_objects": n_g, "n_pred_objects": n_p, "count_error": n_p - n_g, "pairs": pairs}
