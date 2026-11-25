"""Chỉ số chất lượng cạnh có dung sai khoảng cách.

    Precision_tau = |{p in P : d(p, G) <= tau}| / |P|
    Recall_tau    = |{g in G : d(g, P) <= tau}| / |G|          (khớp nhiều-một, 'many')
    one-to-one    : TP = cỡ cặp ghép cực đại trong đồ thị hai phía {(p, g): |p - g| <= tau}
                    (Hopcroft–Karp, scipy.sparse.csgraph); phạt cạnh dày/kép, giống tinh thần BSDS
    FOM (Pratt)   = 1/max(|P|,|G|) * sum_p 1 / (1 + alpha d(p,G)^2), alpha = 1/9
    Chamfer       = trung bình hai chiều của d;  HD95 = phân vị 95 của mọi khoảng cách hai chiều
Ca biên (evaluation-rules.md mục 5): P = G = rỗng -> P = R = F1 = 1; P rỗng -> P = 1, R = 0;
G rỗng -> P = 0, R = 1. d tính bằng biến đổi khoảng cách Euclid (O(N)).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

FOM_ALPHA = 1.0 / 9.0


def distance_to(mask: np.ndarray) -> np.ndarray:
    """Khoảng cách Euclid từ mỗi pixel tới pixel True gần nhất của ``mask`` (inf nếu mask rỗng)."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndi.distance_transform_edt(~mask)


def _prf(tp_p: int, n_p: int, tp_g: int, n_g: int) -> tuple[float, float, float]:
    if n_p == 0 and n_g == 0:
        return 1.0, 1.0, 1.0
    precision = tp_p / n_p if n_p else 1.0
    recall = tp_g / n_g if n_g else 1.0
    if n_g == 0:
        precision = 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def one_to_one_matches(pred: np.ndarray, gt: np.ndarray, tolerance: float) -> int:
    """Số cặp (p, g) ghép một-một tối đa với |p - g| <= tolerance."""
    py, px = np.nonzero(pred)
    gy, gx = np.nonzero(gt)
    if py.size == 0 or gy.size == 0:
        return 0
    h, w = gt.shape
    gt_index = -np.ones(gt.shape, dtype=np.int64)
    gt_index[gy, gx] = np.arange(gy.size)
    r = int(np.floor(tolerance))
    rows, cols = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > tolerance * tolerance:
                continue
            ny, nx = py + dy, px + dx
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            idx = np.where(ok, gt_index[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
            hit = idx >= 0
            rows.append(np.nonzero(hit)[0])
            cols.append(idx[hit])
    rows_a, cols_a = np.concatenate(rows), np.concatenate(cols)
    graph = csr_matrix((np.ones(rows_a.size, dtype=np.int8), (rows_a, cols_a)), shape=(py.size, gy.size))
    match = maximum_bipartite_matching(graph, perm_type="column")
    return int((match >= 0).sum())


def edge_metrics(pred: np.ndarray, gt: np.ndarray, tolerance: float = 1.5) -> dict[str, float]:
    """Tất cả chỉ số cạnh có ground truth. ``tolerance`` tính bằng pixel."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} và gt {gt.shape} phải cùng kích thước")
    n_p, n_g = int(pred.sum()), int(gt.sum())
    d_to_gt = distance_to(gt)
    d_to_pred = distance_to(pred)
    dp = d_to_gt[pred]
    dg = d_to_pred[gt]
    p_many, r_many, f_many = _prf(int((dp <= tolerance).sum()), n_p, int((dg <= tolerance).sum()), n_g)
    tp = one_to_one_matches(pred, gt, tolerance)
    p_one, r_one, f_one = _prf(tp, n_p, tp, n_g)
    out = {"precision": p_many, "recall": r_many, "f1": f_many,
           "precision_1to1": p_one, "recall_1to1": r_one, "f1_1to1": f_one,
           "n_pred": n_p, "n_gt": n_g, "tolerance": tolerance}
    if n_p and n_g:
        out["fom"] = float(np.sum(1.0 / (1.0 + FOM_ALPHA * dp ** 2)) / max(n_p, n_g))
        out["chamfer"] = float(0.5 * (dp.mean() + dg.mean()))
        out["hd95"] = float(np.percentile(np.concatenate([dp, dg]), 95))
    else:
        out["fom"] = 1.0 if n_p == n_g == 0 else 0.0
        out["chamfer"] = 0.0 if n_p == n_g == 0 else float("inf")
        out["hd95"] = 0.0 if n_p == n_g == 0 else float("inf")
    return out


def continuity(edges: np.ndarray) -> dict[str, float]:
    """Tính liên tục (không cần ground truth): số thành phần 8-liên thông, độ dài trung bình/trung vị,
    tỉ lệ phân mảnh = số thành phần trên 100 px cạnh, tỉ lệ pixel cạnh (chỉ mô tả)."""
    edges = np.asarray(edges, dtype=bool)
    labels, n = ndi.label(edges, np.ones((3, 3), bool))
    total = int(edges.sum())
    sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0)
    return {"components": int(n), "mean_length": float(sizes.mean()) if n else 0.0,
            "median_length": float(np.median(sizes)) if n else 0.0,
            "fragmentation": float(n / (total / 100.0)) if total else 0.0,
            "edge_ratio": float(total / edges.size)}


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Độ khớp pixel-chính xác giữa hai bản đồ cạnh (kiểm chứng Manual vs thư viện)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return {"iou_pixel": float((a & b).sum() / union) if union else 1.0, "n_diff": int((a ^ b).sum()),
            "identical": bool(np.array_equal(a, b))}
