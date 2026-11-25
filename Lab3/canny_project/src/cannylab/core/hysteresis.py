"""Ngưỡng kép và hysteresis tự cài đặt (bước 4–5). Toán học: docs/mathematics/04_nguong_kep_va_hysteresis.md.

Hysteresis = giữ các thành phần liên thông (8 lân cận) của tập ứng viên có chứa ít nhất một điểm mạnh.
Bản Manual dùng BFS bằng ngăn xếp: O(N) thời gian, O(N) bộ nhớ trong trường hợp xấu nhất.
Bản Optimized (gán nhãn thành phần liên thông) nằm ở ``cannylab.backends.hysteresis_label``.
"""

from __future__ import annotations

import numpy as np

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBORS_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def double_threshold(values: np.ndarray, low: float, high: float, *, strict: bool) -> tuple[np.ndarray, np.ndarray]:
    """Trả ``(strong, candidate)``. strict=True: v > T (OpenCV); strict=False: v >= T (skimage).

    ``candidate`` gồm cả điểm mạnh và điểm yếu. Pixel có ``values == 0`` (bị NMS loại) không bao giờ là ứng viên.
    """
    if low > high:
        raise ValueError(f"low ({low}) phải <= high ({high})")
    nonzero = values > 0
    if strict:
        return nonzero & (values > high), nonzero & (values > low)
    return nonzero & (values >= high), nonzero & (values >= low)


def reachable(seeds: np.ndarray, allowed: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Tập pixel trong ``allowed`` đến được từ ``seeds`` bằng đường đi trong ``allowed`` (BFS/DFS bằng ngăn xếp)."""
    if seeds.shape != allowed.shape:
        raise ValueError("seeds và allowed phải cùng kích thước")
    offsets = NEIGHBORS_8 if connectivity == 8 else NEIGHBORS_4 if connectivity == 4 else None
    if offsets is None:
        raise ValueError("connectivity phải là 4 hoặc 8")
    h, w = allowed.shape
    allowed_flat = allowed.ravel()
    out = np.zeros(h * w, dtype=bool)
    start = np.flatnonzero(seeds.ravel() & allowed_flat)
    out[start] = True
    stack = start.tolist()
    while stack:
        idx = stack.pop()
        y, x = divmod(idx, w)
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                nidx = ny * w + nx
                if allowed_flat[nidx] and not out[nidx]:
                    out[nidx] = True
                    stack.append(nidx)
    return out.reshape(h, w)


def hysteresis_bfs(strong: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Hysteresis bản Manual: lan từ điểm mạnh qua các ứng viên 8 lân cận."""
    return reachable(strong, candidate | strong, connectivity=8)
