"""Hình thái học nhị phân và gán nhãn thành phần liên thông tự cài đặt (phục vụ P4a).

Định nghĩa theo OpenCV (phần tử cấu trúc SE có tâm ở giữa):
    dilate(A)[p] = OR_{s in SE} A[p + s - c]      (ngoài ảnh coi là False)
    erode(A)[p]  = AND_{s in SE} A[p + s - c]     (ngoài ảnh coi là True, nên viền không bị ăn mòn)
    closing = erode(dilate(A)),  opening = dilate(erode(A))
Độ phức tạp O(N |SE|).
"""

from __future__ import annotations

import numpy as np

from .hysteresis import NEIGHBORS_4, NEIGHBORS_8, reachable

SHAPES = ("rect", "cross", "ellipse")


def structuring_element(shape: str, size: int) -> np.ndarray:
    """Phần tử cấu trúc kích thước size x size, cùng công thức với cv2.getStructuringElement."""
    if size < 1:
        raise ValueError("size phải >= 1")
    if shape not in SHAPES:
        raise ValueError(f"shape phải thuộc {SHAPES}")
    se = np.zeros((size, size), dtype=bool)
    r = c = size // 2
    if shape == "rect":
        se[:] = True
    elif shape == "cross":
        se[r, :] = True
        se[:, c] = True
    else:
        inv_r2 = 1.0 / (r * r) if r else 0.0
        for i in range(size):
            dy = i - r
            if abs(dy) <= r:
                dx = int(np.rint(c * np.sqrt((r * r - dy * dy) * inv_r2)))
                se[i, max(c - dx, 0):min(c + dx + 1, size)] = True
    return se


def _apply(mask: np.ndarray, se: np.ndarray, op: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    ry, rx = se.shape[0] // 2, se.shape[1] // 2
    fill = op == "erode"
    padded = np.pad(mask, ((ry, se.shape[0] - 1 - ry), (rx, se.shape[1] - 1 - rx)), constant_values=fill)
    h, w = mask.shape
    out = np.full((h, w), fill, dtype=bool)
    for dy, dx in zip(*np.nonzero(se)):
        window = padded[dy:dy + h, dx:dx + w]
        if op == "erode":
            out &= window
        else:
            out |= window
    return out


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return _apply(mask, se, "dilate")


def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return _apply(mask, se, "erode")


def closing(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Đóng: lấp khe hở nhỏ hơn SE, nối đoạn cạnh bị đứt (ví dụ NMS làm đứt ở góc)."""
    return erode(dilate(mask, se), se)


def opening(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return dilate(erode(mask, se), se)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Lấp lỗ: nền nào không nối được ra viền (4 lân cận) thì là lỗ. Tiền cảnh 8 lân cận ↔ nền 4 lân cận."""
    mask = np.asarray(mask, dtype=bool)
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    outside = reachable(border & ~mask, ~mask, connectivity=4)
    return mask | ~outside


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Gán nhãn thành phần liên thông bằng BFS. Nhãn 1..n theo thứ tự quét raster; 0 là nền."""
    mask = np.asarray(mask, dtype=bool)
    offsets = NEIGHBORS_8 if connectivity == 8 else NEIGHBORS_4
    h, w = mask.shape
    flat = mask.ravel()
    labels = np.zeros(h * w, dtype=np.int32)
    current = 0
    for start in np.flatnonzero(flat):
        if labels[start]:
            continue
        current += 1
        labels[start] = current
        stack = [int(start)]
        while stack:
            idx = stack.pop()
            y, x = divmod(idx, w)
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nidx = ny * w + nx
                    if flat[nidx] and not labels[nidx]:
                        labels[nidx] = current
                        stack.append(nidx)
    return labels.reshape(h, w), current


def remove_small(labels: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Bỏ vùng có diện tích < min_area và đánh lại nhãn liên tục 1..k."""
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    new_ids = np.zeros(areas.size, dtype=np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1)
    return new_ids[labels], int(keep.sum())
