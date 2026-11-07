"""Canny + phân đoạn (P4a): cạnh -> closing -> lấp lỗ -> gán nhãn -> lọc diện tích.

Giả thiết: đối tượng là vùng khép kín có biên tương phản với nền. Closing nối các đoạn cạnh bị NMS
làm đứt ở góc (docs/mathematics/03, LIMITATIONS). Giới hạn: hai vật chạm nhau bị gộp thành một vùng.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import morphology as morph


@dataclass
class SegmentationResult:
    labels: np.ndarray      # int32, 0 = nền
    count: int
    closed: np.ndarray      # cạnh sau closing
    filled: np.ndarray      # sau lấp lỗ


def segment_by_edges(edges: np.ndarray, close_size: int = 5, close_shape: str = "ellipse",
                     min_area: int = 50, backend: str = "manual") -> SegmentationResult:
    """``backend='manual'`` dùng core.morphology; ``'opencv'`` dùng cv2/SciPy (bản Optimized)."""
    edges = np.asarray(edges, dtype=bool)
    if backend == "manual":
        se = morph.structuring_element(close_shape, close_size)
        closed = morph.closing(edges, se)
        filled = morph.fill_holes(closed)
        labels, _ = morph.label_components(filled, 8)
    elif backend == "opencv":
        from .. import backends
        se = backends.structuring_element_opencv(close_shape, close_size)
        closed = backends.closing_opencv(edges, se)
        filled = backends.fill_holes_scipy(closed)
        labels, _ = backends.label_scipy(filled)
    else:
        raise ValueError("backend phải là 'manual' hoặc 'opencv'")
    labels, count = morph.remove_small(labels, min_area)
    # Bỏ vùng chạm viền ảnh chiếm gần cả ảnh (nền bị khép kín do cạnh viền)
    h, w = labels.shape
    for lab in range(1, count + 1):
        if (labels == lab).sum() > 0.5 * h * w:
            labels[labels == lab] = 0
    labels, count = morph.remove_small(labels, 1)
    return SegmentationResult(labels=labels, count=count, closed=closed, filled=filled)
