"""Chỉ số chất lượng ảnh ĐẦU VÀO: chỉ dùng mô tả mức suy giảm, KHÔNG dùng chấm bản đồ cạnh
(evaluation-rules.md mục 3). Ảnh float [0, 1]: PSNR = 10 log10(1 / MSE)."""

from __future__ import annotations

import numpy as np

from ..imaging.convert import to_float


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((to_float(a) - to_float(b)) ** 2))


def psnr(reference: np.ndarray, degraded: np.ndarray) -> float:
    err = mse(reference, degraded)
    return float("inf") if err == 0 else float(10 * np.log10(1.0 / err))
