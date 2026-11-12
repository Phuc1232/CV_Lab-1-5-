"""Đo hiệu năng (experiment-rules.md mục 6).

Thời gian: chạy khởi động rồi lấy trung vị và IQR của ``repeats`` lần (time.perf_counter).
Bộ nhớ: đỉnh tracemalloc (NumPy có báo cấp phát cho tracemalloc; bộ nhớ C++ của OpenCV KHÔNG được thấy)
và chênh lệch RSS của tiến trình (psutil, gồm cả OpenCV nhưng thô vì allocator giữ lại bộ nhớ).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable

import numpy as np
import psutil


def measure(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1, memory: bool = True) -> dict[str, float]:
    if repeats < 1:
        raise ValueError("repeats phải >= 1")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    out = {"time_median_s": float(med), "time_iqr_s": float(q3 - q1), "time_min_s": float(min(times)),
           "repeats": repeats}
    if memory:
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["mem_peak_tracemalloc_mb"] = peak / 2 ** 20
        out["mem_rss_delta_mb"] = max(0, proc.memory_info().rss - rss0) / 2 ** 20
    return out


def fit_power_law(sizes: list[float], times: list[float]) -> dict[str, float]:
    """Khớp t = c N^k trên log-log (bình phương tối thiểu). k ≈ 1 nghĩa là tuyến tính theo số pixel."""
    x, y = np.log(np.asarray(sizes, float)), np.log(np.asarray(times, float))
    k, logc = np.polyfit(x, y, 1)
    return {"exponent": float(k), "coefficient": float(np.exp(logc))}
