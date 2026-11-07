"""Canny cho ảnh màu (câu E3). Toán học: docs/mathematics/07_canny_anh_mau_va_video.md.

Bốn phương pháp:
    gray         chuyển xám Y = 0.299R + 0.587G + 0.114B rồi Canny (mất cạnh đẳng sáng)
    or           Canny từng kênh rồi hợp (OR)
    max_channel  mỗi pixel lấy gradient của kênh có độ lớn lớn nhất (cách cv2.Canny xử lý ảnh 3 kênh;
                 với hồ sơ OPENCV trùng từng pixel với cv2.Canny(ảnh RGB))
    di_zenzo     ma trận cấu trúc J = sum_c grad(I_c) grad(I_c)^T; độ lớn sqrt(lambda_max),
                 hướng theta = 0.5 atan2(2 gxy, gxx - gyy). Với 1 kênh quy về Canny xám (L2).
"""

from __future__ import annotations

import numpy as np

from ..imaging.convert import rgb_to_gray
from ..imaging.validate import check_image, dtype_max
from .canny import SKIMAGE, CannyProfile, CannyResult, canny, canny_from_gradient, prepare, resolve_profile
from .gradient import sobel
from .hysteresis import hysteresis_bfs
from .thresholds import ThresholdContext

COLOR_METHODS = ("gray", "or", "max_channel", "di_zenzo")


def structure_tensor_gradient(gxs: list[np.ndarray], gys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Từ gradient từng kênh, trả (gx_eq, gy_eq, magnitude) với magnitude = sqrt(lambda_max).

    (gx_eq, gy_eq) = magnitude * (cos theta, sin theta) là "gradient tương đương" dùng cho NMS.
    """
    gxx = sum(g.astype(np.float64) ** 2 for g in gxs)
    gyy = sum(g.astype(np.float64) ** 2 for g in gys)
    gxy = sum(a.astype(np.float64) * b.astype(np.float64) for a, b in zip(gxs, gys))
    lam = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy ** 2))
    mag = np.sqrt(np.maximum(lam, 0.0))
    theta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
    return mag * np.cos(theta), mag * np.sin(theta), mag


def max_channel_gradient(gxs: list[np.ndarray], gys: list[np.ndarray], norm: str) -> tuple[np.ndarray, np.ndarray]:
    """Chọn kênh có độ lớn lớn nhất (L1: |gx|+|gy|; L2: gx^2+gy^2); hòa thì lấy kênh đầu (như OpenCV)."""
    gx = np.stack(gxs, -1)
    gy = np.stack(gys, -1)
    score = np.abs(gx) + np.abs(gy) if norm == "L1" else gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2
    best = np.argmax(score, axis=-1)[..., None]
    return np.take_along_axis(gx, best, -1)[..., 0], np.take_along_axis(gy, best, -1)[..., 0]


def canny_color(image: np.ndarray, method: str = "di_zenzo", sigma: float | None = None,
                low_threshold: float | None = None, high_threshold: float | None = None, *,
                profile: CannyProfile | str = SKIMAGE, norm: str | None = None,
                threshold_strategy: str = "fixed", strategy_params: dict | None = None,
                hysteresis=hysteresis_bfs) -> CannyResult:
    """Canny trên ảnh RGB (H, W, 3). Ngưỡng ``fixed`` theo đơn vị dtype đầu vào như :func:`canny`."""
    image = check_image(image)
    if image.ndim != 3:
        raise ValueError("canny_color cần ảnh RGB (H, W, 3); ảnh xám dùng canny()")
    if method not in COLOR_METHODS:
        raise ValueError(f"method phải thuộc {COLOR_METHODS}, nhận {method!r}")
    kwargs = dict(profile=profile, norm=norm, threshold_strategy=threshold_strategy,
                  strategy_params=strategy_params, hysteresis=hysteresis)
    if method == "gray":
        return canny(rgb_to_gray(image), sigma, low_threshold, high_threshold, **kwargs)
    if method == "or":
        results = [canny(np.ascontiguousarray(image[..., c]), sigma, low_threshold, high_threshold, **kwargs)
                   for c in range(3)]
        first = results[0]
        first.edges = results[0].edges | results[1].edges | results[2].edges
        first.extra["per_channel_edges"] = [r.edges for r in results]
        return first

    prof = resolve_profile(profile, norm)
    if method == "di_zenzo" and prof.nms == "quantized":
        raise ValueError("di_zenzo cho gradient thực nên cần NMS nội suy (hồ sơ 'skimage')")
    smoothed, gxs, gys, used_sigma = [], [], [], sigma
    for c in range(3):
        sm, used_sigma = prepare(np.ascontiguousarray(image[..., c]), prof, sigma)
        gx, gy = sobel(sm, mode=prof.sobel_mode)
        smoothed.append(sm)
        gxs.append(gx)
        gys.append(gy)
    input_scale = dtype_max(image) if prof.scale == "unit" else 1.0
    stack = np.stack(smoothed, -1)
    ctx = ThresholdContext(image=stack.mean(-1), sigma=used_sigma,
                           scale_max=255.0 if prof.scale == "uint8" else 1.0, raw=rgb_to_gray(image))
    common = dict(low_threshold=low_threshold, high_threshold=high_threshold, input_scale=input_scale,
                  threshold_strategy=threshold_strategy, strategy_params=strategy_params, context=ctx,
                  hysteresis=hysteresis, smoothed=stack, sigma=used_sigma)
    if method == "max_channel":
        gx, gy = max_channel_gradient(gxs, gys, prof.norm)
        return canny_from_gradient(gx, gy, prof, **common)
    gx, gy, mag = structure_tensor_gradient(gxs, gys)
    return canny_from_gradient(gx, gy, prof, magnitude=mag, **common)
