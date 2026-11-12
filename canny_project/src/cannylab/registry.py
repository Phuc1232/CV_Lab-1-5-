"""Registry thuật toán: tên -> hàm ``fn(image, **params) -> bool[H, W]``.

Experiment Engine chỉ gọi thuật toán qua tên, nên thêm thuật toán mới chỉ cần thêm một hàm
``@register`` ở đây (hoặc ở module khác rồi import), không phải sửa runner.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import backends
from .core import baselines, canny
from .core.color import canny_color
from .core.denoise import bilateral_filter, median_filter
from .core.hysteresis import hysteresis_bfs

EdgeFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    fn: EdgeFn
    family: str          # canny | baseline | color
    implementation: str  # manual | opencv | skimage | scipy | hybrid
    description: str
    color: bool = False  # nhận ảnh RGB


REGISTRY: dict[str, AlgorithmSpec] = {}


def register(name: str, family: str, implementation: str, description: str, color: bool = False):
    def decorator(fn: EdgeFn) -> EdgeFn:
        if name in REGISTRY:
            raise ValueError(f"Thuật toán {name!r} đã được đăng ký")
        REGISTRY[name] = AlgorithmSpec(name, fn, family, implementation, description, color)
        return fn
    return decorator


def get(name: str) -> AlgorithmSpec:
    if name not in REGISTRY:
        hint = difflib.get_close_matches(name, REGISTRY, n=3)
        raise KeyError(f"Không có thuật toán {name!r}. Gợi ý: {hint or sorted(REGISTRY)}")
    return REGISTRY[name]


def run(name: str, image: np.ndarray, **params) -> np.ndarray:
    edges = get(name).fn(image, **params)
    if edges.dtype != bool:
        raise TypeError(f"{name} phải trả mảng bool, nhận {edges.dtype}")
    return edges


# ------------------------------------------------------------------ Canny xám
_CANNY_KEYS = {"sigma", "low_threshold", "high_threshold", "profile", "norm", "threshold_strategy"}
_PREFILTERS = {"median": median_filter, "bilateral": bilateral_filter,
               "median_opencv": backends.median_opencv, "bilateral_opencv": backends.bilateral_opencv}


def _split_params(params: dict) -> tuple[dict, dict, dict]:
    """Tách tham số Canny / tham số chiến lược ngưỡng / tham số tiền lọc (tiền tố 'prefilter_').

    ``low_ratio`` (nếu có): low_threshold = low_ratio * high_threshold (quét T_h với tỉ lệ cố định).
    """
    params = dict(params)
    if "low_ratio" in params:
        ratio = params.pop("low_ratio")
        if "high_threshold" not in params:
            raise ValueError("low_ratio cần high_threshold")
        params["low_threshold"] = ratio * params["high_threshold"]
    canny_kw = {k: v for k, v in params.items() if k in _CANNY_KEYS}
    prefilter_kw = {k[len("prefilter_"):]: v for k, v in params.items() if k.startswith("prefilter_")}
    strategy_kw = {k: v for k, v in params.items()
                   if k not in _CANNY_KEYS and not k.startswith("prefilter_") and k != "prefilter"}
    return canny_kw, strategy_kw, prefilter_kw


def _canny_manual(image: np.ndarray, hysteresis, **params) -> np.ndarray:
    canny_kw, strategy_kw, prefilter_kw = _split_params(params)
    prefilter = params.get("prefilter")
    if prefilter:
        if prefilter not in _PREFILTERS:
            raise ValueError(f"prefilter phải thuộc {sorted(_PREFILTERS)}")
        image = _PREFILTERS[prefilter](image, **prefilter_kw)
    return canny(image, strategy_params=strategy_kw or None, hysteresis=hysteresis, **canny_kw).edges


@register("canny_manual", "canny", "manual", "Canny tự cài NumPy (hồ sơ opencv|skimage), hysteresis BFS")
def canny_manual(image: np.ndarray, **params) -> np.ndarray:
    return _canny_manual(image, hysteresis_bfs, **params)


@register("canny_manual_fast", "canny", "hybrid", "Canny tự cài, hysteresis bằng scipy.ndimage.label")
def canny_manual_fast(image: np.ndarray, **params) -> np.ndarray:
    return _canny_manual(image, backends.hysteresis_label, **params)


register("canny_opencv", "canny", "opencv", "cv2.Canny (sigma>0: GaussianBlur trước)")(backends.canny_opencv)
register("canny_skimage", "canny", "skimage", "skimage.feature.canny")(backends.canny_skimage)


# ------------------------------------------------------------------ Ảnh màu
@register("canny_color_manual", "color", "manual", "Canny màu tự cài: gray|or|max_channel|di_zenzo", color=True)
def canny_color_manual(image: np.ndarray, method: str = "di_zenzo", **params) -> np.ndarray:
    canny_kw, strategy_kw, _ = _split_params(params)
    return canny_color(image, method, strategy_params=strategy_kw or None,
                       hysteresis=backends.hysteresis_label, **canny_kw).edges


@register("canny_opencv_color", "color", "opencv", "cv2.Canny trên ảnh 3 kênh", color=True)
def canny_opencv_color(image: np.ndarray, **params) -> np.ndarray:
    canny_kw, rest, _ = _split_params(params)
    canny_kw.pop("profile", None)
    return backends.canny_opencv_color(image, **canny_kw, **rest)


# ------------------------------------------------------------------ Đối chứng (T1b)
register("sobel_threshold_manual", "baseline", "manual", "Sobel L2 + một ngưỡng (không NMS)")(baselines.sobel_threshold)
register("sobel_threshold_opencv", "baseline", "opencv", "cv2.Sobel L2 + một ngưỡng")(backends.sobel_threshold_opencv)
register("log_manual", "baseline", "manual", "LoG + điểm qua 0 (Marr–Hildreth), tự cài")(baselines.log_zero_crossing)
register("log_scipy", "baseline", "scipy", "scipy gaussian_laplace + điểm qua 0")(backends.log_scipy)


def available(family: str | None = None) -> list[AlgorithmSpec]:
    return [s for s in REGISTRY.values() if family is None or s.family == family]
