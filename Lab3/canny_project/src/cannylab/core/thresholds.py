"""Chiến lược chọn ngưỡng tự động (câu E2). Toán học: docs/mathematics/04_nguong_kep_va_hysteresis.md.

Mỗi chiến lược nhận (magnitude, nms_values, context, profile, **params) và trả (low, high) theo
CÙNG THANG với ``magnitude`` của hồ sơ.

    quantile : T = phân vị q của toàn bộ độ lớn (giống skimage use_quantiles)
    otsu     : T_h = Otsu trên độ lớn tại cực đại địa phương; T_l = ratio * T_h
    noise    : T_h = k * sigma_g (mặc định k = sqrt(2 ln N), ngưỡng phổ quát), T_l = 2 sigma_g; sigma_g = sigma_n * sqrt(sum((Sobel_x ⊛ G_sigma)^2)) là độ lệch chuẩn
               CHÍNH XÁC của một thành phần gradient khi nhiễu trắng; sigma_n ước lượng bằng Immerkær (1996)
               có lặp loại pixel cạnh
    median   : heuristic phổ biến (không có cơ sở): T = (1 ∓ spread) * median(I), I là cường độ ảnh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..imaging.validate import dtype_max
from .convolution import correlate2d
from .gaussian import gaussian_kernel1d

if TYPE_CHECKING:  # tránh import vòng
    from .canny import CannyProfile

IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


@dataclass
class ThresholdContext:
    image: np.ndarray | None      # ảnh đã đưa về thang hồ sơ và làm trơn
    sigma: float | None
    scale_max: float              # 255 (hồ sơ uint8) hoặc 1 (hồ sơ unit)
    raw: np.ndarray | None = None  # ảnh gốc trước làm trơn (để ước lượng nhiễu)


def estimate_noise_sigma(image: np.ndarray, iterations: int = 3, edge_factor: float = 4.0) -> float:
    """Ước lượng độ lệch chuẩn nhiễu Gauss (Immerkær 1996) có loại pixel cạnh.

    sigma = sqrt(pi/2) * mean_{p in mask} |(I ⊛ N)(p)| / 6, N = [[1,-2,1],[-2,4,-2],[1,-2,1]].
    N triệt tiêu thành phần bậc 0 và 1 nhưng KHÔNG triệt tiêu cạnh bậc thang, nên ảnh nhiều cạnh / nhiễu
    nhỏ bị ước lượng thừa (đo được: 0.018 thay vì 0.01). Cách khắc phục: lặp, loại pixel có độ lớn Sobel
    > edge_factor * sqrt(12) * sigma_hiện_tại (nhiễu thuần vượt mức này với xác suất e^{-8} theo Rayleigh).
    Đơn vị: cùng thang với ``image``.
    """
    img = image.astype(np.float64)
    response = np.abs(correlate2d(img, IMMERKAER_KERNEL, mode="reflect")[1:-1, 1:-1])
    gx = correlate2d(img, np.outer([1, 2, 1], [-1, 0, 1]).astype(np.float64), mode="reflect")[1:-1, 1:-1]
    gy = correlate2d(img, np.outer([-1, 0, 1], [1, 2, 1]).astype(np.float64), mode="reflect")[1:-1, 1:-1]
    grad = np.hypot(gx, gy)
    mask = np.ones(response.shape, dtype=bool)
    sigma = float(np.sqrt(np.pi / 2) * response.mean() / 6.0)
    for _ in range(iterations):
        mask = grad <= edge_factor * np.sqrt(12.0) * sigma
        if mask.sum() < 0.25 * mask.size:  # quá ít pixel còn lại: giữ ước lượng hiện tại
            break
        sigma = float(np.sqrt(np.pi / 2) * response[mask].mean() / 6.0)
    return sigma


def gradient_noise_gain(sigma: float | None) -> float:
    """sqrt(sum K^2) với K = Sobel_x ⊛ G_sigma (2D). sigma None/0: chỉ Sobel, gain = sqrt(12)."""
    sobel_x = np.outer([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0])
    if not sigma:
        return float(np.sqrt((sobel_x ** 2).sum()))
    g = gaussian_kernel1d(sigma, 4.0)
    gauss2d = np.outer(g, g)
    # tích chập đầy đủ hai kernel (kích thước nhỏ, tính trực tiếp)
    kh, kw = gauss2d.shape
    full = np.zeros((kh + 2, kw + 2))
    for dy in range(3):
        for dx in range(3):
            full[dy:dy + kh, dx:dx + kw] += sobel_x[dy, dx] * gauss2d
    return float(np.sqrt((full ** 2).sum()))


def _quantile(mag, nms_values, ctx, profile, low_q: float = 0.8, high_q: float = 0.9):
    if not (0 <= low_q <= high_q <= 1):
        raise ValueError("cần 0 <= low_q <= high_q <= 1")
    lo, hi = np.percentile(mag, [100.0 * low_q, 100.0 * high_q])
    return float(lo), float(hi)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Ngưỡng Otsu tự cài: tối đa phương sai giữa hai lớp sigma_b^2(t) = w0 w1 (mu0 - mu1)^2."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    vmin, vmax = values.min(), values.max()
    if vmax <= vmin:
        return float(vmin)
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    mu0 = np.divide(s0, w0, out=np.zeros_like(s0), where=w0 > 0)
    mu1 = np.divide(s0[-1] - s0, w1, out=np.zeros_like(s0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(edges[int(np.argmax(between)) + 1])


def _otsu(mag, nms_values, ctx, profile, ratio: float = 0.5):
    values = nms_values[nms_values > 0]
    high = otsu_threshold(values)
    return ratio * high, high


def _raw_in_profile_scale(ctx: ThresholdContext) -> tuple[np.ndarray, float]:
    """Ảnh gốc (chưa làm trơn) và hệ số đổi sang thang của hồ sơ."""
    if ctx.raw is None:
        return ctx.image, 1.0
    return ctx.raw, ctx.scale_max / dtype_max(ctx.raw)


def _noise(mag, nms_values, ctx, profile, k: float | None = None, k_low: float = 2.0,
           ratio: float | None = None):
    """T_h = k sigma_g, T_l = k_low sigma_g (hoặc ratio * T_h).

    Mặc định k = sqrt(2 ln N) (ngưỡng phổ quát Donoho–Johnstone): với M Rayleigh, số pixel nhiễu thuần
    vượt T_h có kỳ vọng N exp(-k^2/2) = 1, tức gần như không có hạt giống giả. Mặc định cũ k = 3.5,
    T_l = 0.4 T_h cho khoảng 0.2% N hạt giống giả (~140 trên ảnh 256^2) nên Precision chỉ 0.2–0.7
    (thực nghiệm X2, xem docs/experiments). k_low = 2: mật độ ứng viên nhiễu ~10%, cụm nhỏ (đã đo).
    """
    source, factor = _raw_in_profile_scale(ctx)
    sigma_n = estimate_noise_sigma(source) * factor
    sigma_g = sigma_n * gradient_noise_gain(ctx.sigma)
    if k is None:
        k = float(np.sqrt(2.0 * np.log(mag.size)))
    high = k * sigma_g
    low = ratio * high if ratio is not None else min(k_low * sigma_g, high)
    return low, high


def _median(mag, nms_values, ctx, profile, spread: float = 0.33):
    source, factor = _raw_in_profile_scale(ctx)
    med = float(np.median(source)) * factor
    return max(0.0, (1 - spread) * med), min(ctx.scale_max, (1 + spread) * med)


THRESHOLD_STRATEGIES: dict[str, Callable] = {
    "quantile": _quantile,
    "otsu": _otsu,
    "noise": _noise,
    "median": _median,
}
