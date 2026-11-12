"""Canny Edge Detector tự cài đặt, có hồ sơ tái tạo OpenCV hoặc scikit-image.

Pipeline (docs/mathematics/01–05):
    1. làm trơn Gaussian (tùy hồ sơ)      gaussian.py
    2. gradient Sobel, độ lớn, hướng      gradient.py
    3. NMS (lượng tử hóa / nội suy)       nms.py
    4. ngưỡng kép (cố định hoặc tự động)  hysteresis.py, thresholds.py
    5. hysteresis (BFS mặc định)          hysteresis.py

Hồ sơ là dữ liệu (dataclass), không phải lớp con: OPENCV và SKIMAGE chỉ khác nhau ở các lựa chọn.
Tiêu chí khớp (đã kiểm thử): OPENCV với sigma=None trùng từng pixel với cv2.Canny; SKIMAGE trùng
skimage.feature.canny trừ điểm hòa do sai số dấu phẩy động.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..imaging.convert import to_float, to_uint8
from ..imaging.validate import check_gray, dtype_max
from .gaussian import gaussian_blur
from .gradient import magnitude as grad_magnitude
from .gradient import sobel
from .hysteresis import double_threshold, hysteresis_bfs
from .nms import nms_bilinear, nms_quantized
from .thresholds import THRESHOLD_STRATEGIES, ThresholdContext

Hysteresis = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CannyProfile:
    """Bộ lựa chọn rời rạc mô tả một cách cài Canny.

    scale       'uint8': làm việc trên số nguyên 0..255 (OpenCV); 'unit': float [0, 1] (skimage)
    smoothing   có bước Gaussian bên trong hay không; default_sigma dùng khi người dùng không truyền
    blur_mode   chế độ biên của Gaussian; bleedover: chuẩn hóa theo mặt nạ (chỉ với 'constant')
    sobel_mode  chế độ biên của Sobel
    norm        'L1' hoặc 'L2'
    nms         'quantized' (cần gradient nguyên) hoặc 'bilinear'
    strict      True: so ngưỡng bằng '>' (OpenCV); False: '>=' (skimage)
    drop_border bỏ viền 1 px (skimage)
    default_low/high  ngưỡng mặc định theo thang của hồ sơ (None: bắt buộc truyền)
    """

    name: str
    scale: str
    smoothing: bool
    default_sigma: float | None
    blur_mode: str
    bleedover: bool
    truncate: float
    sobel_mode: str
    norm: str
    nms: str
    strict: bool
    drop_border: bool
    default_low: float | None = None
    default_high: float | None = None

    def __post_init__(self) -> None:
        if self.scale not in ("uint8", "unit"):
            raise ValueError("scale phải là 'uint8' hoặc 'unit'")
        if self.nms not in ("quantized", "bilinear"):
            raise ValueError("nms phải là 'quantized' hoặc 'bilinear'")
        if self.nms == "quantized" and self.scale != "uint8":
            raise ValueError("NMS lượng tử hóa kiểu OpenCV cần gradient nguyên (scale='uint8')")
        if self.norm not in ("L1", "L2"):
            raise ValueError("norm phải là 'L1' hoặc 'L2'")


OPENCV = CannyProfile(name="opencv", scale="uint8", smoothing=False, default_sigma=None,
                      blur_mode="reflect101", bleedover=False, truncate=3.0, sobel_mode="replicate",
                      norm="L1", nms="quantized", strict=True, drop_border=False)
SKIMAGE = CannyProfile(name="skimage", scale="unit", smoothing=True, default_sigma=1.0,
                       blur_mode="constant", bleedover=True, truncate=4.0, sobel_mode="reflect",
                       norm="L2", nms="bilinear", strict=False, drop_border=True,
                       default_low=0.1, default_high=0.2)
PROFILES = {"opencv": OPENCV, "skimage": SKIMAGE}


@dataclass
class CannyResult:
    """Kết quả từng bước (phục vụ minh họa T1a và kiểm thử từng bước)."""

    smoothed: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray
    nms: np.ndarray            # độ lớn tại cực đại địa phương, 0 nơi khác
    strong: np.ndarray
    candidate: np.ndarray      # mạnh ∪ yếu
    edges: np.ndarray
    low: float                 # ngưỡng hiệu dụng, cùng thang với ``magnitude``
    high: float
    profile: CannyProfile
    sigma: float | None
    extra: dict = field(default_factory=dict)

    @property
    def weak(self) -> np.ndarray:
        return self.candidate & ~self.strong

    @property
    def direction(self) -> np.ndarray:
        return np.arctan2(self.gy.astype(np.float64), self.gx.astype(np.float64))


def resolve_profile(profile: CannyProfile | str, norm: str | None = None) -> CannyProfile:
    """Nhận tên hoặc đối tượng hồ sơ; ``norm`` (nếu có) ghi đè chuẩn của hồ sơ."""
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(f"profile không có: {profile!r}; chọn {sorted(PROFILES)}")
        profile = PROFILES[profile]
    return replace(profile, norm=norm) if norm and norm != profile.norm else profile


def prepare(image: np.ndarray, profile: CannyProfile, sigma: float | None) -> tuple[np.ndarray, float | None]:
    """Bước 1: đưa ảnh về thang của hồ sơ và làm trơn. Trả (ảnh đã làm trơn, sigma thực dùng)."""
    if sigma is None:
        sigma = profile.default_sigma
    if sigma is not None and sigma < 0:
        raise ValueError(f"sigma phải >= 0, nhận {sigma}")
    if profile.scale == "uint8":
        img = to_uint8(image)
        if sigma:  # OpenCV không có sigma: làm mờ trước (như Lab3), bản Manual xấp xỉ cv2.GaussianBlur
            blurred = gaussian_blur(img.astype(np.float64), sigma, truncate=profile.truncate, mode=profile.blur_mode)
            img = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
        return img.astype(np.int64), sigma
    img = to_float(image)
    if not profile.smoothing or not sigma:
        # sigma = 0 trong skimage: Gaussian bán kính 0 là đồng nhất, vẫn chia cho (1 + eps) nếu bleedover
        return (img / (1.0 + np.finfo(np.float64).eps) if profile.bleedover else img), sigma
    return gaussian_blur(img, sigma, truncate=profile.truncate, mode=profile.blur_mode,
                         bleedover=profile.bleedover), sigma


def comparison_values(gx: np.ndarray, gy: np.ndarray, profile: CannyProfile) -> tuple[np.ndarray, np.ndarray]:
    """Trả (độ lớn để báo cáo, độ lớn để so sánh). OpenCV-L2 so sánh bình phương nguyên gx^2 + gy^2."""
    mag = grad_magnitude(gx, gy, profile.norm)
    if profile.nms == "quantized" and profile.norm == "L2":
        return mag, gx.astype(np.int64) ** 2 + gy.astype(np.int64) ** 2
    return mag, mag


def comparison_thresholds(low: float, high: float, profile: CannyProfile) -> tuple[float, float]:
    """Đổi ngưỡng (thang của ``magnitude``) sang thang so sánh. Với OpenCV: floor, và bình phương nếu L2."""
    if profile.nms != "quantized":
        return low, high
    if profile.norm == "L2":
        low, high = min(32767.0, low), min(32767.0, high)
        low = low * low if low > 0 else low
        high = high * high if high > 0 else high
    return float(np.floor(low)), float(np.floor(high))


def canny_from_gradient(gx: np.ndarray, gy: np.ndarray, profile: CannyProfile, *,
                        low_threshold: float | None, high_threshold: float | None,
                        input_scale: float, threshold_strategy: str = "fixed",
                        strategy_params: dict | None = None, context: ThresholdContext | None = None,
                        hysteresis: Hysteresis = hysteresis_bfs, magnitude: np.ndarray | None = None,
                        smoothed: np.ndarray | None = None, sigma: float | None = None) -> CannyResult:
    """Bước 3–5 trên gradient có sẵn (dùng chung cho ảnh xám và các phương pháp ảnh màu).

    ``input_scale``: hệ số chia ngưỡng người dùng (255 cho ảnh uint8 vào hồ sơ 'unit', 1 nơi khác).
    ``magnitude``: truyền khi độ lớn không suy ra được từ (gx, gy) theo chuẩn của hồ sơ (vd Di Zenzo).
    """
    if magnitude is None:
        mag, cmp = comparison_values(gx, gy, profile)
    else:
        mag = cmp = magnitude
    h, w = mag.shape
    mask = np.ones((h, w), dtype=bool)
    if profile.drop_border:
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False

    # NMS độc lập với ngưỡng: tính cực đại địa phương một lần, áp ngưỡng sau
    if profile.nms == "quantized":
        local_max = nms_quantized(gx, gy, cmp, low=-1)
        nms_values = np.where(local_max, mag, 0)
    else:
        nms_values = nms_bilinear(gx, gy, mag, low=0.0, mask=mask)
        local_max = nms_values > 0

    if threshold_strategy == "fixed":
        low = profile.default_low if low_threshold is None else low_threshold / input_scale
        high = profile.default_high if high_threshold is None else high_threshold / input_scale
        if low is None or high is None:
            raise ValueError(f"Hồ sơ {profile.name!r} bắt buộc truyền low_threshold và high_threshold")
    else:
        if threshold_strategy not in THRESHOLD_STRATEGIES:
            raise ValueError(f"threshold_strategy không có: {threshold_strategy!r}; "
                             f"chọn {['fixed', *sorted(THRESHOLD_STRATEGIES)]}")
        ctx = context or ThresholdContext(image=smoothed, sigma=sigma, scale_max=255.0 if profile.scale == "uint8" else 1.0)
        low, high = THRESHOLD_STRATEGIES[threshold_strategy](mag, nms_values, ctx, profile, **(strategy_params or {}))

    if profile.strict:
        low, high = min(low, high), max(low, high)  # OpenCV tự hoán đổi
    elif high < low:
        raise ValueError("low_threshold phải <= high_threshold")

    cmp_low, cmp_high = comparison_thresholds(low, high, profile)
    cmp_at_max = np.where(local_max, cmp, 0)
    strong, candidate = double_threshold(cmp_at_max, cmp_low, cmp_high, strict=profile.strict)
    if profile.nms == "quantized":
        # OpenCV: ứng viên là m > low kể cả khi low âm; cmp = 0 tại cực đại vẫn hợp lệ nếu low < 0
        candidate = local_max & (cmp > cmp_low)
        strong = candidate & (cmp > cmp_high)
    edges = hysteresis(strong, candidate)
    return CannyResult(smoothed=smoothed if smoothed is not None else np.empty(0), gx=gx, gy=gy,
                       magnitude=mag, nms=np.where(local_max, mag, 0), strong=strong, candidate=candidate,
                       edges=edges, low=float(low), high=float(high), profile=profile, sigma=sigma)


def canny(image: np.ndarray, sigma: float | None = None, low_threshold: float | None = None,
          high_threshold: float | None = None, *, profile: CannyProfile | str = SKIMAGE,
          norm: str | None = None, threshold_strategy: str = "fixed", strategy_params: dict | None = None,
          hysteresis: Hysteresis = hysteresis_bfs) -> CannyResult:
    """Canny trên ảnh xám 2D.

    Ngưỡng ``fixed`` truyền theo đơn vị của **dtype đầu vào** (giống skimage): ảnh uint8 với hồ sơ
    'skimage' thì 25.5 tương đương 0.1 trên thang [0,1]. Với hồ sơ 'opencv', ngưỡng so trực tiếp với
    độ lớn Sobel trên ảnh 0..255 (giống cv2.Canny). Các chiến lược tự động: xem ``thresholds.py``.
    """
    image = check_gray(image)
    prof = resolve_profile(profile, norm)
    smoothed, used_sigma = prepare(image, prof, sigma)
    gx, gy = sobel(smoothed, mode=prof.sobel_mode)
    input_scale = dtype_max(image) if prof.scale == "unit" else 1.0
    ctx = ThresholdContext(image=smoothed, sigma=used_sigma, scale_max=255.0 if prof.scale == "uint8" else 1.0,
                           raw=image)
    return canny_from_gradient(gx, gy, prof, low_threshold=low_threshold, high_threshold=high_threshold,
                               input_scale=input_scale, threshold_strategy=threshold_strategy,
                               strategy_params=strategy_params, context=ctx, hysteresis=hysteresis,
                               smoothed=smoothed, sigma=used_sigma)
