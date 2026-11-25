"""Bản Optimized: bọc OpenCV / scikit-image / SciPy. Đây là nơi DUY NHẤT trong gói gọi các thư viện này
cho thuật toán (quy tắc 2). Mọi hàm phát hiện cạnh trả mảng bool.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature

from .imaging.convert import to_float, to_uint8
from .imaging.validate import check_gray, check_image

STRUCT_8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------- Canny
def canny_opencv(image: np.ndarray, low_threshold: float = 100, high_threshold: float = 200,
                 sigma: float = 0.0, l2gradient: bool = False, aperture_size: int = 3) -> np.ndarray:
    """cv2.Canny. cv2 không có sigma: sigma > 0 thì gọi cv2.GaussianBlur trước (cách làm của Lab3)."""
    img = to_uint8(check_gray(image))
    if sigma and sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.Canny(img, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=l2gradient) > 0


def canny_opencv_color(image: np.ndarray, low_threshold: float = 100, high_threshold: float = 200,
                       l2gradient: bool = False) -> np.ndarray:
    """cv2.Canny trên ảnh 3 kênh (chọn kênh có gradient lớn nhất tại mỗi pixel)."""
    img = to_uint8(check_image(image))
    return cv2.Canny(np.ascontiguousarray(img), low_threshold, high_threshold, L2gradient=l2gradient) > 0


def canny_skimage(image: np.ndarray, sigma: float = 1.0, low_threshold: float | None = None,
                  high_threshold: float | None = None, use_quantiles: bool = False) -> np.ndarray:
    return feature.canny(check_gray(image), sigma=sigma, low_threshold=low_threshold,
                         high_threshold=high_threshold, use_quantiles=use_quantiles)


def hysteresis_label(strong: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Hysteresis bằng gán nhãn thành phần liên thông (SciPy, C): O(N alpha(N)). Trùng tuyệt đối với BFS."""
    allowed = candidate | strong
    labels, count = ndi.label(allowed, STRUCT_8)
    if count == 0:
        return np.zeros_like(allowed)
    good = np.zeros(count + 1, dtype=bool)
    good[np.unique(labels[strong])] = True
    good[0] = False
    return good[labels]


# ---------------------------------------------------------------- Các bước riêng lẻ (so sánh Manual)
def gaussian_blur_scipy(image: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    return ndi.gaussian_filter(to_float(image), sigma, mode=mode, truncate=4.0)


def gaussian_blur_opencv(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(to_float(image).astype(np.float32), (0, 0), sigma)


def sobel_opencv(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = to_uint8(image)
    return (cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
            cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE))


def median_opencv(image: np.ndarray, size: int = 3) -> np.ndarray:
    return to_float(cv2.medianBlur(to_uint8(check_gray(image)), size))


def bilateral_opencv(image: np.ndarray, diameter: int = 5, sigma_color: float = 0.1,
                     sigma_space: float = 2.0) -> np.ndarray:
    img = to_float(check_gray(image)).astype(np.float32)
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space).astype(np.float64)


def closing_opencv(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se.astype(np.uint8)) > 0


def structuring_element_opencv(shape: str, size: int) -> np.ndarray:
    code = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}[shape]
    return cv2.getStructuringElement(code, (size, size)) > 0


def fill_holes_scipy(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


def label_scipy(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndi.label(mask, STRUCT_8)
    return labels.astype(np.int32), int(n)


def boundary_opencv(region: np.ndarray) -> np.ndarray:
    """Đường bao ngoài (x, y) của vùng lớn nhất bằng cv2.findContours (CHAIN_APPROX_NONE)."""
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.int64)
    return max(contours, key=len)[:, 0, :].astype(np.int64)


def approx_polygon_opencv(points: np.ndarray, epsilon: float) -> np.ndarray:
    return cv2.approxPolyDP(points.reshape(-1, 1, 2).astype(np.int32), epsilon, True)[:, 0, :]


def log_scipy(image: np.ndarray, sigma: float = 2.0, threshold_rel: float = 0.05) -> np.ndarray:
    """LoG bằng scipy.ndimage.gaussian_laplace + cùng quy tắc điểm qua 0 với bản Manual."""
    from .core.baselines import zero_crossings
    response = ndi.gaussian_laplace(to_float(check_gray(image)), sigma, mode="reflect")
    return zero_crossings(response, threshold_rel * float(np.abs(response).max()))


def sobel_threshold_opencv(image: np.ndarray, threshold: float = 0.3, sigma: float = 0.0) -> np.ndarray:
    img = to_float(check_gray(image))
    if sigma > 0:
        img = ndi.gaussian_filter(img, sigma, mode="reflect")
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return np.hypot(gx, gy) > threshold
