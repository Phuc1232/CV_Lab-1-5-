"""Đệm biên và tương quan (correlation) 1D/2D tự cài đặt.

Tương quan: out[i] = sum_k w[k] * x[i + k - r], r = len(w) // 2. Với kernel đối xứng
(Gaussian) tương quan = tích chập; với kernel Sobel đạo hàm [-1, 0, 1] ta dùng tương quan
để out[i] = x[i+1] - x[i-1] (cùng quy ước dấu với cv2.Sobel và scipy.ndimage.sobel).

Độ phức tạp: correlate1d O(N k), correlate2d O(N k_h k_w); bộ nhớ O(N) cho ảnh đệm.
"""

from __future__ import annotations

import numpy as np

# Tên chế độ biên của dự án -> tên trong numpy.pad
#   constant   : ...0 0 | a b c d | 0 0...      (scipy 'constant', OpenCV BORDER_CONSTANT)
#   replicate  : ...a a | a b c d | d d...      (scipy 'nearest', OpenCV BORDER_REPLICATE)
#   reflect    : ...b a | a b c d | d c...      (scipy 'reflect', OpenCV BORDER_REFLECT)
#   reflect101 : ...c b | a b c d | c b...      (scipy 'mirror',  OpenCV BORDER_REFLECT_101, mặc định của cv2)
PAD_MODES = {"constant": "constant", "replicate": "edge", "reflect": "symmetric", "reflect101": "reflect"}


def pad(image: np.ndarray, pad_y: int, pad_x: int, mode: str, cval: float = 0.0) -> np.ndarray:
    """Đệm ``pad_y`` hàng trên/dưới và ``pad_x`` cột trái/phải theo ``mode``."""
    if mode not in PAD_MODES:
        raise ValueError(f"mode biên không hỗ trợ: {mode!r}; chọn trong {sorted(PAD_MODES)}")
    widths = [(pad_y, pad_y), (pad_x, pad_x)] + [(0, 0)] * (image.ndim - 2)
    if mode == "constant":
        return np.pad(image, widths, mode="constant", constant_values=cval)
    return np.pad(image, widths, mode=PAD_MODES[mode])


def _accum_dtype(image: np.ndarray, kernel: np.ndarray) -> np.dtype:
    """Số nguyên × số nguyên tích lũy bằng int64 (chính xác tuyệt đối, không tràn); còn lại float64."""
    if np.issubdtype(image.dtype, np.integer) and np.issubdtype(kernel.dtype, np.integer):
        return np.dtype(np.int64)
    return np.dtype(np.float64)


def correlate1d(image: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "reflect",
                cval: float = 0.0) -> np.ndarray:
    """Tương quan 1D theo trục ``axis`` (0 = hàng/y, 1 = cột/x). Kernel phải có độ dài lẻ."""
    kernel = np.asarray(kernel)
    if kernel.ndim != 1 or kernel.size % 2 == 0:
        raise ValueError("kernel 1D phải có độ dài lẻ")
    if axis not in (0, 1):
        raise ValueError("axis phải là 0 hoặc 1")
    r = kernel.size // 2
    dtype = _accum_dtype(image, kernel)
    padded = pad(image.astype(dtype, copy=False), r if axis == 0 else 0, r if axis == 1 else 0, mode, cval)
    n = image.shape[axis]
    out = np.zeros(image.shape, dtype=dtype)
    for k, w in enumerate(kernel):
        if w == 0:
            continue
        window = padded[k:k + n] if axis == 0 else padded[:, k:k + n]
        out += w * window
    return out


def correlate_separable(image: np.ndarray, kernel_y: np.ndarray, kernel_x: np.ndarray,
                        mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """Tương quan với kernel tách được K = kernel_y ⊗ kernel_x: lọc theo hàng (trục 0) trước, rồi theo cột."""
    return correlate1d(correlate1d(image, kernel_y, 0, mode, cval), kernel_x, 1, mode, cval)


def correlate2d(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """Tương quan 2D tổng quát (kernel kích thước lẻ x lẻ), cài bằng tổng các ảnh dịch chuyển."""
    kernel = np.asarray(kernel)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel 2D phải có kích thước lẻ x lẻ")
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    dtype = _accum_dtype(image, kernel)
    padded = pad(image.astype(dtype, copy=False), ry, rx, mode, cval)
    h, w = image.shape[:2]
    out = np.zeros(image.shape, dtype=dtype)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            wgt = kernel[dy, dx]
            if wgt != 0:
                out += wgt * padded[dy:dy + h, dx:dx + w]
    return out
