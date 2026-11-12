"""Non-Maximum Suppression tự cài đặt (bước 3). Toán học: docs/mathematics/03_non_maximum_suppression.md.

Hai biến thể, cài vector hóa bằng NumPy (không có vòng lặp Python), O(N) thời gian:

``nms_quantized`` (quy ước OpenCV): lượng tử hóa hướng gradient thành 4 cung bằng so sánh số
nguyên với tan(22.5°) * 2^15 (không cần atan2):
    ngang  : |gy|<<15 <  TG22*|gx|                   giữ nếu m >  trái  và m >= phải
    dọc    : |gy|<<15 >  TG22*|gx| + (|gx|<<16)       giữ nếu m >  trên  và m >= dưới
    chéo   : còn lại; s = -1 nếu gx, gy trái dấu      giữ nếu m >  hai lân cận chéo (cả hai nghiêm ngặt)
Ngoài ảnh coi độ lớn = 0. Điều kiện ứng viên: m > T_low.

``nms_bilinear`` (quy ước skimage): nội suy tuyến tính độ lớn tại hai điểm dọc hướng gradient,
giữ nếu cả hai giá trị nội suy <= m (điểm hòa được giữ ở cả hai phía). Điều kiện: m >= T_low,
và pixel nằm trong ``mask`` (skimage bỏ viền 1 px).
"""

from __future__ import annotations

import numpy as np

CANNY_SHIFT = 15
TG22 = int(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5)  # = 13573


def _shifted(padded: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Mảng lân cận (y+dy, x+dx) cắt từ ảnh đã đệm 1 pixel."""
    h, w = padded.shape[0] - 2, padded.shape[1] - 2
    return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def nms_quantized(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray, low: float) -> np.ndarray:
    """NMS 4 hướng theo đúng quy tắc của cv2.Canny. Trả mặt nạ bool các pixel ứng viên (cực đại và m > low).

    ``gx, gy`` là gradient nguyên; ``mag`` là độ lớn dùng để so sánh (L1 nguyên, hoặc bình phương L2 nguyên).
    """
    gx = gx.astype(np.int64, copy=False)
    gy = gy.astype(np.int64, copy=False)
    m = mag
    p = np.pad(m, 1, mode="constant", constant_values=0)
    ax = np.abs(gx)
    ay = np.abs(gy) << CANNY_SHIFT
    tg22x = ax * TG22
    tg67x = tg22x + (ax << (CANNY_SHIFT + 1))
    horizontal = ay < tg22x
    vertical = ~horizontal & (ay > tg67x)
    diagonal = ~horizontal & ~vertical
    same_sign = (gx ^ gy) >= 0  # s = +1: so với (y-1, x-1) và (y+1, x+1)

    keep_h = (m > _shifted(p, 0, -1)) & (m >= _shifted(p, 0, 1))
    keep_v = (m > _shifted(p, -1, 0)) & (m >= _shifted(p, 1, 0))
    keep_d = np.where(same_sign,
                      (m > _shifted(p, -1, -1)) & (m > _shifted(p, 1, 1)),
                      (m > _shifted(p, -1, 1)) & (m > _shifted(p, 1, -1)))
    local_max = (horizontal & keep_h) | (vertical & keep_v) | (diagonal & keep_d)
    return local_max & (m > low)


def nms_bilinear(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray, low: float,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """NMS nội suy song tuyến tính theo quy tắc của skimage. Trả mảng float: m tại pixel được giữ, 0 nơi khác.

    Ký hiệu như skimage: ``isobel`` = gy (trục 0), ``jsobel`` = gx (trục 1).
    """
    isobel = gy.astype(np.float64, copy=False)
    jsobel = gx.astype(np.float64, copy=False)
    m = mag.astype(np.float64, copy=False)
    if mask is None:
        mask = np.ones(m.shape, dtype=bool)
    p = np.pad(m, 1, mode="constant", constant_values=0.0)

    is_down = isobel <= 0
    is_up = isobel >= 0
    is_left = jsobel <= 0
    is_right = jsobel >= 0
    cond1 = (is_up & is_right) | (is_down & is_left)   # hai thành phần cùng dấu
    cond2 = (is_down & is_right) | (is_up & is_left)   # trái dấu
    abs_i = np.abs(isobel)
    abs_j = np.abs(jsobel)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Nhánh cond1
        i_major = abs_i > abs_j
        w1 = np.where(i_major, abs_j / abs_i, abs_i / abs_j)
        n11 = np.where(i_major, _shifted(p, 1, 0), _shifted(p, 0, 1))
        n12 = _shifted(p, 1, 1)
        n21 = np.where(i_major, _shifted(p, -1, 0), _shifted(p, 0, -1))
        n22 = _shifted(p, -1, -1)
        keep1 = ((n12 * w1 + n11 * (1.0 - w1)) <= m) & ((n22 * w1 + n21 * (1.0 - w1)) <= m)
        # Nhánh cond2
        j_major = abs_i < abs_j
        w2 = np.where(j_major, abs_i / abs_j, abs_j / abs_i)
        m11 = np.where(j_major, _shifted(p, 0, 1), _shifted(p, -1, 0))
        m12 = _shifted(p, -1, 1)
        m21 = np.where(j_major, _shifted(p, 0, -1), _shifted(p, 1, 0))
        m22 = _shifted(p, 1, -1)
        keep2 = ((m12 * w2 + m11 * (1.0 - w2)) <= m) & ((m22 * w2 + m21 * (1.0 - w2)) <= m)

    selected = mask & (m >= low) & ((cond1 & keep1) | (cond2 & keep2))
    return np.where(selected, m, 0.0)
