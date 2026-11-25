"""Sinh ảnh tổng hợp có ground truth CHÍNH XÁC (thay cho BSDS, xem docs/datasets/README.md).

Ảnh sinh từ ảnh nhãn (label map) vẽ bằng cv2 (không khử răng cưa), nên biên vùng biết chính xác.
Quy ước ground truth cạnh: pixel p là cạnh nếu nhãn của p khác nhãn của lân cận phải hoặc dưới
(đường 1 px, nằm phía trên/trái của biên — trùng quy ước của cv2.Canny khi gặp điểm hòa).
Suy giảm (nhiễu, mờ, tương phản) KHÔNG thay đổi ground truth.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import noise as nz

SHAPE_TYPES = ("circle", "triangle", "rectangle", "pentagon", "hexagon", "star")


def edges_from_labels(labels: np.ndarray) -> np.ndarray:
    """Ground truth cạnh 1 px từ ảnh nhãn (quy ước ở docstring module)."""
    e = np.zeros(labels.shape, dtype=bool)
    e[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    e[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return e


def _polygon(cx: float, cy: float, radius: float, n: int, rotation: float, inner: float | None = None) -> np.ndarray:
    k = 2 * n if inner else n
    ang = rotation + np.arange(k) * 2 * np.pi / k
    rad = np.where(np.arange(k) % 2 == 1, radius * inner, radius) if inner else np.full(k, radius)
    return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).round().astype(np.int32)


def draw_shape(labels: np.ndarray, shape: str, cx: float, cy: float, radius: float, rotation: float,
               value: int) -> None:
    """Vẽ hình (tô kín) lên ảnh nhãn bằng giá trị ``value``."""
    if shape == "circle":
        cv2.circle(labels, (int(round(cx)), int(round(cy))), int(round(radius)), value, -1, lineType=cv2.LINE_8)
        return
    if shape == "rectangle":
        w, h = radius * 1.6, radius * 1.0
        box = cv2.boxPoints(((cx, cy), (w, h), np.degrees(rotation))).round().astype(np.int32)
        cv2.fillPoly(labels, [box], value, lineType=cv2.LINE_8)
        return
    n = {"triangle": 3, "pentagon": 5, "hexagon": 6, "star": 5}[shape]
    pts = _polygon(cx, cy, radius, n, rotation, inner=0.45 if shape == "star" else None)
    cv2.fillPoly(labels, [pts], value, lineType=cv2.LINE_8)


def shapes_scene(size: int = 256, n_shapes: int = 6, rng: np.random.Generator | None = None,
                 shape_types: tuple[str, ...] = SHAPE_TYPES, min_contrast: float = 0.25,
                 radius_range: tuple[float, float] = (14, 30), max_tries: int = 2000) -> dict:
    """Cảnh gồm các hình không chạm nhau trên nền phẳng.

    Trả dict: image (float [0,1]), labels (int32, 0 = nền), edges (bool), classes {nhãn: tên hình}.
    """
    rng = rng or np.random.default_rng(0)
    if size < 24:
        raise ValueError("size phải >= 24 để đặt được hình")
    # Giới hạn bán kính theo kích thước ảnh (ảnh nhỏ không chứa nổi bán kính mặc định)
    r_hi = min(radius_range[1], size / 4 - 4)
    radius_range = (min(radius_range[0], r_hi), r_hi)
    labels = np.zeros((size, size), dtype=np.int32)
    background = float(rng.uniform(0.15, 0.35))
    placed: list[tuple[float, float, float]] = []
    classes: dict[int, str] = {}
    intensities = {0: background}
    tries = 0
    while len(placed) < n_shapes and tries < max_tries:
        tries += 1
        r = float(rng.uniform(*radius_range))
        cx, cy = rng.uniform(r + 4, size - r - 4, 2)
        if any(np.hypot(cx - x, cy - y) < r + rr + 6 for x, y, rr in placed):
            continue
        lab = len(placed) + 1
        shape = shape_types[(lab - 1) % len(shape_types)] if len(shape_types) else "circle"
        draw_shape(labels, shape, cx, cy, r, float(rng.uniform(0, 2 * np.pi)), lab)
        placed.append((cx, cy, r))
        classes[lab] = shape
        intensities[lab] = float(np.clip(background + rng.choice([-1, 1]) * rng.uniform(min_contrast, 0.5)
                                         if background > 0.3 else background + rng.uniform(min_contrast, 0.6), 0, 1))
    lut = np.array([intensities[i] for i in range(len(placed) + 1)])
    return {"image": lut[labels], "labels": labels, "edges": edges_from_labels(labels), "classes": classes}


def step_edge(size: int = 128, angle_deg: float = 0.0, low: float = 0.3, high: float = 0.7) -> dict:
    """Cạnh bậc thang thẳng qua tâm ảnh, nghiêng ``angle_deg`` so với trục dọc."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    t = np.radians(angle_deg)
    labels = ((xx - size / 2 + 0.5) * np.cos(t) + (yy - size / 2 + 0.5) * np.sin(t) >= 0).astype(np.int32)
    return {"image": np.where(labels == 1, high, low), "labels": labels, "edges": edges_from_labels(labels),
            "classes": {}}


def isoluminant_scene(size: int = 128, rng: np.random.Generator | None = None) -> dict:
    """Ảnh RGB: hình màu có cùng độ sáng Y với nền (Y lệch < 0.5/255), dùng cho E3."""
    rng = rng or np.random.default_rng(0)
    scene = shapes_scene(size, 3, rng, ("circle", "rectangle", "triangle"), radius_range=(16, 26))
    labels = scene["labels"]
    bg = np.array([200, 100, 50], dtype=np.float64) / 255  # Y ≈ 124.2/255
    fg = np.array([55, 170, 70], dtype=np.float64) / 255   # Y ≈ 124.2/255
    rgb = np.where(labels[..., None] > 0, fg, bg)
    return {"image": rgb, "labels": labels, "edges": scene["edges"], "classes": scene["classes"]}


def degrade(image: np.ndarray, rng: np.random.Generator, *, noise_sigma: float = 0.0, salt_pepper: float = 0.0,
            contrast: float = 1.0, blur: float = 0.0, texture: float = 0.0, texture_scale: float = 1.5) -> np.ndarray:
    """Suy giảm theo thứ tự vật lý: mờ quang học -> giảm tương phản -> texture -> nhiễu Gauss -> muối tiêu."""
    img = nz.optical_blur(image, blur) if blur > 0 else np.asarray(image, dtype=np.float64)
    if contrast != 1.0:
        img = nz.reduce_contrast(img, contrast, center=float(img.mean()))
    if texture > 0:
        tex = nz.correlated_texture(img.shape[:2], texture_scale, texture, rng)
        img = np.clip(img + (tex[..., None] if img.ndim == 3 else tex), 0, 1)
    if noise_sigma > 0:
        img = nz.add_gaussian_noise(img, noise_sigma, rng)
    if salt_pepper > 0:
        img = nz.add_salt_pepper(img, salt_pepper, rng)
    return img


def moving_video(n_frames: int = 12, size: int = 128, noise_sigma: float = 0.05, speed: float = 0.0,
                 rng: np.random.Generator | None = None) -> list[dict]:
    """Chuỗi khung hình: cảnh cố định (speed = 0) hoặc tịnh tiến ``speed`` px/khung, nhiễu độc lập mỗi khung."""
    rng = rng or np.random.default_rng(0)
    base = shapes_scene(size + int(abs(speed) * n_frames) + 2, 4, rng, radius_range=(12, 20))
    frames = []
    for t in range(n_frames):
        off = int(round(speed * t))
        labels = base["labels"][:size, off:off + size]
        clean = base["image"][:size, off:off + size]
        frames.append({"image": nz.add_gaussian_noise(clean, noise_sigma, rng), "labels": labels,
                       "edges": edges_from_labels(labels), "classes": base["classes"], "frame": t})
    return frames
