"""Nhận dạng hình dạng (P4b): đường bao -> Douglas–Peucker -> đặc trưng -> nhãn.

Tự cài đặt:
    trace_boundary   dò biên Moore 8 lân cận, tiêu chí dừng Jacob; O(chu vi)
    douglas_peucker  xấp xỉ đa giác (giữ điểm xa đoạn thẳng nhất nếu > epsilon); O(n log n) trung bình
    convex_hull      Andrew monotone chain; O(n log n)
Đặc trưng: số đỉnh (eps = 0.04 chu vi, như Lab3; gộp cạnh ngắn giả), độ tròn 4πA/P², độ đặc
(solidity) = diện tích đa giác biên / diện tích bao lồi (cùng hệ tọa độ tâm pixel),
hệ số biến thiên bán kính CV_r = std(r)/mean(r) với r là khoảng cách tâm -> biên.
CV_r tách được tròn (≈0.01) với lục giác (≈0.04) tốt hơn số đỉnh, vì DP với eps lớn biến đường tròn
thành 5–6 đỉnh (docs/algorithms/07_phan_doan_nhan_dang.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPE_CLASSES = ("circle", "triangle", "rectangle", "pentagon", "hexagon", "star")
# Hướng 8 lân cận theo chiều kim đồng hồ (y hướng xuống), bắt đầu từ Tây
_DIRS = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]


def trace_boundary(region: np.ndarray) -> np.ndarray:
    """Dò biên ngoài của vùng (thành phần 8-liên thông). Trả mảng (n, 2) tọa độ (x, y) theo thứ tự."""
    region = np.asarray(region, dtype=bool)
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    h, w = region.shape
    start = (int(ys[0]), int(xs[0]))  # điểm đầu theo raster: lân cận Tây chắc chắn là nền
    contour = [start]
    cur, back = start, 0  # back: hướng từ cur tới pixel nền vừa xét
    first_move = None
    for _ in range(4 * region.size + 8):
        found = None
        for i in range(1, 9):
            d = (back + i) % 8
            ny, nx = cur[0] + _DIRS[d][0], cur[1] + _DIRS[d][1]
            if 0 <= ny < h and 0 <= nx < w and region[ny, nx]:
                found = (d, (ny, nx))
                break
        if found is None:  # vùng một pixel
            break
        d, nxt = found
        if first_move is None:
            first_move = nxt
        elif cur == start and nxt == first_move:
            contour.pop()  # điểm start bị thêm lại ở bước trước
            break
        pd = (d - 1) % 8
        prev = (cur[0] + _DIRS[pd][0], cur[1] + _DIRS[pd][1])
        back = _DIRS.index((prev[0] - nxt[0], prev[1] - nxt[1]))
        cur = nxt
        contour.append(cur)
    pts = np.array([(x, y) for y, x in contour], dtype=np.int64)
    return pts


def _point_line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    norm = np.hypot(*ab)
    if norm == 0:
        return np.hypot(*(points - a).T)
    return np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / norm


def _dp_open(points: np.ndarray, epsilon: float) -> list[int]:
    """Douglas–Peucker trên đường mở; trả chỉ số các điểm được giữ (dùng ngăn xếp, không đệ quy)."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dist = _point_line_distance(points[i + 1:j].astype(np.float64), points[i].astype(np.float64),
                                    points[j].astype(np.float64))
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            idx = i + 1 + k
            keep.add(idx)
            stack.extend([(i, idx), (idx, j)])
    return sorted(keep)


def douglas_peucker(points: np.ndarray, epsilon: float, closed: bool = True) -> np.ndarray:
    """Xấp xỉ đa giác. Đường khép kín: tách tại cặp điểm xa nhau (a = xa điểm 0 nhất, b = xa a nhất),
    cả hai gần như chắc chắn là đỉnh thật, rồi chạy DP trên hai nửa. Tách tại điểm 0 (điểm đầu raster,
    thường nằm giữa một cạnh) sẽ sinh thêm một đỉnh giả."""
    points = np.asarray(points)
    if len(points) < 3:
        return points.copy()
    if not closed:
        return points[_dp_open(points, epsilon)]
    a = int(np.argmax(np.hypot(*(points - points[0]).T)))
    points = np.roll(points, -a, axis=0)
    b = int(np.argmax(np.hypot(*(points - points[0]).T)))
    first = points[:b + 1]
    second = np.vstack([points[b:], points[:1]])
    idx1 = _dp_open(first, epsilon)
    idx2 = [b + i for i in _dp_open(second, epsilon)][1:-1]
    return remove_redundant_vertices(points[idx1 + idx2], epsilon)


def remove_redundant_vertices(poly: np.ndarray, epsilon: float) -> np.ndarray:
    """Lượt 2 cho đường khép kín (như cv2.approxPolyDP): bỏ đỉnh cách đường nối hai đỉnh kề <= epsilon.

    Lượt DP đầu chỉ xét từng nửa riêng nên đỉnh tại điểm tách (a, b) và các đỉnh sát chúng có thể thừa.
    Lặp đến khi không bỏ được nữa; O(n^2) với n đỉnh (n nhỏ).
    """
    poly = np.asarray(poly)
    changed = True
    while changed and len(poly) > 3:
        changed = False
        for k in range(len(poly)):
            prev, nxt = poly[k - 1].astype(np.float64), poly[(k + 1) % len(poly)].astype(np.float64)
            if _point_line_distance(poly[k:k + 1].astype(np.float64), prev, nxt)[0] <= epsilon:
                poly = np.delete(poly, k, axis=0)
                changed = True
                break
    return poly


def merge_short_edges(poly: np.ndarray, min_ratio: float = 0.25) -> np.ndarray:
    """Bỏ đỉnh ở hai đầu cạnh quá ngắn (< min_ratio x cạnh dài nhất): góc nhọn bị raster cắt cụt tạo ra
    cạnh ngắn giả. Trong hai đầu mút, bỏ đỉnh có góc trong gần 180° hơn. Đa giác đều không bị ảnh hưởng."""
    poly = np.asarray(poly, dtype=np.float64)
    while len(poly) > 3:
        sides = np.hypot(*(np.roll(poly, -1, 0) - poly).T)
        i = int(np.argmin(sides))
        if sides[i] >= min_ratio * sides.max():
            break
        j = (i + 1) % len(poly)

        def straightness(k: int) -> float:
            u, v = poly[k - 1] - poly[k], poly[(k + 1) % len(poly)] - poly[k]
            return float(np.dot(u, v) / (np.hypot(*u) * np.hypot(*v) + 1e-12))  # -1 = thẳng hàng

        drop = i if straightness(i) < straightness(j) else j
        poly = np.delete(poly, drop, axis=0)
    return poly


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, np.asarray(points).tolist()))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(points: np.ndarray) -> float:
    x, y = np.asarray(points, dtype=np.float64).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return float(np.hypot(*(np.roll(p, -1, axis=0) - p).T).sum())


@dataclass
class ShapeFeatures:
    area: float
    perimeter: float
    vertices: int
    circularity: float
    solidity: float
    radial_cv: float


def shape_features(region: np.ndarray, epsilon_ratio: float = 0.04, backend: str = "manual") -> ShapeFeatures:
    if backend == "manual":
        contour = trace_boundary(region)
        approx = douglas_peucker(contour, epsilon_ratio * polygon_perimeter(contour)) if len(contour) >= 3 else contour
    elif backend == "opencv":
        from .. import backends
        contour = backends.boundary_opencv(region)
        approx = backends.approx_polygon_opencv(contour, epsilon_ratio * polygon_perimeter(contour)) \
            if len(contour) >= 3 else contour
    else:
        raise ValueError("backend phải là 'manual' hoặc 'opencv'")
    if len(approx) > 3:
        approx = merge_short_edges(approx)
    area = float(np.asarray(region, bool).sum())
    perim = polygon_perimeter(contour) if len(contour) >= 2 else 0.0
    hull = convex_hull(contour) if len(contour) >= 3 else contour
    hull_area = polygon_area(hull) if len(hull) >= 3 else 0.0
    contour_area = polygon_area(contour) if len(contour) >= 3 else 0.0
    ys, xs = np.nonzero(region)
    r = np.hypot(contour[:, 0] - xs.mean(), contour[:, 1] - ys.mean()) if len(contour) else np.zeros(1)
    return ShapeFeatures(area=area, perimeter=perim, vertices=int(len(approx)),
                         circularity=float(4 * np.pi * area / perim ** 2) if perim else 0.0,
                         solidity=float(min(1.0, contour_area / hull_area)) if hull_area else 0.0,  # cùng hệ tâm pixel
                         radial_cv=float(r.std() / r.mean()) if r.mean() > 0 else 0.0)


def classify_shape(f: ShapeFeatures, circle_cv: float = 0.03, star_solidity: float = 0.75) -> str:
    """Luật quyết định (thứ tự quan trọng): sao (lõm) -> tròn (bán kính gần hằng) -> theo số đỉnh."""
    if f.solidity < star_solidity:
        return "star"
    if f.radial_cv < circle_cv:
        return "circle"
    return {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon"}.get(f.vertices,
                                                                           "circle" if f.vertices > 6 else "unknown")


def recognize_shapes(labels: np.ndarray, count: int | None = None, backend: str = "manual",
                     epsilon_ratio: float = 0.04) -> list[dict]:
    """Nhận dạng từng vùng của ảnh nhãn. Trả list dict {label, shape, features}."""
    count = int(labels.max()) if count is None else count
    out = []
    for lab in range(1, count + 1):
        region = labels == lab
        if not region.any():
            continue
        f = shape_features(region, epsilon_ratio, backend)
        out.append({"label": lab, "shape": classify_shape(f), "features": f.__dict__})
    return out
