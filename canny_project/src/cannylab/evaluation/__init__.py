"""Đánh giá: cạnh (có dung sai), vùng, phân loại, chất lượng ảnh đầu vào, hiệu năng.

Module này không biết gì về thuật toán; chỉ nhận mảng bool / nhãn. Định nghĩa toán học và giới hạn:
docs/mathematics/06_chi_so_danh_gia.md, docs/evaluation/README.md.
"""

from .edge_metrics import continuity, edge_metrics
from .region_metrics import dice, iou, match_objects

__all__ = ["edge_metrics", "continuity", "iou", "dice", "match_objects"]
