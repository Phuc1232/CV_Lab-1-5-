# 05. Ngưỡng kép và hysteresis

**WHAT / WHY / MATHEMATICS.** Xem [../mathematics/04](../mathematics/04_nguong_kep_va_hysteresis.md), gồm chứng minh BFS ≡ gán nhãn và tính đơn điệu theo ngưỡng.

**PSEUDOCODE (Manual, BFS bằng ngăn xếp).**
```
out[strong] = True ; stack = strong pixels
while stack: p = pop(); for q in 8-neighbors(p): if candidate[q] and not out[q]: out[q] = True; push(q)
```
**Optimized:** `ndi.label(candidate, 3×3)`, giữ nhãn nào chứa điểm mạnh (cách skimage làm).

**IMPLEMENTATION.** [`core/hysteresis.py`](../../src/cannylab/core/hysteresis.py): `double_threshold(strict=…)`, `reachable` (dùng lại cho lấp lỗ), `hysteresis_bfs`. [`backends.hysteresis_label`](../../src/cannylab/backends.py). Thuật toán `canny_manual_fast` = Canny tự cài + `hysteresis_label`.

**PARAMETERS.** $T_l\le T_h$ (ValueError nếu ngược với hồ sơ skimage; hồ sơ OpenCV tự hoán đổi như cv2), `strict`.

**COMPLEXITY.** BFS $O(N)$ nhưng chạy bằng Python (hằng số lớn); `ndi.label` $O(N\alpha(N))$ chạy bằng C. Số đo ở [../experiments/README.md](../experiments/README.md) (X9).

**LIMITATIONS.** Texture dài chạm điểm mạnh vẫn lọt qua; cạnh thật nhưng toàn bộ yếu bị loại.

**KẾT QUẢ.** `test_hysteresis_bfs_equals_label`: 20 ảnh ngẫu nhiên, trùng tuyệt đối. X1 `bfs_vs_label`: n_diff = 0 trên 6 ảnh thật. `test_edges_monotone_in_thresholds`: không vi phạm.
