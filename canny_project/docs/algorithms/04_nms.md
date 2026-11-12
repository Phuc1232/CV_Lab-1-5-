# 04. Non-Maximum Suppression

**WHAT / WHY / MATHEMATICS.** Xem [../mathematics/03](../mathematics/03_non_maximum_suppression.md).

**HOW.** Hai biến thể, cả hai **vector hóa hoàn toàn** (không có vòng lặp Python): dựng 8 mảng lân cận từ độ lớn đã đệm 0, rồi kết hợp các điều kiện bằng `np.where`.

| | `nms_quantized` (OpenCV) | `nms_bilinear` (skimage) |
|---|---|---|
| Chọn hướng | So sánh nguyên $|g_y|\ll15$ với $13573\,|g_x|$ (tan 22.5°·2¹⁵) | Hướng thật, trọng số $w=\min/\max(|g_x|,|g_y|)$ |
| Lân cận | 4 cung: ngang / dọc / 2 đường chéo | Nội suy tuyến tính giữa lân cận thẳng và chéo |
| Điểm hòa | ngang/dọc: `>` một phía, `>=` phía kia; chéo: `>` cả hai | `<=` cả hai phía (giữ) |
| Ngưỡng thấp | `m > T_l` | `m >= T_l`, trong mặt nạ bỏ viền |

**PSEUDOCODE (lượng tử hóa).**
```
p = pad(M, 1, 0)
horizontal = |gy|<<15 <  TG22·|gx|
vertical   = |gy|<<15 >  TG22·|gx| + (|gx|<<16)
diagonal   = còn lại;  same_sign = (gx XOR gy) >= 0
keep_h = M > left  & M >= right
keep_v = M > up    & M >= down
keep_d = same_sign ? (M > upleft & M > downright) : (M > upright & M > downleft)
```

**IMPLEMENTATION.** [`core/nms.py`](../../src/cannylab/core/nms.py). Pipeline tính NMS **một lần** với ngưỡng trung tính rồi mới áp ngưỡng (NMS không phụ thuộc ngưỡng). Nhờ vậy các chiến lược ngưỡng tự động dùng được độ lớn tại cực đại địa phương.

**COMPLEXITY.** $O(N)$ thời gian; khoảng 10 mảng tạm cỡ $N$.

**LIMITATIONS.** Đứt cạnh ở góc và chỗ giao nhau. Biến thể nội suy cho cạnh dày 2 px khi cạnh rơi đúng giữa hai pixel.

**KẾT QUẢ.** `test_nms_tie_rules_on_ideal_step` (skimage giữ {5, 6}, OpenCV giữ {5}). `test_nms_invariant_to_magnitude_scaling`. Đúng từng pixel qua toàn pipeline ([06](06_canny_pipeline.md)).
