# 08. Canny ảnh màu (E3)

**WHAT / WHY / MATHEMATICS.** Xem [../mathematics/07](../mathematics/07_canny_anh_mau_va_video.md).

| method | Cách làm | Ghi chú |
|---|---|---|
| `gray` | Y = 0.299R + 0.587G + 0.114B rồi Canny | mất cạnh đẳng sáng |
| `or` | Canny từng kênh, hợp kết quả | hợp nhiễu của 3 kênh |
| `max_channel` | mỗi pixel lấy (gx, gy) của kênh có độ lớn lớn nhất | ≡ `cv2.Canny` 3 kênh với hồ sơ OPENCV |
| `di_zenzo` | $\sqrt{\lambda_{max}}$ của ma trận cấu trúc; hướng $\tfrac12\operatorname{atan2}(2g_{xy},g_{xx}-g_{yy})$ | cần NMS nội suy (hồ sơ SKIMAGE) |

**IMPLEMENTATION.** [`core/color.py`](../../src/cannylab/core/color.py). Phương pháp `max_channel` và `di_zenzo` tạo gradient rồi dùng chung `canny_from_gradient` (bước 3–5) với ảnh xám.

**COMPLEXITY.** Khoảng 3 lần chi phí gradient của ảnh xám.

**LIMITATIONS.** Ngưỡng của Di Zenzo khác thang với ảnh xám: 3 kênh giống nhau cho độ lớn gấp √3 lần.

**KẾT QUẢ.**
- `test_color_max_channel_matches_opencv_3channel`: 0 px lệch, cả L1 lẫn L2.
- `test_di_zenzo_gray_image_equals_gray_canny`: lệch ≤ 0.1% (điểm hòa do làm tròn cos/sin).
- `test_isoluminant_edges_lost_in_gray_but_found_in_color`: chuyển xám cho 0 px cạnh, Di Zenzo cho > 100 px.
- Định lượng ở X7.
