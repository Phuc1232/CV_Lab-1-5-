# Quy tắc dữ liệu ảnh

1. **Đầu vào thuật toán:** ảnh xám 2D. `uint8` [0,255] hoặc `float32/64` [0,1]. Kiểm tra bằng `imaging.validate.check_gray`.
2. **Màu:** trong gói luôn là RGB. `imaging.io.load_image` đọc bằng OpenCV rồi đổi BGR→RGB ngay. Chuyển xám dùng $Y=0.299R+0.587G+0.114B$, cùng hệ số với OpenCV.
3. **dtype:**
   - `to_float(img)` đưa `uint8` về [0,1] bằng cách chia 255.
   - `to_uint8(img)` dùng `np.clip(np.rint(img*255), 0, 255)`. Cấm ép kiểu trực tiếp `astype(np.uint8)` trên `float` vì sẽ quấn số.
4. **Miền giá trị:** ảnh `float` nằm ngoài [0,1] (sai số ±1e-6 được chấp nhận rồi clip) thì báo lỗi; không âm thầm co giãn.
5. **Ảnh hỏng:** file không đọc được, ảnh rỗng (0 pixel), kích thước < 3×3, chứa NaN/Inf, dtype int64/bool/complex thì ném `InvalidImageError`. Bộ nạp dataset ghi lại ảnh hỏng vào `skipped`, không dừng cả lô.
6. **Bản đồ cạnh và mặt nạ:** luôn là `bool`. Chuyển từ OpenCV bằng `edges > 0`.
7. **Nhiễu tổng hợp:** dùng `np.random.Generator` có seed; cộng nhiễu trên `float` rồi clip về [0,1].
8. **Metadata cần ghi:** kích thước, số kênh, dtype, min/max/mean/std, nguồn, seed, tham số suy giảm.
