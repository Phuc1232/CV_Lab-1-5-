# Quy tắc toán học

1. Ký hiệu thống nhất với `docs/mathematics/`:
   - $I$: ảnh; $G_\sigma$: Gaussian; $G_x, G_y$: gradient; $M$: độ lớn; $\theta$: hướng; $N$: đầu ra NMS.
   - $T_l, T_h$: ngưỡng; $S, W$: tập mạnh và yếu; $P, G$: cạnh dự đoán và ground truth.
2. Trục: `axis 0` = hàng = $y$ hướng xuống; `axis 1` = cột = $x$. $G_x$ là đạo hàm theo cột.
3. Mọi công thức có giả thiết về miền áp dụng (ví dụ "σ ≥ 2" cho xấp xỉ liên tục) thì ghi rõ giả thiết đó.
4. Mệnh đề mới phải có kiểm chứng số (script hoặc test) và chỉ ghi con số lấy từ lần chạy thật.
5. Phân biệt rõ: **định lý** (có chứng minh), **xấp xỉ** (có sai số đo được), **heuristic** (không có cơ sở và được nói rõ là heuristic).
6. So sánh số thực: không dùng `==` với dữ liệu `float`; dùng dung sai tuyệt đối/tương đối được ghi lý do.
7. Ngưỡng luôn đi kèm thang đo, ví dụ "L2, ảnh 0–255" hoặc "[0,1]".
