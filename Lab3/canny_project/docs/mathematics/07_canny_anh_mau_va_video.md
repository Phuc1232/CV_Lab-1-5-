# 07. Canny cho ảnh màu (E3) và video (E4)

Số liệu: [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), mục `8_anh_mau`.

## Phần A: Ảnh màu (E3)

### WHY: chuyển sang ảnh xám có thể làm mất cạnh
Ảnh xám $Y=0.299R+0.587G+0.114B$ là một phép chiếu tuyến tính $\mathbb R^3\to\mathbb R$. Mọi cặp màu nằm trên cùng một mặt phẳng $Y=\text{const}$ (**đẳng sáng**, isoluminant) sẽ bị chiếu vào cùng một giá trị xám.

**Ví dụ đã kiểm chứng:** hai màu (R, G, B) = (200, 100, 50) và (55, 170, 70) có $Y=124.20$ và $124.21$. Khác nhau rõ về màu nhưng ảnh xám gần như phẳng: Canny trên ảnh xám cho **0 pixel** cạnh, trong khi `cv2.Canny` trên ảnh 3 kênh cho **64 pixel** (đủ một đường cạnh 64 hàng).

### HOW: các cách mở rộng
| # | Cách | Công thức | Nhận xét |
|---|---|---|---|
| 1 | Xám rồi Canny | $\nabla Y$ | Mất cạnh đẳng sáng |
| 2 | Canny từng kênh rồi OR | $E=E_R\cup E_G\cup E_B$ | Hợp nhiễu của 3 kênh; cạnh có thể lệch nhau 1 px giữa các kênh nên bị nhân đôi |
| 3 | Chọn kênh có gradient lớn nhất (cách của OpenCV) | $c^*(p)=\arg\max_c M_{L1}^{(c)}(p)$, dùng $(G_x^{(c^*)},G_y^{(c^*)})$ cho NMS + hysteresis | **Đã kiểm chứng:** `cv2.Canny` trên ảnh 3 kênh trùng **100%** (0 px lệch trên `astronaut`) với cách này, khi Sobel dùng biên replicate |
| 4 | Gradient Di Zenzo (structure tensor) | xem bên dưới | Có cơ sở hình học; cộng gộp thông tin của cả 3 kênh |

Trên `astronaut`: IoU(cách 3, cách 2) = 0.773, IoU(cách 3, cách 1) = 0.718. Ba cách cho kết quả thực sự khác nhau.

### MATHEMATICS: Di Zenzo (1986)
Xét ảnh màu như ánh xạ $f:\mathbb R^2\to\mathbb R^3$. Bình phương độ biến thiên theo hướng đơn vị $\mathbf n$ là $\lVert J\mathbf n\rVert^2=\mathbf n^\top(J^\top J)\mathbf n$, với ma trận cấu trúc
$$J^\top J=\begin{bmatrix}g_{xx}&g_{xy}\\g_{xy}&g_{yy}\end{bmatrix},\quad g_{xx}=\sum_c(\partial_xI_c)^2,\ g_{yy}=\sum_c(\partial_yI_c)^2,\ g_{xy}=\sum_c\partial_xI_c\,\partial_yI_c$$
Hướng biến thiên lớn nhất là vector riêng ứng với trị riêng lớn nhất:
$$\lambda_+=\tfrac12\Big(g_{xx}+g_{yy}+\sqrt{(g_{xx}-g_{yy})^2+4g_{xy}^2}\Big),\qquad\theta=\tfrac12\operatorname{atan2}(2g_{xy},\,g_{xx}-g_{yy})$$
Thay $(M,\theta)$ của Canny xám bằng $(\sqrt{\lambda_+},\theta)$ rồi chạy NMS và hysteresis như cũ.

- Với 1 kênh: $\lambda_+=I_x^2+I_y^2$, tức trở về Canny xám (kiểm thử được).
- Gradient các kênh ngược chiều nhau không triệt tiêu nhau, khác với lấy tổng gradient, vì ma trận cấu trúc là dạng toàn phương.
- θ chỉ xác định theo modulo π, nhưng NMS cũng chỉ cần modulo π.

**Không gian màu:** khoảng cách Euclid trong RGB không phản ánh khác biệt cảm nhận. Chạy Di Zenzo trên CIELAB thì $\sqrt{\lambda_+}$ gần với $\Delta E$ hơn. Khi dùng phải kiểm soát chặt kiểu dữ liệu: L\*a\*b\* `uint8` của OpenCV đã bị dịch và co giãn (xem Lab1), nên phải đổi sang `float` trước.

## Phần B: Video (E4)

### HOW
Video là dãy khung hình $I_t$, $t=1..T$. Cách cơ bản là chạy Canny độc lập trên từng khung hình theo lô từ file video. Không làm webcam hay hiển thị realtime (ngoài phạm vi dự án).

### Vấn đề toán học: nhấp nháy (flicker)
Một pixel có $M_t$ dao động quanh ngưỡng (do nhiễu cảm biến hoặc nén) sẽ bật/tắt giữa các khung hình. Với $M_t=\mu+\varepsilon_t$, $\varepsilon_t\sim\mathcal N(0,s^2)$ độc lập theo thời gian, xác suất trạng thái đổi giữa 2 khung hình liên tiếp là $2p(1-p)$ với $p=P(M_t>T)$. Xác suất này lớn nhất khi $\mu\approx T$.

### Giải pháp (ứng viên, Phase 3 sẽ chọn)
1. **Làm trơn theo thời gian** trước Canny, bằng Gaussian 3D với $\sigma_t$. Phương sai nhiễu giảm thêm $\propto1/\sigma_t$ nhưng vật chuyển động bị nhòe.
2. **Hysteresis theo thời gian:** pixel là cạnh ở khung $t-1$ thì được dùng ngưỡng thấp hơn ở khung $t$. Đây là mở rộng tự nhiên của bước 5 sang trục thời gian.
3. **Ngưỡng toàn cục cố định** thay vì ngưỡng thích nghi theo từng khung hình (phân vị hoặc Otsu), để tránh ngưỡng tự nhảy theo nội dung.

### Chỉ số
- **Độ ổn định thời gian** trên cảnh tĩnh: $\overline{\text{IoU}_\tau(E_t,E_{t+1})}$, với IoU có dung sai như ở [06](06_chi_so_danh_gia.md).
- **Thông lượng:** số khung hình/giây theo kích thước khung.

Dữ liệu video sẽ được **tổng hợp**: một cảnh tĩnh cộng nhiễu độc lập mỗi khung hình, và một vật chuyển động với quỹ đạo đã biết (biết trước ground truth). Như vậy không cần tải video từ ngoài.

## LIMITATIONS
- Di Zenzo tốn gấp khoảng 3 lần chi phí gradient, và ngưỡng có thang khác ngưỡng xám nên phải chọn lại.
- Hysteresis theo thời gian làm cạnh "trễ" khi vật chuyển động nhanh.
