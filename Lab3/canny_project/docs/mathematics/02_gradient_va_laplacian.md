# 02. Gradient (bước 2 của Canny) và so sánh với Laplacian (câu T1b)

Số liệu: [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), mục `2_gradient`, `3_nguong_opencv`.

## WHAT
Ước lượng vector gradient $\nabla I=(I_x,I_y)$ tại mỗi pixel, từ đó suy ra **độ lớn** $M$ (độ mạnh của cạnh) và **hướng** $\theta$ (pháp tuyến của cạnh).

## WHY
Cạnh là nơi cường độ thay đổi nhanh nhất. Theo định nghĩa, $\nabla I$ chỉ về hướng tăng nhanh nhất và $\lVert\nabla I\rVert$ cho biết tốc độ tăng. Hướng θ là thông tin bắt buộc của bước NMS.

## HOW

### Sai phân hữu hạn
Khai triển Taylor:
$$\frac{I(x+1)-I(x-1)}{2}=I'(x)+\frac{I'''(x)}{6}+O(h^4)$$
Sai phân trung tâm có sai số bậc 2. Kernel tương ứng (bỏ hệ số ½) là $d=[-1,0,1]$.

### Sobel = làm trơn ⊗ đạo hàm
$$K_x=\begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}-1&0&1\end{bmatrix}=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\qquad K_y=K_x^\top$$
- $[1,2,1]$ là Gauss nhị thức bậc 2, đáp ứng tần số $4\cos^2(\omega/2)$, bằng 0 tại tần số Nyquist. Thành phần này làm trơn theo hướng vuông góc với đạo hàm.
- $[-1,0,1]$ có đáp ứng $2i\sin\omega$, xấp xỉ đạo hàm lý tưởng $i\omega$ ở tần số thấp.
- **Hệ số khuếch đại 8:** trên mặt dốc $I=x$, ta có $K_x*I=2\cdot4=8$. Đo được đúng 8.000.
- **Đỉnh trên cạnh bậc thang** cao $h$ là $4h$. Đo được 4.000. Với ảnh `uint8` ($h\le255$), $|G_x|\le1020$, nên kết quả **bắt buộc** phải lưu ở `int16`/`float`, không lưu ở `uint8` (quy tắc tránh tràn số).
- Prewitt dùng $[1,1,1]$ thay $[1,2,1]$: làm trơn yếu hơn và kém đẳng hướng hơn.

`scipy.ndimage.sobel`, công cụ skimage dùng, **trùng hoàn toàn** với `cv2.Sobel(ksize=3)` ở phần trong ảnh (sai lệch tối đa 0.0). Hai thư viện chỉ khác nhau ở biên và ở thang giá trị ảnh ([05](05_khac_biet_opencv_skimage.md)).

### Độ lớn và hướng
$$M_{L2}=\sqrt{G_x^2+G_y^2},\qquad M_{L1}=|G_x|+|G_y|,\qquad\theta=\operatorname{atan2}(G_y,G_x)$$

**Bất đẳng thức chuẩn:** $M_{L2}\le M_{L1}\le\sqrt2\,M_{L2}$. Dấu bằng thứ nhất xảy ra khi cạnh nằm ngang hoặc dọc, dấu bằng thứ hai khi cạnh chéo 45°. Đo trên `camera`: tỉ lệ min 1.0000, max 1.4142, trung bình 1.259.

**Hệ quả:** chuẩn L1 (mặc định của `cv2.Canny`) **không đẳng hướng**. Cùng một độ tương phản, cạnh chéo được khuếch đại √2 lần so với cạnh ngang. Ngưỡng L1 vì thế ưu ái cạnh chéo. Không tồn tại hệ số quy đổi chính xác giữa ngưỡng L1 và L2; chỉ có khoảng $[T,\sqrt2T]$.

**Quy ước tọa độ:** trục $y$ của ảnh hướng xuống. NMS chỉ cần hướng theo modulo π (θ và θ+π cho cùng một cặp lân cận), nên dấu của $y$ không ảnh hưởng đến NMS. Tuy vậy dấu này phải nhất quán khi vẽ vector gradient hoặc khi dùng θ cho đặc trưng hình dạng.

### Khuếch đại nhiễu
Với nhiễu trắng σₙ, $\operatorname{std}(K*n)=\sigma_n\sqrt{\sum K_{ij}^2}$:

| Toán tử | $\sum K^2$ | std lý thuyết | std đo |
|---|---|---|---|
| Sobel $K_x$ | 12 | 3.464 | 3.453 |
| Laplacian 4 lân cận | 20 | 4.472 | 4.473 |

SNR của Sobel đơn (không có Gaussian) trên cạnh bậc thang: $4h/(\sqrt{12}\sigma_n)\approx1.15\,h/\sigma_n$. Có Gaussian σ thì SNR tăng lên khoảng $2h\sigma/\sigma_n$ ([01](01_lam_tron_gaussian.md)).

## So sánh với Laplacian / LoG (T1b)

### Laplacian
$$\nabla^2I=I_{xx}+I_{yy},\qquad L=\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$$
- Đẳng hướng, không cho hướng, là một số vô hướng. Cạnh nằm ở **điểm qua 0** (zero-crossing) của $\nabla^2I$, vì đó là điểm uốn của mặt cắt cường độ.
- Đáp ứng tần số $-(\omega_x^2+\omega_y^2)$, tức tăng theo **bình phương** tần số, trong khi đạo hàm bậc 1 chỉ tăng tuyến tính. Do đó Laplacian nhạy nhiễu hơn hẳn: phương sai nhiễu đầu ra lớn hơn Sobel (20 so với 12), và nhiễu tần số cao chiếm ưu thế.
- **Cạnh ảo (phantom edges):** zero-crossing xuất hiện cả ở những chỗ cường độ thay đổi từ từ (gradient nhỏ). Phải kiểm tra thêm độ lớn gradient mới loại được.
- Ưu điểm: zero-crossing luôn tạo đường **khép kín** và mảnh 1 pixel mà không cần NMS.

### LoG (Marr–Hildreth, 1980)
$$\nabla^2(G_\sigma*I)=(\nabla^2G_\sigma)*I,\qquad\nabla^2G_\sigma(x,y)=-\frac{1}{\pi\sigma^4}\Big(1-\frac{x^2+y^2}{2\sigma^2}\Big)e^{-\frac{x^2+y^2}{2\sigma^2}}$$
Có thể xấp xỉ bằng hiệu hai Gauss (DoG) với tỉ lệ $\sigma_2/\sigma_1\approx1.6$.

### Bảng so sánh lý thuyết
Thực nghiệm kiểm chứng bảng này sẽ làm ở Phase 7–8, đo trên ảnh tổng hợp có ground truth.

| Tiêu chí | Sobel + ngưỡng | Laplacian / LoG + zero-crossing | Canny |
|---|---|---|---|
| Bậc đạo hàm | 1 | 2 | 1 (đạo hàm của Gaussian) |
| Độ dày cạnh | Dày 2–3 px | 1 px | 1 px (nhờ NMS) |
| Hướng cạnh | Có | Không | Có, và được dùng trong NMS |
| Liên tục | Đứt hoặc dính, phụ thuộc 1 ngưỡng | Khép kín nhưng nhiều cạnh ảo | Nối bằng hysteresis (2 ngưỡng) |
| Nhạy nhiễu | Trung bình | Cao (Laplacian), trung bình (LoG) | Thấp, điều khiển được qua σ |
| Chi phí | Thấp nhất ($O(N)$, 2 tích chập) | Thấp ($O(N)$) | Cao hơn: thêm NMS và hysteresis, nhưng vẫn $O(N)$ |
| Số tham số | 1 | 1–2 | 3 (σ, $T_l$, $T_h$) |

## PARAMETERS
`apertureSize` (3/5/7) trong OpenCV là kích thước Sobel; kernel lớn hơn thì làm trơn mạnh hơn và độ lớn có thang khác. Chuẩn L1/L2. Kiểu dữ liệu đầu ra (`int16`/`float32`).

## COMPLEXITY
Sobel tách được: khoảng $6N$ phép cộng/nhân cho mỗi hướng. `atan2` và `sqrt` tốn $O(N)$ nhưng hằng số lớn. OpenCV tránh hoàn toàn `atan2` bằng thủ thuật so sánh với tan 22.5° ([03](03_non_maximum_suppression.md)).

## LIMITATIONS
- Sobel 3×3 có sai số góc nhỏ nhưng khác 0. Scharr ($[3,10,3]$) đẳng hướng hơn.
- Gradient không phân biệt được cạnh vật thể với cạnh bóng đổ hay texture. Đây là giới hạn chung của mọi phương pháp cục bộ.
