# THEORY – CƠ SỞ LÝ THUYẾT XỬ LÝ ẢNH

## Mục lục

- [1. Tổng quan về xử lý ảnh số](#1-tổng-quan-về-xử-lý-ảnh-số)
- [2. Point Operations](#2-point-operations)
- [3. Linear Filtering](#3-linear-filtering)
- [4. Nonlinear Filtering](#4-nonlinear-filtering)
- [5. Edge Detection](#5-edge-detection)
- [6. Noise Models](#6-noise-models)
- [7. Image Quality Assessment](#7-image-quality-assessment)
- [8. So sánh các thuật toán](#8-so-sánh-các-thuật-toán)
- [9. Phạm vi áp dụng và giới hạn diễn giải](#9-phạm-vi-áp-dụng-và-giới-hạn-diễn-giải)
- [10. Tổng kết](#10-tổng-kết)

> **Quy ước trích dẫn:** Các mã `[R1]`, `[R2]`, ... tham chiếu đến danh mục trong [reference.md](./reference.md). Nguồn học thuật được dùng để hỗ trợ nguyên lý; tài liệu OpenCV được dùng để xác nhận hành vi API. Phiên bản tài liệu thư viện không nhất thiết trùng phiên bản runtime của project.

## 1. Tổng quan về xử lý ảnh số

Ảnh số có thể được mô hình hóa thành hàm rời rạc hai chiều \(f(x,y)\), trong đó \(x,y\) là tọa độ pixel và \(f\) biểu diễn cường độ sáng hoặc một vector màu. Ảnh grayscale thường được lưu dưới dạng ma trận có kích thước \(H\times W\); ảnh màu thường có dạng \(H\times W\times C\), với \(C\) là số kênh màu. Các khái niệm nền tảng về ảnh số, lấy mẫu, lượng tử hóa và biểu diễn ảnh được trình bày trong [R6].

Trong project Lab2, phần lớn thuật toán xử lý nhận ảnh `numpy.ndarray` kiểu `uint8`. Với kiểu này, mỗi kênh có giá trị nguyên trong khoảng \([0,255]\). Ảnh grayscale biểu diễn một kênh cường độ. Ảnh màu ba kênh có thể được biểu diễn theo RGB hoặc BGR; hai cách biểu diễn có cùng số lượng kênh nhưng thứ tự ý nghĩa khác nhau. OpenCV thường sử dụng BGR ở các API đọc ảnh mặc định, trong khi nhiều thư viện hiển thị sử dụng RGB. Vì vậy, cần chuyển đổi đúng thứ tự kênh trước khi đổi màu hoặc hiển thị.

## 2. Point Operations

Point Operations là nhóm phép biến đổi trong đó giá trị đầu ra tại một pixel chỉ phụ thuộc vào giá trị đầu vào tại chính pixel đó, không cần lấy mẫu vùng lân cận [R6]. Với ảnh màu, phép biến đổi thường được áp dụng độc lập trên từng kênh.

### 2.1. Brightness Adjustment

Điều chỉnh độ sáng bằng cách cộng một độ lệch \(\beta\) vào cường độ pixel:

\[
g(x,y)=\operatorname{clip}\big(f(x,y)+\beta,\;0,\;255\big).
\]

Trong đó:

- \(f(x,y)\): giá trị pixel đầu vào.
- \(g(x,y)\): giá trị pixel sau biến đổi.
- \(\beta\): độ lệch sáng.
- \(\operatorname{clip}(z,0,255)\): giới hạn \(z\) vào khoảng từ 0 đến 255.

Nếu \(\beta>0\), ảnh sáng hơn; nếu \(\beta<0\), ảnh tối hơn. Phép clipping bảo đảm kết quả nằm trong miền biểu diễn của `uint8`.

### 2.2. Contrast Adjustment

Điều chỉnh tương phản bằng hệ số tỷ lệ \(\alpha\):

\[
g(x,y)=\operatorname{clip}\big(\alpha f(x,y),\;0,\;255\big).
\]

Trong đó \(\alpha>0\) là hệ số tương phản. Khi \(\alpha>1\), khoảng cách cường độ so với mức 0 được khuếch đại; khi \(0<\alpha<1\), dải cường độ bị nén.

Trong project, điều chỉnh độ sáng và tương phản được kết hợp thành một phép biến đổi:

\[
g(x,y)=\operatorname{round}\!\left(
\operatorname{clip}\big(\alpha f(x,y)+\beta,\;0,\;255\big)
\right).
\]

Sau khi clipping, chương trình làm tròn giá trị và chuyển về `uint8`. Các pixel vượt miền biểu diễn bị bão hòa về 0 hoặc 255.

### 2.3. Negative Transformation

Với ảnh có \(L=256\) mức cường độ (ảnh 8-bit), phép biến đổi âm bản được xác định bởi [R6]:

\[
g(x,y)=L-1-f(x,y)=255-f(x,y).
\]

Các pixel có cường độ thấp trở thành cao và ngược lại. Đối với ảnh màu, phép biến đổi được áp dụng độc lập cho từng kênh; đây không phải phép chuyển đổi sang một không gian màu cảm nhận.

### 2.4. Binary Thresholding

Phân ngưỡng nhị phân chuyển ảnh grayscale thành ảnh chỉ có hai mức cường độ. Với ngưỡng \(T\):

\[
g(x,y)=
\begin{cases}
255, & f(x,y)\geq T,\\
0, & f(x,y)<T.
\end{cases}
\]

Ngưỡng \(T\) phân chia pixel thành hai lớp theo cường độ. Trong project, điều kiện là lớn hơn **hoặc bằng** ngưỡng; ảnh đầu vào phải là grayscale. Kết quả chỉ chứa hai giá trị 0 và 255.

## 3. Linear Filtering

### 3.1. Spatial Filtering và Kernel

Lọc không gian tạo giá trị đầu ra tại một vị trí bằng cách kết hợp các pixel trong vùng lân cận. Kernel (hay filter mask) là ma trận trọng số quy định mức đóng góp của từng pixel lân cận [R6].

Trong định nghĩa tích chập hai chiều, phép tính có thể viết:

\[
g(x,y)=\sum_{u=-a}^{a}\sum_{v=-b}^{b}
h(u,v)\,f(x-u,y-v).
\]

Tuy nhiên, hàm `cv2.filter2D()` của OpenCV thực hiện **tương quan chéo (cross-correlation)**: kernel không bị lật trước khi áp dụng [R2]. Với kernel đối xứng, tích chập và tương quan chéo cho cùng kết quả. Với kernel bất đối xứng, chẳng hạn kernel Emboss, hướng và dấu của hiệu ứng có thể khác nếu nhầm lẫn hai phép toán.

Cách hiểu trực quan:

```text
Vùng ảnh 3×3 quanh pixel p       Kernel h 3×3
┌─────┬─────┬─────┐             ┌─────┬─────┬─────┐
│ f11 │ f12 │ f13 │             │ h11 │ h12 │ h13 │
├─────┼─────┼─────┤  nhân và    ├─────┼─────┼─────┤
│ f21 │ f22 │ f23 │  cộng  ───> │ h21 │ h22 │ h23 │
├─────┼─────┼─────┤             ├─────┼─────┼─────┤
│ f31 │ f32 │ f33 │             │ h31 │ h32 │ h33 │
└─────┴─────┴─────┘             └─────┴─────┴─────┘
                         g(p) = Σ h_ij × f_ij
```

Khi vùng lân cận vượt ra ngoài biên ảnh, thuật toán cần quy tắc mở rộng biên (border extrapolation). Các chế độ thường dùng gồm constant, replicate, reflect và reflect-101. OpenCV cung cấp tham số `borderType` cho các phép lọc không gian [R2].

### 3.2. Mean Filter

Mean Filter (box filter) thay pixel trung tâm bằng trung bình cộng của các pixel trong cửa sổ \(k\times k\). Kernel chuẩn hóa có trọng số đồng nhất:

\[
h(u,v)=\frac{1}{k^2},\qquad
 g(x,y)=\frac{1}{k^2}
\sum_{(u,v)\in W_k}f(x+u,y+v).
\]

Trong đó \(W_k\) là tập vị trí thuộc cửa sổ lọc. Với kernel 3×3:

\[
K_{\mathrm{mean}}=\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}.
\]

Mean Filter đơn giản và có chi phí tính toán thấp. Nó có thể làm giảm dao động cường độ nhỏ, nhưng các pixel ngoại lai có ảnh hưởng lớn đến trung bình; đồng thời việc trộn cường độ có thể làm nhòe biên và chi tiết. Khi tăng kích thước kernel, vùng lân cận lớn hơn và mức làm mượt thường tăng [R6], [R2].

### 3.3. Gaussian Filter

Gaussian Filter sử dụng trọng số lớn hơn cho pixel gần tâm kernel và giảm dần theo khoảng cách. Hàm Gaussian liên tục hai chiều có dạng [R6]:

\[
G(x,y)=\frac{1}{2\pi\sigma^2}
\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right),
\qquad \sigma>0.
\]

Trong đó \(x,y\) là độ lệch tọa độ so với tâm và \(\sigma\) là độ lệch chuẩn. \(\sigma\) lớn làm phân bố rộng hơn; kích thước kernel xác định phạm vi mẫu rời rạc được sử dụng.

Một kernel rời rạc 3×3 thường dùng để minh họa xấp xỉ Gaussian là:

\[
K_{\mathrm{Gaussian}}=\frac{1}{16}
\begin{bmatrix}
1&2&1\\
2&4&2\\
1&2&1
\end{bmatrix}.
\]

Ma trận trên là ví dụ minh họa dạng trọng số Gaussian rời rạc, không phải kernel duy nhất cho mọi giá trị \(\sigma\). Trong project, `cv2.GaussianBlur()` nhận kích thước kernel và các tham số sigma. Theo tài liệu OpenCV, kích thước kernel có thể bằng 0 để suy ra từ sigma; nếu sigma bằng 0, OpenCV có thể suy ra sigma từ kích thước kernel [R2].

### 3.4. Sharpen Filter

Sharpen Filter làm nổi bật biến thiên cường độ cục bộ để tăng cảm nhận độ sắc nét. Một kernel được sử dụng trong project là:

\[
K_{\mathrm{sharpen}}=
\begin{bmatrix}
0&-1&0\\
-1&5&-1\\
0&-1&0
\end{bmatrix}.
\]

Trọng số dương ở tâm và trọng số âm ở các pixel lân cận làm tăng thành phần biến thiên. Tổng trọng số của kernel minh họa bằng 1, do đó vùng phẳng lý tưởng được giữ nguyên trước ảnh hưởng của biên, lượng tử hóa và clipping. Sharpen có thể làm nổi bật nhiễu cùng với chi tiết; giá trị đầu ra vượt miền `uint8` có thể bị bão hòa nếu hàm lọc lưu trực tiếp về kiểu này.

### 3.5. Custom Kernel

Custom Kernel cho phép người dùng xác định ma trận trọng số riêng. Một số quy tắc thiết kế thường gặp:

- Kernel làm mờ thường có tổng trọng số bằng 1 để giữ mức cường độ vùng phẳng.
- Kernel sai phân hoặc phát hiện biến thiên thường có tổng trọng số gần 0.
- Trọng số âm và dương kết hợp có thể tăng cường cạnh hoặc tạo hiệu ứng nổi.

Đây là quy tắc thiết kế phổ biến, không phải điều kiện bắt buộc cho mọi kernel. Trong project, các kernel mẫu gồm Box Blur, Sharpen, Laplacian và Emboss. Hàm OpenCV `filter2D()` áp dụng ma trận kernel bằng phép tương quan chéo; `ddepth=-1` giữ độ sâu dữ liệu đầu ra theo ảnh nguồn [R2]. Với ảnh `uint8`, kết quả lưu trực tiếp có thể bị bão hòa về miền 0–255.

## 4. Nonlinear Filtering

### 4.1. Median Filter

Median Filter thay pixel trung tâm bằng trung vị của các giá trị trong vùng lân cận \(W\):

\[
g(x,y)=\operatorname{median}
\left\{f(x+u,y+v):(u,v)\in W\right\}.
\]

Trung vị là phần tử ở giữa sau khi sắp xếp các giá trị. Vì trung vị ít bị chi phối bởi một số giá trị ngoại lai cực lớn hoặc cực nhỏ hơn trung bình cộng, Median Filter thường hữu ích khi xử lý nhiễu xung (impulse noise), gồm Salt-and-Pepper Noise [R6]. Tuy vậy, cửa sổ lớn có thể loại bỏ chi tiết mảnh hoặc làm biến dạng cấu trúc nhỏ.

```text
Cửa sổ pixel          Sắp xếp các giá trị        Giá trị trung vị
┌────┬────┬────┐       0, 10, 11, 12, 12, ...   ┌────┬────┬────┐
│ 10 │ 12 │ 11 │       12, 12, 13, 14, 255      │ .. │ .. │ .. │
├────┼────┼────┤  ──────────────────────────>   ├────┼────┼────┤
│  0 │ 13 │ 12 │                                │ .. │ 12 │ .. │
├────┼────┼────┤                                ├────┼────┼────┤
│ 12 │255 │ 14 │                                │ .. │ .. │ .. │
└────┴────┴────┘                                └────┴────┴────┘
```

Các số trong sơ đồ chỉ minh họa việc chọn trung vị, không phải kết quả đo từ ảnh Lab2. Trong OpenCV, `medianBlur()` áp dụng median filter trên vùng lân cận; với ảnh nhiều kênh, phép lọc được thực hiện độc lập theo kênh [R2].

### 4.2. Bilateral Filter

Bilateral Filter kết hợp hai loại trọng số: trọng số không gian và trọng số tương đồng cường độ. Dạng tổng quát cho ảnh grayscale có thể viết [R7]:

\[
w(p,q)=
\exp\!\left(-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}\right)
\exp\!\left(-\frac{(I(p)-I(q))^2}{2\sigma_r^2}\right),
\]

\[
I_{\mathrm{out}}(p)=
\frac{\displaystyle\sum_{q\in\Omega_p}w(p,q)I(q)}
{\displaystyle\sum_{q\in\Omega_p}w(p,q)}.
\]

Trong đó:

- \(p,q\): vị trí pixel.
- \(I(p),I(q)\): cường độ tại các vị trí tương ứng.
- \(\sigma_s\): độ lệch chuẩn không gian, chi phối ảnh hưởng của khoảng cách vị trí.
- \(\sigma_r\): độ lệch chuẩn cường độ, chi phối mức khác biệt cường độ được chấp nhận.
- \(\Omega_p\): vùng lân cận được xét quanh pixel \(p\).

Pixel càng xa về không gian hoặc càng khác biệt về cường độ thường nhận trọng số càng nhỏ. Vì vậy, Bilateral Filter có thể làm mượt trong vùng tương đối đồng nhất mà hạn chế trộn qua một số biên. Hiệu quả phụ thuộc ảnh và tham số; bộ lọc không bảo đảm loại bỏ mọi loại nhiễu. Bài báo của Tomasi và Manduchi trình bày phương pháp bilateral cho ảnh grayscale và ảnh màu [R7].

## 5. Edge Detection

### 5.1. Gradient và đạo hàm ảnh

Gradient của ảnh vô hướng \(f(x,y)\) biểu diễn tốc độ thay đổi cường độ theo hai trục:

\[
\nabla f=
\begin{bmatrix}
\dfrac{\partial f}{\partial x}\\[2pt]
\dfrac{\partial f}{\partial y}
\end{bmatrix}
=
\begin{bmatrix}G_x\\G_y\end{bmatrix},
\qquad
G=\sqrt{G_x^2+G_y^2}.
\]

\(G_x\) và \(G_y\) là các thành phần gradient theo hai hướng; \(G\) là độ lớn gradient. Vùng có độ lớn gradient cao thường tương ứng với biến thiên cường độ mạnh. Vì ảnh số là rời rạc, đạo hàm được xấp xỉ bằng các toán tử/kernel rời rạc [R6].

```text
Ảnh xám f(x,y) ── Kx ──> Gx ──┐
                              ├── sqrt(Gx² + Gy²) ──> độ lớn gradient
             └── Ky ──> Gy ──┘
```

**Phân biệt quan trọng:** ảnh độ lớn gradient không đồng nghĩa với ảnh cạnh nhị phân. Trong project, hai hàm Prewitt và Sobel tính độ lớn gradient rồi chuẩn hóa min–max về `[0,255]`; chúng không thực hiện bước threshold để tạo mặt nạ cạnh chỉ gồm 0 và 255.

### 5.2. Prewitt Operator

Prewitt sử dụng hai kernel 3×3 để xấp xỉ biến thiên theo hai hướng:

\[
K_x=
\begin{bmatrix}
-1&0&1\\
-1&0&1\\
-1&0&1
\end{bmatrix},
\qquad
K_y=
\begin{bmatrix}
-1&-1&-1\\
0&0&0\\
1&1&1
\end{bmatrix}.
\]

Áp dụng các kernel để thu \(G_x\) và \(G_y\), sau đó tính độ lớn gradient. Chương trình Lab2 sử dụng dữ liệu trung gian `float32` để giữ được giá trị gradient có dấu, rồi chuẩn hóa độ lớn thành ảnh `uint8`. Công trình Prewitt được trình bày trong chương “Object Enhancement and Extraction” [R8].

### 5.3. Sobel Operator

Sobel dùng các kernel có hệ số 2 ở hàng hoặc cột giữa:

\[
K_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
K_y=
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}.
\]

Cấu trúc trọng số này kết hợp phép lấy sai phân theo một hướng với làm trơn theo hướng vuông góc. Project gọi `cv2.Sobel()` cho hai thành phần đạo hàm; OpenCV cho phép chỉ định bậc đạo hàm và kích thước aperture/kernel [R2]. Sau đó project tính độ lớn gradient và chuẩn hóa min–max trên toàn ảnh. Kết quả là ảnh cường độ gradient, không phải bản đồ cạnh nhị phân.

## 6. Noise Models

### 6.1. Salt-and-Pepper Noise

Salt-and-Pepper Noise (nhiễu muối tiêu) là dạng nhiễu xung, trong đó một số pixel bị thay bằng giá trị cực tiểu hoặc cực đại. Mô hình khái quát cho ảnh 8-bit:

\[
I_n(i)=
\begin{cases}
0, & i\in S_0,\\
255, & i\in S_{255},\\
I(i), & \text{các vị trí còn lại}.
\end{cases}
\]

Trong mô hình xác suất tổng quát, xác suất thay pixel bằng 0 và 255 có thể được mô tả bởi \(P(I_n=0)=p_0\), \(P(I_n=255)=p_{255}\), và \(P(I_n=I)=1-p_0-p_{255}\), với \(p_0+p_{255}\leq 1\). Tuy nhiên, helper trong `tests/test_image_processing.py` của Lab2 triển khai theo cách chọn số lượng cố định: chọn \(n=\lfloor N\cdot a\rfloor\) vị trí không lặp sau khi flatten ảnh, với \(N\) là tổng số phần tử và `amount` \(a=0.05\) theo mặc định; sau đó chia ngẫu nhiên các vị trí thành nhóm gán 0 và nhóm gán 255. Vì vậy, đây là cách tạo nhiễu với số vị trí được chọn cố định, không phải phép thử Bernoulli độc lập tại từng pixel.

### 6.2. Gaussian Noise

Gaussian Noise là nhiễu cộng có phân phối chuẩn:

\[
I_n(x,y)=\operatorname{clip}
\big(I(x,y)+n(x,y),\;0,\;255\big),
\qquad
n(x,y)\sim\mathcal{N}(\mu,\sigma^2).
\]

Trong đó \(\mu\) là trung bình và \(\sigma\) là độ lệch chuẩn của nhiễu. Helper của Lab2 sử dụng \(\mu=0\), \(\sigma=20\) theo mặc định; cộng nhiễu trên dữ liệu `float32`, giới hạn về `[0,255]` rồi chuyển về `uint8`. Cả hai helper tạo nhiễu trong test đều khởi tạo bộ sinh ngẫu nhiên với seed 42, nhờ đó có thể tái tạo kết quả khi dùng cùng đầu vào và tham số.

## 7. Image Quality Assessment

Các chỉ số sai số cần được tính giữa ảnh tham chiếu và ảnh kết quả có cùng kích thước, số kênh và quy ước biểu diễn. Gọi \(N\) là tổng số phần tử pixel được so sánh trên toàn bộ ảnh/kênh; \(I_i\) là phần tử thứ \(i\) của ảnh tham chiếu và \(K_i\) là phần tử tương ứng của ảnh kết quả.

### 7.1. Mean Squared Error (MSE)

\[
\operatorname{MSE}=\frac{1}{N}\sum_{i=1}^{N}(I_i-K_i)^2.
\]

MSE là trung bình bình phương sai khác giữa các phần tử tương ứng. Sai khác lớn bị phạt mạnh hơn sai khác nhỏ. Giá trị tốt nhất là 0, đạt được khi hai ảnh giống hệt nhau. **MSE càng thấp càng tốt** khi cùng một ảnh tham chiếu và cùng quy ước dữ liệu được sử dụng.

### 7.2. Mean Absolute Error (MAE)

\[
\operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|I_i-K_i|.
\]

MAE là sai khác tuyệt đối trung bình, có đơn vị cùng thang cường độ pixel. Giá trị tốt nhất là 0. **MAE càng thấp càng tốt** khi các điều kiện so sánh giống nhau.

### 7.3. Peak Signal-to-Noise Ratio (PSNR)

Với ảnh `uint8`, giá trị cực đại mỗi kênh là \(L=255\). PSNR được tính từ MSE:

\[
\operatorname{PSNR}=10\log_{10}
\left(\frac{L^2}{\operatorname{MSE}}\right)\;\text{dB},
\qquad L=255.
\]

PSNR biểu diễn tỷ số tín hiệu cực đại so với sai số bình phương trung bình theo thang logarit. Khi MSE giảm, PSNR tăng. Nếu MSE bằng 0, hai ảnh giống hệt nhau và PSNR tiến tới \(+\infty\). **PSNR càng cao càng tốt** khi dùng cùng ảnh tham chiếu, dải động và cách tính.

### 7.4. Execution Time

Nếu \(T_j\) là thời gian đo của lần chạy thứ \(j\) (đơn vị giây) và \(r\) là số lần lặp được đo, thời gian trung bình tính bằng mili giây là:

\[
\overline{T}_{\mathrm{ms}}=
\frac{1000}{r}\sum_{j=1}^{r}T_j.
\]

Trong `compare_filters()` của Lab2, mỗi bộ lọc được gọi một lần warm-up trước khi đo; lần warm-up không được tính vào trung bình. Các lần đo dùng `time.perf_counter()`. **Execution Time càng thấp càng nhanh**, nhưng chỉ có ý nghĩa so sánh khi điều kiện chạy tương đương: cùng ảnh, độ phân giải, cấu hình tham số, môi trường phần mềm/phần cứng và cách benchmark.

### 7.5. Giới hạn của các chỉ số

MSE, MAE và PSNR là các chỉ số dựa trên sai khác pixel so với ảnh tham chiếu. Chúng hữu ích cho đánh giá full-reference nhưng không mô hình hóa đầy đủ cảm nhận thị giác của con người. Một ảnh có sai số pixel thấp hơn không nhất thiết luôn được người xem đánh giá đẹp hơn trong mọi tình huống. Wang và cộng sự đề xuất Structural Similarity Index (SSIM) như một cách đánh giá bổ sung dựa trên suy giảm thông tin cấu trúc [R9]. **SSIM không được triển khai trong Lab2**; tài liệu chỉ nêu để chỉ ra rằng MSE/MAE/PSNR không phải toàn bộ các khía cạnh của chất lượng cảm nhận.

## 8. So sánh các thuật toán

Bảng dưới đây tổng hợp đặc điểm lý thuyết của các phương pháp, không phải thứ hạng đo được trên ảnh Lab2.

| Phương pháp | Nguyên lý | Khả năng làm mượt/khử nhiễu | Bảo toàn biên | Ưu điểm | Hạn chế | Trường hợp thường phù hợp |
|---|---|---|---|---|---|---|
| Mean | Trung bình đều trong cửa sổ | Giảm dao động cường độ nhỏ | Thấp; dễ trộn qua biên | Đơn giản, chi phí thấp | Nhòe chi tiết; nhạy với pixel ngoại lai | Làm mượt tổng quát, nhiễu cường độ nhẹ |
| Gaussian | Trung bình có trọng số Gaussian | Làm mượt thành phần biến thiên tần số cao | Thấp đến trung bình | Trọng số giảm theo khoảng cách, có thể tách thành các kernel 1D | Vẫn làm mờ biên; phụ thuộc sigma và kích thước kernel | Làm mượt trước các bước phân tích ảnh |
| Median | Trung vị trong vùng lân cận | Thường hiệu quả với nhiễu xung cực trị | Thường giữ cấu trúc biên tốt hơn mean trong nhiều trường hợp | Ít bị ảnh hưởng bởi giá trị ngoại lai đơn lẻ | Có thể loại bỏ chi tiết mảnh; phụ thuộc kích thước cửa sổ | Salt-and-Pepper Noise |
| Bilateral | Kết hợp khoảng cách không gian và tương đồng cường độ | Làm mượt trong vùng có cường độ tương đồng | Có xu hướng hạn chế làm mờ qua biên | Kết hợp thông tin không gian và cường độ | Nhạy với tham số; thường tốn tính toán hơn lọc tuyến tính | Làm mượt khi cần giữ biên |
| Sobel/Prewitt | Xấp xỉ đạo hàm theo X/Y | Không phải bộ khử nhiễu | Làm nổi bật vùng biến thiên cường độ | Tạo thành phần gradient và độ lớn gradient | Nhạy với nhiễu; cần tiền xử lý/threshold tùy mục tiêu | Phân tích biên sau tiền xử lý phù hợp |

Không có bộ lọc nào luôn tốt nhất cho mọi ảnh, mọi loại nhiễu và mọi mục tiêu. Việc lựa chọn cần căn cứ vào loại nhiễu, mức bảo toàn chi tiết mong muốn, tham số và kết quả thực nghiệm.

## 9. Phạm vi áp dụng và giới hạn diễn giải

1. Các công thức trong tài liệu mô tả nguyên lý chung. Hành vi cụ thể của Lab2 còn phụ thuộc cách gọi OpenCV, tham số mặc định, quy tắc validation, kiểu dữ liệu và bước chuẩn hóa đầu ra; các chi tiết này được mô tả trong [algorithm.md](./algorithm.md).
2. Khi đánh giá khả năng khử nhiễu, ảnh tham chiếu nên là ảnh sạch tương ứng (ground truth). Nếu so ảnh sau lọc với ảnh nhiễu đầu vào, MSE/MAE/PSNR chỉ đo mức khác biệt so với ảnh nhiễu, không trực tiếp đo mức phục hồi ảnh sạch.
3. Kết luận thực nghiệm chỉ có thể áp dụng cho dữ liệu, tham số, seed, cấu hình và quy trình đo đã ghi trong [experimentLog.md](./experimentLog.md). Không suy rộng một lượt benchmark trên một ảnh thành kết luận phổ quát.
4. Tài liệu tham khảo xác định tại [reference.md](./reference.md). Nguồn học thuật đã xác minh và nguồn tài liệu API được phân loại riêng; không xem tài liệu API là bằng chứng cho nguồn gốc lịch sử của mọi thuật toán.

## 10. Tổng kết

Point Operations biến đổi độc lập từng pixel; Linear Filtering kết hợp các pixel lân cận bằng trọng số; Median và Bilateral là các bộ lọc phi tuyến; Sobel và Prewitt ước lượng gradient để làm nổi bật vùng biến thiên cường độ. MSE, MAE, PSNR và Execution Time hỗ trợ đánh giá sai số và hiệu năng, nhưng việc diễn giải cần dùng đúng ảnh tham chiếu và điều kiện thực nghiệm. Phần triển khai cụ thể của Lab2 nằm trong [algorithm.md](./algorithm.md), nguồn tài liệu nằm trong [reference.md](./reference.md), còn kết quả đo thực tế nằm trong [experimentLog.md](./experimentLog.md).
