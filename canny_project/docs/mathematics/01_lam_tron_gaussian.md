# 01. Làm trơn Gaussian (bước 1 của Canny)

Số liệu trong file này lấy từ [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), mục `1_gaussian`, `2_gradient`, `7_snr_gradient`.

## WHAT
Tích chập ảnh với hàm Gauss 2D để làm suy giảm thành phần tần số cao, gồm cả nhiễu và chi tiết nhỏ.

## WHY
Đạo hàm khuếch đại tần số cao (mục [02](02_gradient_va_laplacian.md)). Nếu lấy gradient trực tiếp trên ảnh nhiễu thì nhiễu tạo ra hàng loạt cạnh giả. Canny (1986) chứng minh rằng bộ lọc tối ưu cho cạnh bậc thang trong nhiễu trắng xấp xỉ **đạo hàm của Gaussian**, tức "làm trơn Gaussian rồi lấy đạo hàm".

## HOW

### Liên tục
$$g_\sigma(x)=\frac{1}{\sqrt{2\pi}\,\sigma}e^{-x^2/(2\sigma^2)},\qquad G_\sigma(x,y)=g_\sigma(x)\,g_\sigma(y),\qquad I_s=G_\sigma*I$$

**Tính tách được:** vì $G_\sigma(x,y)=g_\sigma(x)g_\sigma(y)$ nên $G_\sigma*I=g_\sigma^{(x)}*(g_\sigma^{(y)}*I)$. Một tích chập 2D kích thước $k\times k$ được thay bằng hai tích chập 1D kích thước $k$.

**Tính nửa nhóm (semigroup):** $G_{\sigma_1}*G_{\sigma_2}=G_{\sqrt{\sigma_1^2+\sigma_2^2}}$. Một ảnh vốn đã mờ với $\sigma_0$, sau khi làm trơn thêm $\sigma$, có độ mờ hiệu dụng $\sqrt{\sigma_0^2+\sigma^2}$.

**Miền tần số:** $\hat g_\sigma(\omega)=e^{-\sigma^2\omega^2/2}$. Đây là bộ lọc thông thấp không gợn sóng (không có ringing), tần số cắt cỡ $1/\sigma$.

### Rời rạc
Lấy mẫu $g_\sigma$ tại các số nguyên $x\in[-r,r]$ rồi **chuẩn hóa cho tổng bằng 1**, để vùng ảnh phẳng giữ nguyên độ sáng:
$$\tilde g[x]=\frac{g_\sigma(x)}{\sum_{u=-r}^{r}g_\sigma(u)}$$

Phần khối lượng bị cắt bỏ ngoài $[-k\sigma,k\sigma]$ bằng $\operatorname{erfc}(k/\sqrt2)$: khoảng 0.27% với $k=3$ và khoảng $6.3\times10^{-5}$ với $k=4$.

| Thư viện | Bán kính $r$ | Ghi chú (đã kiểm chứng) |
|---|---|---|
| OpenCV `GaussianBlur(ksize=(0,0))`, ảnh `uint8` | $k = \operatorname{round}(6\sigma+1)\,\lvert\,1$ (tức $r\approx3\sigma$) | Trùng kết quả với kernel dự đoán ở mọi σ thử. Với σ = 0.5 và 1.5, kernel nhỏ hơn 2 đơn vị **cũng** cho kết quả trùng, vì hệ số ngoài cùng bị làm tròn về 0 trong số học điểm cố định |
| OpenCV, ảnh `float32` | $k = \operatorname{round}(8\sigma+1)\,\lvert\,1$ ($r\approx4\sigma$) | Trùng chính xác; kernel nhỏ hơn thì khác |
| SciPy / skimage | $r=\lfloor4\sigma+0.5\rfloor$ (`truncate=4`) | skimage `canny` dùng `mode='constant'` **có chuẩn hóa bleed-over**: chia cho Gaussian của mặt nạ toàn 1, nên ở biên ảnh tương đương "trung bình có trọng số trên phần ảnh có thật", không phải đệm 0 |

Với σ nhỏ (dưới khoảng 0.8), Gauss lấy mẫu là một xấp xỉ kém, vì kernel gần như chỉ còn tâm. Ví dụ σ = 0.5 cho $[0.107, 0.787, 0.107]$.

## MATHEMATICS: tác động định lượng lên cạnh và nhiễu

### (a) Nhiễu giảm tỉ lệ với $1/\sigma$
Với nhiễu trắng phương sai $\sigma_n^2$:
$$\operatorname{Var}[G_\sigma*n]=\sigma_n^2\iint G_\sigma^2=\frac{\sigma_n^2}{4\pi\sigma^2}\ \Rightarrow\ \operatorname{std}=\frac{\sigma_n}{2\sqrt\pi\,\sigma}$$

| σ | std đo (σₙ = 1) | lý thuyết |
|---|---|---|
| 1 | 0.2829 | 0.2821 |
| 2 | 0.1415 | 0.1410 |
| 3 | 0.0942 | 0.0940 |
| 5 | 0.0562 | 0.0564 |

### (b) Độ lớn gradient tại cạnh cũng giảm tỉ lệ với $1/\sigma$
Cạnh bậc thang cao $h$ sau Gaussian có dạng hàm erf, đạo hàm cực đại bằng $h/(\sqrt{2\pi}\sigma)$. Sobel khuếch đại 8 lần ([02](02_gradient_va_laplacian.md)), nên:
$$M_{\text{đỉnh}}\approx\frac{8h}{\sqrt{2\pi}\,\sigma}\approx\frac{3.19\,h}{\sigma}\qquad(\sigma\gtrsim2),\qquad M_{\text{đỉnh}}\le4h$$

| σ | đỉnh đo (h = 1) | xấp xỉ $3.19/\sigma$ |
|---|---|---|
| 0.5 | 3.572 | 6.383 (sai, vì bị chặn bởi 4) |
| 1 | 2.564 | 3.192 |
| 2 | 1.502 | 1.596 |
| 3 | 1.035 | 1.064 |
| 5 | 0.632 | 0.638 |

**Hệ quả 1 (giải thích một kết quả của Lab3):** Lab3 ghi nhận "OpenCV với σ = 5 không còn cạnh nào ở ngưỡng 100/200". Với σ = 5, cạnh mạnh nhất có thể có ($h=255$) chỉ cho $M\approx0.632\times255=161<200$, nên **không pixel nào vượt được ngưỡng cao**. Đây là hệ quả toán học, không phải lỗi.

**Hệ quả 2 (ngưỡng phải đi cùng σ):** giữ ngưỡng cố định trong khi tăng σ thì cạnh biến mất vì *hai* lý do chồng lên nhau: nhiễu giảm (điều mong muốn) và tín hiệu giảm (điều không mong muốn). Thí nghiệm "chỉ đổi σ" trong P2 phải nói rõ điều này. Experiment Engine sẽ có thêm chế độ "ngưỡng chuẩn hóa theo σ" (đề xuất cho Phase 3).

### (c) SNR của gradient tăng tuyến tính theo σ (trường hợp 2D)
Phương sai nhiễu sau đạo hàm Gaussian:
$$\operatorname{Var}[\partial_x(G_\sigma*n)]=\sigma_n^2\iint(\partial_xG_\sigma)^2=\frac{\sigma_n^2}{8\pi\sigma^4}$$
Nhân với hệ số Sobel 8, ta được $\operatorname{std}\approx1.596\,\sigma_n/\sigma^2$. Chia cho đỉnh ở (b):
$$\text{SNR}=\frac{M_{\text{đỉnh}}}{\operatorname{std}_{\text{nhiễu}}}\approx\frac{2\,h\,\sigma}{\sigma_n}$$

| σ | SNR đo (h = 1, σₙ = 1) | lý thuyết 2σ |
|---|---|---|
| 1 | 2.24 | 2 |
| 2 | 4.11 | 4 |
| 3 | 6.03 | 6 |
| 5 | 9.80 | 10 |

Trong 1D, Canny thu được SNR ∝ √σ. Trong 2D, làm trơn **dọc theo** cạnh còn lấy trung bình nhiễu theo chiều thứ hai, nên SNR ∝ σ.

### (d) Đổi lại là mất định vị
Canny (1986) chỉ ra rằng với họ đạo hàm Gaussian, tích (độ phát hiện × độ định vị) **không phụ thuộc σ**. Tăng σ làm tăng SNR nhưng giảm độ chính xác vị trí, giống một nguyên lý bất định. Các biểu hiện hình học:
- Hai cạnh cách nhau $d$ bắt đầu ảnh hưởng lẫn nhau khi $\sigma\gtrsim d/2$: đỉnh gradient của chúng chồng lên nhau, lệch vị trí, rồi hợp làm một.
- Góc nhọn bị bo tròn với bán kính cỡ σ.
- Thực nghiệm Lab3: 2562 pixel cạnh có ở σ = 2 nhưng **không** có ở σ = 1. Cạnh không chỉ mất đi mà còn **dịch chỗ**, nên tập cạnh không đơn điệu theo σ (khác với ngưỡng, xem [04](04_nguong_kep_va_hysteresis.md)).

## PARAMETERS
| Tham số | Miền | Ảnh hưởng |
|---|---|---|
| σ | > 0; thực tế 0.5–5 với ảnh cỡ 512² | SNR ∝ σ, định vị ∝ 1/σ, đỉnh gradient ∝ 1/σ |
| bán kính $r$ | $3\sigma$ đến $4\sigma$ | Sai số cắt bỏ so với chi phí $O(r)$ |
| xử lý biên | replicate / reflect / constant + chuẩn hóa | Cạnh giả hoặc mất cạnh ở viền ảnh |

## COMPLEXITY
Ảnh $N=H\cdot W$ pixel, kernel $k=2r+1$:
- Tích chập 2D trực tiếp: $O(Nk^2)$.
- Tách được: $O(2Nk)$.
- Qua FFT: $O(N\log N)$, chỉ có lợi khi $k$ rất lớn.

Bộ nhớ: $O(N)$ cho một ảnh trung gian `float`.

## LIMITATIONS
- Đẳng hướng: làm mờ cạnh theo mọi hướng như nhau, kể cả hướng vuông góc với cạnh. Bilateral hoặc anisotropic diffusion giữ cạnh tốt hơn (ứng viên cho E2).
- Một σ cho toàn ảnh: ảnh vừa có vùng nhiễu vừa có vùng chi tiết thì không σ nào tối ưu cho cả hai (P3, ảnh "nhiều chi tiết").
- Không xử lý tốt nhiễu xung (muối tiêu): giá trị cực trị bị trải ra chứ không bị loại. Median xử lý tốt hơn.

## WHICH (khi nào chọn σ lớn / nhỏ)
- σ lớn: ảnh nhiều nhiễu Gauss, cần đường bao đối tượng lớn (P4).
- σ nhỏ: ảnh sạch, cần chi tiết mảnh, cần vị trí cạnh chính xác (đo đạc).
- Ảnh nhiều chi tiết (texture): σ lớn để triệt texture, kết hợp **hạ ngưỡng** để bù phần đỉnh gradient bị giảm. Phương trình (b) cho biết hạ bao nhiêu.
