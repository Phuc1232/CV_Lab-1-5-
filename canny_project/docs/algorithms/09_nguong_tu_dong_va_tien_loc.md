# 09. Ngưỡng tự động và tiền lọc (E2)

**Chiến lược ngưỡng** ([`core/thresholds.py`](../../src/cannylab/core/thresholds.py)). Mỗi chiến lược trả $(T_l,T_h)$ theo thang của độ lớn gradient.

| Tên | Công thức | Cơ sở |
|---|---|---|
| `quantile` | $T=$ phân vị $q$ của $M$ | ≡ skimage `use_quantiles` (đã kiểm chứng trùng tuyệt đối) |
| `otsu` | $T_h=$ Otsu trên $M$ tại cực đại địa phương, $T_l=0.5T_h$ | tối đa phương sai giữa lớp; Otsu tự cài |
| `noise` | $T_h=k\,s$ với $k=\sqrt{2\ln N}$ (ngưỡng phổ quát), $T_l=2s$, $s=\hat\sigma_n\sqrt{\sum(K_x*G_\sigma)^2}$ | nhiễu trắng: $P(M>ks)=e^{-k^2/2}$ (Rayleigh), kỳ vọng 1 hạt giống giả trên cả ảnh |
| `median` | $T=(1\mp0.33)\,\operatorname{median}(I)$ | **heuristic** trên cường độ, không có cơ sở; giữ làm đối chứng |

**Ước lượng nhiễu $\hat\sigma_n$.** Immerkær (1996) cho $\sqrt{\pi/2}\cdot\overline{|I*N|}/6$, có **lặp loại pixel cạnh**: bỏ pixel có $|\nabla I|>4\sqrt{12}\hat\sigma$.

| σₙ thật | Immerkær gốc | Có loại cạnh |
|---|---|---|
| 0.01 (ảnh hình, có cạnh) | 0.0180 | **0.0099** |
| 0.05 | không đo | 0.0499 |
| 0.10 | không đo | 0.1014 |
| 0.02 / 0.10 (nhiễu thuần) | không đo | 0.0201 / 0.0993 |

Lỗi của bản gốc được phát hiện nhờ test `test_immerkaer_noise_estimate[0.01]` fail.

**Hệ số khuếch đại nhiễu chính xác.** `gradient_noise_gain(σ)` tính $\sqrt{\sum K^2}$ của kernel $K_x*G_\sigma$ thay vì dùng xấp xỉ liên tục $1.6/\sigma^2$, vốn sai khi σ < 2. Kiểm chứng: sai số < 3% so với đo trực tiếp ở σ ∈ {0, 1, 2, 3}.

**Tiền lọc** ([`core/denoise.py`](../../src/cannylab/core/denoise.py)):
- `median_filter`: sliding window, biên reflect101. Trùng `cv2.medianBlur` ở phần trong ảnh.
- `bilateral_filter`: cửa sổ tròn bán kính d/2 như OpenCV. Sai lệch trung bình < 0.01 so với `cv2.bilateralFilter`, vì OpenCV dùng bảng tra và float32.

Gọi qua registry bằng `prefilter="median"|"bilateral"`, tham số có tiền tố `prefilter_`.

**COMPLEXITY.** Median $O(Nk^2\log k)$, bilateral $O(Nd^2)$, quantile $O(N\log N)$, Otsu $O(N)$, noise $O(N)$.

**KẾT QUẢ.** X6, X3, X2 ([../experiments/README.md](../experiments/README.md)).

**Thay đổi sau thực nghiệm (Phase 9).** Mặc định ban đầu `k = 3.5, ratio = 0.4` cho F1 khoảng 0.5 vì có khoảng 140 hạt giống giả trên ảnh 256². Đã đổi sang ngưỡng phổ quát, và chọn bộ tham số mới trên **seed 100** (tách khỏi dữ liệu thực nghiệm):

| Cấu hình | σ = 1 | σ = 2 | σ = 3 |
|---|---|---|---|
| k = 3.5, T_l = 0.4T_h (cũ) | 0.317 | 0.450 | 0.524 |
| k = √(2 ln N), k_low = 2 (mới) | 0.756 | 0.770 | 0.760 |
| k_low = 1.5 / 3.0 | 0.739 / 0.765 | 0.767 / 0.772 | 0.759 / 0.763 |

k_low ít ảnh hưởng nên giữ 2. Các tham số `k`, `k_low`, `ratio` vẫn ghi đè được (X6 dùng để so sánh).
