# x8_video

Câu hỏi: **E4**

Video tổng hợp cảnh tĩnh, nhiễu độc lập mỗi khung (sigma = 0.08). So sánh Canny từng khung với làm trơn theo
thời gian (Gaussian 1D trục khung, temporal_sigma) về F1 với ground truth và độ ổn định F1 giữa khung liên tiếp.

| run | params | suy giảm | n | f1 | precision | recall |
|---|---|---|---|---|---|---|
| tung_khung | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.1835 ± 0.0078 | 0.1011 ± 0.0047 | 0.9922 ± 0.0028 |
| tung_khung_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {} 128 | 32 | 0.9823 ± 0.021 | 0.9735 ± 0.04 | 0.9921 ± 0.0029 |
| lam_tron_thoi_gian_1 | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.8226 ± 0.22 | 0.7473 ± 0.24 | 0.9922 ± 0.0023 |
| lam_tron_thoi_gian_2 | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.9322 ± 0.16 | 0.913 ± 0.22 | 0.9921 ± 0.002 |
| lam_tron_thoi_gian_2_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {} 128 | 32 | 0.9895 ± 0.013 | 0.9873 ± 0.025 | 0.9921 ± 0.002 |

## Độ ổn định theo thời gian (F1 giữa khung liên tiếp)

| cấu hình | F1 TB | std | số cặp |
|---|---|---|---|
| tung_khung sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.7813 | 0.014 | 30 |
| tung_khung_nguong_nhieu sigma=1.5,threshold_strategy=noise | 0.9717 | 0.029 | 30 |
| lam_tron_thoi_gian_1 sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.7874 | 0.188 | 30 |
| lam_tron_thoi_gian_2 sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.9335 | 0.164 | 30 |
| lam_tron_thoi_gian_2_nguong_nhieu sigma=1.5,threshold_strategy=noise | 0.9897 | 0.018 | 30 |
