# x8b_video_chuyen_dong

Câu hỏi: **E4**

Video tổng hợp có vật tịnh tiến 2 px/khung, nhiễu độc lập mỗi khung (sigma = 0.08). So sánh Canny từng khung với làm trơn theo
thời gian (Gaussian 1D trục khung, temporal_sigma) về F1 với ground truth và độ ổn định F1 giữa khung liên tiếp.

| run | params | suy giảm | n | f1 | precision | recall |
|---|---|---|---|---|---|---|
| tung_khung | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.1354 ± 0.034 | 0.07305 ± 0.02 | 0.9914 ± 0.0055 |
| tung_khung_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {} 128 | 32 | 0.9753 ± 0.026 | 0.9612 ± 0.048 | 0.9912 ± 0.0053 |
| lam_tron_thoi_gian_1 | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.7682 ± 0.23 | 0.6747 ± 0.25 | 0.9871 ± 0.0044 |
| lam_tron_thoi_gian_2 | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {} 128 | 32 | 0.8564 ± 0.19 | 0.8127 ± 0.22 | 0.9612 ± 0.034 |
| lam_tron_thoi_gian_2_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {} 128 | 32 | 0.9231 ± 0.043 | 0.889 ± 0.058 | 0.9612 ± 0.034 |

## Độ ổn định theo thời gian (F1 giữa khung liên tiếp)

| cấu hình | F1 TB | std | số cặp |
|---|---|---|---|
| tung_khung sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.7645 | 0.013 | 30 |
| tung_khung_nguong_nhieu sigma=1.5,threshold_strategy=noise | 0.7426 | 0.044 | 30 |
| lam_tron_thoi_gian_1 sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.5854 | 0.148 | 30 |
| lam_tron_thoi_gian_2 sigma=1.5,low_threshold=0.04,high_threshold=0.1 | 0.7642 | 0.147 | 30 |
| lam_tron_thoi_gian_2_nguong_nhieu sigma=1.5,threshold_strategy=noise | 0.8285 | 0.066 | 30 |
