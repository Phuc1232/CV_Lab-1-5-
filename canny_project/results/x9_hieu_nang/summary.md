# x9_hieu_nang

Câu hỏi: **T3a (tốc độ), Manual vs Optimized**

Thời gian và bộ nhớ theo kích thước ảnh (128² … 1024²), cv2.setNumThreads(0) (chạy tuần tự; trên macOS/GCD giá trị 1 bị bỏ qua) để so sánh công bằng.
Khớp t ∝ N^k trên log–log; kỳ vọng k ≈ 1 vì mọi bước của Canny là O(N).

| run | params | suy giảm | n | time_median_s | time_iqr_s | mem_peak_tracemalloc_mb | mem_rss_delta_mb |
|---|---|---|---|---|---|---|---|
| manual_bfs | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 128 | 1 | 0.002642 ± 0 | 3.8e-05 ± 0 | 2.181 ± 0 | 0.03125 ± 0 |
| manual_bfs | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 1 | 0.01371 ± 0 | 0.001201 ± 0 | 8.2 ± 0 | 1.594 ± 0 |
| manual_bfs | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 512 | 1 | 0.06079 ± 0 | 0.005761 ± 0 | 32.77 ± 0 | 0 ± 0 |
| manual_bfs | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 1024 | 1 | 0.2372 ± 0 | 0.01705 ± 0 | 131 ± 0 | 0.125 ± 0 |
| manual_label | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 128 | 1 | 0.00183 ± 0 | 3.338e-05 ± 0 | 2.181 ± 0 | 0.1719 ± 0 |
| manual_label | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 1 | 0.006361 ± 0 | 0.00109 ± 0 | 8.2 ± 0 | 0 ± 0 |
| manual_label | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 512 | 1 | 0.03294 ± 0 | 0.0009069 ± 0 | 32.77 ± 0 | 0 ± 0 |
| manual_label | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 1024 | 1 | 0.1411 ± 0 | 0.004685 ± 0 | 131 ± 0 | 0.2344 ± 0 |
| skimage | high_threshold=0.1, low_threshold=0.04, sigma=1.5 | {"noise_sigma": 0.05} 128 | 1 | 0.0007893 ± 0 | 2.942e-05 ± 0 | 0.8177 ± 0 | 0 ± 0 |
| skimage | high_threshold=0.1, low_threshold=0.04, sigma=1.5 | {"noise_sigma": 0.05} 256 | 1 | 0.004136 ± 0 | 6.904e-05 ± 0 | 3.069 ± 0 | 0 ± 0 |
| skimage | high_threshold=0.1, low_threshold=0.04, sigma=1.5 | {"noise_sigma": 0.05} 512 | 1 | 0.01772 ± 0 | 0.0006424 ± 0 | 12.07 ± 0 | 0 ± 0 |
| skimage | high_threshold=0.1, low_threshold=0.04, sigma=1.5 | {"noise_sigma": 0.05} 1024 | 1 | 0.07272 ± 0 | 0.0008972 ± 0 | 48.09 ± 0 | 0 ± 0 |
| manual_opencv_profile | high_threshold=100, low_threshold=40, profile=opencv | {"noise_sigma": 0.05} 128 | 1 | 0.00979 ± 0 | 0.002371 ± 0 | 1.352 ± 0 | 0.1094 ± 0 |
| manual_opencv_profile | high_threshold=100, low_threshold=40, profile=opencv | {"noise_sigma": 0.05} 256 | 1 | 0.03891 ± 0 | 0.004374 ± 0 | 5.323 ± 0 | 0 ± 0 |
| manual_opencv_profile | high_threshold=100, low_threshold=40, profile=opencv | {"noise_sigma": 0.05} 512 | 1 | 0.1307 ± 0 | 0.001326 ± 0 | 21.27 ± 0 | 0 ± 0 |
| manual_opencv_profile | high_threshold=100, low_threshold=40, profile=opencv | {"noise_sigma": 0.05} 1024 | 1 | 0.5455 ± 0 | 0.0238 ± 0 | 85.03 ± 0 | 0 ± 0 |
| opencv | high_threshold=100, low_threshold=40 | {"noise_sigma": 0.05} 128 | 1 | 0.0003283 ± 0 | 9.875e-06 ± 0 | 0.2507 ± 0 | 0 ± 0 |
| opencv | high_threshold=100, low_threshold=40 | {"noise_sigma": 0.05} 256 | 1 | 0.001389 ± 0 | 0.0001972 ± 0 | 1.001 ± 0 | 0 ± 0 |
| opencv | high_threshold=100, low_threshold=40 | {"noise_sigma": 0.05} 512 | 1 | 0.006124 ± 0 | 0.001582 ± 0 | 4.001 ± 0 | 0 ± 0 |
| opencv | high_threshold=100, low_threshold=40 | {"noise_sigma": 0.05} 1024 | 1 | 0.02455 ± 0 | 0.0005847 ± 0 | 16 ± 0 | 0.2344 ± 0 |

## Hệ số mũ thời gian t ∝ N^k

- manual_bfs: k = 1.081
- manual_label: k = 1.059
- skimage: k = 1.084
- manual_opencv_profile: k = 0.957
- opencv: k = 1.041
