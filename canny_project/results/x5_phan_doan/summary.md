# x5_phan_doan

Câu hỏi: **P4a, P4b**

Canny -> closing (ellipse 5x5) -> lấp lỗ -> lọc diện tích -> nhận dạng hình (Douglas–Peucker + đặc trưng).
Đánh giá mức đối tượng (ghép Hungarian, IoU >= 0.5) và accuracy nhận dạng 6 lớp hình.

| run | params | suy giảm | n | object_f1 | object_recall | mean_iou_matched | count_error | f1 |
|---|---|---|---|---|---|---|---|---|
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=1.5 | {} 256 | 10 | 1 ± 0 | 1 ± 0 | 0.9246 ± 0.0053 | 0 ± 0 | 0.9972 ± 0.0011 |
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 10 | 0.8559 ± 0.065 | 1 ± 0 | 0.9187 ± 0.0094 | 2.1 ± 1 | 0.8103 ± 0.047 |
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=1.5 | {"noise_sigma": 0.1} 256 | 10 | 0 ± 0 | 0 ± 0 | 0 ± 0 | -6 ± 0 | 0.09544 ± 0.007 |
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=2.5 | {} 256 | 10 | 1 ± 0 | 1 ± 0 | 0.9199 ± 0.0067 | 0 ± 0 | 0.9832 ± 0.0019 |
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=2.5 | {"noise_sigma": 0.05} 256 | 10 | 0.9833 ± 0.05 | 0.9833 ± 0.05 | 0.9181 ± 0.0067 | 0 ± 0 | 0.9838 ± 0.0019 |
| canny_manual | high_threshold=0.12, low_ratio=0.4, profile=skimage, sigma=2.5 | {"noise_sigma": 0.1} 256 | 10 | 0.95 ± 0.076 | 0.95 ± 0.076 | 0.9163 ± 0.0082 | 0 ± 0 | 0.9819 ± 0.0038 |
| opencv_lab3 | high_threshold=90, low_threshold=30, sigma=1.5 | {} 256 | 10 | 1 ± 0 | 1 ± 0 | 0.9238 ± 0.0064 | 0 ± 0 | 0.9986 ± 0.00069 |
| opencv_lab3 | high_threshold=90, low_threshold=30, sigma=1.5 | {"noise_sigma": 0.05} 256 | 10 | 1 ± 0 | 1 ± 0 | 0.9197 ± 0.0058 | 0 ± 0 | 0.9986 ± 0.00052 |
| opencv_lab3 | high_threshold=90, low_threshold=30, sigma=1.5 | {"noise_sigma": 0.1} 256 | 10 | 0.9317 ± 0.09 | 0.9667 ± 0.067 | 0.9107 ± 0.011 | 0.5 ± 0.67 | 0.9463 ± 0.034 |

## Tốt nhất theo lưới tham số (cực đại object_f1 trung bình, kiểu ODS)

| run | suy giảm | params tốt nhất | object_f1 |
|---|---|---|---|
| canny_manual | {} 256 | {"high_threshold": 0.12, "low_ratio": 0.4, "profile": "skimage", "sigma": 1.5} | 1.0000 |
| canny_manual | {"noise_sigma": 0.05} 256 | {"high_threshold": 0.12, "low_ratio": 0.4, "profile": "skimage", "sigma": 2.5} | 0.9833 |
| canny_manual | {"noise_sigma": 0.1} 256 | {"high_threshold": 0.12, "low_ratio": 0.4, "profile": "skimage", "sigma": 2.5} | 0.9500 |

## Nhận dạng hình

| cấu hình | accuracy | macro-F1 | n |
|---|---|---|---|
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=1.5 @ {} | 1.0000 | 1.0000 | 60 |
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=1.5 @ {"noise_sigma": 0.05} | 0.9333 | 0.9273 | 60 |
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=1.5 @ {"noise_sigma": 0.1} | 0.0000 | 0.0000 | 60 |
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=2.5 @ {} | 0.9167 | 0.9111 | 60 |
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=2.5 @ {"noise_sigma": 0.05} | 0.8833 | 0.8857 | 60 |
| canny_manual low_ratio=0.4,high_threshold=0.12,sigma=2.5 @ {"noise_sigma": 0.1} | 0.8000 | 0.8107 | 60 |
| opencv_lab3 sigma=1.5,low_threshold=30,high_threshold=90 @ {} | 1.0000 | 1.0000 | 60 |
| opencv_lab3 sigma=1.5,low_threshold=30,high_threshold=90 @ {"noise_sigma": 0.05} | 1.0000 | 1.0000 | 60 |
| opencv_lab3 sigma=1.5,low_threshold=30,high_threshold=90 @ {"noise_sigma": 0.1} | 0.9167 | 0.9322 | 60 |
