# x1_thu_vien

Câu hỏi: **P1**

Canny tự cài (2 hồ sơ) so với cv2.Canny và skimage.feature.canny trên ảnh thật (skimage.data),
và hysteresis BFS so với gán nhãn. Kỳ vọng: n_diff = 0 ở mọi cấu hình (docs/mathematics/05).

| run | params | suy giảm | n | n_diff | iou_pixel | edge_ratio |
|---|---|---|---|---|---|---|
| manual_opencv_L1 | high_threshold=150, low_threshold=50, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1384 ± 0.099 |
| manual_opencv_L1 | high_threshold=200, low_threshold=50, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1316 ± 0.099 |
| manual_opencv_L1 | high_threshold=150, low_threshold=100, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1228 ± 0.096 |
| manual_opencv_L1 | high_threshold=200, low_threshold=100, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.116 ± 0.095 |
| manual_opencv_L2 | high_threshold=120, low_threshold=50, norm=L2, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1351 ± 0.1 |
| manual_opencv_L2 | high_threshold=200, low_threshold=50, norm=L2, profile=opencv | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1212 ± 0.099 |
| manual_skimage | profile=skimage, sigma=0.5 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1584 ± 0.094 |
| manual_skimage | profile=skimage, sigma=1.0 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1247 ± 0.083 |
| manual_skimage | profile=skimage, sigma=2.0 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.06831 ± 0.055 |
| manual_skimage | profile=skimage, sigma=3.0 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.04058 ± 0.026 |
| bfs_vs_label | high_threshold=40, low_threshold=10, profile=skimage, sigma=1.5 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.1041 ± 0.069 |
| bfs_vs_label | high_threshold=40, low_threshold=20, profile=skimage, sigma=1.5 | {}  | 6 | 0 ± 0 | 1 ± 0 | 0.09959 ± 0.07 |
