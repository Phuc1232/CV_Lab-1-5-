# x7_mau

Câu hỏi: **E3**

Ảnh màu đẳng sáng (hình và nền cùng độ sáng Y): chuyển xám làm mất cạnh. So sánh gray / OR từng kênh /
kênh lớn nhất / Di Zenzo (tự cài) và cv2.Canny 3 kênh. Mỗi phương pháp quét ngưỡng riêng (các độ lớn
khác thang nhau: Di Zenzo cộng năng lượng 3 kênh) và so sánh bằng F1 tốt nhất trên lưới.

| run | params | suy giảm | n | f1 | precision | recall | fom |
|---|---|---|---|---|---|---|---|
| gray | high_threshold=0.08, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.08, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.08974 ± 0.031 | 0.06621 ± 0.023 | 0.1417 ± 0.055 | 0.1632 ± 0.031 |
| gray | high_threshold=0.12, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.12, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.18, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.18, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.27, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.27, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.4, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.4, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.6, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| gray | high_threshold=0.6, low_ratio=0.4, method=gray, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| or | high_threshold=0.08, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9399 ± 0.031 |
| or | high_threshold=0.08, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.09039 ± 0.0061 | 0.04734 ± 0.0033 | 0.9995 ± 0.001 | 0.121 ± 0.0072 |
| or | high_threshold=0.12, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9399 ± 0.031 |
| or | high_threshold=0.12, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.3357 ± 0.026 | 0.202 ± 0.019 | 0.9995 ± 0.001 | 0.2538 ± 0.023 |
| or | high_threshold=0.18, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9399 ± 0.031 |
| or | high_threshold=0.18, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9863 ± 0.0093 | 0.9736 ± 0.019 | 0.9995 ± 0.001 | 0.9485 ± 0.0087 |
| or | high_threshold=0.27, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9394 ± 0.031 |
| or | high_threshold=0.27, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9968 ± 0.0017 | 0.9941 ± 0.0037 | 0.9995 ± 0.001 | 0.9612 ± 0.0023 |
| or | high_threshold=0.4, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9394 ± 0.031 |
| or | high_threshold=0.4, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9993 ± 0.0011 | 1 ± 0 | 0.9986 ± 0.0021 | 0.9672 ± 0.0035 |
| or | high_threshold=0.6, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9394 ± 0.031 |
| or | high_threshold=0.6, low_ratio=0.4, method=or, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9993 ± 0.0011 | 1 ± 0 | 0.9986 ± 0.0021 | 0.9672 ± 0.0035 |
| max_channel | high_threshold=0.08, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.08, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.1218 ± 0.0089 | 0.06486 ± 0.005 | 0.9977 ± 0.0026 | 0.1297 ± 0.0078 |
| max_channel | high_threshold=0.12, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.12, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.1729 ± 0.016 | 0.09473 ± 0.0096 | 0.9977 ± 0.0026 | 0.1481 ± 0.013 |
| max_channel | high_threshold=0.18, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.18, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.93 ± 0.032 |
| max_channel | high_threshold=0.27, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.27, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.93 ± 0.032 |
| max_channel | high_threshold=0.4, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.4, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.93 ± 0.032 |
| max_channel | high_threshold=0.6, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| max_channel | high_threshold=0.6, low_ratio=0.4, method=max_channel, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.93 ± 0.032 |
| di_zenzo | high_threshold=0.08, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.08, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.1172 ± 0.0095 | 0.0623 ± 0.0054 | 0.9977 ± 0.0026 | 0.1284 ± 0.0082 |
| di_zenzo | high_threshold=0.12, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.12, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.1331 ± 0.014 | 0.07134 ± 0.0079 | 0.9977 ± 0.0026 | 0.1319 ± 0.011 |
| di_zenzo | high_threshold=0.18, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.18, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9731 ± 0.039 | 0.9523 ± 0.071 | 0.9977 ± 0.0026 | 0.8921 ± 0.047 |
| di_zenzo | high_threshold=0.27, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.27, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9296 ± 0.031 |
| di_zenzo | high_threshold=0.4, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.4, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9296 ± 0.031 |
| di_zenzo | high_threshold=0.6, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9348 ± 0.03 |
| di_zenzo | high_threshold=0.6, low_ratio=0.4, method=di_zenzo, profile=skimage, sigma=1.0 | {"noise_sigma": 0.03} 128 | 6 | 0.9988 ± 0.0013 | 1 ± 0 | 0.9977 ± 0.0026 | 0.9296 ± 0.031 |
| opencv_3_kenh | high_threshold=40, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=40, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 0.1225 ± 0.0089 | 0.0653 ± 0.0051 | 1 ± 0 | 0.1576 ± 0.0098 |
| opencv_3_kenh | high_threshold=60, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=60, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 0.1234 ± 0.0092 | 0.06577 ± 0.0053 | 1 ± 0 | 0.1576 ± 0.01 |
| opencv_3_kenh | high_threshold=90, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=90, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 0.1378 ± 0.01 | 0.07402 ± 0.0058 | 1 ± 0 | 0.1609 ± 0.011 |
| opencv_3_kenh | high_threshold=135, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=135, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 0.4922 ± 0.051 | 0.3279 ± 0.045 | 1 ± 0 | 0.3961 ± 0.037 |
| opencv_3_kenh | high_threshold=200, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=200, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 0.9869 ± 0.0054 | 0.9742 ± 0.011 | 1 ± 0 | 0.9482 ± 0.013 |
| opencv_3_kenh | high_threshold=300, low_ratio=0.4 | {} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.9147 ± 0.03 |
| opencv_3_kenh | high_threshold=300, low_ratio=0.4 | {"noise_sigma": 0.03} 128 | 6 | 1 ± 0 | 1 ± 0 | 1 ± 0 | 0.935 ± 0.02 |

## Tốt nhất theo lưới tham số (cực đại f1 trung bình, kiểu ODS)

| run | suy giảm | params tốt nhất | f1 |
|---|---|---|---|
| gray | {} 128 | {"high_threshold": 0.08, "low_ratio": 0.4, "method": "gray", "profile": "skimage", "sigma": 1.0} | 0.0000 |
| gray | {"noise_sigma": 0.03} 128 | {"high_threshold": 0.08, "low_ratio": 0.4, "method": "gray", "profile": "skimage", "sigma": 1.0} | 0.0897 |
| or | {} 128 | {"high_threshold": 0.08, "low_ratio": 0.4, "method": "or", "profile": "skimage", "sigma": 1.0} | 0.9988 |
| or | {"noise_sigma": 0.03} 128 | {"high_threshold": 0.4, "low_ratio": 0.4, "method": "or", "profile": "skimage", "sigma": 1.0} | 0.9993 |
| max_channel | {} 128 | {"high_threshold": 0.08, "low_ratio": 0.4, "method": "max_channel", "profile": "skimage", "sigma": 1.0} | 0.9988 |
| max_channel | {"noise_sigma": 0.03} 128 | {"high_threshold": 0.18, "low_ratio": 0.4, "method": "max_channel", "profile": "skimage", "sigma": 1.0} | 0.9988 |
| di_zenzo | {} 128 | {"high_threshold": 0.08, "low_ratio": 0.4, "method": "di_zenzo", "profile": "skimage", "sigma": 1.0} | 0.9988 |
| di_zenzo | {"noise_sigma": 0.03} 128 | {"high_threshold": 0.27, "low_ratio": 0.4, "method": "di_zenzo", "profile": "skimage", "sigma": 1.0} | 0.9988 |
| opencv_3_kenh | {} 128 | {"high_threshold": 40, "low_ratio": 0.4} | 1.0000 |
| opencv_3_kenh | {"noise_sigma": 0.03} 128 | {"high_threshold": 300, "low_ratio": 0.4} | 1.0000 |
