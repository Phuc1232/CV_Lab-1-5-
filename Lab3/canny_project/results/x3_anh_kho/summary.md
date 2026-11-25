# x3_anh_kho

Câu hỏi: **P3**

Ba loại ảnh khó: nhiều nhiễu (noise_sigma), tương phản thấp (contrast), nhiều chi tiết (texture).
So sánh tham số mặc định của hai thư viện với cấu hình chỉnh tay và hai chiến lược ngưỡng tự động.

| run | params | suy giảm | n | f1 | precision | recall | fom | fragmentation |
|---|---|---|---|---|---|---|---|---|
| skimage_mac_dinh |  | {"noise_sigma": 0.1} 256 | 6 | 0.07719 ± 0.0072 | 0.04016 ± 0.0039 | 0.9985 ± 0.0012 | 0.08752 ± 0.0061 | 1.78 ± 0.11 |
| skimage_mac_dinh |  | {"noise_sigma": 0.2} 256 | 6 | 0.07401 ± 0.0068 | 0.03845 ± 0.0037 | 0.9975 ± 0.0019 | 0.09262 ± 0.0076 | 1.96 ± 0.052 |
| skimage_mac_dinh |  | {"contrast": 0.3, "noise_sigma": 0.01} 256 | 6 | 0.9995 ± 0.00058 | 1 ± 0 | 0.9989 ± 0.0012 | 0.9509 ± 0.02 | 0.7729 ± 0.097 |
| skimage_mac_dinh |  | {"contrast": 0.12, "noise_sigma": 0.01} 256 | 6 | 0.1606 ± 0.18 | 1 ± 0 | 0.09812 ± 0.11 | 0.09241 ± 0.11 | 0.4675 ± 0.49 |
| skimage_mac_dinh |  | {"texture": 0.08} 256 | 6 | 0.09577 ± 0.0077 | 0.05031 ± 0.0043 | 0.9986 ± 0.00079 | 0.1003 ± 0.0071 | 3.059 ± 0.17 |
| skimage_mac_dinh |  | {"texture": 0.15} 256 | 6 | 0.09156 ± 0.0055 | 0.04804 ± 0.003 | 0.9773 ± 0.019 | 0.1033 ± 0.0072 | 3.807 ± 0.17 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"noise_sigma": 0.1} 256 | 6 | 0.08998 ± 0.0093 | 0.04713 ± 0.0051 | 1 ± 0 | 0.1035 ± 0.0068 | 6.978 ± 0.33 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"noise_sigma": 0.2} 256 | 6 | 0.08641 ± 0.0064 | 0.04517 ± 0.0035 | 0.9985 ± 0.0011 | 0.1044 ± 0.0071 | 7.153 ± 0.2 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"contrast": 0.3, "noise_sigma": 0.01} 256 | 6 | 0.6177 ± 0.35 | 1 ± 0 | 0.5273 ± 0.32 | 0.5096 ± 0.3 | 0.6291 ± 0.33 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"contrast": 0.12, "noise_sigma": 0.01} 256 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"texture": 0.08} 256 | 6 | 0.1362 ± 0.011 | 0.07311 ± 0.0065 | 1 ± 0 | 0.1269 ± 0.0086 | 2.853 ± 0.08 |
| opencv_100_200 | high_threshold=200, low_threshold=100 | {"texture": 0.15} 256 | 6 | 0.09296 ± 0.007 | 0.04878 ± 0.0039 | 0.9937 ± 0.0091 | 0.1072 ± 0.0067 | 1.607 ± 0.12 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"noise_sigma": 0.1} 256 | 6 | 0.1361 ± 0.011 | 0.07314 ± 0.0063 | 0.9831 ± 0.0025 | 0.09843 ± 0.0067 | 2.173 ± 0.11 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"noise_sigma": 0.2} 256 | 6 | 0.1055 ± 0.008 | 0.05576 ± 0.0045 | 0.9827 ± 0.0036 | 0.08908 ± 0.0078 | 3.178 ± 0.15 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"contrast": 0.3, "noise_sigma": 0.01} 256 | 6 | 0.9934 ± 0.0015 | 1 ± 0 | 0.987 ± 0.0029 | 0.9551 ± 0.014 | 0.7273 ± 0.069 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"contrast": 0.12, "noise_sigma": 0.01} 256 | 6 | 0.5108 ± 0.39 | 1 ± 0 | 0.4352 ± 0.35 | 0.4228 ± 0.34 | 0.5264 ± 0.39 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"texture": 0.08} 256 | 6 | 0.1184 ± 0.0096 | 0.06303 ± 0.0055 | 0.9753 ± 0.0088 | 0.0978 ± 0.0076 | 4.611 ± 0.18 |
| chinh_tay_sigma2 | high_threshold=0.08, low_threshold=0.03, profile=skimage, sigma=2.0 | {"texture": 0.15} 256 | 6 | 0.1052 ± 0.0038 | 0.05582 ± 0.0022 | 0.9104 ± 0.042 | 0.1007 ± 0.0075 | 5.498 ± 0.2 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.1} 256 | 6 | 0.9859 ± 0.0093 | 0.9889 ± 0.019 | 0.9831 ± 0.0025 | 0.9505 ± 0.018 | 0.8749 ± 0.12 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.2} 256 | 6 | 0.9684 ± 0.023 | 0.9578 ± 0.04 | 0.9799 ± 0.0062 | 0.9262 ± 0.036 | 0.9009 ± 0.2 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.3, "noise_sigma": 0.01} 256 | 6 | 0.9843 ± 0.0098 | 0.9818 ± 0.02 | 0.987 ± 0.0029 | 0.9469 ± 0.019 | 0.772 ± 0.082 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.12, "noise_sigma": 0.01} 256 | 6 | 0.9766 ± 0.014 | 0.9675 ± 0.029 | 0.9864 ± 0.0029 | 0.9286 ± 0.023 | 0.8184 ± 0.073 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"texture": 0.08} 256 | 6 | 0.1149 ± 0.0089 | 0.06109 ± 0.005 | 0.9755 ± 0.0087 | 0.09635 ± 0.0074 | 5.95 ± 0.18 |
| tu_dong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"texture": 0.15} 256 | 6 | 0.1047 ± 0.0039 | 0.05553 ± 0.0023 | 0.9117 ± 0.041 | 0.1004 ± 0.0075 | 5.861 ± 0.18 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"noise_sigma": 0.1} 256 | 6 | 0.991 ± 0.0018 | 0.999 ± 0.0022 | 0.9831 ± 0.0025 | 0.9597 ± 0.0082 | 0.8241 ± 0.11 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"noise_sigma": 0.2} 256 | 6 | 0.9464 ± 0.059 | 0.921 ± 0.1 | 0.9794 ± 0.0064 | 0.8919 ± 0.092 | 1.33 ± 0.67 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"contrast": 0.3, "noise_sigma": 0.01} 256 | 6 | 0.9934 ± 0.0015 | 1 ± 0 | 0.987 ± 0.0029 | 0.9551 ± 0.014 | 0.7273 ± 0.069 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"contrast": 0.12, "noise_sigma": 0.01} 256 | 6 | 0.9929 ± 0.0016 | 0.9996 ± 0.0006 | 0.9864 ± 0.0029 | 0.9588 ± 0.011 | 0.7161 ± 0.064 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"texture": 0.08} 256 | 6 | 0.7889 ± 0.09 | 0.6765 ± 0.13 | 0.9662 ± 0.013 | 0.6679 ± 0.12 | 2.848 ± 1.1 |
| tu_dong_phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"texture": 0.15} 256 | 6 | 0.4274 ± 0.095 | 0.2943 ± 0.073 | 0.7888 ± 0.13 | 0.3331 ± 0.063 | 6.425 ± 0.99 |

## PSNR ảnh đầu vào (mô tả mức suy giảm)

- {"noise_sigma": 0.1}: 20.16 dB
- {"noise_sigma": 0.2}: 14.90 dB
- {"contrast": 0.3, "noise_sigma": 0.01}: 21.39 dB
- {"contrast": 0.12, "noise_sigma": 0.01}: 19.43 dB
- {"texture": 0.08}: 22.01 dB
- {"texture": 0.15}: 16.98 dB
