# x6_cai_tien

Câu hỏi: **E2**

Các biến thể cải thiện Canny: ngưỡng theo nhiễu (Immerkær + hệ số khuếch đại chính xác), phân vị, Otsu,
heuristic median±33% (đối chứng), tiền lọc median / bilateral (đi kèm Otsu, vì sau tiền lọc nhiễu không còn
là nhiễu trắng Gauss nên chiến lược noise mất cơ sở), so với ngưỡng cố định.

| run | params | suy giảm | n | f1 | precision | recall | fom |
|---|---|---|---|---|---|---|---|
| co_dinh | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=2.0 | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| co_dinh | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=2.0 | {"noise_sigma": 0.15} 256 | 6 | 0.1188 ± 0.015 | 0.0633 ± 0.0086 | 0.982 ± 0.0042 | 0.09041 ± 0.012 |
| co_dinh | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=2.0 | {"salt_pepper": 0.02} 256 | 6 | 0.444 ± 0.033 | 0.2873 ± 0.027 | 0.9841 ± 0.0056 | 0.2969 ± 0.027 |
| co_dinh | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=2.0 | {"salt_pepper": 0.06} 256 | 6 | 0.1427 ± 0.016 | 0.07702 ± 0.0094 | 0.9846 ± 0.0039 | 0.1037 ± 0.011 |
| co_dinh | high_threshold=0.1, low_threshold=0.04, profile=skimage, sigma=2.0 | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9129 ± 0.045 | 0.9997 ± 0.00058 | 0.8431 ± 0.076 | 0.8273 ± 0.077 |
| nguong_nhieu_pho_quat | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9847 ± 0.011 | 0.9848 ± 0.024 | 0.9849 ± 0.0044 | 0.9462 ± 0.019 |
| nguong_nhieu_pho_quat | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9879 ± 0.0027 | 0.9944 ± 0.0063 | 0.9816 ± 0.0038 | 0.9575 ± 0.0041 |
| nguong_nhieu_pho_quat | profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.02} 256 | 6 | 0.1521 ± 0.023 | 0.08257 ± 0.014 | 0.9841 ± 0.0056 | 0.1079 ± 0.016 |
| nguong_nhieu_pho_quat | profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.06} 256 | 6 | 0.1166 ± 0.012 | 0.06199 ± 0.0069 | 0.9849 ± 0.004 | 0.09017 ± 0.0091 |
| nguong_nhieu_pho_quat | profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9854 ± 0.011 | 0.9851 ± 0.018 | 0.9858 ± 0.0042 | 0.9495 ± 0.016 |
| nguong_nhieu_k | k=3.5, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.6769 ± 0.029 | 0.5166 ± 0.036 | 0.9849 ± 0.0044 | 0.5084 ± 0.032 |
| nguong_nhieu_k | k=3.5, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.7098 ± 0.028 | 0.5566 ± 0.035 | 0.9818 ± 0.0039 | 0.5503 ± 0.029 |
| nguong_nhieu_k | k=3.5, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.02} 256 | 6 | 0.1472 ± 0.02 | 0.07969 ± 0.012 | 0.9841 ± 0.0056 | 0.105 ± 0.014 |
| nguong_nhieu_k | k=3.5, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.06} 256 | 6 | 0.1125 ± 0.013 | 0.05972 ± 0.0071 | 0.9849 ± 0.004 | 0.08835 ± 0.0094 |
| nguong_nhieu_k | k=3.5, profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.7234 ± 0.043 | 0.5733 ± 0.056 | 0.9858 ± 0.0042 | 0.5636 ± 0.057 |
| nguong_nhieu_k | k=5.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| nguong_nhieu_k | k=5.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9879 ± 0.0027 | 0.9944 ± 0.0063 | 0.9816 ± 0.0038 | 0.9575 ± 0.0041 |
| nguong_nhieu_k | k=5.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.02} 256 | 6 | 0.1544 ± 0.024 | 0.08397 ± 0.014 | 0.9841 ± 0.0056 | 0.1093 ± 0.016 |
| nguong_nhieu_k | k=5.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.06} 256 | 6 | 0.1191 ± 0.012 | 0.06341 ± 0.0071 | 0.9849 ± 0.004 | 0.09148 ± 0.0094 |
| nguong_nhieu_k | k=5.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9928 ± 0.002 | 0.9998 ± 0.00045 | 0.9858 ± 0.0042 | 0.9584 ± 0.0098 |
| nguong_nhieu_k | k=6.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| nguong_nhieu_k | k=6.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9881 ± 0.0025 | 0.9947 ± 0.0058 | 0.9816 ± 0.0038 | 0.9578 ± 0.0039 |
| nguong_nhieu_k | k=6.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.02} 256 | 6 | 0.1618 ± 0.028 | 0.0884 ± 0.016 | 0.9841 ± 0.0056 | 0.1137 ± 0.018 |
| nguong_nhieu_k | k=6.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.06} 256 | 6 | 0.1296 ± 0.013 | 0.06943 ± 0.0076 | 0.9846 ± 0.0039 | 0.09682 ± 0.0095 |
| nguong_nhieu_k | k=6.0, profile=skimage, sigma=2.0, threshold_strategy=noise | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9928 ± 0.002 | 0.9998 ± 0.00045 | 0.9858 ± 0.0042 | 0.9584 ± 0.0098 |
| nguong_nhieu_mac_dinh_cu | k=3.5, profile=skimage, ratio=0.4, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.5111 ± 0.038 | 0.3461 ± 0.035 | 0.9849 ± 0.0044 | 0.346 ± 0.032 |
| nguong_nhieu_mac_dinh_cu | k=3.5, profile=skimage, ratio=0.4, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.5479 ± 0.023 | 0.3803 ± 0.023 | 0.9818 ± 0.0039 | 0.3859 ± 0.021 |
| nguong_nhieu_mac_dinh_cu | k=3.5, profile=skimage, ratio=0.4, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.02} 256 | 6 | 0.1472 ± 0.02 | 0.07966 ± 0.012 | 0.9841 ± 0.0056 | 0.105 ± 0.014 |
| nguong_nhieu_mac_dinh_cu | k=3.5, profile=skimage, ratio=0.4, sigma=2.0, threshold_strategy=noise | {"salt_pepper": 0.06} 256 | 6 | 0.1116 ± 0.013 | 0.0592 ± 0.0072 | 0.9849 ± 0.004 | 0.08783 ± 0.0094 |
| nguong_nhieu_mac_dinh_cu | k=3.5, profile=skimage, ratio=0.4, sigma=2.0, threshold_strategy=noise | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.5611 ± 0.061 | 0.3949 ± 0.062 | 0.9858 ± 0.0042 | 0.395 ± 0.063 |
| phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"noise_sigma": 0.15} 256 | 6 | 0.9816 ± 0.018 | 0.9823 ± 0.035 | 0.9816 ± 0.0038 | 0.9459 ± 0.032 |
| phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"salt_pepper": 0.02} 256 | 6 | 0.9901 ± 0.0046 | 0.9967 ± 0.0041 | 0.9836 ± 0.0063 | 0.9593 ± 0.0073 |
| phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"salt_pepper": 0.06} 256 | 6 | 0.9794 ± 0.012 | 0.9746 ± 0.022 | 0.9845 ± 0.0039 | 0.9412 ± 0.022 |
| phan_vi | high_q=0.95, low_q=0.88, profile=skimage, sigma=2.0, threshold_strategy=quantile | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9928 ± 0.002 | 0.9998 ± 0.00045 | 0.9858 ± 0.0042 | 0.9584 ± 0.0098 |
| otsu | profile=skimage, sigma=2.0, threshold_strategy=otsu | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| otsu | profile=skimage, sigma=2.0, threshold_strategy=otsu | {"noise_sigma": 0.15} 256 | 6 | 0.9893 ± 0.0026 | 0.9971 ± 0.0033 | 0.9816 ± 0.0038 | 0.9595 ± 0.0036 |
| otsu | profile=skimage, sigma=2.0, threshold_strategy=otsu | {"salt_pepper": 0.02} 256 | 6 | 0.9911 ± 0.0029 | 0.9987 ± 0.0017 | 0.9836 ± 0.0063 | 0.9593 ± 0.0073 |
| otsu | profile=skimage, sigma=2.0, threshold_strategy=otsu | {"salt_pepper": 0.06} 256 | 6 | 0.9881 ± 0.0041 | 0.9918 ± 0.0074 | 0.9845 ± 0.0039 | 0.957 ± 0.0065 |
| otsu | profile=skimage, sigma=2.0, threshold_strategy=otsu | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9928 ± 0.002 | 0.9998 ± 0.00045 | 0.9858 ± 0.0042 | 0.9584 ± 0.0098 |
| median_33 | profile=skimage, sigma=2.0, threshold_strategy=median | {"noise_sigma": 0.05} 256 | 6 | 0.9923 ± 0.0022 | 0.9998 ± 0.00045 | 0.9849 ± 0.0044 | 0.957 ± 0.0086 |
| median_33 | profile=skimage, sigma=2.0, threshold_strategy=median | {"noise_sigma": 0.15} 256 | 6 | 0.9896 ± 0.0018 | 0.9981 ± 0.0016 | 0.9812 ± 0.0039 | 0.9598 ± 0.0024 |
| median_33 | profile=skimage, sigma=2.0, threshold_strategy=median | {"salt_pepper": 0.02} 256 | 6 | 0.9911 ± 0.0029 | 0.9987 ± 0.0017 | 0.9836 ± 0.0063 | 0.9593 ± 0.0073 |
| median_33 | profile=skimage, sigma=2.0, threshold_strategy=median | {"salt_pepper": 0.06} 256 | 6 | 0.9288 ± 0.085 | 0.8907 ± 0.14 | 0.9845 ± 0.0039 | 0.8619 ± 0.13 |
| median_33 | profile=skimage, sigma=2.0, threshold_strategy=median | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0 ± 0 | 1 ± 0 | 0 ± 0 | 0 ± 0 |
| tien_loc_median | prefilter=median, prefilter_size=3, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.05} 256 | 6 | 0.9972 ± 0.00086 | 1 ± 0 | 0.9944 ± 0.0017 | 0.9527 ± 0.0091 |
| tien_loc_median | prefilter=median, prefilter_size=3, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.15} 256 | 6 | 0.9335 ± 0.12 | 0.9018 ± 0.17 | 0.9907 ± 0.0058 | 0.8789 ± 0.16 |
| tien_loc_median | prefilter=median, prefilter_size=3, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.02} 256 | 6 | 0.996 ± 0.0014 | 1 ± 0 | 0.992 ± 0.0027 | 0.9493 ± 0.0085 |
| tien_loc_median | prefilter=median, prefilter_size=3, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.06} 256 | 6 | 0.9952 ± 0.0025 | 0.9992 ± 0.0012 | 0.9913 ± 0.005 | 0.962 ± 0.0061 |
| tien_loc_median | prefilter=median, prefilter_size=3, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.997 ± 0.00099 | 1 ± 0 | 0.994 ± 0.002 | 0.9483 ± 0.014 |
| tien_loc_median | prefilter=median, prefilter_size=5, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.05} 256 | 6 | 0.9847 ± 0.003 | 0.9996 ± 0.0006 | 0.9702 ± 0.006 | 0.9247 ± 0.012 |
| tien_loc_median | prefilter=median, prefilter_size=5, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.15} 256 | 6 | 0.978 ± 0.0088 | 0.988 ± 0.013 | 0.9684 ± 0.0059 | 0.9501 ± 0.014 |
| tien_loc_median | prefilter=median, prefilter_size=5, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.02} 256 | 6 | 0.9815 ± 0.0034 | 0.9993 ± 0.00065 | 0.9642 ± 0.0064 | 0.9244 ± 0.011 |
| tien_loc_median | prefilter=median, prefilter_size=5, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.06} 256 | 6 | 0.9816 ± 0.0039 | 1 ± 0 | 0.9639 ± 0.0076 | 0.9244 ± 0.011 |
| tien_loc_median | prefilter=median, prefilter_size=5, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9849 ± 0.0022 | 0.9996 ± 0.0006 | 0.9707 ± 0.0047 | 0.9266 ± 0.012 |
| tien_loc_bilateral | prefilter=bilateral, prefilter_diameter=7, prefilter_sigma_color=0.15, prefilter_sigma_space=2.0, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.05} 256 | 6 | 0.9989 ± 0.00059 | 1 ± 0 | 0.9978 ± 0.0012 | 0.9551 ± 0.0079 |
| tien_loc_bilateral | prefilter=bilateral, prefilter_diameter=7, prefilter_sigma_color=0.15, prefilter_sigma_space=2.0, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"noise_sigma": 0.15} 256 | 6 | 0.9359 ± 0.13 | 0.9054 ± 0.19 | 0.9967 ± 0.0023 | 0.882 ± 0.17 |
| tien_loc_bilateral | prefilter=bilateral, prefilter_diameter=7, prefilter_sigma_color=0.15, prefilter_sigma_space=2.0, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.02} 256 | 6 | 0.8414 ± 0.26 | 0.7952 ± 0.3 | 0.9976 ± 0.0023 | 0.7875 ± 0.27 |
| tien_loc_bilateral | prefilter=bilateral, prefilter_diameter=7, prefilter_sigma_color=0.15, prefilter_sigma_space=2.0, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"salt_pepper": 0.06} 256 | 6 | 0.6613 ± 0.27 | 0.5516 ± 0.28 | 0.9959 ± 0.0029 | 0.5617 ± 0.26 |
| tien_loc_bilateral | prefilter=bilateral, prefilter_diameter=7, prefilter_sigma_color=0.15, prefilter_sigma_space=2.0, profile=skimage, sigma=1.0, threshold_strategy=otsu | {"contrast": 0.2, "noise_sigma": 0.01} 256 | 6 | 0.9956 ± 0.0011 | 1 ± 0 | 0.9913 ± 0.0022 | 0.9584 ± 0.0077 |

## Tốt nhất theo lưới tham số (cực đại f1 trung bình, kiểu ODS)

| run | suy giảm | params tốt nhất | f1 |
|---|---|---|---|
| nguong_nhieu_k | {"noise_sigma": 0.05} 256 | {"k": 5.0, "profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.9923 |
| nguong_nhieu_k | {"noise_sigma": 0.15} 256 | {"k": 6.0, "profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.9881 |
| nguong_nhieu_k | {"salt_pepper": 0.02} 256 | {"k": 6.0, "profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.1618 |
| nguong_nhieu_k | {"salt_pepper": 0.06} 256 | {"k": 6.0, "profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.1296 |
| nguong_nhieu_k | {"contrast": 0.2, "noise_sigma": 0.01} 256 | {"k": 5.0, "profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.9928 |
| tien_loc_median | {"noise_sigma": 0.05} 256 | {"prefilter": "median", "prefilter_size": 3, "profile": "skimage", "sigma": 1.0, "threshold_strategy": "otsu"} | 0.9972 |
| tien_loc_median | {"noise_sigma": 0.15} 256 | {"prefilter": "median", "prefilter_size": 5, "profile": "skimage", "sigma": 1.0, "threshold_strategy": "otsu"} | 0.9780 |
| tien_loc_median | {"salt_pepper": 0.02} 256 | {"prefilter": "median", "prefilter_size": 3, "profile": "skimage", "sigma": 1.0, "threshold_strategy": "otsu"} | 0.9960 |
| tien_loc_median | {"salt_pepper": 0.06} 256 | {"prefilter": "median", "prefilter_size": 3, "profile": "skimage", "sigma": 1.0, "threshold_strategy": "otsu"} | 0.9952 |
| tien_loc_median | {"contrast": 0.2, "noise_sigma": 0.01} 256 | {"prefilter": "median", "prefilter_size": 3, "profile": "skimage", "sigma": 1.0, "threshold_strategy": "otsu"} | 0.9970 |
