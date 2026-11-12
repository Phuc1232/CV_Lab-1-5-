# x2_tham_so

Câu hỏi: **P2, T2a, T2b**

Khảo sát từng tham số, giữ cố định phần còn lại (cấu hình cơ sở: sigma=1.5, T_l=0.05, T_h=0.15, thang [0,1]).
sigma_nguong_co_dinh kiểm chứng hệ quả "đỉnh gradient ∝ 1/sigma" (docs/mathematics/01): ngưỡng cố định thì
sigma lớn làm mất cạnh; sigma_nguong_nhieu dùng ngưỡng tỉ lệ với độ lệch chuẩn nhiễu của gradient.

| run | params | suy giảm | n | f1 | precision | recall | fom | fragmentation |
|---|---|---|---|---|---|---|---|---|
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=0.5 | {"noise_sigma": 0.05} 256 | 6 | 0.06659 ± 0.0051 | 0.03445 ± 0.0027 | 0.9998 ± 0.00039 | 0.09204 ± 0.0069 | 3.137 ± 0.24 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=0.5 | {"noise_sigma": 0.15} 256 | 6 | 0.07892 ± 0.0049 | 0.04109 ± 0.0027 | 0.9996 ± 0.00058 | 0.1013 ± 0.0075 | 3.324 ± 0.25 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=1.0 | {"noise_sigma": 0.05} 256 | 6 | 0.08529 ± 0.0074 | 0.04456 ± 0.004 | 0.9986 ± 0.00082 | 0.08326 ± 0.0061 | 0.9422 ± 0.058 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=1.0 | {"noise_sigma": 0.15} 256 | 6 | 0.07235 ± 0.0051 | 0.03754 ± 0.0028 | 0.9986 ± 0.00078 | 0.0906 ± 0.0068 | 2.122 ± 0.07 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9938 ± 0.0069 | 0.9927 ± 0.013 | 0.995 ± 0.0025 | 0.9573 ± 0.011 | 0.7685 ± 0.08 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.09092 ± 0.0074 | 0.04766 ± 0.0041 | 0.9938 ± 0.003 | 0.08785 ± 0.0068 | 1.861 ± 0.031 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=2.0 | {"noise_sigma": 0.05} 256 | 6 | 0.993 ± 0.0015 | 1 ± 0 | 0.9862 ± 0.003 | 0.957 ± 0.0069 | 0.7999 ± 0.12 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=2.0 | {"noise_sigma": 0.15} 256 | 6 | 0.2085 ± 0.026 | 0.1168 ± 0.016 | 0.9835 ± 0.0039 | 0.1401 ± 0.017 | 1.753 ± 0.077 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=3.0 | {"noise_sigma": 0.05} 256 | 6 | 0.9727 ± 0.0029 | 0.9968 ± 0.0014 | 0.9497 ± 0.0061 | 0.9343 ± 0.012 | 0.8597 ± 0.16 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=3.0 | {"noise_sigma": 0.15} 256 | 6 | 0.9705 ± 0.006 | 0.9951 ± 0.0024 | 0.947 ± 0.01 | 0.9368 ± 0.011 | 0.8324 ± 0.17 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=4.0 | {"noise_sigma": 0.05} 256 | 6 | 0.9308 ± 0.0086 | 0.9791 ± 0.0042 | 0.8871 ± 0.013 | 0.8871 ± 0.019 | 0.9614 ± 0.19 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=4.0 | {"noise_sigma": 0.15} 256 | 6 | 0.9294 ± 0.0081 | 0.9757 ± 0.0051 | 0.8873 ± 0.012 | 0.8885 ± 0.014 | 1.006 ± 0.19 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=5.0 | {"noise_sigma": 0.05} 256 | 6 | 0.8679 ± 0.015 | 0.9351 ± 0.0098 | 0.8098 ± 0.02 | 0.8378 ± 0.016 | 1.009 ± 0.16 |
| sigma_nguong_co_dinh | high_threshold=0.15, low_threshold=0.05, profile=skimage, sigma=5.0 | {"noise_sigma": 0.15} 256 | 6 | 0.8601 ± 0.026 | 0.9396 ± 0.0069 | 0.794 ± 0.041 | 0.8187 ± 0.045 | 1.038 ± 0.14 |
| sigma_nguong_nhieu | profile=skimage, sigma=0.5, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9788 ± 0.0089 | 0.9588 ± 0.017 | 0.9998 ± 0.00039 | 0.9387 ± 0.015 | 0.9881 ± 0.13 |
| sigma_nguong_nhieu | profile=skimage, sigma=0.5, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.7989 ± 0.18 | 0.8715 ± 0.02 | 0.782 ± 0.25 | 0.7887 ± 0.22 | 1.536 ± 0.54 |
| sigma_nguong_nhieu | profile=skimage, sigma=1.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9911 ± 0.01 | 0.9839 ± 0.02 | 0.9986 ± 0.00082 | 0.9512 ± 0.018 | 0.8078 ± 0.11 |
| sigma_nguong_nhieu | profile=skimage, sigma=1.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.956 ± 0.024 | 0.9226 ± 0.039 | 0.9926 ± 0.0065 | 0.9162 ± 0.028 | 0.9455 ± 0.15 |
| sigma_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9948 ± 0.0049 | 0.9947 ± 0.0086 | 0.995 ± 0.0025 | 0.9592 ± 0.0073 | 0.7701 ± 0.081 |
| sigma_nguong_nhieu | profile=skimage, sigma=1.5, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9731 ± 0.015 | 0.9545 ± 0.027 | 0.9928 ± 0.0035 | 0.9294 ± 0.024 | 0.9066 ± 0.2 |
| sigma_nguong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9895 ± 0.0057 | 0.993 ± 0.012 | 0.9862 ± 0.003 | 0.9531 ± 0.011 | 0.8323 ± 0.1 |
| sigma_nguong_nhieu | profile=skimage, sigma=2.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9773 ± 0.013 | 0.9714 ± 0.024 | 0.9835 ± 0.0039 | 0.9386 ± 0.025 | 0.9139 ± 0.24 |
| sigma_nguong_nhieu | profile=skimage, sigma=3.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9673 ± 0.0088 | 0.9858 ± 0.019 | 0.9497 ± 0.0061 | 0.9306 ± 0.016 | 0.8891 ± 0.14 |
| sigma_nguong_nhieu | profile=skimage, sigma=3.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.962 ± 0.012 | 0.9776 ± 0.018 | 0.947 ± 0.01 | 0.9287 ± 0.0095 | 0.9174 ± 0.26 |
| sigma_nguong_nhieu | profile=skimage, sigma=4.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.9308 ± 0.0086 | 0.9791 ± 0.0042 | 0.8871 ± 0.013 | 0.8871 ± 0.019 | 0.9614 ± 0.19 |
| sigma_nguong_nhieu | profile=skimage, sigma=4.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.9252 ± 0.012 | 0.9665 ± 0.015 | 0.8873 ± 0.012 | 0.8889 ± 0.014 | 1.085 ± 0.25 |
| sigma_nguong_nhieu | profile=skimage, sigma=5.0, threshold_strategy=noise | {"noise_sigma": 0.05} 256 | 6 | 0.8594 ± 0.027 | 0.9155 ± 0.037 | 0.8098 ± 0.02 | 0.8381 ± 0.016 | 1.1 ± 0.28 |
| sigma_nguong_nhieu | profile=skimage, sigma=5.0, threshold_strategy=noise | {"noise_sigma": 0.15} 256 | 6 | 0.8678 ± 0.013 | 0.9361 ± 0.0086 | 0.8089 ± 0.017 | 0.8365 ± 0.016 | 1.149 ± 0.2 |
| nguong_thap | high_threshold=0.15, low_threshold=0.01, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9912 ± 0.0094 | 0.9876 ± 0.018 | 0.995 ± 0.0025 | 0.9528 ± 0.015 | 0.7646 ± 0.081 |
| nguong_thap | high_threshold=0.15, low_threshold=0.01, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.08979 ± 0.0072 | 0.04704 ± 0.004 | 0.9938 ± 0.003 | 0.08726 ± 0.0067 | 1.799 ± 0.032 |
| nguong_thap | high_threshold=0.15, low_threshold=0.02, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9912 ± 0.0094 | 0.9876 ± 0.018 | 0.995 ± 0.0025 | 0.9528 ± 0.015 | 0.7646 ± 0.081 |
| nguong_thap | high_threshold=0.15, low_threshold=0.02, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.08979 ± 0.0072 | 0.04704 ± 0.004 | 0.9938 ± 0.003 | 0.08726 ± 0.0067 | 1.799 ± 0.032 |
| nguong_thap | high_threshold=0.15, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9933 ± 0.0076 | 0.9918 ± 0.014 | 0.995 ± 0.0025 | 0.9564 ± 0.012 | 0.7677 ± 0.08 |
| nguong_thap | high_threshold=0.15, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.09019 ± 0.0073 | 0.04725 ± 0.004 | 0.9938 ± 0.003 | 0.08744 ± 0.0067 | 1.816 ± 0.027 |
| nguong_thap | high_threshold=0.15, low_threshold=0.06, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9949 ± 0.0047 | 0.9949 ± 0.0082 | 0.995 ± 0.0025 | 0.9594 ± 0.007 | 0.7703 ± 0.081 |
| nguong_thap | high_threshold=0.15, low_threshold=0.06, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.0925 ± 0.0075 | 0.04852 ± 0.0041 | 0.9938 ± 0.003 | 0.08871 ± 0.0069 | 1.962 ± 0.044 |
| nguong_thap | high_threshold=0.15, low_threshold=0.09, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9957 ± 0.0034 | 0.9964 ± 0.0054 | 0.995 ± 0.0025 | 0.9608 ± 0.0048 | 0.7715 ± 0.082 |
| nguong_thap | high_threshold=0.15, low_threshold=0.09, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.1042 ± 0.0079 | 0.05501 ± 0.0044 | 0.9938 ± 0.003 | 0.09511 ± 0.0071 | 3.032 ± 0.18 |
| nguong_thap | high_threshold=0.15, low_threshold=0.12, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9968 ± 0.0018 | 0.9987 ± 0.0019 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.7734 ± 0.082 |
| nguong_thap | high_threshold=0.15, low_threshold=0.12, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.1282 ± 0.0092 | 0.06852 ± 0.0053 | 0.9938 ± 0.003 | 0.1085 ± 0.0075 | 6.164 ± 0.41 |
| nguong_thap | high_threshold=0.15, low_threshold=0.15, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9972 ± 0.0015 | 0.9994 ± 0.00089 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.774 ± 0.082 |
| nguong_thap | high_threshold=0.15, low_threshold=0.15, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.1708 ± 0.014 | 0.0935 ± 0.0082 | 0.9936 ± 0.003 | 0.133 ± 0.01 | 13.27 ± 1 |
| nguong_cao | high_threshold=0.05, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.1207 ± 0.011 | 0.06429 ± 0.0061 | 0.995 ± 0.0025 | 0.0969 ± 0.0085 | 5.793 ± 0.11 |
| nguong_cao | high_threshold=0.05, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.08379 ± 0.0066 | 0.04375 ± 0.0036 | 0.9938 ± 0.003 | 0.08598 ± 0.0066 | 3.795 ± 0.12 |
| nguong_cao | high_threshold=0.08, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.1772 ± 0.019 | 0.09739 ± 0.011 | 0.995 ± 0.0025 | 0.1241 ± 0.014 | 3.018 ± 0.13 |
| nguong_cao | high_threshold=0.08, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.08509 ± 0.0068 | 0.04446 ± 0.0037 | 0.9938 ± 0.003 | 0.08651 ± 0.0067 | 2.984 ± 0.039 |
| nguong_cao | high_threshold=0.12, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.7382 ± 0.078 | 0.5928 ± 0.099 | 0.995 ± 0.0025 | 0.5834 ± 0.096 | 1.305 ± 0.25 |
| nguong_cao | high_threshold=0.12, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.08761 ± 0.0071 | 0.04584 ± 0.0039 | 0.9938 ± 0.003 | 0.08694 ± 0.0068 | 2.203 ± 0.028 |
| nguong_cao | high_threshold=0.18, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9975 ± 0.0012 | 1 ± 0 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.7365 ± 0.089 |
| nguong_cao | high_threshold=0.18, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.09406 ± 0.0074 | 0.04939 ± 0.0041 | 0.9934 ± 0.0032 | 0.08845 ± 0.0066 | 1.494 ± 0.059 |
| nguong_cao | high_threshold=0.25, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9975 ± 0.0012 | 1 ± 0 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.7365 ± 0.089 |
| nguong_cao | high_threshold=0.25, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.1129 ± 0.0092 | 0.0599 ± 0.0052 | 0.9928 ± 0.0035 | 0.09388 ± 0.0055 | 0.8421 ± 0.072 |
| nguong_cao | high_threshold=0.35, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9975 ± 0.0012 | 1 ± 0 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.7365 ± 0.089 |
| nguong_cao | high_threshold=0.35, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.4961 ± 0.2 | 0.3565 ± 0.2 | 0.9928 ± 0.0035 | 0.3709 ± 0.19 | 0.615 ± 0.25 |
| nguong_cao | high_threshold=0.5, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.05} 256 | 6 | 0.9975 ± 0.0012 | 1 ± 0 | 0.995 ± 0.0025 | 0.9628 ± 0.0034 | 0.7365 ± 0.089 |
| nguong_cao | high_threshold=0.5, low_threshold=0.04, profile=skimage, sigma=1.5 | {"noise_sigma": 0.15} 256 | 6 | 0.8685 ± 0.16 | 0.804 ± 0.22 | 0.9928 ± 0.0035 | 0.7924 ± 0.21 | 0.6257 ± 0.18 |

## Tốt nhất theo lưới tham số (cực đại f1 trung bình, kiểu ODS)

| run | suy giảm | params tốt nhất | f1 |
|---|---|---|---|
| sigma_nguong_co_dinh | {"noise_sigma": 0.05} 256 | {"high_threshold": 0.15, "low_threshold": 0.05, "profile": "skimage", "sigma": 1.5} | 0.9938 |
| sigma_nguong_co_dinh | {"noise_sigma": 0.15} 256 | {"high_threshold": 0.15, "low_threshold": 0.05, "profile": "skimage", "sigma": 3.0} | 0.9705 |
| sigma_nguong_nhieu | {"noise_sigma": 0.05} 256 | {"profile": "skimage", "sigma": 1.5, "threshold_strategy": "noise"} | 0.9948 |
| sigma_nguong_nhieu | {"noise_sigma": 0.15} 256 | {"profile": "skimage", "sigma": 2.0, "threshold_strategy": "noise"} | 0.9773 |
| nguong_thap | {"noise_sigma": 0.05} 256 | {"high_threshold": 0.15, "low_threshold": 0.15, "profile": "skimage", "sigma": 1.5} | 0.9972 |
| nguong_thap | {"noise_sigma": 0.15} 256 | {"high_threshold": 0.15, "low_threshold": 0.15, "profile": "skimage", "sigma": 1.5} | 0.1708 |
| nguong_cao | {"noise_sigma": 0.05} 256 | {"high_threshold": 0.18, "low_threshold": 0.04, "profile": "skimage", "sigma": 1.5} | 0.9975 |
| nguong_cao | {"noise_sigma": 0.15} 256 | {"high_threshold": 0.5, "low_threshold": 0.04, "profile": "skimage", "sigma": 1.5} | 0.8685 |

## PSNR ảnh đầu vào (mô tả mức suy giảm)

- {"noise_sigma": 0.05}: 26.06 dB
- {"noise_sigma": 0.15}: 16.92 dB
