# 10. Bộ phát hiện đối chứng: Sobel-ngưỡng và LoG (T1b)

**WHAT.**
- `sobel_threshold`: cạnh = $\{M_{L2}>T\}$ trên ảnh [0,1], có thể làm trơn σ trước. Không có NMS, không có hysteresis.
- `log_zero_crossing`: điểm qua 0 của $\nabla^2(G_\sigma*I)$ (Laplacian 4 lân cận), xét cặp lân cận phải và dưới, có chênh lệch $>T_{rel}\max|\text{LoG}|$; đánh dấu pixel có |giá trị| nhỏ hơn.

**WHY.** Đề (T1b) yêu cầu so sánh Canny với Sobel và Laplacian. Hai bộ này cho thấy vai trò của từng thành phần trong Canny: NMS (cạnh mỏng), hysteresis (liên tục), Gaussian (chống nhiễu).

**MATHEMATICS.** [../mathematics/02](../mathematics/02_gradient_va_laplacian.md).

**IMPLEMENTATION.** [`core/baselines.py`](../../src/cannylab/core/baselines.py); bản Optimized trong [`backends.py`](../../src/cannylab/backends.py): `sobel_threshold_opencv` trùng tuyệt đối, `log_scipy` dùng `ndi.gaussian_laplace`.

**LIMITATIONS.** Sobel-ngưỡng cho cạnh dày 2–3 px nên Precision một-một thấp. Ngưỡng tương đối của LoG phụ thuộc vào cực đại của ảnh.

**KẾT QUẢ.**
- `test_baselines_manual_vs_optimized`: Sobel trùng tuyệt đối. LoG IoU > 0.85 (Manual dùng Laplacian rời rạc 3×3 trên ảnh đã làm trơn, SciPy dùng đạo hàm bậc 2 giải tích của Gaussian; khác biệt là bản chất).
- So sánh định lượng ở X4.
