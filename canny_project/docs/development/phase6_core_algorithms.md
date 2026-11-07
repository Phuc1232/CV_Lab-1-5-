# Phase 6: Thuật toán cốt lõi

- **Hoàn thành (Manual, `core/`):** tương quan + 4 chế độ biên; Gaussian + bleed-over; Sobel; NMS lượng tử hóa (quy tắc OpenCV) và nội suy (quy tắc skimage); ngưỡng kép, BFS hysteresis; pipeline Canny 2 hồ sơ; 4 chiến lược ngưỡng tự động; Canny màu (4 cách); Sobel-ngưỡng, LoG; hình thái học, lấp lỗ, gán nhãn; median, bilateral. **Analysis:** phân đoạn; dò biên Moore, Douglas–Peucker, bao lồi, đặc trưng, phân loại 6 hình. **Optimized (`backends.py`):** OpenCV / skimage / SciPy cho từng thuật toán. **Registry:** 10 bộ phát hiện cạnh.
- **Quyết định:**
  - NMS vector hóa hoàn toàn;
  - NMS tính **một lần** độc lập với ngưỡng, rồi mới áp ngưỡng (để có chiến lược ngưỡng tự động);
  - hồ sơ OpenCV-L2 so bình phương nguyên với $\lfloor T^2\rfloor$ đúng như OpenCV;
  - `canny_manual_fast` = Canny tự cài + hysteresis bằng `ndi.label`.
- **Kết quả kiểm chứng:**
  - **0 px lệch** với `cv2.Canny` (L1, L2, ngưỡng hoán đổi, ngưỡng lẻ) và `skimage.feature.canny` (σ 0.5–3, uint8 / uint16 / float, quantile) trên 4–6 ảnh thật;
  - Canny 3 kênh 0 px lệch; BFS ≡ label;
  - hình thái học ≡ OpenCV; SE ≡ `getStructuringElement` (3 hình × 8 kích thước).
- **Vấn đề phát sinh:** nhận dạng hình lần đầu chỉ đúng 62/72, do ba lỗi:
  - tách DP tại điểm đầu raster;
  - độ đặc tính sai hệ tọa độ;
  - góc nhọn bị cắt cụt.

  Sau khi sửa và thêm lượt 2 của DP như `cv2.approxPolyDP`: Manual 180/180, OpenCV 178/180 trên ảnh sạch. Chi tiết: [../algorithms/07](../algorithms/07_phan_doan_nhan_dang.md).
- **Tiếp theo:** Phase 7, dữ liệu và Experiment Engine.
