# Phase 8: Evaluation Engine

- **Hoàn thành:**
  - P/R/F1 có dung sai, khớp nhiều-một (biến đổi khoảng cách) và một-một (Hopcroft–Karp trên đồ thị thưa);
  - FOM, Chamfer, HD95; độ liên tục; agreement;
  - IoU / Dice / ghép đối tượng Hungarian;
  - accuracy, macro-F1, ma trận nhầm lẫn;
  - PSNR (chỉ cho đầu vào);
  - thời gian (trung vị, IQR), bộ nhớ (tracemalloc, ΔRSS), khớp hệ số mũ.
- **Quyết định:**
  - báo cáo cả hai kiểu khớp;
  - chỉ báo IoU (Dice là hàm của IoU);
  - định nghĩa rõ ca biên;
  - bộ nhớ chỉ đọc như cận dưới.

  Tài liệu: [../evaluation/README.md](../evaluation/README.md).
- **Kết quả kiểm thử:** 14 test giá trị đã biết. Ví dụ: lệch 1 px cho F1 = 1 với τ = 1.5 và 0 với τ = 0.5; cạnh kép cho P = 1 (nhiều-một) nhưng 0.5 (một-một); FOM = 1/(1 + 1/9).
- **Phát hiện có giá trị:** X4 xác nhận sự cần thiết của khớp một-một. Sobel-ngưỡng đạt F1 nhiều-một 0.946 nhưng F1 một-một chỉ 0.689.
- **Tiếp theo:** Phase 9, kiểm thử và tối ưu.
