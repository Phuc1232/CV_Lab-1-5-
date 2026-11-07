# Phase 2: Phân tích toán học (báo cáo)

Ngày: 2026-11-07 · Trạng thái: chờ duyệt · Tài liệu chính: [../mathematics/README.md](../mathematics/README.md)

## Công việc hoàn thành
- 7 tài liệu toán học (`docs/mathematics/01`–`07`) bao phủ cả 5 bước của Canny, so sánh với Laplacian/LoG, khác biệt giữa hai thư viện, chỉ số đánh giá, ảnh màu và video.
- Script kiểm chứng `docs/mathematics/kiem_chung/kiem_chung_phase2.py` kiểm 15 mệnh đề bằng số; log và JSON được lưu kèm.

## Quyết định kỹ thuật
1. **Hai hồ sơ cho bản tự cài** (`opencv`, `skimage`), mỗi hồ sơ là một bộ lựa chọn: làm trơn, biên, thang, chuẩn, kiểu NMS, quy ước điểm hòa, `>`/`>=`, viền.
2. **Tiêu chí chấp nhận Phase 6:** hồ sơ `opencv` trùng từng pixel với `cv2.Canny`; hồ sơ `skimage` trùng từng pixel trừ các điểm có $|M-T|<10^{-9}$. So sánh giữa hai thư viện thì không yêu cầu trùng (IoU khoảng 0.78–0.88 là bản chất).
3. **Hysteresis:** Manual dùng BFS, Optimized dùng `ndi.label`. Hai bản phải trùng tuyệt đối, vì đã chứng minh chúng tương đương.
4. **Thuộc tính kiểm thử:** cạnh đơn điệu theo ngưỡng (property-based test); Di Zenzo với 1 kênh quy về Canny xám; `cv2.Canny` 3 kênh tương đương "chọn kênh max L1".
5. **Chỉ số:** P/R/F có dung sai (cả khớp nhiều-một lẫn một-một), FOM, chamfer/HD95, chỉ số liên tục. Không dùng PSNR/SSIM/MSE cho bản đồ cạnh. IoU và Dice chỉ báo cáo một trong hai.
6. **Ngưỡng theo nhiễu** ($T_h=k\sigma_g$, $\sigma_g\approx1.6\sigma_n/\sigma^2$) là ứng viên có cơ sở toán học cho E2. "Median ± 33%" chỉ giữ làm baseline.

## Vấn đề phát sinh
- Phép đo ksize trên ảnh `uint8` lần đầu cho kết quả sai: kernel nhỏ hơn vẫn trùng vì hệ số ngoài cùng bị làm tròn về 0. Đã đổi sang kiểm tra "kernel dự đoán trùng với auto".
- Thử ngưỡng đúng bằng $M$ trên skimage ra 0 pixel, ngược với mã nguồn (`>=`). Nguyên nhân là bước chuẩn hóa bleed-over làm $M$ lệch $-8.9\times10^{-16}$. Bài học được ghi vào tài liệu 04.
- Thử Canny 3 kênh lần đầu lệch 98 px, toàn bộ do Sobel mặc định dùng biên reflect-101. Đổi sang replicate thì 0 px lệch. Đây cũng là bằng chứng OpenCV dùng biên replicate bên trong.
- Các công thức liên tục (đỉnh gradient, SNR) sai lệch đáng kể khi σ < 2. Tài liệu ghi rõ miền áp dụng.

## Kết quả kiểm thử
15/15 mệnh đề khớp lý thuyết trong miền áp dụng đã nêu. Bảng tóm tắt ở [../mathematics/README.md](../mathematics/README.md).

## Công việc tiếp theo: Phase 3 (mở rộng dự án)
Chọn module dựa trên tài liệu này: bộ sinh ảnh tổng hợp có ground truth, các biến thể E2 (ngưỡng theo nhiễu, phân vị, Otsu, tiền lọc bilateral/median), Canny màu (3 cách + Di Zenzo), video tổng hợp, baseline Sobel/LoG. Đồng thời quyết định không làm những gì.
