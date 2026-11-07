# Phase 3: Mở rộng dự án

Ngày: 2026-11-07 · Đầu vào: [phase1](phase1_assignment_analysis.md), [phase2](phase2_mathematical_analysis.md)

Nguyên tắc chọn module: **mỗi module phải trả lời một yêu cầu cụ thể của đề** (mã P/T/E ở Phase 1), hoặc là hạ tầng bắt buộc để trả lời yêu cầu đó (dữ liệu, thực nghiệm, đánh giá). Không thêm module chỉ để cho đủ danh sách.

## 1. Module được chọn

| Module (theo đề cương) | Làm gì trong dự án | Trả lời |
|---|---|---|
| Image Acquisition | Đọc file ảnh/video, kiểm tra hợp lệ; ảnh mẫu từ `skimage.data` | P1, P3 |
| Image Preprocessing | Chuyển xám/màu có kiểm soát BGR/RGB, chuẩn hóa dtype và miền giá trị | P1, quy tắc 4–6 |
| Image Restoration (mức tiền lọc) | Tiền lọc thay cho Gaussian: median, bilateral | E2 |
| Spatial Domain Processing / Filtering | Tích chập tách được tự viết, Gaussian, Sobel, các chế độ biên | T1a, P1 |
| Edge Detection | Canny tự cài (2 hồ sơ), wrapper OpenCV/skimage, baseline Sobel-ngưỡng và LoG zero-crossing, Canny màu (3 cách + Di Zenzo) | P1, T1b, E3 |
| Image Segmentation | Cạnh → closing → tô kín → vùng; hình thái học tự viết + OpenCV | P4a |
| Feature Extraction / Image Analysis | Contour → Douglas–Peucker tự viết → số đỉnh + độ tròn → nhãn hình | P4b |
| Image Quality Assessment | PSNR để mô tả mức suy giảm của đầu vào (không dùng chấm cạnh) | P3, E1 |
| Batch Processing | Chạy pipeline trên cả bộ dữ liệu và trên chuỗi khung hình video | P3, E4 |
| Experiment Management | Cấu hình TOML → lưới tham số → chạy → lưu cấu hình + môi trường + kết quả | P2, P3, E2 |
| Dataset Management | Bộ sinh ảnh tổng hợp có ground truth, nạp ảnh thư mục, metadata, thống kê, chia tập | E1, P3, P4 |
| Evaluation | Chỉ số cạnh có dung sai, FOM, chamfer/HD95, liên tục, IoU/Dice, accuracy, hiệu năng | E1, T3a |

## 2. Module KHÔNG làm, và lý do
| Module | Lý do |
|---|---|
| Frequency Domain Processing | Không có bước nào của Canny cần FFT. Gaussian tách được ($O(Nk)$) nhanh hơn FFT ở mọi σ ≤ 5 đang dùng. Phần lý thuyết tần số đã có trong tài liệu toán học |
| Image Transformation (hình học) | Không thuộc đề Canny |
| Image Enhancement (cân bằng histogram, CLAHE) | Ảnh tương phản thấp (P3) được giải quyết bằng ngưỡng phân vị hoặc ngưỡng theo nhiễu, vốn bất biến với tương phản, không cần biến đổi ảnh. Thêm CLAHE làm khuếch đại nhiễu và lẫn tác động |
| PyTorch | Không có thành phần học sâu; tích chập NumPy/OpenCV đủ nhanh. Thêm PyTorch vi phạm quy tắc "không tạo module dư thừa" |
| GUI, webcam, dashboard | Ngoài phạm vi theo đề cương |
| BSDS500 | Theo quyết định ở Phase 1: không tải dữ liệu ngoài |

## 3. Các biến thể cải thiện Canny (E2) được chọn để thực nghiệm
| Biến thể | Cơ sở | Kỳ vọng kiểm chứng |
|---|---|---|
| Ngưỡng theo nhiễu: $T_h=k\hat\sigma_g$, $\hat\sigma_g=1.6\hat\sigma_n/\sigma^2$, $\hat\sigma_n$ ước lượng bằng MAD | [04](../mathematics/04_nguong_kep_va_hysteresis.md) | F1 ổn định hơn ngưỡng cố định khi mức nhiễu thay đổi |
| Ngưỡng phân vị (`use_quantiles`) | [04](../mathematics/04_nguong_kep_va_hysteresis.md) | Tốt trên ảnh tương phản thấp |
| Otsu trên độ lớn gradient | [04](../mathematics/04_nguong_kep_va_hysteresis.md) | Đối chứng |
| Median ± 33% (heuristic trên mạng) | Không có cơ sở | Baseline để so |
| Tiền lọc median / bilateral thay Gaussian | [01](../mathematics/01_lam_tron_gaussian.md) LIMITATIONS | Median thắng trên nhiễu muối tiêu |

## 4. Bộ dữ liệu tổng hợp (thay BSDS)
Mọi ảnh sinh ra từ **mô tả hình học** nên ground truth cạnh chính xác: cạnh = biên giữa các vùng của ảnh nhãn, lấy theo quy ước pixel phía trong (giống OpenCV khi gặp điểm hòa).

| Họ ảnh | Nội dung | Phục vụ |
|---|---|---|
| `step` | Cạnh bậc thang thẳng, góc tùy ý | Định vị, đẳng hướng L1/L2 |
| `shapes` | Hình tròn, tam giác, chữ nhật, ngũ giác, lục giác, sao trên nền; có mặt nạ vùng và nhãn | P4a, P4b, E1 |
| `texture` | Hình trên nền nhiều chi tiết (nhiễu có tương quan) | P3 "nhiều chi tiết" |
| `isoluminant` | Vùng màu cùng độ sáng | E3 |
| Suy giảm | Nhiễu Gauss σₙ, nhiễu muối tiêu, giảm tương phản, mờ | P3, độ bền |
| `video` | Cảnh tĩnh + nhiễu độc lập mỗi khung; vật chuyển động theo quỹ đạo đã biết | E4 |

Ảnh thật từ `skimage.data` (`camera`, `coins`, `astronaut`, `grass`) dùng để **tái tạo kết quả Lab3** và đánh giá định tính, không tính chỉ số có ground truth.

## 5. Danh sách thực nghiệm (sẽ có file cấu hình riêng)
| Mã | Câu hỏi | Thiết kế |
|---|---|---|
| X1 | P1: hai thư viện và bản tự cài khớp nhau tới đâu? | 3 bản cài × ảnh thật + tổng hợp; IoU, số pixel lệch |
| X2 | P2/T2: ảnh hưởng σ, $T_l$, $T_h$ | Quét từng tham số, giữ cố định phần còn lại; F1, FOM, độ liên tục |
| X3 | P3: ba loại ảnh khó | Họ ảnh × mức suy giảm × (mặc định, đã chỉnh, ngưỡng tự động) |
| X4 | T1b/T3a: Canny so với Sobel và LoG | Độ chính xác, độ bền theo σₙ, thời gian |
| X5 | P4a/P4b: phân đoạn và nhận dạng hình | IoU vùng, đếm đối tượng, accuracy, ma trận nhầm lẫn |
| X6 | E2: các biến thể cải thiện | Theo bảng ở mục 3 |
| X7 | E3: ảnh màu | 4 cách × ảnh đẳng sáng + ảnh màu thật |
| X8 | E4: video | Theo từng khung so với làm trơn theo thời gian; độ ổn định, fps |
| X9 | Hiệu năng Manual vs Optimized | Thời gian, bộ nhớ theo kích thước ảnh; hệ số mũ log–log |

## Báo cáo phase
- **Hoàn thành:** chọn 12 module, loại 6, chọn 5 biến thể E2, 6 họ dữ liệu, 9 thực nghiệm.
- **Quyết định:** không làm miền tần số, CLAHE, PyTorch; ground truth theo quy ước pixel phía trong.
- **Vấn đề:** ground truth của ảnh có mờ quang học: cạnh vẫn nằm ở biên vùng gốc. Ghi rõ trong tài liệu dữ liệu.
- **Tiếp theo:** Phase 4, kiến trúc.
