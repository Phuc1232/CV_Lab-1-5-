# Evaluation Engine

Mã: [`src/cannylab/evaluation/`](../../src/cannylab/evaluation/). Định nghĩa toán và phân tích giới hạn: [../mathematics/06_chi_so_danh_gia.md](../mathematics/06_chi_so_danh_gia.md). Quy tắc: [rules/evaluation-rules.md](../../rules/evaluation-rules.md).

## Câu hỏi → chỉ số
| Câu hỏi | Chỉ số | Hàm |
|---|---|---|
| Cạnh đúng hay sai? (E1, P3) | Precision / Recall / F1 có dung sai τ (khớp nhiều-một) | `edge_metrics` |
| Có phản hồi kép, cạnh dày không? (tiêu chí đáp ứng đơn) | P/R/F1 **một-một** (ghép cực đại Hopcroft–Karp) | `edge_metrics` (`*_1to1`) |
| Vị trí chính xác không? (T2a) | Pratt FOM (α = 1/9), Chamfer, HD95 | `edge_metrics` |
| Cạnh liền hay vụn? (T2b) | số thành phần, độ dài TB, phân mảnh / 100 px | `continuity` |
| Bản tự cài có khớp thư viện? (P1) | n_diff, IoU pixel, identical | `agreement` |
| Tìm đủ vùng? (P4a) | P/R/F1 đối tượng (Hungarian, IoU ≥ 0.5), IoU TB, sai số đếm | `match_objects` |
| Gọi đúng tên hình? (P4b) | accuracy, macro-F1, ma trận nhầm lẫn | `classification_report` |
| Đầu vào suy giảm đến mức nào? | PSNR (chỉ mô tả đầu vào) | `psnr` |
| Nhanh và tốn bộ nhớ không? (T3a) | trung vị + IQR thời gian, đỉnh tracemalloc, ΔRSS, hệ số mũ log–log | `measure`, `fit_power_law` |

## Lựa chọn và lý do
- **τ = 1.5 px.** Biên của ảnh tổng hợp nằm giữa hai pixel, nên cạnh lệch 1 px vẫn đúng. Chọn 1.5 thay vì 1 để chấp nhận cả lệch chéo (√2).
- **Báo cáo cả hai kiểu khớp.** Thực nghiệm X4 cho thấy điều này cần thiết: Sobel + ngưỡng có F1 nhiều-một ngang Canny nhưng F1 một-một thấp hơn hẳn, vì cạnh dày 2–3 px ([../experiments/README.md](../experiments/README.md)).
- **F1 tốt nhất trên lưới (kiểu ODS).** So sánh hai phương pháp ở một bộ ngưỡng tùy ý là không công bằng, vì các phương pháp có thang ngưỡng khác nhau. Bảng "tốt nhất theo lưới" trong `summary.md` chọn cấu hình tốt nhất cho từng (run, mức suy giảm). Đây là giới hạn trên mang tính lạc quan, vì ngưỡng được chọn trên chính dữ liệu đánh giá; ODS thật cần tách tập (`Dataset.split`).
- **Ca biên** (có test): P = G = ∅ cho P = R = F1 = 1; P = ∅ cho P = 1, R = 0; G = ∅ cho P = 0, R = 1.
- **Không dùng** MSE/PSNR/SSIM cho bản đồ cạnh, cũng không dùng entropy hay Tenengrad. Lý do ở [06 §7](../mathematics/06_chi_so_danh_gia.md).

## Giới hạn đo hiệu năng
`tracemalloc` không thấy bộ nhớ C++ của OpenCV (đỉnh "16 MB" của cv2 ở ảnh 1024² thực ra là mảng kết quả NumPy). ΔRSS gần 0 vì allocator giữ lại bộ nhớ của các lần chạy trước. Vì vậy hai số bộ nhớ chỉ nên đọc như **cận dưới**, không dùng để xếp hạng chính xác.
