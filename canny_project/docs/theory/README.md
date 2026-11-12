# Trả lời các câu hỏi lý thuyết của đề (phần I và III)

Mỗi câu trả lời dựa trên tài liệu toán ([../mathematics/](../mathematics/README.md)) và **số liệu thực nghiệm** ([../experiments/](../experiments/README.md)). Tài liệu chỉ nói những gì đã đo được, không suy diễn thuần lý thuyết.

## I.1a. Các bước của Canny
| Bước | Làm gì | Vì sao | Chi tiết |
|---|---|---|---|
| 1. Giảm nhiễu | $G_\sigma*I$ | Đạo hàm khuếch đại nhiễu; làm trơn tăng SNR ≈ 2hσ/σₙ | [01](../mathematics/01_lam_tron_gaussian.md) |
| 2. Gradient | Sobel → $M$, $\theta$ | Cạnh là nơi cường độ đổi nhanh nhất | [02](../mathematics/02_gradient_va_laplacian.md) |
| 3. NMS | giữ cực đại theo hướng θ | Tiêu chí *đáp ứng đơn*: cạnh dày 1 px | [03](../mathematics/03_non_maximum_suppression.md) |
| 4. Ngưỡng kép | mạnh ($\ge T_h$), yếu ($[T_l,T_h)$) | Một ngưỡng thì vừa đứt cạnh vừa lọt nhiễu | [04](../mathematics/04_nguong_kep_va_hysteresis.md) |
| 5. Hysteresis | giữ thành phần liên thông chứa điểm mạnh | Cạnh thật liên tục, nhiễu rời rạc | [04](../mathematics/04_nguong_kep_va_hysteresis.md) |

**Thư viện làm gì thực sự.** `cv2.Canny` làm bước 2–5 (không có bước 1). `skimage.feature.canny` làm đủ 5 bước. Bản tự cài của dự án tái tạo **từng pixel** cả hai (X1: 0 px lệch trên 6 ảnh × 12 cấu hình). Khác biệt giữa hai thư viện: [05](../mathematics/05_khac_biet_opencv_skimage.md).

## I.1b. So sánh với Sobel, Laplacian
Số liệu X4, F1 tốt nhất trên lưới:

| | Canny | Sobel + ngưỡng | LoG + điểm qua 0 |
|---|---|---|---|
| F1 một-một, ảnh sạch | 0.919 | 0.657 | 0.917 |
| F1 một-một, σₙ = 0.2 | **0.919** | 0.689 | 0.536 |
| Độ dày cạnh | 1 px | 2–3 px | 1 px |
| Thời gian 256² | khoảng 5.5 ms | khoảng 2 ms | khoảng 2 ms |

Sobel không có NMS nên cạnh dày (F1 một-một thấp). Laplacian là đạo hàm bậc 2 nên khuếch đại nhiễu mạnh hơn (√20 so với √12) và suy giảm nhanh nhất khi có nhiễu. Canny chậm hơn khoảng 2.7 lần nhưng chính xác và bền với nhiễu nhất.

## I.2a. Ảnh hưởng của σ
- **Chứng minh và đo được:** σ tăng thì nhiễu trên gradient giảm ∝ 1/σ², **nhưng đỉnh gradient tại cạnh cũng giảm ∝ 1/σ** (≈ 3.19h/σ). SNR tăng ≈ 2hσ/σₙ.
- **Hệ quả 1:** giữ ngưỡng cố định mà tăng σ thì cạnh biến mất (Lab3: OpenCV σ = 5 không còn cạnh ở 100/200, vì đỉnh lớn nhất chỉ còn 161 < 200). X2: Recall giảm từ 0.950 (σ = 3) xuống 0.810 (σ = 5).
- **Hệ quả 2:** σ cần thiết tăng theo nhiễu. X2: σₙ = 0.05 cần σ ≥ 1.5, σₙ = 0.15 cần σ ≥ 3.
- **Đánh đổi định vị:** σ lớn bo tròn góc. X5: nhận dạng hình trên ảnh sạch giảm từ 100% (σ = 1.5) xuống 91.7% (σ = 2.5).

## I.2b. Ngưỡng thấp và ngưỡng cao
- **$T_h$ quyết định số hạt giống.** Nhiễu vượt $T_h$ là hạt giống giả. Ngưỡng phổ quát $T_h=s\sqrt{2\ln N}$ cho kỳ vọng 1 hạt giả trên cả ảnh.
- **$T_l$ quyết định độ liền.** X2: khi $T_l\to T_h$ ở σₙ = 0.15, độ phân mảnh tăng từ 1.8 lên 13.3 thành phần / 100 px.
- **Tính đơn điệu (định lý, đã kiểm chứng):** tăng bất kỳ ngưỡng nào thì tập cạnh chỉ nhỏ đi (112 cặp, 0 vi phạm). Điều này **không đúng với σ**: cạnh dịch chỗ, 2562 pixel mới xuất hiện khi σ tăng từ 1 lên 2.

## I.3. Ưu và nhược điểm, ứng dụng
**Ưu điểm** (đã đo): cạnh mảnh 1 px; liên tục nhờ hysteresis; bền với nhiễu Gauss khi chọn ngưỡng đúng; độ phức tạp O(N) (X9: k ≈ 1); tái lập được giữa các thư viện khi hiểu đúng quy ước.

**Nhược điểm** (đã đo):
- 3 tham số phụ thuộc lẫn nhau, và tham số mặc định thất bại trên ảnh nhiễu hay tương phản thấp (X3: F1 < 0.1).
- Không phân biệt được biên vật thể với texture (X3: F1 tốt nhất 0.43–0.79).
- Đứt cạnh ở góc, nên phân đoạn phải dùng closing.
- Mất cạnh đẳng sáng nếu chuyển xám (X7: F1 = 0).
- Không bền với nhiễu muối tiêu nếu không có tiền lọc median (X6).

**Lĩnh vực dùng nhiều nhất (T3b):** thị giác công nghiệp và đo lường (kiểm tra kích thước, phát hiện lỗi bề mặt), tiền xử lý cho phát hiện đường thẳng và đường tròn (biến đổi Hough), robot và xe tự hành (vạch kẻ đường), ảnh y tế (đường bao cơ quan), OCR và ảnh tài liệu.

**Ví dụ cụ thể (T3c):**
- Đếm và phân loại vật trên băng chuyền (X5 làm đúng bài toán này với 6 loại hình: phân đoạn 60/60, nhận dạng 100% trên ảnh sạch).
- Phát hiện làn đường: Canny rồi biến đổi Hough trên vùng quan tâm.
- Căn chỉnh tài liệu: Canny, tìm đường bao tứ giác lớn nhất, rồi biến đổi phối cảnh.
- Đếm đồng xu trong Lab3 (24/24).

## III.1. Đánh giá chất lượng cạnh (E1)
Xem [../evaluation/README.md](../evaluation/README.md). Những điểm chính:
1. Cần ground truth, nên dự án dùng ảnh tổng hợp có biên chính xác.
2. Dùng chỉ số **có dung sai** (τ = 1.5 px), vì cạnh lệch 1 px vẫn đúng.
3. **Báo cáo cả khớp nhiều-một và một-một.** X4 cho thấy chỉ dùng nhiều-một sẽ kết luận sai rằng Sobel ngang Canny.
4. Bổ sung FOM và chamfer cho định vị, độ phân mảnh cho độ liền.
5. Không dùng PSNR/SSIM cho bản đồ cạnh.
6. So sánh phương pháp ở F1 tốt nhất trên lưới, không ở một ngưỡng tùy ý.

## III.2. Cải thiện hiệu suất Canny (E2)
Số liệu X6:
- **Chọn ngưỡng tự động** là cải thiện lớn nhất. Ngưỡng theo nhiễu (ngưỡng phổ quát) đưa F1 từ 0.119 lên 0.988 với nhiễu Gauss. Otsu và phân vị bền trên mọi loại suy giảm (≥ 0.979).
- **Tiền lọc theo loại nhiễu:** median 3×3 cho nhiễu muối tiêu (F1 0.995); bilateral giữ biên khi nhiễu Gauss nhẹ (0.999).
- **Tốc độ:** hysteresis bằng gán nhãn nhanh hơn BFS khoảng 30 lần. OpenCV (SIMD) nhanh hơn bản NumPy tốt nhất khoảng 5.6 lần (X9).
- Heuristic phổ biến "median ± 33%" **không nên dùng**: F1 = 0 trên ảnh tương phản thấp.

## III.3. Canny cho ảnh màu (E3)
Có, nhưng **không chuyển xám trước**: hai màu cùng độ sáng thì ảnh xám không còn cạnh (X7: F1 = 0). Có ba cách: Canny từng kênh rồi OR; chọn kênh có gradient lớn nhất (cách `cv2.Canny` làm với ảnh 3 kênh, đã kiểm chứng 0 px lệch); gradient Di Zenzo (ma trận cấu trúc). Cả ba đạt F1 0.999 khi ngưỡng được chọn đúng thang của từng toán tử. Chi tiết: [07](../mathematics/07_canny_anh_mau_va_video.md).

## III.4. Canny cho video (E4)
Chạy từng khung theo lô. Vấn đề là cạnh nhấp nháy giữa các khung. X8 / X8b:
- **Ngưỡng theo nhiễu trên từng khung** đã cho F1 0.982 và độ ổn định 0.972.
- **Làm trơn theo thời gian** (Gaussian trục khung) tăng thêm khi cảnh tĩnh (F1 0.989, độ ổn định 0.990) nhưng **gây nhòe khi vật chuyển động** (Recall giảm từ 0.991 xuống 0.961, F1 0.923 so với 0.975 khi xử lý từng khung).
- Chọn σ_t theo tốc độ chuyển động. Tốc độ xử lý: hồ sơ OpenCV / `cv2.Canny` khoảng 1 ms cho khung 256² (X9), đủ cho thời gian thực.
