# 06. Chỉ số đánh giá (câu E1: đánh giá chất lượng cạnh thế nào?)

File này định nghĩa toán học và nêu giới hạn của từng chỉ số. Phase 8 sẽ cài đặt và kiểm chứng. Nguyên tắc: **không dùng chỉ số một cách máy móc**. Mỗi chỉ số trả lời một câu hỏi cụ thể và có điểm mù riêng.

Ký hiệu: $P$ là tập pixel cạnh dự đoán, $G$ là tập pixel cạnh ground truth, $d(p,A)=\min_{a\in A}\lVert p-a\rVert_2$ (tính bằng distance transform trong $O(N)$), $\tau$ là dung sai khớp (pixel).

## 1. Độ chính xác phát hiện có dung sai

$$\text{Precision}_\tau=\frac{|\{p\in P: d(p,G)\le\tau\}|}{|P|},\qquad\text{Recall}_\tau=\frac{|\{g\in G: d(g,P)\le\tau\}|}{|G|},\qquad F_1=\frac{2PR}{P+R}$$

- **Ý nghĩa:** Precision là tỉ lệ cạnh báo ra là thật (ngược với cạnh giả), Recall là tỉ lệ cạnh thật được tìm thấy (ngược với bỏ sót).
- **Vì sao cần dung sai:** cạnh lệch 1 px vẫn là phát hiện đúng về mặt nhận thức, nhưng so khớp pixel-chính xác sẽ tính là 1 FP + 1 FN.
- **Giới hạn 1, khớp nhiều-một:** định nghĩa trên cho phép nhiều $p$ cùng khớp một $g$. Cạnh dày 2 px (NMS của skimage trên bậc thang, xem [03](03_non_maximum_suppression.md)) hoặc phản hồi kép vẫn đạt Precision gần 1, tức là **không phạt vi phạm tiêu chí đáp ứng đơn**. Cách khắc phục: **khớp một-một** (bipartite matching có dung sai, như BSDS benchmark), trong đó mỗi $g$ chỉ được ghép với tối đa một $p$. Sẽ cài đặt cả hai và báo cáo cả hai.
- **Giới hạn 2:** kết quả phụ thuộc mạnh vào $\tau$, nên luôn công bố $\tau$. BSDS dùng $\tau=0.0075\times$ đường chéo ảnh, khoảng 4 px với ảnh 481×321. Ảnh tổng hợp có ground truth chính xác dưới pixel nên dùng $\tau\in\{1,2\}$.
- **Giới hạn 3:** $F_1$ coi P và R quan trọng như nhau. Ứng dụng đo đạc ưu tiên P, ứng dụng phân đoạn cần đường bao kín thì ưu tiên R. Khi cần thì báo cáo $F_\beta$.

**Đường cong P–R theo ngưỡng.** Quét $T_h$ (với $T_l=T_h/k$ cố định) ta được đường P–R. Tổng hợp bằng **ODS** (F tốt nhất với một ngưỡng chung cho cả bộ dữ liệu) và **OIS** (ngưỡng tốt nhất riêng cho từng ảnh). Chênh lệch OIS − ODS đo mức độ phụ thuộc vào việc chỉnh ngưỡng theo từng ảnh. Đây là thước đo định lượng cho kết luận "mỗi loại ảnh cần tham số riêng" của P3.

## 2. Định vị

**Pratt's Figure of Merit (1978):**
$$\text{FOM}=\frac{1}{\max(|G|,|P|)}\sum_{p\in P}\frac{1}{1+\alpha\,d(p,G)^2},\qquad\alpha=\tfrac19$$
- Thuộc $(0,1]$, bằng 1 khi $P=G$. Phạt đồng thời cạnh lệch xa (qua $d$), cạnh thừa và cạnh thiếu (qua $\max$).
- **Giới hạn:** một số duy nhất gộp ba loại lỗi nên không biết lỗi nào chiếm ưu thế; không nhạy với việc cạnh bị đứt (hai đoạn rời và một đường liền có thể cho cùng FOM); hằng số α là quy ước.

**Khoảng cách Chamfer (trung bình hai chiều) và Hausdorff:**
$$D_{\text{ch}}=\tfrac12\Big(\overline{d(p,G)}_{p\in P}+\overline{d(g,P)}_{g\in G}\Big),\qquad D_H=\max\Big(\max_{p}d(p,G),\ \max_{g}d(g,P)\Big)$$
Hausdorff bị chi phối bởi một điểm ngoại lai duy nhất. Thay vào đó dùng **Hausdorff phân vị 95** ($HD_{95}$).

**Sai số định vị có dấu (riêng cho ảnh tổng hợp):** với cạnh thẳng hoặc tròn đã biết phương trình, đo độ lệch vuông góc trung bình và độ lệch chuẩn. Chỉ số này kiểm chứng trực tiếp phân tích "σ làm lệch vị trí" ở [01](01_lam_tron_gaussian.md)(d).

## 3. Tính liên tục (không cần ground truth)
- Số thành phần liên thông (8 lân cận) của $P$, độ dài trung bình và trung vị của mỗi thành phần.
- **Tỉ lệ phân mảnh** = số thành phần / (tổng độ dài cạnh / 100 px).

Các chỉ số này trực tiếp đo tác dụng của hysteresis (T2b). **Giới hạn:** "liền hơn" chưa chắc đã "đúng hơn"; texture bị nối thành mạng lưới cũng cho độ liền cao. Phải đọc cùng với Precision.

## 4. Độ bền với nhiễu (T3a)
Đường $F_1(\sigma_n)$ và $\text{FOM}(\sigma_n)$ theo mức nhiễu, trên cùng một ảnh tổng hợp. Mức nhiễu của ảnh đầu vào được mô tả bằng **PSNR** của ảnh nhiễu so với ảnh sạch:
$$\text{PSNR}=10\log_{10}\frac{255^2}{\text{MSE}}$$
PSNR và SSIM ở đây **chỉ dùng để mô tả mức suy giảm của đầu vào**, không dùng để đánh giá bản đồ cạnh. Chúng đo độ giống ảnh cường độ, không có nghĩa với ảnh nhị phân thưa.

## 5. Phân đoạn (P4a) và nhận dạng hình (P4b)
$$\text{IoU}=\frac{|A\cap B|}{|A\cup B|},\qquad\text{Dice}=\frac{2|A\cap B|}{|A|+|B|}=\frac{2\,\text{IoU}}{1+\text{IoU}}$$
- Dice và IoU là hàm đơn điệu của nhau, nên **chỉ cần báo cáo một** cho mỗi phép so sánh, kèm công thức quy đổi. Báo cáo cả hai không thêm thông tin.
- Mức đối tượng: một đối tượng ground truth được coi là tìm thấy khi tồn tại vùng dự đoán có IoU ≥ 0.5 với nó (ghép một-một). Từ đó tính Precision/Recall/F1 đối tượng và số đếm (như "24/24 đồng xu" của Lab3).
- Nhận dạng hình: accuracy, **ma trận nhầm lẫn**, macro-F1 (các lớp có thể lệch số lượng).
- **Giới hạn:** IoU pixel trên vật nhỏ rất nhạy, vì lệch 1 px viền đã là một tỉ lệ lớn của diện tích. Nên báo cáo thêm sai số đường bao (chamfer) cho vật nhỏ.

## 6. Hiệu năng
- **Thời gian:** trung vị của $n\ge7$ lần chạy sau 1–2 lần chạy khởi động, dùng `time.perf_counter`. Báo cáo thêm IQR, cùng kích thước ảnh và số luồng (OpenCV đa luồng nên phải ghi `cv2.getNumThreads()`).
- **Bộ nhớ:** `tracemalloc` chỉ thấy cấp phát qua allocator của Python/NumPy, **không thấy** bộ nhớ C++ của OpenCV. Vì vậy dùng thêm đỉnh RSS (`resource.getrusage`) trong tiến trình con riêng cho mỗi lần đo. Ghi rõ đo bằng phương pháp nào.
- **Độ phức tạp:** đo thời gian theo $N$ trên nhiều kích thước ảnh rồi khớp hệ số mũ trên đồ thị log–log. Mọi bước của Canny là $O(N)$ nên kỳ vọng hệ số khoảng 1. Gaussian có thêm hệ số $O(\sigma)$.

## 7. Những chỉ số KHÔNG dùng, và lý do
| Chỉ số | Lý do không dùng |
|---|---|
| Tỉ lệ pixel cạnh (Lab3) | Chỉ mô tả "nhiều/ít"; giữ lại làm thông tin phụ, không dùng để kết luận chất lượng |
| MSE/MAE/PSNR/SSIM giữa hai bản đồ cạnh | Bản đồ nhị phân thưa: lệch 1 px bị phạt như sai hoàn toàn, và cạnh rỗng có MSE thấp |
| Entropy, Variance of Laplacian, Tenengrad | Là chỉ số độ nét của ảnh, không phải của bản đồ cạnh. Có thể dùng làm *đặc trưng mô tả ảnh đầu vào* (ảnh "nhiều chi tiết" có entropy cao) khi phân loại các ca P3 |
