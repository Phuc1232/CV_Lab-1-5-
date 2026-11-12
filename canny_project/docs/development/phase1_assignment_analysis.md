# Phase 1: Phân tích đề bài (Canny Edge Detector)

Ngày: 2026-11-07 · Trạng thái: chờ duyệt

Nguồn đề: slide `tai_lieu/Chapter2. ImageProcessingMethods_part2.pdf`, trang 26, "Bài thực hành chương 2 (phần 2): CANNY EDGE DETECTOR".
Bài đã nộp: `Lab3/` (notebook + `bai_thuc_hanh_canny.py`). Dự án này **mở rộng** Lab3 và không sửa `Lab3/`.

---

## 1. Đề bài gốc (chép nguyên ý, đánh mã để truy vết)

Đề có 3 phần. Phần II là phần bắt buộc phải chạy được; phần I và III là câu hỏi lý thuyết phải trả lời được, và nên trả lời bằng thực nghiệm thay vì chỉ nói lý thuyết.

### I. Lý thuyết

| Mã | Yêu cầu |
|---|---|
| T1a | Giải thích chi tiết từng bước: giảm nhiễu, tính gradient, non-maximum suppression, ngưỡng kép, theo dõi cạnh |
| T1b | So sánh với Sobel, Laplacian |
| T2a | Ảnh hưởng của sigma trong bộ lọc Gaussian |
| T2b | Ảnh hưởng của ngưỡng thấp, ngưỡng cao |
| T3 | Ưu/nhược điểm và ứng dụng thực tế |
| T3a | So sánh với thuật toán khác về **độ chính xác, tốc độ, khả năng xử lý nhiễu** |
| T3b | Canny được dùng phổ biến nhất trong lĩnh vực nào |
| T3c | Ví dụ ứng dụng cụ thể |

### II. Bài tập thực hành (bắt buộc)

| Mã | Yêu cầu |
|---|---|
| P1 | Chạy Canny bằng `cv2.Canny` (OpenCV) và `skimage.feature.canny` (scikit-image) |
| P2 | Thay đổi sigma, ngưỡng thấp, ngưỡng cao; quan sát và so sánh với giá trị mặc định |
| P3 | Áp dụng cho ảnh nhiều nhiễu, ảnh tương phản thấp, ảnh nhiều chi tiết; đánh giá và rút ra kết luận |
| P4a | Kết hợp Canny với thuật toán phân đoạn để tìm các vùng trong ảnh |
| P4b | Kết hợp Canny với thuật toán nhận dạng hình dạng để tìm đối tượng |

### III. Câu hỏi mở rộng

| Mã | Câu hỏi |
|---|---|
| E1 | Đánh giá chất lượng cạnh do Canny phát hiện như thế nào? |
| E2 | Có những phương pháp nào để cải thiện hiệu suất của Canny? |
| E3 | Canny có dùng được cho ảnh màu không? Nếu có thì làm thế nào? |
| E4 | Áp dụng Canny cho video như thế nào? |

**Nhận xét:** phần III gần như là bản đề cương cho phần mở rộng. E1 cần Evaluation Engine, E2 cần Experiment Engine để so sánh biến thể, E3 cần gradient ảnh màu, E4 cần xử lý theo lô / theo chuỗi khung hình. Khi mở rộng sẽ bám vào 4 câu này để mọi module thêm vào đều trả lời một câu hỏi của đề.

---

## 2. Bài toán cốt lõi

**Phát hiện cạnh (edge detection):** cho ảnh xám $I:\Omega\subset\mathbb{Z}^2\to\mathbb{R}$, tìm tập điểm $E\subset\Omega$ nằm trên chỗ cường độ thay đổi đột ngột, sao cho:

1. **Phát hiện tốt (good detection):** ít bỏ sót cạnh thật, ít báo cạnh giả.
2. **Định vị tốt (good localization):** điểm báo cạnh nằm sát cạnh thật.
3. **Đáp ứng đơn (single response):** mỗi cạnh chỉ cho ra một đường mảnh 1 pixel.

Đây là 3 tiêu chí Canny (1986) dùng để suy ra bộ lọc tối ưu. Kết quả gần đúng của bộ lọc đó là **đạo hàm của Gaussian**, và vì thế pipeline có dạng "Gaussian rồi gradient".

Canny gồm 5 bước. Ba bước sau là phi tuyến:

```
I (xám) ─► [1] Gaussian G_σ * I ─► [2] Gradient (Gx, Gy) → (M, θ)
        ─► [3] NMS: giữ cực đại theo hướng θ ─► [4] Ngưỡng kép (T_low, T_high)
        ─► [5] Hysteresis: giữ cạnh yếu nếu liên thông với cạnh mạnh ─► E (nhị phân)
```

---

## 3. Kiến thức xử lý ảnh liên quan

| Chủ đề | Dùng ở đâu |
|---|---|
| Tích chập 2D, bộ lọc tách được (separable) | Bước 1 (Gaussian), bước 2 (Sobel) |
| Phân phối Gauss, độ lệch chuẩn, quan hệ σ với kích thước kernel | Bước 1, tham số sigma |
| Đạo hàm rời rạc, sai phân, toán tử Sobel/Prewitt | Bước 2, so sánh T1b |
| Toán tử Laplacian / LoG, zero-crossing | So sánh T1b (đạo hàm bậc 2) |
| Hình học vector: độ lớn, hướng, `atan2`, lượng tử hóa góc | Bước 2–3 |
| Nội suy tuyến tính (NMS nội suy dưới pixel) | Bước 3, biến thể của skimage |
| Ngưỡng, histogram, phân vị (quantile), Otsu | Bước 4, cải thiện E2 |
| Liên thông 8 lân cận, gán nhãn thành phần liên thông, BFS/DFS | Bước 5 |
| Hình thái học: dilate, erode, closing, phần tử cấu trúc | P4a (đóng viền), nằm trong slide chương 2 phần 2 |
| Contour, xấp xỉ đa giác (Douglas–Peucker), diện tích, chu vi | P4b |
| Không gian màu, gradient ảnh màu (structure tensor) | E3 |
| Mô hình nhiễu: Gauss, muối tiêu; tương phản; mờ | P3, kiểm thử |
| Thống kê phân loại nhị phân: Precision/Recall/F1, IoU/Dice | E1, P4a |

---

## 4. Công thức toán học (bản tóm tắt; chứng minh và phân tích sâu để dành cho Phase 2)

**Bước 1: Gaussian.**
$$G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\Big(-\frac{x^2+y^2}{2\sigma^2}\Big)=g_\sigma(x)\,g_\sigma(y),\qquad I_s=G_\sigma * I$$
Vì tách được nên chi phí giảm từ $O(k^2)$ xuống $O(2k)$ phép nhân mỗi pixel. Bán kính kernel thường lấy $r=\lceil 3\sigma\rceil$ hoặc $\lceil 4\sigma\rceil$ (skimage dùng `truncate=4`). Kernel rời rạc phải chuẩn hóa để tổng bằng 1.

**Bước 2: Gradient (Sobel 3×3).**
$$K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\;K_y=K_x^\top,\quad G_x=K_x*I_s,\;G_y=K_y*I_s$$
$$M=\sqrt{G_x^2+G_y^2}\ \ (\text{L2})\quad\text{hoặc}\quad M=|G_x|+|G_y|\ \ (\text{L1, mặc định của cv2.Canny}),\qquad \theta=\operatorname{atan2}(G_y,G_x)$$

**Bước 3: NMS.** Lượng tử hóa $\theta$ về 4 hướng {0°, 45°, 90°, 135°} (hoặc nội suy theo góc thật). Gọi $p^\pm$ là 2 lân cận dọc theo hướng gradient:
$$N(p)=\begin{cases}M(p) & M(p)\ge M(p^+)\ \wedge\ M(p)\ge M(p^-)\\ 0 & \text{ngược lại}\end{cases}$$

**Bước 4: Ngưỡng kép.** Mạnh $S=\{p: N(p)\ge T_h\}$, yếu $W=\{p: T_l\le N(p)<T_h\}$. Canny khuyến nghị $T_h/T_l\in[2,3]$.

**Bước 5: Hysteresis.** $E=S\ \cup\ \{p\in W:\ p \text{ liên thông 8 lân cận với } S \text{ qua đường đi trong } S\cup W\}$, tức là lấy các thành phần liên thông của $S\cup W$ có chứa ít nhất một điểm mạnh.

**So sánh (T1b).** Laplacian: $\nabla^2 I = I_{xx}+I_{yy}$, cạnh là zero-crossing. LoG: $\nabla^2(G_\sigma * I)=(\nabla^2G_\sigma)*I$.

---

## 5. Thuật toán được yêu cầu và thuật toán liên quan

| Thuật toán | Đề yêu cầu? | Vai trò |
|---|---|---|
| Canny (5 bước) | **Bắt buộc** (P1–P4) | Thuật toán trung tâm |
| Gaussian smoothing | Bắt buộc (bước 1, tham số σ) | Thành phần của Canny |
| Sobel gradient | Bắt buộc (bước 2) và so sánh (T1b) | Thành phần + baseline |
| NMS, ngưỡng kép, hysteresis | Bắt buộc (bước 3–5) | Thành phần của Canny |
| Laplacian (và LoG) | Phải so sánh (T1b) | Baseline |
| Hình thái học (closing, dilate, erode) | Dùng trong P4a, có trong slide | Hậu xử lý cạnh → vùng |
| Phân đoạn vùng (tô contour, gán nhãn liên thông) | P4a, đề không chỉ định thuật toán | Cạnh → vùng |
| Nhận dạng hình dạng (xấp xỉ đa giác, đặc trưng hình học) | P4b, đề không chỉ định thuật toán | Vùng → nhãn hình |
| Canny cho ảnh màu | E3 | Mở rộng |
| Ngưỡng tự động (Otsu, theo median, theo quantile) | E2 | Mở rộng |
| Chỉ số đánh giá cạnh | E1 | Mở rộng |

---

## 6. Input / Output

| | Mô tả | Ràng buộc cần kiểm soát |
|---|---|---|
| **Input chính** | Ảnh xám 2D | `ndim == 2`; ảnh màu phải chuyển rõ ràng (BGR→Gray với OpenCV, RGB→Gray với skimage) |
| Kiểu dữ liệu | `uint8` cho `cv2.Canny` (bắt buộc); skimage nhận `uint8`/`float` (không nhận int64/uint64) | skimage đổi ảnh về `float [0,1]` bên trong và chia ngưỡng người dùng truyền vào cho max của dtype. Vì vậy ngưỡng phải truyền **theo đơn vị của dtype đầu vào** (25.5 cho `uint8`, 0.1 cho `float`). Đã kiểm tra: `camera` ở `uint8` và `float` cho cùng 25 934 pixel cạnh |
| Miền giá trị | `uint8` ∈ [0,255], `float` ∈ [0,1] | Tính gradient phải dùng `float32/64` để tránh tràn số (Sobel trên `uint8` có thể ra tới ±1020) |
| **Output chính** | Ảnh cạnh nhị phân, cùng kích thước | OpenCV trả về `uint8` {0,255}; skimage trả về `bool`. Phải chuẩn hóa về một kiểu chung trước khi so sánh |
| Output trung gian | $I_s, G_x, G_y, M, \theta, N$, mặt nạ mạnh/yếu | Cần cho giải thích từng bước (T1a) và kiểm thử từng bước |
| Output P4a | Mặt nạ vùng / ảnh nhãn, số vùng | — |
| Output P4b | Danh sách đối tượng (contour, nhãn hình) | — |
| Input không hợp lệ | `None`, sai đường dẫn, ảnh rỗng, sai số chiều, NaN | Báo lỗi rõ ràng, không im lặng trả kết quả giả |

---

## 7. Tham số

| Tham số | Ý nghĩa | `cv2.Canny` | `skimage.feature.canny` |
|---|---|---|---|
| σ | Độ lệch chuẩn Gaussian: lớn thì ít nhiễu nhưng mất chi tiết và cạnh bị lệch vị trí | **Không có**, phải gọi `cv2.GaussianBlur` trước | `sigma=1.0` |
| $T_l$ | Ngưỡng thấp, quyết định cạnh yếu nào được nối | `threshold1` (bắt buộc) | `low_threshold=None` → 0.1 trên ảnh đã chuẩn hóa [0,1] (tương đương 25.5 với `uint8`) |
| $T_h$ | Ngưỡng cao, quyết định điểm xuất phát (cạnh mạnh) | `threshold2` (bắt buộc) | `high_threshold=None` → 0.2 trên ảnh đã chuẩn hóa (tương đương 51 với `uint8`) |
| Kích thước Sobel | Khẩu độ đạo hàm | `apertureSize=3` (3/5/7) | Cố định 3×3 |
| Chuẩn gradient | L1 hoặc L2 | `L2gradient=False` (L1) | L2 |
| Ngưỡng theo phân vị | Ngưỡng là phân vị của $M$, không phải giá trị tuyệt đối | — | `use_quantiles=False` |
| Xử lý biên | Cách đệm biên | `BORDER_REPLICATE` nội bộ | `mode='constant'` |
| Mặt nạ | Giới hạn vùng tính | — | `mask=None` |

**Lưu ý quan trọng:** ngưỡng của hai thư viện **không cùng thang đo**, vì OpenCV so ngưỡng với độ lớn Sobel thô (L1) trên thang 0–255, còn skimage dùng L2 của `ndi.sobel` (chưa chuẩn hóa) trên ảnh đã làm mờ ở thang [0,1] (đã xác minh trong mã nguồn skimage 0.26). Lab3 đã ghi nhận điểm này. Phase 2 sẽ suy ra hệ số quy đổi để bản tự cài đặt có thể tái tạo **đúng** từng thư viện.

**Tham số mặc định đang dùng trong Lab3 (baseline để so sánh):** OpenCV 100/200 không làm mờ; skimage σ=1, 25.5/51; khảo sát σ ∈ {0.5, 1, 2, 3, 5}; P4a: σ=1.5, 30/90, closing ellipse 5×5; P4b: σ=1.5, 50/150, `approxPolyDP` ε = 0.04·chu vi.

---

## 8. Tiêu chí đánh giá

### 8.1 Đề yêu cầu "đánh giá và rút ra kết luận" (P3) nhưng không nêu chỉ số
Lab3 hiện chỉ đo **tỉ lệ pixel cạnh (%)**. Chỉ số này cho biết "nhiều hay ít cạnh", **không** cho biết đúng hay sai. Câu E1 hỏi thẳng vào điểm yếu này.

### 8.2 Chỉ số đề xuất (sẽ chọn và biện luận ở Phase 8)

| Nhóm | Chỉ số | Cần ground truth? | Trả lời câu nào |
|---|---|---|---|
| Độ chính xác cạnh | Precision / Recall / F1 **có dung sai khoảng cách** (khớp cạnh trong bán kính vài pixel, như BSDS) | Có | E1, T3a, P3 |
| Định vị | Pratt's Figure of Merit: $\text{FOM}=\frac{1}{\max(N_I,N_D)}\sum_{i}\frac{1}{1+\alpha d_i^2}$ | Có | E1, T2a |
| Định vị | Khoảng cách chamfer / Hausdorff giữa cạnh dự đoán và cạnh thật | Có | E1 |
| Tính liên tục | Số thành phần liên thông, độ dài trung bình đoạn cạnh, tỉ lệ đứt gãy | Không | T2b (vai trò hysteresis) |
| Độ bền với nhiễu | F1 và FOM theo mức nhiễu σ_n hoặc PSNR của ảnh đầu vào | Có (ảnh tổng hợp) | T3a, P3 |
| Độ khớp với thư viện | Tỉ lệ pixel trùng giữa bản tự cài và OpenCV/skimage, IoU của hai bản đồ cạnh | Không | Kiểm chứng bản tự cài |
| Phân đoạn (P4a) | IoU, Dice, số vùng đúng | Có | P4a |
| Nhận dạng hình (P4b) | Accuracy, ma trận nhầm lẫn | Có | P4b |
| Hiệu năng | Thời gian chạy, bộ nhớ đỉnh, độ phức tạp | Không | T3a ("tốc độ") |

### 8.3 Giới hạn cần nói rõ khi dùng
- Tỉ lệ pixel cạnh chỉ dùng để mô tả, không dùng để kết luận chất lượng.
- P/R/F1 phụ thuộc mạnh vào dung sai khớp, nên phải công bố giá trị dung sai.
- Ground truth cạnh trên ảnh thật mang tính chủ quan (BSDS có nhiều người gán nhãn khác nhau).
- Ảnh tổng hợp có ground truth chính xác nhưng đơn giản hơn ảnh thật. Cần cả hai loại.

---

## 9. Bảng truy vết yêu cầu

Ánh xạ từ yêu cầu của đề sang thành phần dự kiến. Đây là bản sơ bộ, Phase 3–4 sẽ chốt.

| Yêu cầu | Thành phần dự kiến |
|---|---|
| P1 | Lớp bọc (wrapper) thống nhất cho OpenCV và skimage, chuẩn hóa input/output |
| P1 (chiều sâu) | Canny **tự cài bằng NumPy**, có output từng bước, kiểm chứng bằng cách so với cả hai thư viện |
| P2, T2a, T2b | Experiment Engine: quét tham số (giữ cố định các tham số khác), lưu cấu hình + kết quả |
| P3 | Bộ dữ liệu có kiểm soát: nhiễu / tương phản thấp / nhiều chi tiết, kèm mức độ tăng dần |
| P4a | Pipeline Canny → hình thái học → vùng; đánh giá IoU/Dice |
| P4b | Pipeline Canny → contour → đặc trưng hình dạng → nhãn; đánh giá accuracy |
| T1a | Tài liệu thuật toán + hình minh họa từng bước (từ output trung gian) |
| T1b, T3a | Baseline Sobel-ngưỡng, Laplacian/LoG; bảng so sánh độ chính xác / tốc độ / độ bền nhiễu |
| T3, T3b, T3c | Tài liệu lý thuyết (không cần code) |
| E1 | Evaluation Engine (mục 8) |
| E2 | Các biến thể: ngưỡng tự động, tiền lọc khác, σ thích nghi; so sánh bằng Experiment Engine |
| E3 | Canny màu: theo từng kênh rồi gộp, và gradient Di Zenzo (structure tensor) |
| E4 | Xử lý theo lô trên chuỗi khung hình từ file video (không webcam/GUI); đo độ ổn định cạnh theo thời gian |

---

## 10. Vấn đề phát sinh và rủi ro

1. **Ground truth cạnh.** Đánh giá định lượng (E1) cần cạnh thật. Có hai hướng:
   - Ảnh tổng hợp (cạnh bậc, hình tròn, đa giác, đường cong) sinh bằng code: ground truth chính xác, kiểm soát được nhiễu và tương phản. **Đề xuất làm hướng này trước.**
   - Ảnh thật có nhãn (BSDS500): cần tải từ Internet. `Lab3/CLAUDE.md` yêu cầu hỏi người dùng trước khi tải, nên cần bạn quyết định.
2. **Mâu thuẫn quy tắc.** `Lab3/CLAUDE.md` cấm thêm thuật toán ngoài slide. Dự án mới nằm ngoài `Lab3/` nên quy tắc này không áp dụng, nhưng **mọi yêu cầu bắt buộc của đề vẫn được giữ nguyên**, và kết quả P1–P4 vẫn phải tái tạo được.
3. **Thang ngưỡng khác nhau giữa hai thư viện.** Bản tự cài phải có chế độ "tương thích OpenCV" (L1, Sobel thô, biên replicate) và "tương thích skimage" (L2, nội suy NMS, biên constant). Nếu không, phép so sánh với thư viện chuẩn không có nghĩa.
4. **Không khớp 100% từng pixel là bình thường.** OpenCV dùng thủ thuật số nguyên/tan22.5° trong NMS, còn skimage nội suy theo góc. Tiêu chí kiểm chứng phải là "khớp ≥ X% và lệch chỉ ở các điểm hòa (tie)", không phải `array_equal`. Phase 2 sẽ định lượng X.
5. **Hiệu năng của hysteresis tự cài.** BFS bằng Python thuần sẽ chậm trên ảnh lớn. Bản Manual giữ nguyên lý rõ ràng, bản tối ưu có thể dùng `scipy.ndimage.label`. Đây đúng là cặp Manual/Optimized mà đề cương yêu cầu.
6. **Môi trường.** `.venv` hiện có Python 3.14.7, NumPy 2.5.4, OpenCV 5.0.0, scikit-image 0.26.0, SciPy 1.18.1, PyTorch 2.15.0, pytest 9.2.0. OpenCV 5 có thể khác OpenCV 4 ở vài chi tiết. Kết quả và phiên bản sẽ được ghi vào mỗi thí nghiệm.
7. **PyTorch.** Bài toán không cần học sâu. Chỉ cân nhắc dùng PyTorch như một backend tích chập GPU/CPU cho phần so sánh hiệu năng, hoặc không dùng nếu không cần (quy tắc "không tạo module dư thừa").

---

## 11. Phạm vi

**Làm:** thư viện Python + CLI chạy thí nghiệm, sinh hình/bảng ra file, test, tài liệu.
**Không làm:** GUI, web, dashboard, ứng dụng desktop, webcam realtime.

---

## 12. Quyết định đã chốt (2026-11-07)

1. **Dữ liệu:** không tải BSDS500. Dùng ảnh tổng hợp có ground truth chính xác và ảnh `skimage.data`.
2. **Tên thư mục:** `canny_project/`.
3. **Ngôn ngữ tài liệu:** tiếng Việt.

Cập nhật sau Phase 2: rủi ro số 4 (mục 10) đã được định lượng. Hồ sơ `opencv` của bản tự cài có thể đạt **trùng từng pixel** với `cv2.Canny`, xem [../mathematics/05_khac_biet_opencv_skimage.md](../mathematics/05_khac_biet_opencv_skimage.md).
