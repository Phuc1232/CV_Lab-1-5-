# 05. Khác biệt toán học giữa `cv2.Canny` và `skimage.feature.canny`

Đề bài (P1) yêu cầu chạy cả hai thư viện. Lab3 ghi nhận rằng "ngưỡng hai thư viện không cùng thang đo" nhưng chưa định lượng. File này mô tả chính xác từng điểm khác biệt, đưa ra công thức quy đổi và đo xem sau khi quy đổi còn lệch bao nhiêu.

Số liệu: [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), OpenCV 5.0.0, scikit-image 0.26.0.

## 1. Bảng khác biệt (tất cả đã kiểm chứng bằng chạy thử hoặc đọc mã nguồn)

| Khâu | `cv2.Canny(img, T1, T2)` | `skimage.feature.canny(img, σ, Tl, Th)` |
|---|---|---|
| Đầu vào | `uint8`, 1 hoặc 3 kênh | 2D, mọi dtype trừ int64/uint64; nội bộ đổi về `float` [0,1] |
| Bước 1 Gaussian | **Không có**, người dùng tự gọi `GaussianBlur` | Có, `truncate=4`, `mode='constant'` + chuẩn hóa bleed-over |
| Bước 2 gradient | Sobel `apertureSize` (mặc định 3), biên **REPLICATE** | `ndi.sobel` 3×3 (trùng `cv2.Sobel` ở phần trong ảnh) |
| Thang của $M$ | Ảnh 0–255, nên $M\le1020$ (L2) hoặc $\le2040$ (L1) | Ảnh 0–1, nên $M\le4\sqrt2$ |
| Chuẩn | **L1** mặc định (`L2gradient=False`) | **L2** |
| Bước 3 NMS | Lượng tử hóa 4 hướng, số nguyên, xử lý điểm hòa bất đối xứng (bậc thang cho 1 px) | Nội suy song tuyến tính, `>=` hai phía (bậc thang cho 2 px) |
| So sánh ngưỡng | `>` | `>=` |
| Ngưỡng | Theo thang của $M$; tự hoán đổi nếu T1 > T2 | Người dùng truyền **theo đơn vị dtype**, bị chia cho max của dtype; mặc định 0.1 / 0.2 trên thang [0,1]; báo lỗi nếu Th < Tl |
| Bước 5 | 8 lân cận, dùng ngăn xếp | 8 lân cận, `ndi.label` |
| Viền ảnh | Có thể có cạnh ở viền | Viền 1 px luôn bị loại |
| Ảnh 3 kênh | Mỗi pixel chọn kênh có $M_{L1}$ lớn nhất (khớp 100%, 0 px lệch) | Không hỗ trợ |
| Đầu ra | `uint8` {0, 255} | `bool` |

**Mô tả đầy đủ của OpenCV:** `cv2.Canny(img)` trùng **từng pixel** với `cv2.Canny(Sobel_x, Sobel_y)` khi Sobel dùng biên replicate, cả với L1 lẫn L2. Dùng biên reflect-101 (mặc định của `cv2.Sobel`) thì lệch 83–99 px, toàn bộ ở viền.

## 2. Quy đổi ngưỡng

Gọi $I_8$ là ảnh `uint8` và $I_f=I_8/255$. skimage tính $M_{sk}=\lVert\nabla_{\text{Sobel}}(G_\sigma*I_f)\rVert_2$ rồi so với $T/255$, với $T$ là ngưỡng người dùng truyền vào theo đơn vị `uint8`. Do Sobel và Gaussian đều tuyến tính:
$$M_{sk}\ge\frac{T}{255}\iff\underbrace{\lVert\nabla_{\text{Sobel}}(G_\sigma*I_8)\rVert_2}_{M\text{ theo thang OpenCV, chuẩn L2}}\ge T$$

**Kết luận:** con số ngưỡng mà skimage nhận (theo đơn vị `uint8`) có **cùng thang** với ngưỡng của OpenCV khi:
1. bật `L2gradient=True`,
2. làm trơn bằng Gaussian **trên dữ liệu float** (không làm tròn về `uint8`),
3. đưa gradient vào qua overload `cv2.Canny(dx, dy, ...)`.

Với L1 thì không có quy đổi chính xác, chỉ có khoảng $T_{L1}\in[T,\sqrt2T]$; trên `camera`, tỉ lệ trung bình là 1.26.

**Cách Lab3 làm** là `GaussianBlur` trên `uint8` rồi `cv2.Canny` với chuẩn L1 và cùng con số ngưỡng. Cách này vi phạm cả điều kiện 1 lẫn 2. Bảng dưới đo mức lệch so với skimage:

| σ, $T_l$, $T_h$ | IoU cách Lab3 | IoU sau quy đổi | pixel skimage / Lab3 / quy đổi |
|---|---|---|---|
| 1, 25.5, 51 (mặc định skimage) | 0.669 | **0.860** | 25 934 / 31 868 / 26 492 |
| 2, 15, 40 | 0.520 | **0.816** | 9 053 / 13 003 / 9 240 |
| 1, 50, 100 | 0.590 | **0.876** | 9 691 / 14 305 / 10 010 |
| 3, 10, 30 | 0.526 | **0.781** | 6 799 / 7 006 / 6 737 |

Cách của Lab3 cho ra nhiều pixel hơn 23–48%. Chủ yếu là do chuẩn L1 thổi phồng $M$ lên khoảng 1.26 lần, cộng thêm sai số làm tròn `uint8` sau khi làm mờ.

## 3. Phần lệch còn lại (IoU 0.78–0.88) là bản chất, không phải lỗi
Sau khi quy đổi đúng, hai thư viện vẫn khác nhau ở **NMS** (lượng tử hóa so với nội suy, cách xử lý điểm hòa), ở việc **làm tròn gradient về `int16`** khi đưa vào OpenCV, và ở **quy ước `>` / `>=`**. σ càng lớn thì IoU càng thấp, vì cạnh mờ có cao nguyên rộng hơn và các quy tắc NMS phân kỳ nhiều hơn. Cạnh 1 px lệch nhau 1 pixel đã làm IoU giảm mạnh, nên IoU **pixel-chính xác** là thước đo khắt khe. Khi đánh giá với ground truth sẽ dùng chỉ số có dung sai ([06](06_chi_so_danh_gia.md)).

## 4. Hệ quả cho bản tự cài đặt (Phase 6)
1. Bản tự cài có 2 **hồ sơ (profile)**, mỗi hồ sơ là một bộ lựa chọn rời rạc chứ không phải code riêng:
   - `opencv`: không làm trơn bên trong, Sobel biên replicate, L1/L2, NMS lượng tử hóa với điểm hòa bất đối xứng, `>`.
   - `skimage`: Gaussian truncate 4 với chuẩn hóa bleed-over, thang [0,1], L2, NMS nội suy, `>=`, bỏ viền.
2. **Tiêu chí chấp nhận đề xuất:**
   - Hồ sơ `opencv` so với `cv2.Canny`: **trùng từng pixel**. Khả thi vì đầu vào của OpenCV hoàn toàn mô tả được bằng $(G_x,G_y)$ số nguyên, như đã kiểm chứng.
   - Hồ sơ `skimage` so với `skimage.feature.canny`: trùng từng pixel, chấp nhận sai lệch ở các điểm $|M-T|<10^{-9}$ (sai số dấu phẩy động).
3. So sánh *giữa hai thư viện* chỉ báo cáo IoU và chỉ số có dung sai, kèm giải thích ở mục 3, không coi là sai.
