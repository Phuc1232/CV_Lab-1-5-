# 07. Hình thái học, phân đoạn (P4a) và nhận dạng hình (P4b)

## Hình thái học nhị phân
**WHAT.** dilate, erode, closing, opening theo định nghĩa của OpenCV (phần tử cấu trúc SE có tâm ở giữa; erode coi ngoài ảnh là True). Phần tử cấu trúc rect / cross / ellipse dùng cùng công thức với `cv2.getStructuringElement`.
**WHY.** NMS làm đứt cạnh ở góc, nên closing nối lại để đường bao khép kín trước khi lấp lỗ.
**IMPLEMENTATION.** [`core/morphology.py`](../../src/cannylab/core/morphology.py). `fill_holes` = nền không nối được ra viền theo 4 lân cận (dùng lại `reachable` của hysteresis); `label_components` = BFS.
**COMPLEXITY.** $O(N|SE|)$; lấp lỗ và gán nhãn $O(N)$.
**KẾT QUẢ.** SE trùng OpenCV ở 3 hình × 8 kích thước; dilate/erode/closing trùng `cv2`; lấp lỗ trùng `ndi.binary_fill_holes`; nhãn cùng phân hoạch với `ndi.label`.

## Phân đoạn
**PSEUDOCODE.** `edges → closing(ellipse 5×5) → fill_holes → label(8) → bỏ vùng < min_area → bỏ vùng > 50% ảnh`.
**IMPLEMENTATION.** [`analysis/segmentation.py`](../../src/cannylab/analysis/segmentation.py), `backend='manual' | 'opencv'`.
**LIMITATIONS.** Hai vật chạm nhau bị gộp thành một vùng; đường bao hở rộng hơn SE thì không lấp được.

## Nhận dạng hình
**HOW.**
1. Dò biên Moore 8 lân cận (tiêu chí dừng Jacob).
2. Douglas–Peucker với ε = 0.04 × chu vi, như Lab3. Đường khép kín tách tại cặp điểm xa nhau, rồi chạy **lượt 2 bỏ đỉnh thừa** như `cv2.approxPolyDP`.
3. Gộp cạnh ngắn giả: cạnh < 0.25 × cạnh dài nhất, do góc nhọn bị raster cắt cụt.
4. Đặc trưng: số đỉnh, độ tròn $4\pi A/P^2$, độ đặc (diện tích đa giác biên / diện tích bao lồi), hệ số biến thiên bán kính $CV_r$.
5. Luật: $\text{solidity}<0.75$ → sao; $CV_r<0.03$ → tròn; còn lại theo số đỉnh 3/4/5/6.

**Vì sao dùng $CV_r$ cho hình tròn.** DP với ε = 0.04P biến đường tròn thành khoảng 5 đỉnh: độ võng $R(1-\cos(\theta/2))=0.25R$ cho $\theta\approx83°$. Vì vậy số đỉnh không phân biệt được tròn với ngũ/lục giác. Lý thuyết: $CV_r$ của tròn ≈ 0, lục giác ≈ 0.04, ngũ giác ≈ 0.06. Đo được: tròn < 0.03, lục giác 0.041–0.045, ngũ giác 0.055–0.061.

**Quá trình sửa lỗi (ghi lại vì có giá trị khi vấn đáp).** Lần chạy đầu chỉ đúng 62/72, với 3 nguyên nhân:
- tách DP tại điểm đầu raster nên thừa đỉnh;
- độ đặc chia diện tích pixel cho diện tích bao lồi của tâm pixel (hai đại lượng khác hệ);
- tam giác nhỏ bị cắt góc.

Sau khi sửa: Manual 180/180, OpenCV 178/180 trên 30 ảnh sạch (3 seed). 2 lỗi còn lại của OpenCV là ngũ giác bị đếm thành 6 đỉnh.

**IMPLEMENTATION.** [`analysis/shapes.py`](../../src/cannylab/analysis/shapes.py): `trace_boundary`, `douglas_peucker`, `remove_redundant_vertices`, `merge_short_edges`, `convex_hull`, `shape_features`, `classify_shape`, `recognize_shapes`.

**KẾT QUẢ trên dữ liệu nhiễu:** X5 ([../experiments/README.md](../experiments/README.md)).
