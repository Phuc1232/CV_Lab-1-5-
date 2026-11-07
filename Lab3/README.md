# Lab 3: Canny Edge Detector

Bài nộp chính: [Lab3.ipynb](Lab3.ipynb). Script chạy độc lập: [bai_thuc_hanh_canny.py](bai_thuc_hanh_canny.py). Ảnh vào: ảnh có sẵn trong `skimage.data` và `du_lieu/hinh_dang.png` (notebook tự vẽ). Ảnh ra: `ket_qua/<tên bài>/`, mỗi ảnh một file.

Đề: slide `../tai_lieu/Chapter2. ImageProcessingMethods_part2.pdf`, trang 26, phần **II> Bài tập thực hành**. Quy tắc làm bài ghi trong [CLAUDE.md](CLAUDE.md).

## 1. Mục tiêu

1. Chạy Canny bằng `cv2.Canny` (OpenCV) và `skimage.feature.canny` (scikit-image).
2. Thay đổi sigma, ngưỡng thấp, ngưỡng cao rồi so sánh với giá trị mặc định.
3. Áp dụng cho ảnh nhiều nhiễu, ảnh tương phản thấp, ảnh nhiều chi tiết; đánh giá và rút ra kết luận.
4. Kết hợp Canny với phân đoạn (tìm vùng) và nhận dạng hình dạng (tìm đối tượng).

## 2. Lý thuyết

Canny gồm 5 bước:
1. Giảm nhiễu bằng lọc Gaussian (tham số **sigma**).
2. Tính độ lớn và hướng gradient bằng Sobel.
3. Non-Maximum Suppression: chỉ giữ điểm cực đại theo hướng gradient, làm cạnh mỏng còn 1 pixel.
4. Ngưỡng kép: gradient ≥ ngưỡng cao là cạnh mạnh, nằm giữa hai ngưỡng là cạnh yếu.
5. Nối cạnh bằng hysteresis: cạnh yếu chỉ được giữ khi nối với cạnh mạnh.

`cv2.Canny` làm bước 2–5 và không có sigma, nên khi khảo sát sigma phải làm mờ trước bằng `cv2.GaussianBlur`. `skimage.feature.canny` làm đủ 5 bước. Ngưỡng của hai thư viện không cùng thang đo, nên chỉ so sánh kết quả với nhau.

## 3. Cách làm

| Bài | Ảnh | Nội dung |
|---|---|---|
| 1 | `camera` | `cv2.Canny(img, 100, 200)` và `feature.canny(img)` mặc định |
| 2 | `camera` | Khảo sát từng tham số, giữ cố định các tham số còn lại: sigma, ngưỡng thấp, ngưỡng cao; so sánh với mặc định |
| 3 | `camera` + nhiễu σ=25, `camera` nén về 100–140, `grass` | Chạy 4 lần mỗi ảnh: hai thư viện × (mặc định, đã chỉnh) |
| 4.1 | `coins` | Canny → closing (`MORPH_ELLIPSE`) → tô kín contour → lọc diện tích |
| 4.2 | `du_lieu/hinh_dang.png` (8 hình vẽ sẵn) | Canny → closing → `approxPolyDP` đếm số đỉnh để gọi tên hình |

Để so sánh bằng số, notebook đo thêm **tỉ lệ pixel cạnh (%)**.

## 4. Kết quả

| Thư mục trong `ket_qua/` | Nội dung |
|---|---|
| `bai1_opencv_vs_skimage/` | Ảnh gốc, cạnh OpenCV 100/200, cạnh scikit-image mặc định |
| `bai2_sigma/`, `bai2_nguong_thap/`, `bai2_nguong_cao/` | Kết quả khi thay đổi từng tham số |
| `bai2_so_sanh_mac_dinh/` | Mặc định so với bộ tham số chỉnh tay |
| `bai3_anh_dau_vao/` | 3 loại ảnh test |
| `bai3_nhieu/`, `bai3_tuong_phan_thap/`, `bai3_chi_tiet/` | Mặc định so với đã chỉnh cho từng loại ảnh |
| `bai4_phan_doan/` | Các bước tìm vùng đồng xu |
| `bai4_nhan_dang_hinh/` | Ảnh test và kết quả nhận dạng hình |

## 5. Nhận xét

- **Sigma:** càng lớn thì càng ít cạnh vụn nhưng mất chi tiết và cạnh bị lệch vị trí. Với scikit-image, tỉ lệ cạnh giảm 15.78% (σ=0.5) → 9.89% (σ=1) → 2.80% (σ=2) → 1.08% (σ=5). Với OpenCV, σ=5 không còn cạnh nào ở ngưỡng 100/200.
- **Ngưỡng thấp:** càng nhỏ thì viền càng liền nhưng nhiều cạnh vụn. Khi bằng ngưỡng cao thì hysteresis mất tác dụng và cạnh bị đứt.
- **Ngưỡng cao:** quyết định số điểm xuất phát; tăng lên thì chỉ còn các đường viền rõ.
- **Ảnh nhiều nhiễu:** tham số quan trọng nhất là sigma (~32% cạnh giả ở mặc định, còn ~2% khi σ=3). **Tương phản thấp:** phải hạ ngưỡng. **Nhiều chi tiết:** tăng sigma kết hợp hạ ngưỡng.
- **Bài 4:** tìm được 24/24 đồng xu; nhận dạng đúng 8/8 hình trên ảnh test; trên ảnh đồng xu có 23/24 vùng nhận là "Tron", 1 vùng nhận nhầm thành "Ngu giac" vì đường bao bị méo.

Chi tiết số liệu và lý giải có trong phần nhận xét của từng bài trong notebook.

## Chạy lại

```bash
cd Lab3 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab3.ipynb
../.venv/bin/python bai_thuc_hanh_canny.py      # bản script, cùng thư mục ket_qua/
```
