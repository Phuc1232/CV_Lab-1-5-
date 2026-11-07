# Lab 1: Đọc ảnh, không gian màu, cắt/đổi kích thước, vẽ hình

Bài nộp chính: [Lab1.ipynb](Lab1.ipynb). Script chạy độc lập: [Lab1.py](Lab1.py) (OpenCV `imshow`). Ảnh vào: `du_lieu/anh_mau.png` (notebook tự tạo nếu chưa có) và `du_lieu/anh.jpg` (script). Ảnh ra: `ket_qua/` (notebook) và `ket_qua/script/` (script).

## 1. Mục tiêu

- Cài và dùng OpenCV, Pillow để đọc, hiển thị và lưu ảnh ở định dạng khác.
- Chuyển không gian màu: RGB → Grayscale, HSV, CIELAB.
- Cắt xén và thay đổi kích thước ảnh.
- Vẽ hình cơ bản và thêm chữ lên ảnh.

## 2. Lý thuyết

- **Thứ tự kênh:** OpenCV đọc ảnh theo **BGR**, Matplotlib và Pillow dùng **RGB**, nên phải đổi trước khi hiển thị. Mảng NumPy có dạng `(cao, rộng, kênh)`, còn `Pillow.size` là `(rộng, cao)`.
- **Grayscale:** `Y = 0.299R + 0.587G + 0.114B` (tổng có trọng số, không phải trung bình cộng).
- **HSV:** H là sắc màu (0–179 trong OpenCV `uint8`), S là độ bão hòa, V là độ sáng (0–255).
- **CIELAB:** L\* là độ sáng, a\* là trục xanh lá ↔ đỏ, b\* là trục xanh dương ↔ vàng. Với `uint8`: `L* ≈ L8·100/255`, `a* ≈ a8 − 128`, `b* ≈ b8 − 128`.
- **Định dạng lưu:** PNG nén không mất dữ liệu, JPEG nén có mất dữ liệu, BMP không nén.
- **Nội suy khi resize:** thu nhỏ dùng `INTER_AREA`, `INTER_LINEAR` cho trường hợp thông thường, `LANCZOS` (Pillow) cho chất lượng cao.

## 3. Cách làm

| Phần | Nội dung |
|---|---|
| 2 | Đọc ảnh bằng `cv2.imdecode` và `PIL.Image.open`, lưu ra PNG / JPEG / BMP |
| 3 | `cv2.cvtColor` sang Grayscale, HSV, LAB; tách từng kênh, lưu mảng `.npy` |
| 4 | Cắt bằng `image[y1:y2, x1:x2]` và `Image.crop`; resize 50%, ép 400×300, `ImageOps.pad` giữ tỉ lệ |
| 5 | Vẽ bằng `cv2.line/rectangle/circle/putText` và `ImageDraw` (chữ tiếng Việt) |
| 6 | Kiểm tra bằng `assert`: kích thước, kiểu dữ liệu, PNG đọc lại không đổi pixel |

## 4. Kết quả

| Hình | Nội dung |
|---|---|
| `ket_qua/01_anh_doc_opencv.png`, `01_anh_doi_dinh_dang.jpg/.bmp` | Đọc và lưu ở định dạng khác |
| `ket_qua/02_grayscale.png`, `02_grayscale_pillow.png` | Ảnh xám bằng OpenCV và Pillow |
| `ket_qua/03_HSV_kenh_H/S/V.png` | Ba kênh HSV |
| `ket_qua/04_LAB_kenh_L/a/b.png` | Ba kênh LAB |
| `ket_qua/05_crop.png` | Vùng cắt |
| `ket_qua/06_resize_50_percent.png`, `06_resize_fixed.png`, `06_resize_padded.png` | Ba cách đổi kích thước |
| `ket_qua/07_ve_hinh_opencv.png`, `08_ve_hinh_pillow.png` | Vẽ hình và chữ |

## 5. Nhận xét

- Chuyển màu rồi chuyển ngược có thể không khôi phục đúng mọi pixel vì lượng tử hóa 8 bit và làm tròn.
- Ép về kích thước cố định khác tỉ lệ gốc thì ảnh bị méo; `ImageOps.pad` giữ được tỉ lệ nhờ thêm viền.
- `cv2.putText` (font Hershey) không hiển thị được tiếng Việt có dấu; Pillow với font Unicode thì hiển thị được.
- Các lỗi thường gặp: đảo đỏ/xanh do nhầm BGR/RGB, cắt sai vì nhầm `[y, x]` với `[x, y]`, ảnh xám hiện màu vì quên `cmap="gray"`.

## Chạy lại

Mở `Lab1.ipynb` bằng kernel `.venv` ở thư mục gốc `CV_Labs/` rồi chọn **Run All**, hoặc:

```bash
cd Lab1 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab1.ipynb
../.venv/bin/python Lab1.py     # Enter để dùng du_lieu/anh.jpg; mở cửa sổ OpenCV, nhấn phím bất kỳ để qua bước tiếp
```
