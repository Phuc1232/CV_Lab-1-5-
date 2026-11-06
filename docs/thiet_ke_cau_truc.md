# Thiết kế cấu trúc 5 bài lab Computer Vision

Ngày: 2026-11-06
Trạng thái: đã thực hiện (2026-11-06)

## Mục tiêu

Cô yêu cầu 5 bài lab phải có kiến trúc rõ ràng để viết **một report chung** cho cả 5 lab: mỗi lab là một chương, hình kết quả lấy từ thư mục `ket_qua/` của lab đó.

Phạm vi: thống nhất thư mục và code ở mức cần thiết (đường dẫn, lưu hình kết quả). **Không đổi thuật toán.**

## Hiện trạng (trước khi tái cấu trúc)

| Lab | Chủ đề | Dạng bài | Ảnh vào / ra | Tài liệu |
|---|---|---|---|---|
| Lab1 | Đọc ảnh, không gian màu, crop/resize, vẽ | notebook | `du_lieu/` → `ket_qua/` | không có README |
| Lab2 | Toán tử điểm ảnh, bộ lọc | notebook + `.py` | ảnh để ở thư mục gốc, không lưu kết quả | README + agent rule |
| Lab3 | Canny edge detector | notebook + `.py` | `du_lieu/` → `ket_qua/` (chia theo bài) | CLAUDE.md |
| Lab4 | Wavelet Hash so sánh ảnh | chỉ `.py` (3 file) | `dataset/` | README + agent_rule |
| Lab5 | Nhận diện khuôn mặt FaceNet + MTCNN | chỉ `.py` + pytest | `known_faces/` | README + AGENTS + CHEATSHEET |

## Các hướng đã cân nhắc

- **A (đã chọn).** Mỗi lab theo cùng một khuôn và tự chạy được riêng, có README gốc làm mục lục. Ít phải sửa code nhất.
- B. Như A nhưng thêm module `common/` dùng chung. Phải sửa cả 5 lab và khó giải thích khi vấn đáp.
- C. Gom hết kết quả về `ket_qua/labN/` ở thư mục gốc. Lab không còn tự đứng riêng được.

## Cấu trúc thư mục

```
CV_Labs/
├── README.md              # mục lục 5 lab: chủ đề, file chính, hình chính, cách chạy
├── requirements.txt       # thư viện cho Lab1–4, dùng chung một .venv ở gốc
├── .gitignore
├── docs/                  # tài liệu thiết kế (file này)
├── tai_lieu/              # slide của giảng viên
├── Lab1/ … Lab5/
```

Mỗi lab:

```
LabN/
├── README.md        # mẫu chung: Mục tiêu · Lý thuyết · Cách làm · Kết quả · Nhận xét
├── LabN.ipynb       # bài nộp chính, Run All ra toàn bộ hình, là nguồn hình cho report
├── *.py             # script/app chạy độc lập (Lab4, Lab5: notebook import hàm từ đây)
├── du_lieu/         # ảnh đầu vào
├── ket_qua/         # ảnh sinh ra, tên dạng bai<k>_<mo_ta>.png
└── ghi_chu/         # agent_rule.md, AGENTS.md, CHEATSHEET.md (file làm việc, không phải bài nộp)
```

README mỗi lab dùng chung 5 mục, khớp với một chương của report: **Mục tiêu → Lý thuyết → Cách làm → Kết quả → Nhận xét**.

## Thay đổi cụ thể

- Lab2: chuyển `images.jpg` vào `du_lieu/`; thêm bước lưu hình vào `ket_qua/` (hiện chỉ `plt.show()`); bỏ đường dẫn tuyệt đối `c:\Users\TAN LOC\...` trong README.
- Lab3: `CLAUDE.md` giữ nguyên ở gốc lab (để chỗ khác thì không còn tác dụng).
- Lab4: chuyển `dataset/` thành `du_lieu/`, sửa đường dẫn trong 3 file `.py`; thêm lưu hình; viết `Lab4.ipynb` gọi `bam_wavelet`, xuất hình so sánh, ma trận nhầm lẫn và ROC.
- Lab5: chuyển `known_faces/` vào `du_lieu/`, sửa `DEFAULT_DB_DIR`; viết `Lab5.ipynb` chạy trên ảnh tĩnh (phát hiện mặt → embedding → độ tương đồng cosine) và lưu hình. App webcam `Lab5.py` giữ lại để demo.
- Tất cả: đổi tên notebook thành `LabN.ipynb`; chuyển các file agent rule / cheatsheet vào `ghi_chu/`.

## Môi trường

- Lab1–4 dùng chung `.venv` ở `CV_Labs/` (Python 3.14), xóa các `.venv` lẻ trong từng lab. Các `.venv` cũ đã hỏng sau khi chuyển thư mục, vì venv lưu đường dẫn tuyệt đối.
- Lab5 cần `torch` và `facenet-pytorch`. Máy hiện chỉ có Python 3.14, nên có thể không cài được. Nếu không cài được thì Lab5 dùng `.venv` riêng với Python 3.12, ghi rõ trong README.

## Kiểm tra

- Chạy lại cả 5 notebook bằng `jupyter nbconvert --to notebook --execute --inplace`.
- Chạy `pytest` cho Lab5.
- Kiểm tra `ket_qua/` mỗi lab có đủ hình.
- Báo đúng kết quả thật, kể cả chỗ lỗi.

## Ghi chú

- Repo git hiện tại trỏ vào cả thư mục home `/Users/mac`, không phải repo riêng của môn, nên không commit vào đó. Nếu cần thì `git init` riêng trong `CV_Labs/`.
