# AGENTS.md: Quy tắc chung cho canny_project

Áp dụng cho mọi agent và người viết code trong `canny_project/`. Quy tắc chuyên biệt nằm ở `rules/`; khi có mâu thuẫn, file chuyên biệt thắng.

## Bối cảnh
- Dự án mở rộng Lab3 (Canny Edge Detector, slide chương 2 phần 2, trang 26) thành một dự án nghiên cứu thuật toán. `../Lab3/` là bài đã nộp: **không sửa**.
- Yêu cầu của đề có mã P1–P4 (thực hành), T1–T3 (lý thuyết), E1–E4 (mở rộng); xem `docs/development/phase1_assignment_analysis.md`. Mọi tính năng mới phải truy về một mã.
- Tài liệu viết **tiếng Việt**; tên biến, hàm, module viết tiếng Anh `snake_case`.

## 12 quy tắc bắt buộc
1. Không cài thuật toán khi chưa hiểu nguyên lý: trước khi code phải có (hoặc cập nhật) tài liệu toán trong `docs/mathematics/`.
2. Không thay toàn bộ thuật toán bằng OpenCV: `src/cannylab/core/` **cấm** import `cv2` và `skimage` (có test kiểm tra). Thư viện ngoài chỉ được dùng trong `backends.py`.
3. Không đổi công thức tùy tiện: công thức trong code phải khớp tài liệu; đổi công thức thì sửa tài liệu và test trong cùng thay đổi.
4. Kiểm soát không gian màu: ảnh màu trong gói luôn là **RGB**; OpenCV đọc BGR phải đổi ngay tại `imaging/io.py`.
5. Kiểm soát dtype: ảnh `float` phải ở [0, 1]; ảnh `uint8` ở [0, 255]; gradient tính bằng `int32`/`float64`.
6. Tránh tràn số: không làm phép cộng/trừ/nhân trên `uint8`; Sobel trên `uint8` có thể ra ±1020.
7. Xử lý ảnh không hợp lệ: `None`, rỗng, sai số chiều, NaN/Inf, dtype không hỗ trợ thì ném `InvalidImageError` với thông báo rõ ràng. Không trả kết quả giả.
8. Viết unit test cho mọi hàm công khai mới.
9. Kiểm chứng bằng dữ liệu thực tế: ngoài ảnh tổng hợp, chạy trên ảnh `skimage.data`.
10. Đánh giá độ chính xác **và** hiệu suất cho mọi thuật toán mới (đăng ký vào thực nghiệm X9).
11. Không hardcode đường dẫn: dùng `cannylab.utils.project_root()` và tham số dòng lệnh.
12. Không tạo module dư thừa: module mới phải ghi trong `docs/architecture/README.md` kèm mã yêu cầu mà nó phục vụ.

## Quy trình
1. Đọc quy tắc chuyên biệt liên quan trong `rules/`.
2. Sửa tài liệu toán → code → test → chạy `pytest` → cập nhật tài liệu thuật toán / thực nghiệm.
3. Báo cáo trung thực: test fail thì nói fail, kèm output; chưa kiểm chứng thì ghi "chưa kiểm chứng".

## Lệnh
```bash
# ở CV_Labs/
.venv/bin/pip install -e canny_project
.venv/bin/python -m pytest canny_project/tests -q
.venv/bin/python -m cannylab list
.venv/bin/python -m cannylab run canny_project/configs/x1_thu_vien.toml
```

## Bản đồ quy tắc
| File | Phạm vi |
|---|---|
| `rules/algorithm-rules.md` | Cặp Manual/Optimized, hồ sơ, kết quả trung gian |
| `rules/mathematical-rules.md` | Công thức, ký hiệu, kiểm chứng số |
| `rules/python-coding-rules.md` | Phong cách, kiểu, lỗi, phụ thuộc |
| `rules/image-data-rules.md` | dtype, màu, miền giá trị, ảnh hỏng |
| `rules/experiment-rules.md` | Cấu hình, seed, lưu kết quả, tái lập |
| `rules/evaluation-rules.md` | Chọn chỉ số, dung sai, diễn giải |
| `rules/testing-rules.md` | Loại test, ngưỡng sai số, ca biên |
| `rules/documentation-rules.md` | Khuôn tài liệu, số liệu, trích dẫn |
