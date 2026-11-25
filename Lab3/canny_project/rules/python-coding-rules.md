# Quy tắc viết Python

1. Python ≥ 3.11. Có type hints cho mọi hàm công khai. Dùng `from __future__ import annotations`.
2. Hàm thuần, không có trạng thái toàn cục. Cấu hình dùng `dataclass(frozen=True)`.
3. Không bắt lỗi rồi bỏ qua. Ngoại lệ của gói:
   - `InvalidImageError(ValueError)` cho ảnh không hợp lệ;
   - `ValueError` cho tham số sai;
   - `KeyError` có thông báo gợi ý cho tên thuật toán không có.
4. Vector hóa bằng NumPy. Vòng lặp Python chỉ chấp nhận ở chỗ bản chất tuần tự (BFS, Douglas–Peucker) và phải ghi chú độ phức tạp.
5. Không in ra màn hình trong thư viện; dùng `logging`. Chỉ `cli.py` được `print`.
6. Phụ thuộc: numpy, scipy, opencv-python, scikit-image, matplotlib, psutil. Thêm phụ thuộc mới phải ghi lý do vào `docs/architecture/README.md`.
7. Đặt tên rõ nghĩa (`magnitude`, `low_threshold`), không viết tắt khó hiểu. Hằng số viết HOA.
8. Không hardcode đường dẫn; đường dẫn nhận qua tham số kiểu `pathlib.Path`.
9. File ≤ khoảng 400 dòng; vượt quá thì tách theo trách nhiệm, không tách tùy tiện.
