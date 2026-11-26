# Quy tắc cho agent làm việc trong Lab 1

## Phạm vi

- Áp dụng cho `Lab1/`. Đọc `README.md` và `Lab1.py` để xác định mục tiêu bài thực hành, cách chạy và các tệp kết quả.
- Với yêu cầu chỉ liên quan đến tài liệu, giới hạn thay đổi trong các tệp tài liệu được yêu cầu. Viết hướng dẫn bằng tiếng Việt và dùng đúng tên hàm, tham số và tệp trong mã hiện có.

## Quy ước xử lý ảnh

- Xác định hệ màu và quy ước kích thước tại chỗ chuyển giữa thư viện. OpenCV thường đọc ảnh màu theo BGR và lưu kích thước mảng theo hàng, cột. Pillow dùng `(rộng, cao)` và biểu diễn hệ màu qua `mode`. Chuyển BGR và RGB khi trao đổi ảnh màu giữa hai thư viện.
- Trình bày HSV và Lab qua từng kênh hoặc chuyển về hệ màu tương thích khi hiển thị. Ghi chú rõ nếu hình minh họa biểu diễn màu giả.
- Phân biệt cắt vùng giữa bằng `anh[y1:y2, x1:x2]` với đổi kích thước toàn ảnh bằng cùng một hệ số theo hai chiều. Khi mô tả hình vẽ, đối chiếu màu BGR và tọa độ thực tế trong mã.
- Ghi đúng cách nhập ảnh, thao tác qua các cửa sổ OpenCV và vị trí lưu kết quả. Khi kiểm tra bằng cách chạy code, dùng bản sao của `Lab1/` vì chương trình lưu ảnh đầu ra với tên cố định.

## Quy trình

1. Đối chiếu lệnh chạy, ảnh mặc định, trình tự xử lý và danh sách đầu ra với mã nguồn.
2. Khi cần kiểm chứng hành vi thực tế, thử với `anh.jpg` và một ảnh khác có kích thước phù hợp. Ghi rõ phần đã quan sát khi chạy và phần đã đối chiếu từ mã.
3. Bàn giao bản nháp để người dùng duyệt khi họ yêu cầu xem trước.
