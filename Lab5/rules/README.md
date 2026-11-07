# Quy ước kỹ thuật Lab 5

Quy trình phân công và kiểm thử nằm trong [AGENTS.md](../AGENTS.md). [Bản đồ mã nguồn](../docs/README.md#bản-đồ-mã-nguồn) mô tả trách nhiệm của từng mô-đun.

- Đặt thay đổi nhận diện trong `core/`, thay đổi hiển thị trong `imaging/` và thay đổi nhập ảnh mẫu trong `datasets/`.
- Giữ `Lab5.py` và `register_face.py` làm điểm vào cho các lệnh từ thư mục gốc. Đặt phần xử lý tham số trong `src/facelab/cli.py`.
- Dùng đường dẫn ảnh mẫu mặc định từ `src/facelab/paths.py` cho cả nhận diện và đăng ký. Giữ thứ tự ưu tiên dòng lệnh, cấu hình TOML, rồi đến mặc định.
- Kiểm tra biên so khớp theo quy tắc điểm lớn hơn ngưỡng cho `Matched`, điểm bằng hoặc nhỏ hơn ngưỡng cho `Unknown`.
