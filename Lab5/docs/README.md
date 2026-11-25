# Tài liệu Lab 5

- [Bắt đầu](../README.md) — Cài đặt, chạy ứng dụng, đăng ký ảnh và kiểm thử.
- [Báo cáo HTML](bao-cao-lab5.html) — Sơ đồ nhận diện, lưu embedding, cấu hình và kết quả thực nghiệm.
- [Kiến trúc](architecture.md) — Luồng xử lý, toán học và các lưu ý về webcam.
- [Cheatsheet](cheatsheet.md) — Thông số, mã minh họa và phím tắt.
- [Vibe coding](vibe-coding.md) — Mẫu đặc tả yêu cầu với tiêu chí chấp nhận.

## Bản đồ mã nguồn

| Mô-đun | Trách nhiệm |
|---|---|
| [core/recognition.py](../src/facelab/core/recognition.py) | Phát hiện khuôn mặt, tạo embedding và đối sánh |
| [datasets/store.py](../src/facelab/datasets/store.py) | Lưu embedding theo ảnh và ID, đồng bộ chỉ mục SQLite |
| [imaging/overlay.py](../src/facelab/imaging/overlay.py) | Vẽ hộp bao, nhãn và HUD |
| [backends.py](../src/facelab/backends.py) | Mở webcam qua các backend OpenCV |
| [datasets/registration.py](../src/facelab/datasets/registration.py) | Đăng ký ảnh mẫu từ tệp hoặc webcam |
| [cli.py](../src/facelab/cli.py) | Nhận tham số, điều khiển webcam và xử lý phím |
| [config.py](../src/facelab/config.py) | Đọc và kiểm tra cấu hình TOML |
| [paths.py](../src/facelab/paths.py) | Đường dẫn mặc định đến ảnh mẫu |
