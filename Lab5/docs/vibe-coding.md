# Vibe coding với đặc tả cho Lab 5

Đọc [README](../README.md) để biết cách chạy Lab 5 và [AGENTS.md](../AGENTS.md) để biết quy trình sửa mã và kiểm thử. Khi giao việc cho trợ lý lập trình, viết rõ hành vi cần đạt và cách kiểm chứng để người triển khai và người kiểm thử dùng chung một đặc tả. [From Vibe Coding to Spec Coding · Easy-Vibe](https://datawhalechina.github.io/easy-vibe/en/stage-3/core-skills/spec-coding/)

## Mẫu yêu cầu

```markdown
### Mục tiêu
Mô tả vấn đề đang gặp và hành vi mong muốn.

### Phạm vi
Chỉ rõ lệnh, hàm hoặc tệp liên quan và các ràng buộc tương thích.

### Tiêu chí chấp nhận
Ghi các đầu vào, kết quả quan sát được và trường hợp biên.

### Kiểm chứng
Ghi lệnh kiểm thử, bước tái hiện thủ công và điều kiện cần phần cứng.
```

## Ví dụ về ngưỡng nhận diện

- **Mục tiêu** — Kiểm tra quy tắc gắn nhãn khi điểm cosine sát ngưỡng mặc định.
- **Phạm vi** — Đọc `compute_similarity()` trong `src/facelab/core/recognition.py` và ca biên trong `tests/test_lab5.py`.
- **Tiêu chí chấp nhận** — Điểm dưới hoặc bằng ngưỡng cho `Unknown`. Điểm trên ngưỡng cho `Matched`. Ca kiểm thử so sánh theo giá trị dùng khi tính toán vì nhãn trên màn hình làm tròn đến hai chữ số.
- **Kiểm chứng** — Từ thư mục `Lab5`, dùng `.\.venv\Scripts\python.exe -m pytest tests/test_lab5.py::test_strict_boundary_condition -v` theo phân công trong `AGENTS.md`. Ca này dùng tensor để kiểm tra ngưỡng và có thể cần tải trọng số mô hình trong lần chạy đầu.
