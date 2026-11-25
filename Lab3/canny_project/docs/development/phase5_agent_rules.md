# Phase 5: Quy tắc agent

- **Hoàn thành:** [../../AGENTS.md](../../AGENTS.md) (12 quy tắc bắt buộc, quy trình, lệnh) và 8 quy tắc chuyên biệt trong [../../rules/](../../rules/).
- **Quyết định:** mỗi quy tắc phải kiểm tra được bằng test hoặc bằng quy trình:
  - quy tắc 2: test AST;
  - quy tắc 4–7: test của `imaging`;
  - quy tắc tái lập: `test_runner_outputs_and_reproducibility`.
- **Vấn đề:** quy tắc "dùng `cv2.setNumThreads(1)` khi đo" hóa ra sai trên OpenCV/GCD (macOS), nơi giá trị 1 bị bỏ qua. Phát hiện ở Phase 9 và đã sửa trong `rules/experiment-rules.md`.
- **Tiếp theo:** Phase 6, thuật toán cốt lõi.
