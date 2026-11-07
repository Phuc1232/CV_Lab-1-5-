# Phase 4: Kiến trúc hệ thống

Tài liệu chính: [../architecture/README.md](../architecture/README.md).

- **Hoàn thành:** sơ đồ phụ thuộc một chiều (`imaging` ← `core`/`analysis` ← `backends`/`registry` ← `experiments`), hợp đồng dữ liệu (ảnh RGB/xám, bản đồ cạnh `bool`, `CannyResult`, `Sample`), cấu trúc thư mục, định dạng cấu hình và kết quả.
- **Quyết định:**
  - src layout + `pyproject.toml`;
  - TOML (`tomllib`, có sẵn trong stdlib) cho cấu hình;
  - JSONL/CSV cho kết quả (không có pandas);
  - registry hàm thay vì lớp trừu tượng;
  - hồ sơ Canny là dữ liệu;
  - `core` cấm `cv2`/`skimage`, có test AST kiểm tra.
- **Vấn đề:** không có. Quy tắc phụ thuộc được kiểm tra tự động (`test_core_does_not_import_opencv_or_skimage`, `test_dependency_direction_imaging_is_leaf`).
- **Tiếp theo:** Phase 5, quy tắc agent.
