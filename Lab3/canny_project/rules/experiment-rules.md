# Quy tắc thực nghiệm

1. Mỗi thực nghiệm là một file TOML trong `configs/` gồm: `name`, `question` (mã P/T/E), `seed`, `[dataset]`, `[[runs]]` (thuật toán + lưới tham số), `[evaluation]`.
2. **Một yếu tố một lúc:** khảo sát một tham số thì cố định các tham số khác ở cấu hình cơ sở ghi trong file.
3. Thư mục kết quả `results/<name>_<YYYYmmdd-HHMMSS>/` gồm:
   - `config.toml`: bản sao nguyên văn;
   - `environment.json`: phiên bản Python, thư viện, CPU, số luồng OpenCV;
   - `records.jsonl`: một dòng cho mỗi (mẫu × cấu hình);
   - `summary.csv`, `figures/`.
4. **Tái lập:** cùng cấu hình thì ra cùng `records.jsonl`, trừ các khóa `time_*` và `mem_*`. Có test kiểm tra điều này.
5. Không viết kết luận trước khi có số liệu. Kết luận trong `docs/experiments/` phải trích con số từ `summary.csv` của một lần chạy cụ thể (ghi tên thư mục).
6. Đo thời gian:
   - chạy khởi động ≥ 1 lần;
   - báo trung vị của ≥ 5 lần lặp;
   - ghi số luồng OpenCV. So sánh Manual và Optimized thì tắt đa luồng OpenCV để công bằng: `cv2.setNumThreads(0)` (với OpenCV build GCD trên macOS, giá trị 1 bị bỏ qua), rồi kiểm tra `opencv_threads` trong `environment.json`.
7. Thực nghiệm chạy được trong vài phút trên laptop. Bộ dữ liệu lớn chỉ dùng cho X9 (hiệu năng).
