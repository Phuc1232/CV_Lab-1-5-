# Quy tắc đánh giá

1. Chọn chỉ số theo câu hỏi, không theo danh sách. Ánh xạ câu hỏi → chỉ số nằm ở `docs/evaluation/README.md`.
2. **Cạnh:**
   - Chỉ số chính là P/R/F1 có dung sai, luôn ghi rõ τ (mặc định τ = 1.5 px) và kiểu khớp (`many` / `one-to-one`).
   - Chỉ số phụ là FOM (α = 1/9), chamfer, HD95.
3. Không dùng MSE/PSNR/SSIM để chấm bản đồ cạnh. PSNR chỉ dùng mô tả mức suy giảm của ảnh đầu vào.
4. IoU và Dice: chỉ báo cáo IoU (Dice = 2IoU/(1+IoU)).
5. Ca biên phải có định nghĩa rõ, có test, và được ghi trong docstring:
   - Cả P và G rỗng: P = R = F1 = 1.
   - P rỗng, G không rỗng: P = 1 (quy ước), R = 0, F1 = 0.
   - G rỗng, P không rỗng: P = 0, R = 1, F1 = 0.
6. Kết quả tổng hợp báo cáo trung bình **và** độ lệch chuẩn (hoặc trung vị + IQR) trên các mẫu, kèm số mẫu.
7. Tỉ lệ pixel cạnh chỉ là thông tin mô tả, không dùng để kết luận chất lượng.
8. Mỗi chỉ số trong tài liệu phải có: ý nghĩa, công thức, miền giá trị, giới hạn.
