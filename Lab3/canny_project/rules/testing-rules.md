# Quy tắc kiểm thử

1. Công cụ: `pytest`. Thư mục `tests/` chia theo loại:
   - `test_unit_*`: từng hàm.
   - `test_validation_*`: Manual so với thư viện.
   - `test_numerical_*`: mệnh đề toán, sai số.
   - `test_integration_*`: pipeline và runner.
   - `test_regression_*`: so với số liệu đã chốt.
   - `test_performance_*`: đánh dấu `@pytest.mark.slow`.
2. **Ca biên bắt buộc** cho mọi bộ phát hiện cạnh:
   - ảnh đen, ảnh trắng, ảnh hằng: 0 cạnh;
   - ảnh nhiễu, ảnh tương phản thấp, ảnh lớn (≥ 2048²), ảnh nhiều kênh;
   - ảnh không hợp lệ (None, rỗng, NaN, 1×1, int64).
3. Ngưỡng sai số phải có lý do trong comment, ví dụ "1e-12: sai số cộng dồn của tích chập 9 tap trên float64".
4. Property test (tự viết, không có hypothesis):
   - cạnh đơn điệu theo ngưỡng;
   - NMS bất biến khi nhân M với hằng số;
   - Di Zenzo với 1 kênh trùng Canny xám.
5. Test hồi quy lưu số liệu ở `tests/data/regression.json`. Chỉ cập nhật khi thay đổi đã được giải thích trong commit.
6. Test nhanh (không có `slow`) phải chạy xong trong khoảng < 60 s.
7. Không sửa test để test đi qua nếu chưa hiểu vì sao test fail.
