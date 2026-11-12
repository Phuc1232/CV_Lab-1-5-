# Quy tắc thuật toán

1. **Mỗi thuật toán cốt lõi có hai bản:**
   - Manual trong `core/`: NumPy thuần, có thể đọc theo từng bước.
   - Optimized trong `backends.py` (OpenCV/SciPy/skimage), hoặc bản vector hóa/C của cùng thuật toán.

   Cả hai đăng ký vào `registry` với hậu tố `_manual` / `_opencv` / `_skimage`.
2. **Bản Manual phải khớp bản thư viện theo tiêu chí đã ghi trong tài liệu**, kèm test:
   - Hồ sơ `OPENCV` của Canny trùng từng pixel với `cv2.Canny`.
   - Hồ sơ `SKIMAGE` chỉ được lệch ở pixel có |M − T| < 1e-9 hoặc điểm hòa NMS do sai số làm tròn.
   - Hysteresis BFS trùng tuyệt đối với bản `label`.
3. Bản Manual được phép dùng `numpy` và `scipy.ndimage` **chỉ cho hạ tầng không phải thuật toán đang xét**, ví dụ `distance_transform_edt` trong đánh giá. Không được gọi `scipy.ndimage.gaussian_filter` bên trong `core/gaussian.py`.
4. Canny Manual phải trả `CannyResult` với đủ kết quả trung gian để vẽ minh họa từng bước (T1a) và test riêng từng bước.
5. Tham số có giá trị mặc định khớp thư viện tương ứng; tham số không hợp lệ (σ ≤ 0 khi bật làm trơn, `low > high`, chuẩn lạ) thì ném `ValueError`.
6. Mỗi thuật toán có docstring ghi: công thức, độ phức tạp, tham chiếu tới file trong `docs/mathematics/`.
7. Thêm thuật toán mới: viết hàm → `@register` → test → thêm vào ít nhất một cấu hình thực nghiệm → tài liệu `docs/algorithms/`.
