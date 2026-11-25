"""cannylab: Canny Edge Detector tự cài đặt bằng NumPy, kèm hệ thống dữ liệu, thực nghiệm và đánh giá.

Gói con:
    imaging      đọc/kiểm tra/chuyển đổi ảnh (không phụ thuộc gói con khác)
    core         thuật toán tự cài đặt (NumPy, cấm cv2/skimage)
    analysis     phân đoạn và nhận dạng hình dạng dựa trên cạnh (P4)
    backends     bản Optimized dùng OpenCV / scikit-image / SciPy
    datasets     dữ liệu tổng hợp có ground truth, nạp thư mục ảnh
    evaluation   chỉ số cạnh, vùng, phân loại, hiệu năng
    experiments  cấu hình, chạy, lưu và báo cáo thực nghiệm
"""

__version__ = "1.0.0"
