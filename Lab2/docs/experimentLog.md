# EXPERIMENT LOG – NHẬT KÝ THỰC NGHIỆM

## 1. Mục tiêu thực nghiệm

Test của Lab2 kiểm tra các phép biến đổi điểm ảnh, các bộ lọc, phát hiện biên, kernel tùy chỉnh và tính metric. Một test tạo ảnh muối tiêu/Gaussian rồi gọi bốn bộ lọc. Mục tiêu của nhật ký này là ghi lại dữ liệu có thể xác nhận từ source, trạng thái test và các tệp đầu ra đã tồn tại; không suy ra số đo không được lưu.

## 2. Môi trường thực nghiệm

Thông tin runtime được đọc trong môi trường làm việc hiện tại ngày 07-11-2026; đây không phải metadata được lưu cùng từng ảnh kết quả.

| Thành phần         | Thông tin                                            |
| ------------------ | ---------------------------------------------------- |
| Operating System   | Microsoft Windows 11 Home Single Language            |
| Python Version     | 3.11.9                                               |
| OpenCV Version     | 5.0.0                                                |
| NumPy Version      | 2.4.6                                                |
| Pytest Version     | 9.2.0                                                |
| Matplotlib Version | 3.11.2                                               |
| CPU                | AMD Ryzen 5 7535HS with Radeon Graphics              |
| RAM                | 15.21 GiB bộ nhớ vật lý tổng theo thông tin hệ thống |
| Image Resolution   | 678 × 452 pixel (H × W), 3 kênh                      |

`requirement.txt` rỗng nên các phiên bản trên là kết quả đọc từ interpreter hiện tại, không phải phiên bản được khóa trong project.

## 3. Dữ liệu thực nghiệm

- Ảnh: [`data/input/images.jpg`](../data/input/images.jpg), tệp JPEG cục bộ. OpenCV giải mã ảnh thành shape `(678, 452, 3)` và dtype `uint8`.
- Fixture test đọc ảnh bằng `read_image(..., mode="RGB")`, sau đó chuyển sang ảnh xám bằng `cv2.COLOR_RGB2GRAY`.
- Nguồn/tác giả/URL/license của ảnh: **Chưa xác định**.
- Test fixture đang dùng đường dẫn tuyệt đối gắn với máy `C:\Users\TAN LOC\Desktop\CV_Lab-1-5-\Lab2\data\input\images.jpg`; vì vậy test không portable nếu chuyển workspace sang đường dẫn khác.
- Helper nhiễu ở test dùng seed 42. Salt-and-pepper mặc định thay 5% số phần tử ảnh, chia giá trị 0/255 ngẫu nhiên; Gaussian noise có trung bình 0 và sigma 20, sau đó clip về miền 8-bit.

## 4. Các kịch bản thực nghiệm

**Ngày chạy lại bộ test và benchmark:** 07-11-2026. **Môi trường:** cấu hình tại mục 2. **Lệnh test:** `python -m pytest -q -p no:cacheprovider <bản-sao-tạm>\tests\test_image_processing.py`; test chạy trong bản sao tạm gồm `src/` và `tests/`, output test được ghi trong thư mục tạm rồi xóa khi kết thúc. Ảnh nguồn vẫn đọc từ đường dẫn fixture hiện tại. Các lệnh/method test cho từng kịch bản liệt kê bên dưới.

Các tên ảnh trong bảng là artifact đã có trong workspace trước lượt chạy cô lập; không xác định được thời điểm tạo chúng. Lượt test chạy lại ghi 40 ảnh vào thư mục tạm, không ghi vào `output/` của project.

### EXP-01: Brightness và Contrast

- **Mục đích:** xác nhận biến đổi sáng/tương phản và shape/dtype đầu ra.
- **Tham số:** `(beta, alpha)` lần lượt `(50,1.0)`, `(-50,1.0)`, `(0,1.5)`, `(0,0.95)`, `(50,1.5)`.
- **Ảnh đầu vào:** ảnh màu fixture đọc thành RGB.
- **Test/lệnh:** `pytest tests/test_image_processing.py::test_light_and_contrast` trong bản sao tạm.
- **Artifact:** [`output/pointOperations/`](../output/pointOperations/) có `brightness_50.png`, `brightness_minus_50.png`, `contrast_1.5.png`, `contrast_0.95.png`, `brightness_contrast.png`.
- **Kết quả/nhận xét:** test định nghĩa assert shape và dtype; không lưu metric hay log kết quả.

### EXP-02: Negative và Thresholding

- **Mục đích:** kiểm tra âm bản và nhị phân hóa ảnh xám.
- **Tham số:** ngưỡng `64`, `128`, `192` và ngưỡng mặc định `128`.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k binary_threshold` trong bản sao tạm.
- **Artifact:** [`output/pointOperations/`](../output/pointOperations/) có `negative.png`, `binary_64.png`, `binary_128.png`, `binary_192.png`, `binary_default.png`.
- **Kết quả/nhận xét:** test kiểm tra shape, dtype và tập giá trị threshold thuộc `{0,255}`; không có đánh giá phân đoạn định lượng.

### EXP-03: Mean Filter

- **Mục đích:** kiểm tra kích thước cửa sổ và xử lý biên.
- **Tham số:** kernel 3 và 5; border mặc định, `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k mean_filter` trong bản sao tạm.
- **Artifact:** [`output/linearFilters/`](../output/linearFilters/) có `mean_k2.png`, `mean_k5.png`, `mean_constant.png`, `mean_replicate.png`, `mean_reflect.png`.
- **Kết quả/nhận xét:** **lệch nhãn** — test truyền kernel 3 cho `mean_k2.png`; không dùng tên file để suy ra kernel 2.

### EXP-04: Gaussian Filter

- **Mục đích:** kiểm tra Gaussian với sigma tường minh và tự suy ra.
- **Tham số:** `(kernel,sigma)=(0,1.25)` và `(5,0.25)`, `(5,0.5)`, `(5,0.75)`, `(5,1.25)`.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k gaussian_filter` trong bản sao tạm.
- **Artifact:** [`output/linearFilters/`](../output/linearFilters/) có `gaussian_auto.png`, `gaussian_sigma_025.png`, `gaussian_sigma_05.png`, `gaussian_sigma_075.png`, `gaussian_sigma_125.png`.
- **Kết quả/nhận xét:** test xác nhận shape/dtype; không có metric độ mờ/chất lượng.

### EXP-05: Sharpen và Custom Kernel

- **Mục đích:** kiểm tra sharpen mặc định, kernel mẫu và validation kernel.
- **Tham số:** sharpen mặc định; bốn kernel preset; custom sharpen; kernel identity; kernel 2×2 không hợp lệ.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k 'sharpen or custom_kernel or preset_kernels'` trong bản sao tạm.
- **Artifact:** [`output/linearFilters/`](../output/linearFilters/) có `sharpen.png`; [`output/linearFilters/customKernels/`](../output/linearFilters/customKernels/) có `box_blur.png`, `sharpen.png`, `laplacian.png`, `emboss.png`, `custom_kernel.png`.
- **Kết quả/nhận xét:** test kiểm tra shape/dtype; kernel identity phải cho ảnh không đổi và kernel 2×2 phải gây exception.

### EXP-06: Edge Detection

- **Mục đích:** kiểm tra đầu ra Prewitt và Sobel.
- **Tham số:** Prewitt mặc định; Sobel kernel mặc định 3.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k 'prewitt or sobel'` trong bản sao tạm.
- **Artifact:** [`output/edgeDetection/`](../output/edgeDetection/) có `prewitt.png`, `sobel.png`.
- **Kết quả/nhận xét:** test kiểm tra kích thước không gian và dtype; không có đánh giá với ground truth cạnh.

### EXP-07: Nonlinear Filters

- **Mục đích:** kiểm tra Median và Bilateral.
- **Tham số:** Median kernel 5; Bilateral `diameter=5`, `sigma_color=75`, `sigma_space=75`.
- **Test/lệnh:** `pytest tests/test_image_processing.py -k 'median or bilateral'` trong bản sao tạm.
- **Artifact:** [`output/nonlinearFilter/`](../output/nonlinearFilter/) có `median.png`, `bilateral.png`.
- **Kết quả/nhận xét:** test xác nhận shape/dtype; không lưu metric khử nhiễu cho kịch bản này.

### EXP-08: Filter Comparison

- **Mục đích:** chạy bốn bộ lọc trên hai mô hình nhiễu và kiểm tra cấu trúc kết quả/metric.
- **Tham số:** từ cùng ảnh xám gốc tạo salt-and-pepper `amount=0.05` và Gaussian `sigma=20`, seed 42; Mean, Gaussian, Median, Bilateral; `repeat=3` trong test.
- **Dữ liệu:** cùng ảnh xám sạch shape `(678,452)` được dùng làm nguồn tạo từng ảnh nhiễu.
- **Tham số lọc:** `kernel_size=5`, `gaussian_sigma=1.0`, `bilateral_diameter=5`, `bilateral_sigma_color=75`, `bilateral_sigma_space=75`, `BORDER_DEFAULT`.
- **Lệnh test:** `pytest tests/test_image_processing.py::test_compare_filters` trong bản sao tạm. Đo bổ sung gọi `compare_filters(noisy, repeat=3)` trong bộ nhớ và đánh giá cùng ảnh lọc bằng `evaluate_filter_results(clean, filtered)`; không ghi ảnh hay metric ra file.
- **Artifact:** [`output/comparisons/`](../output/comparisons/) có `salt_pepper_input.png`, `salt_pepper_mean.png`, `salt_pepper_gaussian.png`, `salt_pepper_median.png`, `salt_pepper_bilateral.png` và năm tệp tiền tố `gaussian_`, tổng cộng 10 ảnh.
- **Kết quả/nhận xét:** số đo lần chạy cô lập và lần benchmark bổ sung được ghi ở các bảng sau. Test gốc chỉ tính metric so với ảnh nhiễu; bảng riêng dưới đây bổ sung metric so với ảnh sạch.

### Metric so với ảnh gốc sạch

`MSE`, `MAE`, `PSNR` so sánh output lọc với ảnh xám sạch ban đầu. `Execution Time` là thời gian trung bình của 3 lần đo sau một lần warm-up; số liệu được làm tròn đến 3 chữ số thập phân.

| Noise Type    | Filter    |        MSE |       MAE |  PSNR (dB) | Execution Time (ms) |
| ------------- | --------- | ---------: | --------: | ---------: | ------------------: |
| Salt & Pepper | Mean      |    139.576 |     8.508 |     26.683 |               0.322 |
| Salt & Pepper | Gaussian  |    139.548 |     7.612 |     26.684 |           **0.120** |
| Salt & Pepper | Median    | **75.808** | **5.095** | **29.334** |               0.490 |
| Salt & Pepper | Bilateral |    581.236 |     7.721 |     20.487 |               0.510 |
| Gaussian      | Mean      |    100.253 |     7.026 |     28.120 |               0.287 |
| Gaussian      | Gaussian  | **71.853** | **6.305** | **29.566** |           **0.196** |
| Gaussian      | Median    |    105.133 |     7.276 |     27.913 |               0.521 |
| Gaussian      | Bilateral |     74.238 |     6.403 |     29.425 |               0.619 |

### Metric theo cách `compare_filters` hiện trả về: so với ảnh nhiễu

Các chỉ số dưới đây là so sánh ảnh sau lọc với **ảnh nhiễu đầu vào**, không phải ảnh sạch. Thời gian giống lần đo ở bảng trước, được lặp tại đây để tách bạch reference.

| Noise Type    | Filter    | MSE so noisy | MAE so noisy | PSNR so noisy (dB) | Execution Time (ms) |
| ------------- | --------- | -----------: | -----------: | -----------------: | ------------------: |
| Salt & Pepper | Mean      |     1064.158 |       14.039 |             17.861 |               0.322 |
| Salt & Pepper | Gaussian  |      810.747 |       11.744 |             19.042 |           **0.120** |
| Salt & Pepper | Median    |     1101.177 |   **11.091** |             17.712 |               0.490 |
| Salt & Pepper | Bilateral |  **238.073** |        6.875 |         **24.364** |               0.510 |
| Gaussian      | Mean      |      431.902 |       16.457 |             21.777 |               0.287 |
| Gaussian      | Gaussian  |  **313.129** |   **14.057** |         **23.174** |           **0.196** |
| Gaussian      | Median    |      436.668 |       16.017 |             21.729 |               0.521 |
| Gaussian      | Bilateral |      319.205 |       14.262 |             23.090 |               0.619 |

Đây là một lượt benchmark ngắn với ba lần lặp, không có nhiều lượt độc lập, độ lệch chuẩn hay khoảng tin cậy; thời gian chỉ có giá trị tham khảo trên môi trường tại mục 2.

## 5. Kiểm thử tự động bằng Pytest

- Lệnh chạy trong bản sao tạm: `python -m pytest -q -p no:cacheprovider <bản-sao-tạm>\tests\test_image_processing.py`.

- Kết quả: **33 PASS, 0 FAIL, 0 ERROR** trong 1.63 giây; exit code 0.
- Cách bảo toàn dữ liệu: sao chép `src/` và `tests/` vào thư mục tạm, chạy test ở đó. Fixture vẫn đọc ảnh nguồn tại đường dẫn hiện có; 40 ảnh đầu ra được tạo trong thư mục tạm và thư mục được tự dọn khi tiến trình kết thúc. Không có file trong `output/` của workspace bị ghi đè.
- Không có test failure nên không phát hiện lỗi cần khắc phục qua lượt chạy này. Các sai khác đã nêu trong mục 7 là vấn đề đọc từ source, không phải lỗi test được quan sát.
- Test có các kiểm tra cụ thể về phép âm bản, đầu ra phân ngưỡng, kernel identity, kernel 2×2 không hợp lệ, metric trên ảnh nhỏ và metric vô cực khi hai ảnh giống nhau. Test filter comparison kiểm tra số lượng kết quả, kiểu/shape và metric không âm; không khẳng định bộ lọc nào tốt hơn.

## 6. Phân tích và nhận xét kết quả

1. Với salt-and-pepper và metric so ảnh sạch, Median có MSE thấp nhất (75.808), MAE thấp nhất (5.095) và PSNR cao nhất (29.334 dB); Gaussian nhanh nhất trong lượt đo này (0.120 ms).
2. Với Gaussian noise, Gaussian Filter có MSE thấp nhất (71.853), MAE thấp nhất (6.305) và PSNR cao nhất (29.566 dB); cũng nhanh nhất trong lượt đo này (0.196 ms). Bilateral gần với Gaussian về MSE/PSNR trong ảnh này nhưng thời gian đo cao hơn.
3. Đây là quan sát trên một ảnh 678×452, seed 42, một mức nhiễu mỗi loại và chỉ ba lần lặp benchmark; không suy rộng thành kết luận phổ quát. Thời gian rất ngắn nên có thể dao động theo tải máy/backend.
4. Bảng so với ảnh nhiễu khác bảng so với ảnh sạch. Chẳng hạn bilateral có MSE thấp nhất so ảnh nhiễu ở salt-and-pepper nhưng MSE cao nhất so ảnh sạch; chỉ bảng ground-truth đo sai số phục hồi so với ảnh gốc sạch.
5. Tên `mean_k2.png` không khớp kernel 3 mà test truyền vào. Không dùng tên output thay cho tham số đã truyền.

## 7. Vấn đề tồn tại và hướng cải thiện

| Phát hiện                                                                                                                                | Ảnh hưởng đến thực nghiệm/tài liệu                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `requirement.txt` rỗng                                                                                                                   | Không tái lập được phiên bản môi trường từ manifest.                                                   |
| Fixture test dùng đường dẫn tuyệt đối                                                                                                    | Test phụ thuộc vị trí workspace và tên tài khoản trên máy.                                             |
| Test ghi ảnh trực tiếp vào `output/`                                                                                                     | Chạy test trong bản sao tạm để không ghi đè artifact hiện tại.                                         |
| `compare_filters` đối chiếu với ảnh nhiễu                                                                                                | Metric mặc định không đo chất lượng phục hồi so với ảnh sạch; nhật ký lưu riêng cả hai cách đối chiếu. |
| `main.py` đọc BGR nhưng một nhánh dùng `COLOR_RGB2GRAY`; hiển thị mặc định RGB                                                           | Giá trị xám/hiển thị màu có thể lệch với dữ liệu thực tế.                                              |
| `INPUT_PATH` trỏ `data/input/lena.png`, trong khi ảnh hiện có là `data/input/images.jpg`; `main()` vẫn hỏi người dùng nhập đường dẫn mới | Giá trị khai báo không phải ảnh đầu vào mặc định được dùng thực tế.                                    |
| Menu custom kernel truyền chuỗi trực tiếp                                                                                                | Kịch bản nhập kernel từ CLI chưa cung cấp ma trận đúng kiểu cho hàm xử lý.                             |
| Nguồn ảnh và license chưa rõ                                                                                                             | Chưa thể trích dẫn nguồn dataset hoặc xác nhận quyền sử dụng.                                          |

Để tăng độ tin cậy hiệu năng, nên chạy nhiều phiên benchmark độc lập, báo cáo trung vị/độ phân tán, và lưu kết quả có kiểm soát ở artifact riêng; không nên trộn metric theo ảnh nhiễu với metric theo ảnh sạch.

## 8. Kết luận

Lượt chạy cô lập xác nhận 33 test PASS, không có FAIL/ERROR, đồng thời tạo được bảng metric thực tế với ảnh sạch và ảnh nhiễu làm hai reference riêng. Trong thí nghiệm đơn này, Median cho chất lượng tốt nhất trên salt-and-pepper, còn Gaussian Filter tốt nhất trên Gaussian noise theo MSE/MAE/PSNR; Gaussian Filter có thời gian ngắn nhất ở cả hai trường hợp đo. Kết luận chỉ áp dụng cho ảnh, tham số, seed và phiên benchmark đã ghi; nguồn ảnh/license và độ ổn định thời gian qua nhiều lượt vẫn cần bổ sung.
