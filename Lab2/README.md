# Lab 2: Toán tử điểm ảnh và Các bộ lọc xử lý ảnh (Image Processing)

## Tổng quan

Bài tập thực hành tìm hiểu và hiện thực hóa các kỹ thuật cơ bản trong xử lý ảnh số sử dụng thư viện **OpenCV**, **NumPy** và **Matplotlib** tại Python.

## Cấu trúc các hàm chính

### **Phần I: Toán tử điểm ảnh**

1. `lightAndContrast(img, light, contrast)`: hàm dùng để điều chỉnh độ sáng và độ tương phản.
   - **Độ sáng (`light`)**: Cộng hoặc trừ một giá trị cố định cho từng pixel.
   - **Độ tương phản (`contrast`)**: Nhân mỗi pixel với một hằng số phóng đại.
   - _Kỹ thuật xử lý_: Chuyển đổi dữ liệu từ `uint8` sang `float32` trước khi tính toán để tránh hiện tượng tràn số (overflow/underflow) hoặc mất mát dữ liệu. Kết hợp hàm `np.clip()` để giới hạn giá trị pixel an toàn trong dải [0, 255] trước khi ép kiểu ngược lại về `uint8`.

2. `imgNegative(img)`: Tạo ảnh âm bản bằng cách đảo ngược mức xám (255 - img).

3. `imgBinary(img)`: Cắt ngưỡng nhị phân dựa trên giá trị mốc 128.

### **Phần II: Lọc tuyến tính**

1. `filterMean(img)`: Làm mờ tuyến tính bằng bộ lọc trung bình (Box Filter) sử dụng kernel kích thước `5x5` (với trọng số đồng nhất `1/25`).
   - **Đặc điểm**: Khi tăng kích thước kernel lên (`7x7`, `9x9`,...), vùng không gian lấy trung bình mở rộng, khiến ảnh bị nhòe nặng và mất dần chi tiết. Nguyên nhân do giá trị của pixel mới bị san phẳng theo diện tích lân cận lớn.

2. `filterGaussian(img)`: Làm mờ tự nhiên dựa trên hàm phân phối chuẩn (Gaussian). Hiệu ứng phụ thuộc vào 2 tham số cốt lõi:
   - **Kích thước Kernel (`ksize`)**: Kích thước càng lớn, mức độ làm mờ càng mạnh tương tự bộ lọc trung bình.
   - **Độ lệch chuẩn (`sigma`)**: Khi tăng `sigma`, đường cong phân phối xác suất (hình chuông) trải rộng và đều hơn, giúp các điểm ảnh ở xa tâm nhận trọng số lớn hơn -> tạo độ mịn mượt và lan tỏa tự nhiên. Ngược lại, `sigma` quá nhỏ (ví dụ < 1) sẽ khiến phân phối co cụm vào tâm, hiệu ứng làm mờ gần như không đáng kể.

3. `filterSharpen(img)`: Tăng cường độ sắc nét bằng ma trận kernel `3x3` có trọng số tâm dương và các biên xung quanh âm.
   - **Đặc điểm**: Tổng các phần tử trong kernel đúng bằng `1`, giúp bảo toàn cường độ sáng tổng thể của bức ảnh mà không làm biến đổi nền, đồng thời khuếch đại độ tương phản cục bộ để làm nổi bật các chi tiết sắc cạnh tại tâm.

### **Phần III: Bài tập nâng cao**

1. `detectionEdge(img)`: Phát hiện cạnh sử dụng hai toán tử kinh điển là Sobel và Prewitt.
   - **Điểm giống**:
     - Cả hai đều sử dụng 2 ma trận kernel riêng biệt để tính gradient theo 2 hướng: chiều ngang và chiều dọc, sau đó tổng hợp lại để thu được ảnh biên toàn diện cả 2 chiều.
     - Sử dụng hàm `cv2.addWeighted()` để cộng gộp hai ảnh thành phần dựa trên công thức tuyến tính: Result = (ImgNgang _ alpha) + (ImgDoc _ beta) + gamma. Trong đó, ta thiết lập alpha = 0.5 và beta = 0.5 (lấy tỷ lệ cân bằng 50% từ mỗi hình), đồng thời đặt gamma = 0 vì không cần bù thêm độ sáng nền.
   - **Điểm khác biệt & Lưu ý kỹ thuật**:
     - **Toán tử Prewitt**: Sử dụng các hệ số trọng số đơn giản và đồng đều (`-1, 0, 1`).
     - **Toán tử Sobel**: Ma trận kernel biến đổi linh hoạt tùy thuộc vào bậc đạo hàm theo trục X và Y (`dx`, `dy`), đồng thời nhấn mạnh trọng số ở tâm (2) giúp hỗ trợ chống nhiễu tốt hơn.
     - _Lưu ý ràng buộc_: Tổng bậc đạo hàm trong Sobel không được vượt quá kích thước kernel (dx+ dy < ksize), nếu vi phạm OpenCV sẽ báo lỗi do ma trận kernel không đủ không gian tính toán.

2. `designKernelAndFilter(img)`: Thử nghiệm các ma trận kernel tùy chỉnh (làm mờ chuyển động, làm nét mạnh, v.v.).


    + **Nguyên lý thiết kế ma trận Kernel (`3x3`) dựa trên trọng số**:
        - **Trọng số dương (+)**: Giữ lại hoặc khuếch đại cường độ sáng của pixel.
        - **Trọng số âm (-)**: Trừ đi hoặc triệt tiêu cường độ sáng từ các vùng lân cận để tạo độ tương phản.
    + **Quy chuẩn tổng trọng số (Kernel Sum Rule)**:
        - **Làm mịn**: Bản chất là tính trung bình cộng cường độ sáng vùng lân cận để làm mượt gai nhiễu, **tổng tất cả các phần tử trong kernel bằng 1** giúp bảo toàn độ sáng tổng thể của ảnh (không bị cháy sáng hay tối đi).
        - **Làm nét**: Bản chất là tăng cường độ tương phản cục bộ nhằm làm nổi bật pixel trung tâm; cấu trúc kernel có **tâm dương lớn, các ô xung quanh âm** và **tổng tất cả các phần tử bằng 1** để giữ nguyên nền.
        - **Phát hiện biên**: Dựa trên sự thay đổi gradient để dò tìm ranh giới và **xóa bỏ hoàn toàn các vùng đồng màu (vùng phẳng)**, cấu trúc kernel được thiết kế sao cho **tổng tất cả các phần tử bằng 0**.

3. `compareFilter(img, numNoise)`: Thử nghiệm và so sánh hiệu quả xử lý nhiễu muối tiêu (salt-and-pepper noise) giữa các bộ lọc tuyến tính và phi tuyến tính:

- **1. Ảnh có nhiễu (Noisy Image)**: Ảnh gốc sau khi được rắc các hạt nhiễu trắng/đen ngẫu nhiên (`numNoise`), làm suy giảm chất lượng và che lấp chi tiết biên.
- **2. Lọc trung bình (Mean Filter)**:
  - _Đặc điểm_: Do tính chất lấy trung bình cộng trên toàn cửa sổ, các điểm nhiễu cực đoan (0 hoặc 255) bị lan rộng ra các pixel lân cận.
  - _Kết quả_: Nhiễu chưa được khử triệt tiêu hoàn toàn, đồng thời làm thay đổi độ sáng cục bộ và gây nhòe mờ nền ảnh.
- **3. Lọc Gaussian (Gaussian Filter)**:
  - _Đặc điểm_: Là bộ lọc tuyến tính dựa trên trọng số phân phối chuẩn.
  - _Kết quả_: Vẫn còn sót lại các vết mờ do nhiễu, nhưng dải màu chuyển đổi mượt mà, dịu mắt và giữ được độ tự nhiên tốt hơn so với lọc trung bình.
- **4. Lọc Trung vị (Median Filter)**:
  - _Đặc điểm_: Là bộ lọc phi tuyến tính chọn giá trị trung vị trong vùng lân cận.
    Chọn size kernel nhưng kernel lúc này không được dùng để nhân ma trận trọng số nữa mà lúc này nó sẽ chứa các giá trị pixel -> sắp xếp chúng từ nhỏ đến lớn lấy ở giữa vì vậy nên các điểm nhiễu 0,255 sẽ bị xếp đầu cuối không nhận giá trị đó.
  - _Kết quả_: **Khử nhiễu muối tiêu tuyệt đối**, trả lại hình ảnh sạch sẽ hoàn toàn mà không làm ảnh hưởng đến cấu trúc hình học của các hình khối.
- **5. Lọc Bilateral (Bilateral Filter)**:
  - _Đặc điểm_: Bộ lọc phi tuyến tính tích hợp cả khoảng cách không gian và độ chênh lệch cường độ sáng (Edge-preserving filter). Phụ thuộc vào 2 yếu tố:
    - Trọng số không gian (sigmaSpace): giống như gaussian pixel trung tâm về mặt không gian sẽ nhận được trọng số lơn hơn
    - Trọng số màu sắc (sigmaColor): đây là mấu chốt của việc bắt cạnh, so sánh màu sắc của pixel lân cận và tâm, nếu màu sắc tương đồng với tâm nó sẽ lấy trọng số lớn để làm mịn, nếu khác với tâm thì trọng số sẽ bị giảm về gần 0 để không trộn lẫn.
  - _Kết quả_: Vừa làm mịn màng vùng bên trong, vừa **giữ gìn và bắt trọn các đường biên (edges) cực kỳ sắc nét**, không bị hiện tượng nhòe cạnh.

## Chạy chương trình

1. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
