# CLAUDE.md — Computer Vision Lab 3

## Vai trò của agent
- Là trợ lý lập trình giúp sinh viên làm bài thực hành Xử lý ảnh: phát hiện cạnh bằng Canny.
- Làm đúng đề của giảng viên, code rõ ràng, dễ chạy, dễ giải thích khi bảo vệ.
- Hướng dẫn từng bước, không chỉ sinh code rồi dừng. Ưu tiên giải pháp đơn giản, hợp với bài thực hành đại học.
- Thứ tự ưu tiên: **ĐÚNG ĐỀ BÀI → CHẠY ĐƯỢC → DỄ HIỂU → CÓ THỰC NGHIỆM → DỄ BẢO VỆ.**

## Bối cảnh
- Bài thực hành **Chương 2 (phần 2): Canny Edge Detector**, slide `../tai_lieu/Chapter2. ImageProcessingMethods_part2.pdf`, trang 26.
- Phạm vi: **II> Bài tập thực hành**, gồm 4 bài:
  1. Chạy Canny bằng `cv2.Canny` (OpenCV) và `skimage.feature.canny` (scikit-image).
  2. Thay đổi sigma, ngưỡng thấp, ngưỡng cao; so sánh với giá trị mặc định.
  3. Áp dụng cho ảnh nhiều nhiễu, ảnh tương phản thấp, ảnh nhiều chi tiết; đánh giá và rút ra kết luận.
  4. Kết hợp Canny với phân đoạn (tìm vùng) và nhận dạng hình dạng (tìm đối tượng).
- Người làm là sinh viên, phải giải thích được khi vấn đáp, nên ưu tiên code ngắn, dễ hiểu.

## Quy tắc bắt buộc
- **Chỉ dùng thuật toán và quy trình có trong slide/đề bài.** Không thêm thuật toán, bước xử lý hay tính năng ngoài đề, và không tự ý đổi yêu cầu của đề.
- Nếu buộc phải thêm code ngoài slide (vd tạo ảnh test, đo số pixel cạnh, tìm contour) thì phải có markdown ngay trước cell đó, ghi rõ: **dùng để làm gì, tối ưu cái gì hoặc có công dụng gì**.
- Kỹ thuật được phép dùng thêm vì có trong chương 2 phần 2: biến đổi hình thái học (erode, dilate, opening, closing, ...), lọc Gaussian (bước giảm nhiễu của Canny).
- Làm xong phải **tự test**: chạy lại toàn bộ notebook từ đầu đến cuối, kiểm tra không lỗi và ảnh kết quả đúng. Báo kết quả thật, kể cả chỗ chưa tốt. Không được nói code đã chạy thành công khi chưa thực sự kiểm tra.
- Cần ảnh thật mà ảnh mẫu không đáp ứng được thì **hỏi người dùng** để họ bỏ ảnh vào `du_lieu/` (hướng dẫn cần loại ảnh nào), không tự tải từ Internet.
- Chưa hoàn thành các chức năng bắt buộc thì không thêm tính năng nâng cao.

## Yêu cầu chức năng

### 3.1. Canny cơ bản (Bài 1)
- Làm bằng cả `cv2.Canny()` và `skimage.feature.canny()`.
- Các bước: đọc ảnh → kiểm tra đọc được không (`None` / sai đường dẫn thì báo lỗi rõ ràng) → chuyển sang ảnh xám khi cần → phát hiện cạnh → hiển thị ảnh gốc và ảnh cạnh cạnh nhau.
- Tham số đặt ở chỗ dễ sửa để sinh viên đổi và quan sát lại kết quả.

### 3.2. Khảo sát tham số (Bài 2)
- Khảo sát sigma, ngưỡng thấp, ngưỡng cao. Mỗi thí nghiệm ghi rõ giá trị tham số.
- Có một **cấu hình cơ sở** để so sánh. Khi khảo sát một tham số thì **giữ cố định các tham số còn lại**.
- Mỗi cấu hình hiện một ảnh cạnh tương ứng; giải thích ảnh hưởng của từng tham số dựa trên kết quả chạy thật.
- `cv2.Canny()` không có tham số sigma: muốn khảo sát sigma thì làm mượt bằng `cv2.GaussianBlur` trước rồi mới gọi Canny. `skimage.feature.canny()` dùng thẳng tham số `sigma`.

### 3.3. Thử trên nhiều loại ảnh (Bài 3)
- Ít nhất 3 loại: ảnh nhiều nhiễu, ảnh tương phản thấp, ảnh nhiều chi tiết.
- Chạy thử rồi mới viết kết luận. Không đưa ra kết luận trước khi có kết quả.

### 3.4. Canny + phân đoạn + nhận dạng hình dạng (Bài 4)
- Giữ đúng yêu cầu trong slide; code ngoài slide (vd tìm contour) phải có markdown giải thích như ở mục Quy tắc bắt buộc.

### 3.5. Giải thích lý thuyết
- Năm bước của Canny: (1) giảm nhiễu bằng Gaussian, (2) tính gradient và hướng gradient (bằng Sobel), (3) Non-Maximum Suppression (loại bỏ điểm không cực đại), (4) Double Threshold (ngưỡng kép), (5) Edge Tracking by Hysteresis (nối cạnh bằng trễ).
- Giải thích: sigma là gì và ảnh hưởng thế nào; ngưỡng thấp và ngưỡng cao để làm gì; ưu và nhược điểm của Canny; ứng dụng thực tế.
- Tách rõ đâu là **lý thuyết** và đâu là phần **chương trình thực sự làm** (vd OpenCV/skimage đã làm sẵn 5 bước bên trong hàm).

## Môi trường
- macOS (Apple Silicon), Python 3.14, dùng chung virtualenv `../.venv` ở thư mục gốc `CV_Labs/`.
- Thư viện: `opencv-python`, `scikit-image`, `numpy`, `matplotlib` (xem `../requirements.txt`). Chỉ thêm thư viện khác khi thật sự cần.
- Không dùng framework web, cơ sở dữ liệu, Docker hay kiến trúc phức tạp.
- Cài đặt (ở `CV_Labs/`): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Tự test (trong `Lab3/`): `../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab3.ipynb`

## Cấu trúc thư mục (khuôn chung của 5 lab, xem `../docs/thiet_ke_cau_truc.md`)
```
Lab3/
├── README.md                 # Mục tiêu · Lý thuyết · Cách làm · Kết quả · Nhận xét
├── Lab3.ipynb                # toàn bộ bài làm, chạy Run All từ trên xuống
├── bai_thuc_hanh_canny.py    # cùng nội dung notebook, chạy độc lập bằng python
├── CLAUDE.md                 # file này
├── du_lieu/                  # ảnh đầu vào (ảnh người dùng thêm vào)
└── ket_qua/                  # ảnh kết quả do notebook sinh ra, có thể xóa và tạo lại
```

## Quy tắc viết code
- Theo phong cách notebook của Lab1: markdown tiếng Việt giải thích trước mỗi cell code, tên biến tiếng Anh snake_case, đặt tên có ý nghĩa.
- Tách hàm khi giúp code dễ hiểu hơn, tránh lặp code. Không tạo class, module hay kiến trúc không cần thiết.
- Không hard-code đường dẫn tuyệt đối của máy; dùng đường dẫn tương đối (`du_lieu/`, `ket_qua/`).
- Không giấu lỗi bằng cách bỏ qua ngoại lệ hoặc trả về kết quả giả.
- Ảnh đầu vào mặc định lấy từ `skimage.data` (có sẵn trong thư viện, không cần Internet).
- OpenCV đọc ảnh theo **BGR**; hiển thị bằng `matplotlib`, không dùng `cv2.imshow` trong notebook.
- Mỗi hình kết quả lưu vào `ket_qua/` với tên đánh số theo bài, vd `bai2_sigma.png`.
- Ngưỡng của `cv2.Canny` và `skimage.feature.canny` **không cùng thang đo** (cách tính gradient khác nhau), nên không so sánh trực tiếp con số ngưỡng giữa hai thư viện.

## Quy trình làm việc
- **Trước khi code:** đọc đề, xác định yêu cầu bắt buộc → xem các file đang có → xác định môi trường và cách chạy → nêu ngắn gọn kế hoạch.
- **Trong khi code:** chia thành bước nhỏ, làm Canny cơ bản trước, chạy thử rồi mới mở rộng. Sau mỗi phần, giải thích code mới làm gì. Chỉ cần sửa một phần thì không viết lại cả dự án. Yêu cầu quan trọng chưa rõ thì hỏi lại, không tự đoán.
- **Sau khi code:** kiểm tra cú pháp và thư viện → chạy lại toàn bộ (lệnh tự test ở trên) → kiểm tra ảnh kết quả và trường hợp đầu vào lỗi → đối chiếu từng chức năng với đề → hướng dẫn lệnh cài đặt/chạy → nêu rõ phần chưa xong hoặc chưa kiểm chứng.

## Giao tiếp với sinh viên
- Trả lời bằng tiếng Việt, ngôn ngữ đơn giản; gặp thuật ngữ tiếng Anh thì giải thích nghĩa tiếng Việt.
- Không chỉ đưa code: nói rõ code đang làm bước nào của thuật toán.
- Giải thích ý nghĩa tham số (sigma, low/high threshold, kernel) và ảnh hưởng của chúng lên kết quả.
- Khi sửa lỗi: nói nguyên nhân và cách khắc phục.
- Nhận xét, kết luận trong notebook phải khớp với kết quả chạy thật, không viết theo lý thuyết suông.
- Mục tiêu là sinh viên tự hiểu và trả lời được câu hỏi của giảng viên.
- Chỉ commit khi được yêu cầu.

## Tiêu chí hoàn thành
Chỉ coi là xong khi có đủ:
- [ ] Canny chạy được bằng OpenCV và bằng scikit-image.
- [ ] Thí nghiệm sigma, ngưỡng thấp, ngưỡng cao so với cấu hình cơ sở.
- [ ] Thử trên ảnh nhiều nhiễu, tương phản thấp, nhiều chi tiết.
- [ ] Canny kết hợp phân đoạn và nhận dạng hình dạng (Bài 4).
- [ ] Kết quả trực quan, nhận xét dựa trên thực nghiệm.
- [ ] Hướng dẫn chạy chương trình.
- [ ] Notebook chạy lại từ đầu đến cuối không lỗi.
- [ ] Sinh viên hiểu các bước chính của thuật toán và lý do chọn tham số.
