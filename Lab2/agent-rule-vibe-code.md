# Agent Rule: Vibe Coding Lab2

Hướng dẫn này được rút ra từ mã nguồn, notebook, tài liệu và test trong Lab2. Khi quy ước khác nhau giữa các khu vực, hãy ưu tiên phong cách của file đang được chỉnh sửa; không tự ý đồng nhất hoặc tái cấu trúc toàn project.

## A. Coding Philosophy

- Lab2 có hai cách tổ chức đang cùng tồn tại: package chia module trong `src/` và các script/notebook thực hành viết theo luồng từ trên xuống. Chọn cách phù hợp với khu vực đích, không áp đặt kiến trúc mới.
- Ưu tiên giải pháp xử lý ảnh trực tiếp, dễ theo dõi bằng NumPy và API OpenCV; phần điều phối, xử lý ảnh, I/O, kiểm tra dữ liệu và hiển thị đã có các khu vực riêng trong package.
- Giữ nguyên hợp đồng đầu vào/đầu ra và cách người dùng quan sát kết quả, trừ khi yêu cầu nêu rõ cần thay đổi.

## B. Project Structure

- `src/main.py`: menu dòng lệnh và điều phối luồng xử lý.
- `src/imageProcessing/`: các thuật toán theo nhóm như `pointOperations.py`, `linearFilters.py`, `nonlinearFilters.py`, `edgeDetection.py`, `customerKernel.py`, `comparision.py`.
- `src/utils/`: đọc/ghi ảnh, validation, hiển thị và menu.
- `tests/test_image_processing.py`: kiểm thử; ảnh nguồn nằm trong `data/input/`, ảnh kết quả test nằm trong `output/`.
- `lab2.py` và `lab2.ipynb`: cách triển khai thực hành dạng tuần tự, có hàm và phần chạy thử trong cùng script/cell.
- Tài liệu nằm trong `README.md` và `docs/`. Không chuyển module, đổi tên file hiện có hoặc tạo thêm tầng kiến trúc nếu yêu cầu không cần.
- Với chức năng mới thuộc package, đặt code cạnh nhóm thuật toán liên quan và cập nhật điểm export trong `__init__.py` tương ứng nếu hàm cần dùng qua API package.

## C. Coding Convention

- Trong `src/`, tên hàm và biến thường theo `snake_case`; tên module hiện có pha trộn camelCase và cách viết lịch sử (ví dụ `pointOperations.py`, `comparision.py`). Không đổi tên module để chuẩn hóa.
- Trong `lab2.py` và notebook, có các tên camelCase như `lightAndContrast`, `filterMean`, `imgGray`. Khi sửa các khu vực này, giữ cách đặt tên tại chỗ; không đổi tên API cũ tùy tiện.
- Class chưa được dùng trong mã xử lý ảnh đã khảo sát; không tạo class nếu một hàm/module đơn giản đáp ứng yêu cầu.
- Các module package thường nhóm import thư viện chuẩn, thư viện ngoài (`cv2`, `numpy`, ...) rồi mới đến import nội bộ; trong package có cả import tương đối và `from src...` tùy vị trí. Tiếp tục theo mẫu của file đang sửa.
- Mã package dùng type hints ở nhiều hàm, gồm `NDArray[np.uint8]`, alias kiểu ảnh và generic Python như `list[...]`; script/notebook cũ thường không khai báo kiểu. Bám theo phong cách cục bộ.
- Thụt lề 4 dấu cách. Comment và tiêu đề khối thường viết tiếng Việt, có thể dùng thuật ngữ kỹ thuật tiếng Anh. Chỉ thêm comment giải thích bước xử lý hoặc nhóm chức năng.
- Docstring trong package thường viết tiếng Việt, mô tả mục đích và các mục `Args`, `Returns`, `Raises`, đôi khi có `Notes`; hàm nhỏ có thể dùng docstring một dòng.

## D. Function Design

- Mỗi hàm xử lý ảnh nên tập trung vào một phép biến đổi hoặc một bước tiện ích. Dùng các API sẵn có như `cv2.blur`, `cv2.GaussianBlur`, `cv2.filter2D`, `cv2.Sobel`, `cv2.medianBlur`, `cv2.bilateralFilter` và các phép mảng NumPy khi phù hợp.
- Các hàm package thường nhận ảnh `numpy.ndarray` `uint8`, kiểm tra ảnh/tham số trước khi xử lý, rồi trả ảnh có shape và dtype được nêu rõ. Một số thuật toán yêu cầu ảnh grayscale; hãy giữ đúng tiền điều kiện đó.
- Với phép tính số học trên ảnh, chuyển sang `float32` khi cần tránh tràn `uint8`, giới hạn miền giá trị trước khi đổi về `uint8`. Các metric trong `comparision.py` dùng `float64` để tính sai số.
- Dùng validator dùng chung trong `src/utils/` nếu nó đã kiểm tra đúng điều kiện cần thiết; không viết lại kiểm tra riêng trùng lặp.
- Theo mẫu package, dữ liệu sai kiểu dùng `TypeError`, giá trị/cấu trúc không hợp lệ dùng `ValueError`; để lỗi từ OpenCV được truyền lên hàm gọi. Menu bắt các lỗi liên quan và hiển thị thông báo. Các script cũ có chỗ in cảnh báo và trả `None`; chỉ giữ cách này khi tiếp tục sửa đúng luồng cũ.
- Với ảnh màu, xác định rõ thứ tự kênh (`BGR`/`RGB`) tại I/O và hiển thị; với grayscale dùng chế độ `GRAY`. Không giả định hai thứ tự màu có thể thay thế nhau.

## E. Notebook Convention

- Notebook hiện có tổ chức theo chủ đề bài lab: cell Markdown nêu phần/bài, mục đích hoặc cách dùng; cell code kế tiếp định nghĩa hàm, chạy phép xử lý và hiển thị kết quả.
- Markdown và chú thích chủ yếu bằng tiếng Việt, giữ tên thuật toán/thư viện tiếng Anh khi cần. Kết quả thực hành được thể hiện bằng output của cell như ảnh, histogram, bảng hoặc thông tin in ra.
- Notebook hiện chứa các hàm dùng chung và thực hiện tuần tự, dựa trên trạng thái biến giữa các cell. Khi thêm nội dung, đặt cell gần chủ đề tương ứng và đảm bảo các biến cần thiết đã được tạo ở cell trước.
- Không chuyển notebook sang cấu trúc package hoặc xóa/ghi lại output hàng loạt nếu yêu cầu không yêu cầu việc đó.

## F. Implementation Rules

- Đọc hàm, module, cách gọi, export và test liên quan trước khi sửa; tìm cách tái sử dụng validator hoặc helper hiện có.
- Sửa đúng khu vực trách nhiệm: thuật toán ở `src/imageProcessing/`, tiện ích ở `src/utils/`, điều phối tương tác ở `src/main.py`, kiểm thử ở `tests/`.
- Giữ tương thích với tên hàm, tham số, kiểu trả về, thứ tự kết quả và định dạng ảnh mà caller/test đang dùng.
- Tránh thay đổi không liên quan, đổi API, đổi tên file hoặc chuyển đổi toàn bộ code sang một quy ước mới. Với khác biệt giữa script cũ và package, theo phong cách tại chỗ.
- Không áp dụng quy tắc “cấm vòng lặp”: code hiện có dùng vòng lặp cho danh sách kernel, bộ lọc và phép đo lặp; phép tính trên toàn ảnh có thể dùng NumPy/OpenCV để giữ cách triển khai gọn.
- Khi có hành vi mới, thêm hoặc điều chỉnh test tương ứng thay vì chỉ dựa vào việc chương trình chạy mà không báo lỗi.

## G. Visualization and Testing

- Package dùng `display_images` với dictionary `{tên: (ảnh, chế_độ_màu)}`, tiêu đề chung và số cột; ảnh xám hiển thị bằng grayscale, ảnh màu cần xử lý đúng BGR/RGB, trục ảnh được ẩn.
- Script/notebook cũ dùng `displayIMG(titles_list, images_list, ...)`, có tùy chọn `forceGray` và `histogram`. Nếu đang ở luồng này, giữ cách gọi sẵn có thay vì thay bằng giao diện khác.
- Test dùng pytest, fixture `source_images` để chuẩn bị ảnh, `pytest.mark.parametrize` cho nhiều tham số, và helper `save_result` để ghi ảnh qua `write_image`. Các assert phổ biến kiểm tra shape, dtype, tập giá trị hoặc kết quả số học cụ thể.
- Với so sánh bộ lọc, code hiện có dùng MSE, MAE, PSNR và thời gian thực thi; ghi rõ ảnh tham chiếu khi diễn giải metric. Kết quả hình ảnh/histogram hỗ trợ quan sát nhưng không thay thế assert.
- Chạy test liên quan khi môi trường cho phép. Fixture hiện tại có đường dẫn ảnh tuyệt đối gắn với máy; việc chạy test có thể phụ thuộc đường dẫn workspace.

## H. Vibe Coding Instructions

Khi nhận yêu cầu trong Lab2:

1. Đọc code và test liên quan trong Lab2 trước khi chỉnh sửa.
2. Xác định file đang theo kiểu package hay script/notebook và kế thừa đúng phong cách tại đó.
3. Chọn thay đổi nhỏ, dễ hiểu, có thể chạy; không thêm kiến trúc hoặc thư viện nếu không cần thiết.
4. Giữ nhất quán API, dữ liệu ảnh, quy ước kênh màu, hiển thị và test liên quan.
5. Không chỉnh sửa code ngoài phạm vi yêu cầu; nêu ngắn gọn các thay đổi quan trọng và kết quả kiểm tra.
6. Nếu không thể chạy test, kiểm tra cú pháp hoặc giải thích rõ giới hạn thay vì khẳng định đã xác minh.

## Quy ước chưa đủ căn cứ

- Không thấy class trong các module xử lý ảnh đã khảo sát; vì vậy chưa có quy ước thiết kế class.
- `requirement.txt` hiện rỗng và không thấy cấu hình linter/formatter trong Lab2; chưa xác định công cụ hoặc cấu hình format bắt buộc.
- Quy ước tên không hoàn toàn thống nhất giữa package và script/notebook; không có căn cứ để đổi toàn bộ sang một kiểu tên duy nhất.
