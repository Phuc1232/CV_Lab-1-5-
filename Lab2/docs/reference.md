# REFERENCES – TÀI LIỆU THAM KHẢO

## Quy ước trích dẫn và trạng thái xác minh

Tài liệu sử dụng hệ thống trích dẫn đánh số `[R1]`, `[R2]`, ... Các mã này được dẫn trực tiếp trong [theory.md](./theory.md). Danh mục phân biệt ba loại thông tin:

- **Đã xác minh:** metadata hoặc nội dung chính được đối chiếu với trang chính thức của nhà xuất bản, tổ chức/tạp chí, cơ sở dữ liệu học thuật hoặc tài liệu gốc.
- **Tài liệu API:** nguồn chính thức mô tả cách sử dụng thư viện; phiên bản tài liệu được ghi rõ và không mặc nhiên coi là trùng với phiên bản runtime của Lab2.
- **Chưa xác định:** thông tin không có trong workspace hoặc chưa đủ bằng chứng để xác nhận; không được dùng như trích dẫn đã xác minh.

Ngày tra cứu các nguồn trực tuyến trong lần cập nhật này: **07-11-2026**.

## 1. Official Documentation

1. **[R1] Python Software Foundation.** *Python 3 Documentation – The Python Standard Library*. Python Software Foundation. URL: <https://docs.python.org/3/library/>. Tài liệu tham khảo cho các thành phần chuẩn Python, bao gồm `pathlib`, `time` và `typing`. Truy cập ngày 07-11-2026.

2. **[R2] OpenCV.** *Image Filtering – OpenCV 4.11.0 Documentation*. OpenCV. URL: <https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html>. Tài liệu mô tả các API lọc ảnh như `blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`, `filter2D` và `Sobel`; có giải thích border extrapolation, output depth và phép tương quan chéo của `filter2D`. **Lưu ý phiên bản:** tài liệu truy cập được là OpenCV 4.11.0; nhật ký môi trường Lab2 ghi nhận runtime OpenCV 5.0.0. Vì vậy, nguồn này được dùng để tham khảo API và nguyên lý hành vi được mô tả trên trang, không khẳng định là tài liệu khớp phiên bản runtime. Truy cập ngày 07-11-2026.

3. **[R3] NumPy Developers.** *NumPy Documentation*. NumPy. URL: <https://numpy.org/doc/stable/>. Tài liệu tham khảo cho ndarray, kiểu dữ liệu, thao tác mảng và tính toán số. Phiên bản `stable` trực tuyến có thể khác phiên bản runtime trong môi trường Lab2. Truy cập ngày 07-11-2026.

4. **[R4] pytest-dev team.** *pytest Documentation*. pytest. URL: <https://docs.pytest.org/en/stable/>. Tài liệu tham khảo cho fixture, parametrization, assertion và cách chạy kiểm thử. Phiên bản `stable` trực tuyến có thể khác phiên bản runtime trong môi trường Lab2. Truy cập ngày 07-11-2026.

5. **[R5] Matplotlib Development Team.** *matplotlib.pyplot.imshow*. Matplotlib. URL: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>. Tài liệu API tham khảo cho hiển thị ảnh và cách diễn giải dữ liệu ảnh hai chiều/ba chiều. Truy cập ngày 07-11-2026.

## 2. Books and Textbooks

6. **[R6] Rafael C. Gonzalez và Richard E. Woods.** *Digital Image Processing*. Ấn bản thứ 4. Pearson. ISBN-13: **978-0-13-335672-4**. Google Books ghi nhận ấn bản thứ 4, Pearson, 2018; trang sản phẩm Pearson cho cùng ISBN hiện ghi “Published 2017 (2017 update)”. Để tránh nhập nhằng giữa năm bản quyền/năm cập nhật danh mục, thông tin ISBN và số ấn bản được dùng làm định danh chính. URL: <https://www.pearson.com/en-us/subject-catalog/p/Gonzalez-Digital-Image-Processing-Subscription-4th-Edition/P200000003224/9780133356724> và <https://books.google.com/books?id=0F05vgAACAAJ>. Dùng làm tài liệu nền tảng cho xử lý ảnh số, biến đổi điểm ảnh, lọc không gian, lọc phi tuyến, gradient và các phép đo sai số ảnh. Tra cứu ngày 07-11-2026.

## 3. Scientific Papers and Book Chapters

7. **[R7] Carlo Tomasi và Roberto Manduchi.** “Bilateral Filtering for Gray and Color Images.” *Proceedings of the Sixth International Conference on Computer Vision (ICCV)*, 1998, tr. 839–846. DOI: <https://doi.org/10.1109/ICCV.1998.710815>. Metadata về tác giả, tên bài, hội nghị, trang và DOI được đối chiếu với bản ghi IEEE Xplore và các bản ghi thư mục học thuật. Đây là nguồn gốc phù hợp để tham khảo nguyên lý bilateral filtering. Trang IEEE Xplore: <https://ieeexplore.ieee.org/document/710815/>. Truy cập/đối chiếu metadata ngày 07-11-2026.

8. **[R8] J. M. S. Prewitt.** “Object Enhancement and Extraction.” Trong *Picture Processing and Psychopictorics*, biên tập bởi B. S. Lipkin và A. Rosenfeld, Academic Press, 1970, tr. 75–149. Thông tin chương, sách, nhà xuất bản và năm được đối chiếu với danh mục sách của Elsevier và các bản ghi thư mục. Trang nhà xuất bản: <https://shop.elsevier.com/books/picture-processing-and-psychopictorics/lipkin/978-0-12-451550-5>. Bản ghi thư mục bổ sung: <https://cir.nii.ac.jp/crid/1574231873957346816>. Truy cập ngày 07-11-2026.

9. **[R9] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh và Eero P. Simoncelli.** “Image Quality Assessment: From Error Visibility to Structural Similarity.” *IEEE Transactions on Image Processing*, tập 13, số 4, 2004, tr. 600–612. DOI: <https://doi.org/10.1109/TIP.2003.819861>. Metadata được đối chiếu với bản ghi PubMed/NLM. Nguồn này hỗ trợ phần thảo luận về đánh giá chất lượng ảnh dựa trên cấu trúc (SSIM) và giới hạn của việc chỉ dựa vào sai khác pixel. **SSIM không được triển khai trong Lab2**; bài báo chỉ được dẫn để cung cấp bối cảnh học thuật. URL: <https://pubmed.ncbi.nlm.nih.gov/15376593/>. Truy cập ngày 07-11-2026.

## 4. Dataset and Image Sources

| Dữ liệu | Thông tin đã xác nhận | Thông tin còn thiếu |
|---|---|---|
| `data/input/images.jpg` | Tệp JPEG cục bộ trong workspace Lab2; theo nhật ký thực nghiệm, OpenCV giải mã được thành ảnh 3 kênh `uint8`, shape `(678, 452, 3)` theo thứ tự `(height, width, channels)`. | Tác giả/người chụp, nhà cung cấp, URL nguồn ban đầu, tên dataset và giấy phép sử dụng: **Chưa xác định**. Không được gán ảnh này cho một dataset cụ thể nếu chưa có bằng chứng. |

## 5. Bản đồ trích dẫn trong `theory.md`

| Nội dung trong theory.md | Nguồn nên dùng | Phạm vi hỗ trợ |
|---|---|---|
| Biểu diễn ảnh số; point operations; mean/Gaussian/median; kernel và gradient | [R6] | Nền tảng lý thuyết xử lý ảnh số và các phép xử lý ảnh cơ bản. |
| Hành vi API `blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`, `filter2D`, `Sobel`; border và output depth | [R2] | Cách sử dụng và hành vi API OpenCV được mô tả trong tài liệu 4.11.0. |
| Bilateral Filter | [R7] | Nguyên lý kết hợp khoảng cách không gian và tương đồng cường độ. |
| Prewitt Operator | [R8] | Nguồn chương học thuật về tăng cường và trích xuất đối tượng. |
| Đánh giá chất lượng cảm nhận và SSIM | [R9] | Bối cảnh cho metric cấu trúc bổ sung; không phải bằng chứng rằng SSIM có trong source Lab2. |
| Python, NumPy, pytest, Matplotlib | [R1], [R3], [R4], [R5] | Tài liệu công cụ và thư viện được project sử dụng. |

## 6. Giới hạn và ghi chú sử dụng nguồn

- Danh mục này xác minh được metadata thư mục cho [R6]–[R9], nhưng không khẳng định toàn bộ công thức trong `theory.md` được chép trực tiếp từ một trang cụ thể. Các công thức được trình bày lại theo dạng toán học chuẩn và cần đối chiếu nội dung chi tiết của sách/bài báo khi dùng trong báo cáo học thuật chính thức.
- [R2] là tài liệu OpenCV 4.11.0 có nội dung truy cập được; không được ghi nhầm thành tài liệu OpenCV 5.0.0. Nhật ký môi trường Lab2 ghi OpenCV runtime 5.0.0, do đó cần kiểm tra lại API với đúng phiên bản runtime nếu có khác biệt.
- Với [R6], các danh mục thư mục không thống nhất hoàn toàn về năm (Google Books ghi 2018; trang sản phẩm Pearson cho ISBN in ghi bản cập nhật 2017). Vì vậy, cần giữ nguyên ISBN và số ấn bản khi lập danh mục; nếu cơ sở đào tạo yêu cầu một chuẩn trích dẫn cụ thể, hãy theo thông tin trên trang tên sách của bản sách thực tế được sử dụng.
- Nguồn gốc và giấy phép của `data/input/images.jpg` chưa xác định. Người làm project cần bổ sung thông tin từ người cung cấp ảnh hoặc nguồn tải thực tế.
- Tài liệu tham khảo không thay thế việc ghi trích dẫn tại từng đoạn trong `theory.md`. Khi bổ sung nội dung mới, cần gắn `[R#]` ngay sau luận điểm được nguồn hỗ trợ.
