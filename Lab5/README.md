# Lab 5 · Nhận diện khuôn mặt theo thời gian thực

- **Môn học** — Thị giác máy tính (Computer Vision)
- **Đơn vị** — Viện Công nghệ Thông tin, Điện, Điện tử – Trường Đại học Giao thông Vận tải TP.HCM
- **Giảng viên hướng dẫn** — TS. Nguyễn Thị Khánh Tiên (`tienntk@ut.edu.vn`)

MTCNN phát hiện khuôn mặt từ webcam. FaceNet tạo embedding 512 chiều để so sánh với ảnh đã đăng ký. Với ngưỡng mặc định 0.7, điểm cosine lớn hơn 0.7 cho nhãn `Matched` và khung xanh. Điểm bằng hoặc thấp hơn ngưỡng cho nhãn `Unknown` và khung đỏ.

## Cài đặt trên Windows

Mở PowerShell trong thư mục `Lab5` và dùng Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import cv2; from facenet_pytorch import MTCNN, InceptionResnetV1; print('OK')"
```

Khi khởi tạo FaceNet lần đầu, chương trình tải trọng số VGGFace2 nếu bộ nhớ đệm chưa có tệp.

## Đăng ký và nhận diện

Chụp ba ảnh cho cùng một ID. Nhấn `Space` hoặc `c` một lần cho mỗi góc chụp. Có thể bắt đầu với chính diện, nghiêng nhẹ trái và nghiêng nhẹ phải. Nhấn `q` hoặc `Esc` để dừng. Các ảnh đã chụp thành công vẫn được lưu.

```powershell
.\.venv\Scripts\python.exe register_face.py --id "Lan" --select-camera --samples 3
```

Dùng nhiều `--image` để đăng ký các ảnh có sẵn cho cùng ID.

```powershell
.\.venv\Scripts\python.exe register_face.py --id "Lan" --image .\chinh-dien.jpg --image .\nghieng-trai.jpg --image .\nghieng-phai.jpg
```

`--id` và `--name` dùng chung một tham số. Chạy lại lệnh với cùng ID để bổ sung ảnh. Mỗi ảnh hợp lệ có embedding 512 chiều riêng, và chương trình lấy điểm cosine cao nhất trong tất cả mẫu để chọn ID. Ưu tiên ảnh rõ nét của đúng một người, ở những góc và điều kiện ánh sáng thường gặp khi dùng webcam.

Mở webcam để nhận diện.

```powershell
.\.venv\Scripts\python.exe Lab5.py
```

| Tùy chọn của `Lab5.py` | Mặc định | Công dụng |
|---|---|---|
| `--camera N` | `0` | Mở webcam có chỉ số `N` |
| `--list-cameras` | | Kiểm tra camera từ `0` đến `4` và in các chỉ số khả dụng |
| `--select-camera` | | Dò camera và chọn chỉ số trong PowerShell |
| `--db-dir` | `known_faces/` ở gốc Lab 5 | Chọn thư mục ảnh mẫu |
| `--threshold` | `0.7` | Đặt ngưỡng cosine |
| `--frame-skip` | `2` | Đặt khoảng cách giữa hai khung hình chạy suy luận |
| `--device` | Tự chọn CUDA nếu khả dụng, nếu không dùng CPU | Chỉ định `cpu` hoặc `cuda` |
| `--config` | Không nạp cấu hình | Nạp bảng `[recognition]` từ tệp TOML |

### Cơ sở ảnh và embedding

Ảnh đăng ký mới được lưu trong `known_faces/<ID>/`. Vector lưu trong `known_faces/embeddings.sqlite3` và có thể dùng lại sau khi thoát ứng dụng. Khi dùng `--db-dir`, cả ảnh và tệp SQLite nằm trong thư mục đó.

Lab 5 vẫn đọc ảnh đặt trực tiếp trong `known_faces/`. Tệp `Lan.jpg` và `Lan_2.jpg` cùng thuộc ID `Lan` theo quy tắc cũ. Có thể gom các ảnh cũ của cùng người bằng cách chuyển chúng vào một thư mục ID, ví dụ `known_faces/Lan/`. Tên thư mục quyết định ID của các ảnh bên trong. Chương trình đồng bộ các ảnh thêm, sửa hoặc xóa khi khởi động và khi nhấn `r`. Các ảnh giữ nguyên được dùng lại embedding đã lưu.

### Chọn camera

Dùng `--select-camera` để xem danh sách camera khả dụng và nhập chỉ số muốn mở. Nhấn Enter nếu muốn hủy. Nếu đã biết chỉ số, truyền trực tiếp qua `--camera`, kể cả chỉ số ngoài phạm vi dò.

```powershell
.\.venv\Scripts\python.exe Lab5.py --select-camera
.\.venv\Scripts\python.exe Lab5.py --camera 1
.\.venv\Scripts\python.exe register_face.py --name "NguyenVanA" --select-camera
```

Dùng `--list-cameras` với một trong hai lệnh để chỉ xem danh sách. `register_face.py --list-cameras` có thể chạy riêng. Dùng cùng `--db-dir` cho hai lệnh nếu lưu ảnh ngoài thư mục mặc định.

Có thể chạy nhận diện qua mô-đun và chọn cấu hình mẫu.

```powershell
.\.venv\Scripts\python.exe -m src.facelab.cli --config configs/default.toml --camera 0
```

Camera chọn trong PowerShell hoặc qua `--camera` ưu tiên hơn `camera` trong bảng `[recognition]`. Các tùy chọn khác trên dòng lệnh cũng ưu tiên hơn cấu hình TOML. Với giá trị còn thiếu, chương trình dùng mặc định trong bảng trên. `db_dir` tương đối trong TOML được tính từ thư mục của tệp cấu hình. Đường dẫn truyền trực tiếp qua `--db-dir` được tính từ thư mục hiện hành.

| Phím trong cửa sổ nhận diện | Thao tác |
|---|---|
| `s` hoặc `c` | Chụp nhanh và nhập ID trong PowerShell để thêm một mẫu |
| `r` | Đồng bộ ảnh mẫu và embedding với tệp trên đĩa |
| `l` | Bật hoặc tắt điểm mốc khuôn mặt |
| `f` | Bật hoặc tắt chế độ bỏ qua khung hình |
| `q` hoặc `Esc` | Thoát và giải phóng webcam |

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Bộ kiểm thử bao gồm mô hình, embedding, đối sánh cosine, khung hình trống, webcam và cấu hình TOML. Ca webcam thử mở thiết bị và chỉ kiểm tra nội dung ảnh khi đọc được khung hình.

## Bản đồ dự án

| Tệp hoặc thư mục | Nội dung |
|---|---|
| [Lab5.py](Lab5.py) và [register_face.py](register_face.py) | Các lệnh chạy nhận diện và đăng ký |
| [src/facelab/](src/facelab/) | Mã nhận diện, webcam, giao diện, đăng ký và CLI |
| [configs/default.toml](configs/default.toml) | Cấu hình nhận diện mẫu |
| [tests/](tests/) | Kiểm thử bằng pytest |
| [requirements.txt](requirements.txt) | Thư viện để chạy và kiểm thử |
| [known_faces/](known_faces/) | Ảnh mẫu theo ID và tệp SQLite được tạo khi chạy |
| [docs/](docs/README.md) | Kiến trúc, cheatsheet và hướng dẫn vibe coding |
| [rules/](rules/README.md) | Quy ước khi sửa các phần của Lab 5 |
| [results/](results/README.md) | Kết quả từng ghi nhận và mẫu kiểm tra thủ công |
| [AGENTS.md](AGENTS.md) và [llms.txt](llms.txt) | Quy tắc làm việc và mục lục cho trợ lý |
