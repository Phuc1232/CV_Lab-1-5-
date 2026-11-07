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

Đăng ký bằng webcam. Nhấn `Space` hoặc `c` để chụp, `q` hoặc `Esc` để hủy.

```powershell
.\.venv\Scripts\python.exe register_face.py --name "NguyenVanA"
```

Đăng ký từ ảnh có sẵn.

```powershell
.\.venv\Scripts\python.exe register_face.py --name "TranThiB" --image "C:\duong_dan\anh_chan_dung.jpg"
```

Mở webcam để nhận diện.

```powershell
.\.venv\Scripts\python.exe Lab5.py
```

| Tùy chọn của `Lab5.py` | Mặc định | Công dụng |
|---|---|---|
| `--camera` | `0` | Chọn webcam |
| `--db-dir` | `known_faces/` ở gốc Lab 5 | Chọn thư mục ảnh mẫu |
| `--threshold` | `0.7` | Đặt ngưỡng cosine |
| `--frame-skip` | `2` | Đặt khoảng cách giữa hai khung hình chạy suy luận |
| `--device` | Tự chọn CUDA nếu khả dụng, nếu không dùng CPU | Chỉ định `cpu` hoặc `cuda` |
| `--config` | Không nạp cấu hình | Nạp bảng `[recognition]` từ tệp TOML |

`register_face.py` cũng nhận `--camera` và `--db-dir`. Dùng cùng `--db-dir` cho hai lệnh nếu lưu ảnh ngoài thư mục mặc định.

Có thể chạy nhận diện qua mô-đun và chọn cấu hình mẫu.

```powershell
.\.venv\Scripts\python.exe -m src.facelab.cli --config configs/default.toml --camera 0
```

Giá trị truyền trên dòng lệnh ưu tiên hơn cấu hình TOML. Với các tùy chọn còn lại, chương trình dùng giá trị mặc định trong bảng. `db_dir` tương đối trong TOML được tính từ thư mục của tệp cấu hình. Đường dẫn truyền trực tiếp qua `--db-dir` được tính từ thư mục hiện hành.

| Phím trong cửa sổ nhận diện | Thao tác |
|---|---|
| `s` hoặc `c` | Chụp nhanh và nhập tên trong PowerShell |
| `r` | Nạp lại thư mục ảnh mẫu |
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
| [known_faces/](known_faces/) | Ảnh khuôn mặt mẫu |
| [docs/](docs/README.md) | Kiến trúc, cheatsheet và hướng dẫn vibe coding |
| [rules/](rules/README.md) | Quy ước khi sửa các phần của Lab 5 |
| [results/](results/README.md) | Kết quả từng ghi nhận và mẫu kiểm tra thủ công |
| [AGENTS.md](AGENTS.md) và [llms.txt](llms.txt) | Quy tắc làm việc và mục lục cho trợ lý |
