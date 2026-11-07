# Computer Vision: 5 bài lab

Mỗi lab theo cùng một khuôn và chạy độc lập được. Mỗi lab tương ứng một chương trong report, và hình của chương lấy từ thư mục `ket_qua/` của lab đó. Thiết kế cấu trúc: [docs/thiet_ke_cau_truc.md](docs/thiet_ke_cau_truc.md).

| Lab | Chủ đề | Bài nộp | Script / app | Hình chính (`ket_qua/`) |
|---|---|---|---|---|
| [Lab1](Lab1/README.md) | Đọc/lưu ảnh, không gian màu, cắt/resize, vẽ | `Lab1.ipynb` | `Lab1.py` | `02_grayscale.png`, `03_HSV_kenh_*.png`, `06_resize_*.png`, `07_ve_hinh_opencv.png` |
| [Lab2](Lab2/README.md) | Toán tử điểm ảnh, lọc không gian, Sobel/Prewitt, Median/Bilateral | `Lab2.ipynb` | `Lab2.py` | `bai1_sang_tuong_phan.png`, `bai2_loc_gaussian.png`, `bai3_so_sanh_loc_salt.png` |
| [Lab3](Lab3/README.md) | Canny edge detector, phân đoạn, nhận dạng hình | `Lab3.ipynb` | `bai_thuc_hanh_canny.py` | `bai1_opencv_vs_skimage/`, `bai2_sigma/`, `bai3_nhieu/`, `bai4_phan_doan/` |
| [Lab4](Lab4/README.md) | Wavelet Hash so sánh ảnh | `Lab4.ipynb` | `Lab4.py`, `app_tim_kiem.py`, `tao_du_lieu.py` | `bai1_wavelet_hash.png`, `bai2_so_sanh_cap_anh.png`, `bai3_roc_db4.png` |
| [Lab5](Lab5/README.md) | Nhận diện khuôn mặt MTCNN + FaceNet | `Lab5.ipynb` | `Lab5.py` (webcam), `register_face.py` | `bai1_mtcnn_phat_hien.png`, `bai3_nhan_dien_anh_tinh.png` |

**Dự án mở rộng:** [canny_project/](canny_project/README.md) phát triển Lab3 thành dự án nghiên cứu Canny: tự cài đặt bằng NumPy (trùng từng pixel với OpenCV / scikit-image), dữ liệu có ground truth, 10 thực nghiệm, 223 test. Không ảnh hưởng đến thư mục `Lab3/`.

## Cấu trúc

```
CV_Labs/
├── README.md            # file này
├── requirements.txt     # thư viện Lab1–4; Lab5 cài thêm theo Lab5/README.md
├── docs/                # tài liệu thiết kế
├── tai_lieu/            # slide của giảng viên
└── LabN/
    ├── README.md        # Mục tiêu · Lý thuyết · Cách làm · Kết quả · Nhận xét
    ├── LabN.ipynb       # bài nộp chính, Run All ra toàn bộ hình
    ├── *.py             # script / app chạy độc lập
    ├── du_lieu/         # ảnh đầu vào
    ├── ket_qua/         # ảnh sinh ra
    └── ghi_chu/         # file làm việc (agent rule, cheatsheet), không phải bài nộp
```

## Cài đặt

Một `.venv` dùng chung ở thư mục gốc (đã thử với Python 3.14):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Lab5 cần thêm:
.venv/bin/pip install torch torchvision pytest tqdm requests
.venv/bin/pip install --no-deps facenet-pytorch
```

## Chạy lại toàn bộ

```bash
for L in Lab1 Lab2 Lab3 Lab4 Lab5; do
  (cd $L && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace $L.ipynb)
done
(cd Lab5 && ../.venv/bin/python -m pytest test_lab5.py -v)
```

Trong VS Code / Jupyter, chọn kernel `.venv` ở `CV_Labs/` và mở notebook từ đúng thư mục lab, vì đường dẫn `du_lieu/` và `ket_qua/` là đường dẫn tương đối.
