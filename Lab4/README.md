# Lab 4: So sánh sự tương đồng của ảnh bằng Wavelet Hash

Bài nộp chính: [Lab4.ipynb](Lab4.ipynb). Thuật toán nằm trong [Lab4.py](Lab4.py) và được notebook import vào. Ảnh vào: `du_lieu/` (8 ảnh). Ảnh ra: `ket_qua/`.

Tài liệu: slide bài thực hành chương 3, trang 16–18.

## 1. Mục tiêu

- Tạo mã băm nhị phân cho ảnh bằng biến đổi Wavelet (Wavelet Hash).
- So sánh hai ảnh bằng khoảng cách Hamming và luật tác tử với ngưỡng 150.
- Đánh giá trên 12 cặp test (accuracy, độ nhạy, độ đặc hiệu, ma trận nhầm lẫn, ROC/AUC) và so sánh 4 họ wavelet.
- Xây dựng ứng dụng tìm kiếm ảnh giống nhất (Top 3).

## 2. Lý thuyết

1. **Tiền xử lý:** đọc ảnh xám, resize về 256×256 để mã băm luôn có cùng độ dài.
2. **Biến đổi Wavelet** `pywt.wavedec2(img, 'db4', level=3)`: phân rã ảnh thành các dải tần. Chỉ giữ dải xấp xỉ **LL** (tần thấp), vì dải này mang cấu trúc tổng thể của ảnh và ít bị ảnh hưởng bởi nhiễu chi tiết.
3. **Lượng tử hóa:** hệ số LL > **trung vị** thì bit 1, ngược lại bit 0. Trung vị ít bị ảnh hưởng bởi giá trị ngoại lai (vết vẽ bậy trong ảnh `_challenge`) hơn trung bình.
4. **Khoảng cách Hamming:** `Σ (hash1 ≠ hash2)` là số bit khác nhau.
5. **Luật tác tử:** khoảng cách ≤ 150 thì **1 (Tương đồng)**, ngược lại **0 (Khác biệt)**.

## 3. Cách làm

| Hàm (`Lab4.py`) | Việc làm |
|---|---|
| `bam_wavelet(duong_dan, wavelet, level)` | Tạo mã băm (1444 bit với db4, level 3) |
| `tinh_khoang_cach_hamming(h1, h2)` | Đếm số bit khác nhau |
| `agent_rule(distance, threshold)` | Quyết định 1/0 theo ngưỡng |
| `khao_sat_cac_wavelet(data_test, nguong)` | Bảng accuracy / độ nhạy / độ đặc hiệu cho haar, db4, sym4, coif1 |
| `danh_sach_cap_test(thu_muc)` | 12 cặp test: 6 cặp giống (mờ, vẽ bậy, xoay 90°) và 6 cặp khác |

Các file khác: `tao_du_lieu.py` sinh ảnh xoay 90° và ảnh làm mờ từ `anh1.jpg`, `anh2.jpg`; `app_tim_kiem.py` tìm Top 3 ảnh giống nhất qua dòng lệnh.

## 4. Kết quả

| Hình (`ket_qua/`) | Nội dung |
|---|---|
| `bai1_wavelet_hash.png` | Ảnh xám → dải LL 38×38 → mã băm |
| `bai2_so_sanh_cap_anh.png` | 12 cặp ảnh kèm khoảng cách Hamming và kết luận |
| `bai3_ma_tran_nham_lan.png` | Ma trận nhầm lẫn (db4, ngưỡng 150) |
| `bai3_roc_db4.png` | Đường ROC |
| `bai4_tim_kiem_top3.png` | Truy vấn `anh1.jpg` và Top 3 kết quả |

Khoảng cách Hamming của 12 cặp (db4):

| Cặp | Nhãn | Hamming | Dự đoán |
|---|---|---|---|
| anh1 – anh1_challenge / anh2 – anh2_challenge | 1 | 44 / 68 | 1 ✓ |
| anh1 – anh1_mo / anh2 – anh2_mo | 1 | 24 / 20 | 1 ✓ |
| anh1 – anh1_xoay90 / anh2 – anh2_xoay90 | 1 | 712 / 702 | 0 ✗ |
| 6 cặp khác người khác cảnh | 0 | 484 – 580 | 0 ✓ |

Accuracy 83.33%, độ nhạy 66.67%, độ đặc hiệu 100%, **AUC = 0.67**.

## 5. Nhận xét

- Wavelet Hash chịu tốt việc **làm mờ** (Hamming 20–24) và **vẽ bậy** (44–68), vì dải LL giữ cấu trúc tổng thể và trung vị ít bị ảnh hưởng bởi giá trị ngoại lai.
- **Không chịu được phép xoay 90°:** Hamming khoảng 700, còn lớn hơn cả các cặp ảnh khác nhau (484–580). Mã băm phụ thuộc vị trí không gian của từng hệ số LL nên xoay ảnh là sai toàn bộ. Hai lỗi duy nhất trong bảng đến từ đây, và đó cũng là lý do AUC chỉ đạt 0.67.
- Cả 4 họ wavelet (haar, db4, sym4, coif1) cho **cùng một kết quả** trên 12 cặp này. Dữ liệu quá ít để nói họ nào tốt hơn; README trước đây ghi "db4 tốt nhất" nhưng kết quả chạy không cho thấy điều đó.
- Ngưỡng 150 tách rõ cặp giống (≤ 68) với cặp khác (≥ 484) khi ảnh không bị xoay.

## Chạy lại

```bash
cd Lab4 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Lab4.ipynb
../.venv/bin/python Lab4.py           # in bảng so sánh + lưu ket_qua/bai3_roc_db4.png
../.venv/bin/python app_tim_kiem.py   # nhập tên ảnh, ví dụ anh1.jpg
../.venv/bin/python tao_du_lieu.py    # sinh lại ảnh _xoay90 và _mo
```
