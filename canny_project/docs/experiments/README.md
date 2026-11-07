# Kết quả thực nghiệm

Mọi số liệu dưới đây trích từ `summary.csv`/`summary.md`/`extra.json` trong `results/<tên>/` của lần chạy cuối (2026-11-07). Môi trường: Python 3.14.7, NumPy 2.5.4, OpenCV 5.0.0, SciPy 1.18.1, scikit-image 0.26.0, macOS arm64, 8 CPU (`results/*/environment.json`). Chạy lại:

```bash
for c in canny_project/configs/*.toml; do .venv/bin/python -m cannylab run "$c" --out canny_project/results/$(basename "$c" .toml); done
```

Ký hiệu chung:
- F1 là F1 có dung sai τ = 1.5 px, khớp nhiều-một; "F1 1-1" là khớp một-một.
- Mỗi số là trung bình trên n mẫu (độ lệch chuẩn có trong `summary.md`).
- Ngưỡng của hồ sơ skimage tính trên thang [0, 1].

| Mã | Câu hỏi của đề | Cấu hình | Thư mục kết quả |
|---|---|---|---|
| X1 | P1 | [x1_thu_vien.toml](../../configs/x1_thu_vien.toml) | `results/x1_thu_vien/` |
| X2 | P2, T2a, T2b | [x2_tham_so.toml](../../configs/x2_tham_so.toml) | `results/x2_tham_so/` |
| X3 | P3 | [x3_anh_kho.toml](../../configs/x3_anh_kho.toml) | `results/x3_anh_kho/` |
| X4 | T1b, T3a | [x4_doi_chung.toml](../../configs/x4_doi_chung.toml) | `results/x4_doi_chung/` |
| X5 | P4a, P4b | [x5_phan_doan.toml](../../configs/x5_phan_doan.toml) | `results/x5_phan_doan/` |
| X6 | E2 | [x6_cai_tien.toml](../../configs/x6_cai_tien.toml) | `results/x6_cai_tien/` |
| X7 | E3 | [x7_mau.toml](../../configs/x7_mau.toml) | `results/x7_mau/` |
| X8, X8b | E4 | [x8_video.toml](../../configs/x8_video.toml), [x8b](../../configs/x8b_video_chuyen_dong.toml) | `results/x8_video/`, `results/x8b_video_chuyen_dong/` |
| X9 | T3a (tốc độ) | [x9_hieu_nang.toml](../../configs/x9_hieu_nang.toml) | `results/x9_hieu_nang/` |

---

## X1. Bản tự cài so với thư viện (P1)
6 ảnh thật (`camera, coins, moon, page, astronaut, grass`) × 12 cấu hình: hồ sơ OpenCV L1 (4 cặp ngưỡng), L2 (2), hồ sơ skimage (σ = 0.5, 1, 2, 3), BFS so với label (2).

**Kết quả: n_diff = 0 và IoU = 1 ở cả 12 cấu hình.** Bản tự cài tái tạo **từng pixel** cả `cv2.Canny` lẫn `skimage.feature.canny`. Điều này kiểm chứng mô tả toán ở [../mathematics/05](../mathematics/05_khac_biet_opencv_skimage.md): biên replicate, NMS lượng tử hóa số nguyên với quy tắc điểm hòa bất đối xứng, `>` và `>=`, Gaussian có bleed-over.

## X2. Ảnh hưởng của tham số (P2, T2a, T2b)
6 cảnh hình, 2 mức nhiễu: σₙ = 0.05 (PSNR 26.06 dB) và σₙ = 0.15 (16.92 dB). Cấu hình cơ sở: σ = 1.5, T_l = 0.05, T_h = 0.15.

**σ với ngưỡng cố định** (`line_sigma_nguong_co_dinh_f1.png`):

| σ | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| F1, σₙ = 0.05 | 0.067 | 0.085 | **0.994** | 0.993 | 0.973 | 0.931 | 0.868 |
| F1, σₙ = 0.15 | 0.079 | 0.072 | 0.091 | 0.208 | **0.970** | 0.929 | 0.860 |
| Recall, σₙ = 0.05 | 1.000 | 0.999 | 0.995 | 0.986 | 0.950 | 0.887 | 0.810 |

- σ nhỏ: Precision khoảng 0.03–0.05, tức nhiễu vượt ngưỡng. Khi σ đủ lớn, F1 **nhảy vọt**: σₙ = 0.05 cần σ ≥ 1.5, σₙ = 0.15 cần σ ≥ 3. Mức σ cần thiết tăng theo mức nhiễu, khớp với std nhiễu gradient ∝ σₙ/σ² ([01](../mathematics/01_lam_tron_gaussian.md) (c)).
- σ lớn hơn nữa: Recall giảm đều, từ 0.950 (σ = 3) xuống 0.810 (σ = 5), vì đỉnh gradient ∝ 1/σ tụt dưới ngưỡng cố định ở các hình tương phản thấp, và góc bị bo tròn. Đây là hai mặt của đánh đổi SNR – định vị.

**σ với ngưỡng theo nhiễu** (ngưỡng phổ quát, X6): F1 = 0.991 / 0.956 ngay từ σ = 1, đạt 0.995 / 0.977 ở σ = 1.5–2. Khi ngưỡng tự chạy theo σ, F1 không còn "nhảy bậc", và σ chỉ còn điều khiển đánh đổi định vị.

**Ngưỡng thấp** (σ = 1.5, T_h = 0.15):
- σₙ = 0.05: F1 từ 0.991 đến 0.997, độ phân mảnh gần như không đổi (0.765 đến 0.774), vì cạnh thật đủ mạnh.
- σₙ = 0.15: khi T_l tiến tới T_h, **độ phân mảnh tăng từ 1.80 lên 13.27 thành phần / 100 px**. Hysteresis mất tác dụng nên cạnh bị đứt vụn, đúng vai trò của ngưỡng thấp (T2b).

**Ngưỡng cao** (σ = 1.5, T_l = 0.04): σₙ = 0.05 cho F1 từ 0.121 (T_h = 0.05) lên 0.997 (T_h ≥ 0.18); σₙ = 0.15 cho F1 từ 0.084 lên 0.869 (T_h = 0.5). T_h quyết định có bao nhiêu hạt giống nhiễu.

## X3. Ba loại ảnh khó (P3)
6 cảnh × 6 mức suy giảm. Thư viện dùng tham số mặc định: skimage σ = 1, 0.1/0.2; OpenCV 100/200 không làm mờ.

| Ảnh | PSNR | skimage mặc định | OpenCV 100/200 | Chỉnh tay σ = 2, 0.03/0.08 | Tự động: nhiễu | Tự động: phân vị |
|---|---|---|---|---|---|---|
| nhiễu σₙ = 0.1 | 20.16 | 0.077 | 0.090 | 0.136 | 0.986 | **0.991** |
| nhiễu σₙ = 0.2 | 14.90 | 0.074 | 0.086 | 0.106 | **0.968** | 0.946 |
| tương phản ×0.3 | 21.39 | **0.999** | 0.618 | 0.993 | 0.984 | 0.993 |
| tương phản ×0.12 | 19.43 | 0.161 (R = 0.098) | 0.000 | 0.511 | 0.977 | **0.993** |
| texture 0.08 | 22.01 | 0.096 | 0.136 | 0.118 | 0.115 | **0.789** |
| texture 0.15 | 16.98 | 0.092 | 0.093 | 0.105 | 0.105 | **0.427** |

**Kết luận P3 (từ số liệu):**
1. **Nhiều nhiễu:** tham số mặc định của cả hai thư viện thất bại (F1 < 0.1). Ngưỡng phải tỉ lệ với mức nhiễu: chiến lược theo nhiễu cho 0.986 / 0.968.
2. **Tương phản thấp:** ngưỡng tuyệt đối không bất biến với tương phản, nên OpenCV 100/200 cho 0 pixel cạnh ở mức ×0.12. Ngưỡng **phân vị** bất biến với tương phản (0.993 ở cả hai mức), ngưỡng theo nhiễu cũng tốt (0.977), vì cả hai tự co giãn theo dữ liệu.
3. **Nhiều chi tiết (texture):** đây là ca khó nhất. Texture là *cấu trúc thật* chứ không phải nhiễu trắng, nên chiến lược theo nhiễu (vốn giả thiết nhiễu trắng) không giúp được: F1 = 0.115. Phân vị tốt nhất nhưng giảm mạnh khi texture tăng (0.789 xuống 0.427). Canny thuần không phân biệt được biên vật thể với biên texture; cần đặc trưng texture hoặc học máy, nằm ngoài phạm vi đề.
4. Lab3 kết luận định tính rằng "ảnh nhiễu cần σ lớn, tương phản thấp cần hạ ngưỡng". Số liệu xác nhận điều đó và chỉ ra cách làm tự động: ngưỡng theo nhiễu hoặc phân vị thay cho chỉnh tay từng ảnh.

## X4. Canny so với Sobel-ngưỡng và LoG (T1b, T3a)
5 cảnh × 4 mức nhiễu. Mỗi phương pháp quét 8–9 ngưỡng; bảng lấy **F1 tốt nhất trên lưới** cho từng mức nhiễu (kiểu ODS, lạc quan, xem [../evaluation/README.md](../evaluation/README.md)).

| σₙ | Canny F1 / **F1 1-1** | Sobel-ngưỡng F1 / **F1 1-1** | LoG F1 / **F1 1-1** |
|---|---|---|---|
| 0 | 0.993 / **0.919** | 0.984 / 0.657 | 0.998 / 0.917 |
| 0.05 | 0.993 / **0.916** | 0.981 / 0.665 | 0.958 / 0.883 |
| 0.1 | 0.992 / **0.915** | 0.974 / 0.676 | 0.935 / 0.817 |
| 0.2 | 0.990 / **0.919** | 0.946 / 0.689 | 0.628 / 0.536 |
| Thời gian 256² | khoảng 5.5 ms | khoảng 2 ms | khoảng 2 ms |

**Kết luận T1b / T3a:**
- **Độ chính xác.** Theo F1 nhiều-một, Sobel-ngưỡng trông gần ngang Canny (0.946 so với 0.990 ở σₙ = 0.2). **Theo F1 một-một, Sobel chỉ đạt 0.66–0.69**, vì không có NMS nên cạnh dày 2–3 px và vi phạm tiêu chí đáp ứng đơn. Nếu chỉ dùng một kiểu khớp thì kết luận sẽ sai.
- **Khả năng chống nhiễu.** Canny gần như không suy giảm theo nhiễu (F1 1-1 từ 0.919 xuống 0.919) khi ngưỡng được chỉnh. LoG suy giảm mạnh nhất (0.917 xuống 0.536), vì đạo hàm bậc 2 khuếch đại nhiễu với hệ số √20 so với √12 ([02](../mathematics/02_gradient_va_laplacian.md)).
- **Trên ảnh sạch**, LoG khớp vị trí tốt (F1 0.998) nhưng FOM thấp hơn (0.833 so với 0.956), do điểm qua 0 lệch nửa pixel.
- **Tốc độ.** Canny chậm hơn khoảng 2.7 lần, vì có thêm NMS và hysteresis. Mọi phương pháp đều O(N).
- Ngưỡng tốt nhất của Canny tăng theo nhiễu (0.03, 0.12, 0.27, 0.4). Vì vậy mỗi mức nhiễu cần ngưỡng khác nhau, đúng lý do cần ngưỡng tự động (X6).

## X5. Canny + phân đoạn + nhận dạng hình (P4a, P4b)
10 cảnh (60 vật, 6 loại hình) × 3 mức nhiễu. Pipeline: closing ellipse 5×5, lấp lỗ, bỏ vùng < 80 px, Douglas–Peucker 0.04P.

| Cấu hình | Mức nhiễu | F1 đối tượng | Sai số đếm (TB/ảnh) | IoU TB | Accuracy nhận dạng |
|---|---|---|---|---|---|
| Tự cài, σ = 1.5, T_h = 0.12 | sạch / 0.05 / 0.1 | 1.000 / 0.856 / 0.000 | 0 / +2.1 / −6.0 | 0.925 / 0.919 / – | 1.000 / 0.933 / 0.000 |
| Tự cài, σ = 2.5, T_h = 0.12 | sạch / 0.05 / 0.1 | 1.000 / 0.983 / 0.950 | 0 / 0 / 0 | 0.920 / 0.918 / 0.916 | 0.917 / 0.883 / 0.800 |
| OpenCV như Lab3 (σ = 1.5, 30/90) | sạch / 0.05 / 0.1 | 1.000 / 1.000 / 0.932 | 0 / 0 / +0.5 | 0.924 / 0.920 / 0.911 | 1.000 / 1.000 / 0.917 |

**Kết luận P4:**
- Trên ảnh sạch, phân đoạn tìm đủ 60/60 vật, nhận dạng đúng 100% với σ = 1.5. Kết quả này khớp với Lab3 (24/24 đồng xu, 8/8 hình).
- **σ ảnh hưởng ngược chiều ở hai bước.** σ lớn giúp phân đoạn bền với nhiễu (F1 đối tượng 0.950 so với 0.000 ở σₙ = 0.1) nhưng **làm giảm nhận dạng trên ảnh sạch** (0.917 so với 1.000), vì góc bị bo tròn với bán kính cỡ σ nên đa giác dễ bị nhận nhầm. Đây là một minh chứng cụ thể của đánh đổi định vị ([01](../mathematics/01_lam_tron_gaussian.md) (d)).
- Với σ = 1.5 và σₙ = 0.1, cạnh nhiễu nối thành mạng lưới, closing + lấp lỗ biến cả ảnh thành một vùng và vùng đó bị bỏ (> 50% ảnh). Sai số đếm −6 cho thấy cần ngưỡng theo nhiễu cho bước này.
- IoU ≈ 0.92 bị chặn trên bởi viền: vùng từ cạnh đóng lệch khoảng 1 px so với mặt nạ thật, với vật bán kính khoảng 20 px thì 1 px viền tương đương khoảng 8% diện tích.

## X6. Cải thiện Canny (E2)
6 cảnh × 5 suy giảm, σ = 2.

| Chiến lược | Gauss 0.05 | Gauss 0.15 | Muối tiêu 2% | Muối tiêu 6% | Tương phản ×0.2 |
|---|---|---|---|---|---|
| Cố định 0.04/0.1 | 0.992 | 0.119 | 0.444 | 0.143 | 0.913 |
| Theo nhiễu, k = √(2 ln N) (mặc định) | 0.985 | **0.988** | 0.152 | 0.117 | 0.985 |
| Theo nhiễu, k = 5 / 6 | 0.992 / 0.992 | 0.988 / 0.988 | 0.154 / 0.162 | 0.119 / 0.130 | 0.993 / 0.993 |
| Theo nhiễu, mặc định cũ (k = 3.5, T_l = 0.4T_h) | 0.511 | 0.548 | 0.147 | 0.112 | 0.561 |
| Phân vị 0.88/0.95 | 0.992 | 0.982 | 0.990 | 0.979 | 0.993 |
| Otsu | 0.992 | **0.989** | 0.991 | 0.988 | 0.993 |
| Median ± 33% (heuristic) | 0.992 | 0.990 | 0.991 | 0.929 | **0.000** |
| Tiền lọc median 3×3 + Otsu (σ = 1) | **0.997** | 0.934 | **0.996** | **0.995** | **0.997** |
| Tiền lọc median 5×5 + Otsu | 0.985 | 0.978 | 0.981 | 0.982 | 0.985 |
| Tiền lọc bilateral + Otsu | **0.999** | 0.936 | 0.841 | 0.661 | 0.996 |

**Kết luận E2:**
1. **Ngưỡng theo nhiễu với ngưỡng phổ quát** giải quyết nhiễu Gauss (0.119 lên 0.988 ở σₙ = 0.15) nhờ có cơ sở toán học. Mặc định cũ k = 3.5 chỉ đạt khoảng 0.5. Nguyên nhân đã được phân tích bằng số: khoảng 0.2%·N hạt giống giả (so sánh bội), và **không phải** do hiện tượng thấm; giả thuyết thấm đã bị đo và bác bỏ ([../mathematics/04](../mathematics/04_nguong_kep_va_hysteresis.md)).
2. **Nó thất bại với muối tiêu** (0.15), vì nhiễu xung không phải Gauss, nên cả bộ ước lượng lẫn phân phối Rayleigh đều sai giả thiết. Đây là giới hạn của mô hình, không phải lỗi cài đặt.
3. **Otsu và phân vị bền nhất trên mọi loại suy giảm** (≥ 0.979), dù không có mô hình nhiễu.
4. **Heuristic "median ± 33%" thất bại hoàn toàn trên ảnh tương phản thấp** (F1 = 0), vì nó tính ngưỡng *gradient* từ *cường độ*: hai đại lượng khác thứ nguyên. Kết quả này xác nhận nhận định ở Phase 2 rằng heuristic này không có cơ sở.
5. **Tiền lọc median 3×3** tốt nhất với muối tiêu (0.996 / 0.995), đúng lý thuyết: trung vị loại bỏ giá trị cực trị. **Bilateral** giữ biên tốt nhất với nhiễu Gauss nhẹ (0.999) nhưng kém với muối tiêu (0.661 ở 6%), vì điểm xung có màu khác hẳn nên trọng số w_r giữ nguyên chúng.
6. Lần chạy đầu ghép tiền lọc với chiến lược theo nhiễu và cho F1 ≈ 0.08. Đó là **lỗi thiết kế thực nghiệm** (sau median, nhiễu không còn là nhiễu trắng Gauss), đã sửa sang Otsu và ghi lại ở [phase9](../development/phase9_testing_optimization.md).

## X7. Ảnh màu đẳng sáng (E3)
6 cảnh (hình xanh lá trên nền cam đất, cùng Y), không nhiễu và σₙ = 0.03. Mỗi phương pháp quét ngưỡng; bảng lấy F1 tốt nhất.

| Phương pháp | Không nhiễu | σₙ = 0.03 | T_h tốt nhất khi có nhiễu |
|---|---|---|---|
| gray (chuyển xám) | **0.000** | 0.090 | – |
| OR từng kênh | 0.999 | 0.999 | 0.40 |
| max_channel | 0.999 | 0.999 | 0.18 |
| Di Zenzo | 0.999 | 0.999 | 0.27 |
| cv2.Canny 3 kênh | 1.000 | 1.000 | 300 (thang 0..255) |

**Kết luận E3:**
- Chuyển xám **mất hoàn toàn** cạnh đẳng sáng (F1 = 0). Đây là câu trả lời trực tiếp cho câu hỏi "Canny có dùng được cho ảnh màu không": có, nhưng **không được** chuyển xám trước khi phát hiện cạnh.
- Ba cách mở rộng sang màu đạt F1 ngang nhau **sau khi chọn ngưỡng đúng thang**. Trong lần chạy đầu dùng chung ngưỡng 0.05/0.15, Di Zenzo chỉ đạt 0.227 vì độ lớn của nó cộng năng lượng nhiễu của 3 kênh (√3). Bài học: ngưỡng phải được chọn theo thang của từng toán tử. Trên dữ liệu đơn giản này, cách gộp kênh không tạo khác biệt về độ chính xác.

## X8, X8b. Video (E4)
2 video × 16 khung, 128², nhiễu độc lập σₙ = 0.08 mỗi khung. X8: cảnh tĩnh. X8b: vật tịnh tiến 2 px/khung.

| Cấu hình | X8 F1 | X8 ổn định (F1 giữa khung liên tiếp) | X8b F1 | X8b Recall |
|---|---|---|---|---|
| Từng khung, ngưỡng cố định 0.04/0.1 | 0.183 | 0.781 | 0.135 | 0.991 |
| Từng khung, ngưỡng theo nhiễu | 0.982 | 0.972 | **0.975** | 0.991 |
| Làm trơn thời gian σ_t = 1, ngưỡng cố định | 0.823 | 0.787 | 0.768 | 0.987 |
| Làm trơn thời gian σ_t = 2, ngưỡng cố định | 0.932 | 0.934 | 0.856 | 0.961 |
| Làm trơn thời gian σ_t = 2 + ngưỡng theo nhiễu | **0.989** | **0.990** | 0.923 | 0.961 |

**Kết luận E4:**
- **Cảnh tĩnh:** làm trơn theo thời gian giảm nhiễu mà không mất vị trí. Kết hợp với ngưỡng theo nhiễu cho kết quả tốt nhất (F1 0.989, độ ổn định 0.990).
- **Cảnh chuyển động:** làm trơn theo thời gian gây **nhòe chuyển động**: Recall giảm từ 0.991 xuống 0.961 và F1 thấp hơn xử lý từng khung (0.923 so với 0.975). Lựa chọn phụ thuộc tốc độ chuyển động so với σ_t.
- Ngưỡng cố định cho F1 thấp (0.183) vì σₙ = 0.08 vượt mức mà σ = 1.5 chịu được (giống X2). Nhưng độ ổn định 0.781 vẫn tương đối cao, vì nhiễu "ổn định" về số lượng chứ không về vị trí. Độ ổn định chỉ có ý nghĩa khi đọc cùng F1.
- Độ ổn định của X8b không có ý nghĩa, vì cạnh dịch 2 px mỗi khung, lớn hơn τ.
- Std của F1 trong các cấu hình làm trơn thời gian lớn (0.16–0.22), vì khung đầu và cuối chuỗi được làm trơn ít hơn (biên `nearest`).

## X9. Hiệu năng (T3a, Manual so với Optimized)
1 cảnh, 128² đến 1024², OpenCV chạy tuần tự (`opencv_threads = 1` trong environment.json), trung vị 5 lần.

| Ảnh 1024² | Thời gian | Hệ số mũ k (t ∝ Nᵏ) | Đỉnh tracemalloc |
|---|---|---|---|
| Tự cài, hồ sơ skimage, BFS | 0.237 s | 1.08 | 131 MB |
| Tự cài, hồ sơ skimage, label | 0.141 s | 1.06 | 131 MB |
| `skimage.feature.canny` | 0.073 s | 1.08 | 48 MB |
| Tự cài, hồ sơ OpenCV, BFS | 0.545 s | 0.96 | 85 MB |
| `cv2.Canny` | **0.025 s** | 1.04 | 16 MB (chỉ mảng kết quả) |

**Đo từng bước** (1024², nhiễu 0.05):

| Hồ sơ | Chuẩn bị + Gaussian | Sobel | Độ lớn | NMS | Hysteresis BFS | Hysteresis label | % ứng viên / % mạnh |
|---|---|---|---|---|---|---|---|
| OpenCV (40/100) | 5 ms | 19 ms | 2 ms | 13 ms | **479 ms** | 15 ms | 35.9% / 16.3% |
| skimage (σ = 1.5) | 71 ms | 18 ms | 4 ms | 39 ms | 92 ms | 8 ms | 22.5% / 0.45% |

**Kết luận:**
- **Mọi cài đặt đều tuyến tính** (k = 0.96–1.08), khớp phân tích O(N).
- **Điểm nghẽn của bản Manual là BFS bằng Python.** Chi phí của BFS tỉ lệ với số pixel được lan tới chứ không phải N. Hồ sơ OpenCV không làm mờ nên 16% pixel là điểm mạnh, BFS mất 479 ms. Các bước vector hóa (Gaussian, Sobel, NMS) chỉ tốn khoảng 40–130 ms. Thay BFS bằng `ndi.label` giảm hysteresis 30 lần, đưa bản tự cài về mức 2× skimage.
- OpenCV nhanh hơn bản Manual tốt nhất khoảng 5.6 lần (SIMD, không có mảng trung gian). Bộ nhớ của bản Manual cao (131 MB với ảnh 8 MB) vì NMS vector hóa giữ khoảng 10 mảng tạm cỡ N; đây là đánh đổi để dễ đọc.
