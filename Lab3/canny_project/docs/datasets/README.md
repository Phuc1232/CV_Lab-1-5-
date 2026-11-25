# Quản lý dữ liệu

Mã: [`src/cannylab/datasets/`](../../src/cannylab/datasets/), [`src/cannylab/imaging/`](../../src/cannylab/imaging/). Quy tắc: [rules/image-data-rules.md](../../rules/image-data-rules.md).

## 1. Ba nguồn dữ liệu
| Nguồn | Hàm | Ground truth | Dùng cho |
|---|---|---|---|
| Tổng hợp | `Dataset.synthetic(family, count, size, seed, degradations)` | Chính xác (từ ảnh nhãn) | Đánh giá định lượng (X2–X8) |
| `skimage.data` | `Dataset.from_skimage(names)` | Không | Kiểm chứng với thư viện (X1), tái tạo Lab3 |
| Thư mục | `Dataset.from_folder(path)` | Tùy chọn (`<tên>.edges.npy`) | Ảnh của người dùng |

Không tải dữ liệu từ Internet (quyết định ở Phase 1), nên mọi thực nghiệm chạy offline và tái lập được.

## 2. Ảnh tổng hợp
Mỗi ảnh được sinh từ **ảnh nhãn** vẽ bằng `cv2.circle` / `cv2.fillPoly` (`LINE_8`, không khử răng cưa), nên biên vùng được biết chính xác.

| Họ | Mô tả | Tham số |
|---|---|---|
| `shapes` | 6 loại hình (tròn, tam giác, chữ nhật 1.6:1, ngũ giác, lục giác, sao 5 cánh tỉ lệ 0.45), không chạm nhau (cách ≥ 6 px), xoay ngẫu nhiên; nền 0.15–0.35, tương phản ≥ 0.25 | `n_shapes`, `radius_range`, `min_contrast` |
| `step` | cạnh bậc thang thẳng qua tâm, 7 góc 0–90° | `angles` |
| `isoluminant` | ảnh RGB, hình (55,170,70) trên nền (200,100,50): cùng độ sáng Y ≈ 124.2/255 | không |
| `video` | cảnh `shapes` + nhiễu độc lập mỗi khung; `speed` px/khung tịnh tiến | `n_frames`, `noise_sigma`, `speed` |

**Quy ước ground truth cạnh.** Pixel $p$ là cạnh nếu nhãn của $p$ khác nhãn của lân cận phải hoặc dưới. Kết quả là đường 1 px nằm phía trên/trái của biên, trùng với lựa chọn của `cv2.Canny` khi gặp điểm hòa. Biên thật nằm *giữa* hai pixel, nên mọi chỉ số đều dùng dung sai τ = 1.5 px.

**Suy giảm** (`synthetic.degrade`), áp theo thứ tự vật lý: mờ quang học → giảm tương phản (quanh giá trị trung bình) → texture (nhiễu trắng lọc Gauss, độ hạt 1.5 px) → nhiễu Gauss → muối tiêu. Suy giảm **không đổi** ground truth. Riêng mờ quang học làm biên "thật" trở nên mơ hồ; ground truth vẫn giữ ở biên vùng gốc.

**Seed.** `np.random.default_rng(seed)` cho cảnh và `default_rng([seed, i, k])` cho suy giảm của từng mẫu, nên cùng cấu hình luôn cho cùng dữ liệu (`test_synthetic_is_deterministic`).

## 3. Kiểm soát dữ liệu
| Kiểm soát | Cách làm |
|---|---|
| Kích thước | ≥ 3×3 (Sobel); `shapes` cần ≥ 24 px và tự giới hạn bán kính theo kích thước |
| Kênh màu | 2D hoặc (H, W, 3) **RGB**; OpenCV đọc BGR được đổi ngay trong `load_image` |
| Không gian màu | Xám: Y = 0.299R + 0.587G + 0.114B (lệch ≤ 1 mức so với cv2) |
| Kiểu dữ liệu | uint8, uint16, float32, float64; int64, bool, complex bị từ chối |
| Miền giá trị | float phải trong [0, 1] (±1e-6); `to_uint8` làm tròn + clip, không quấn số |
| Ảnh hỏng | `InvalidImageError`; `from_folder` ghi vào `skipped`, không dừng cả lô |

## 4. Metadata, thống kê, chia tập, lưu trữ
- `Sample.describe()`: kích thước, số kênh, dtype, min/max/mean/std, có ground truth không, số vật, cùng meta (họ, seed, suy giảm).
- `Dataset.stats()`: số mẫu, số bị bỏ, các kích thước, kênh, dtype, độ sáng và độ lệch chuẩn trung bình, số vật.
- `Dataset.split(test_fraction, seed)`: chia ngẫu nhiên có seed (dùng để chọn tham số trên tập riêng).
- `Dataset.batches(n)`: xử lý theo lô.
- `Dataset.save(dir)` / `Dataset.load(dir)`: PNG uint8 + `*.gt.npz` (cạnh, nhãn) + `metadata.json` (chuyển định dạng). Lệnh: `python -m cannylab dataset OUT --family shapes --count 8 --degradations '[{"noise_sigma":0.05}]'`.

## 5. Giới hạn
- Hình học đơn giản, nền phẳng; texture tổng hợp là nhiễu có tương quan, không phải vật liệu thật.
- Không có ground truth cho ảnh thật, nên đánh giá ảnh thật chỉ ở mức kiểm chứng với thư viện và định tính.
