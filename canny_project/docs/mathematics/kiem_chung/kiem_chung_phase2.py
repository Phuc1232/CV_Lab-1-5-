"""Kiểm chứng bằng số các công thức trong docs/mathematics (Phase 2).

Chạy (ở thư mục CV_Labs/):
    .venv/bin/python canny_project/docs/mathematics/kiem_chung/kiem_chung_phase2.py

Kết quả in ra màn hình và ghi vào ket_qua_phase2.json cạnh file này.
Script chỉ phục vụ tài liệu Phase 2; code thuật toán chính thức sẽ nằm trong gói ở Phase 6.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import cv2
import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import data, feature

OUT = Path(__file__).with_name("ket_qua_phase2.json")
RNG = np.random.default_rng(0)
results: dict = {
    "moi_truong": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
    }
}


def section(name: str) -> dict:
    print(f"\n=== {name} ===")
    results[name] = {}
    return results[name]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# ---------------------------------------------------------------- 1. Gaussian
r = section("1_gaussian")

# 1a. Kích thước kernel OpenCV tự chọn khi ksize=(0,0).
#     Quy tắc (mã nguồn OpenCV): ksize = round(sigma * (3 nếu uint8, 4 nếu float) * 2 + 1) | 1.
#     Kiểm tra: kernel dự đoán phải cho kết quả TRÙNG với ksize=(0,0). Với float32, kernel nhỏ hơn
#     dự đoán phải KHÁC (chứng tỏ dự đoán đúng là kích thước thật). Với uint8, các hệ số ngoài
#     cùng có thể bị làm tròn về 0 trong số học điểm cố định, nên kernel nhỏ hơn đôi khi vẫn trùng.
def ksize_check(sigma: float, dtype, factor: int) -> dict:
    img = (RNG.random((101, 101)) * (255 if dtype == np.uint8 else 1.0)).astype(dtype)
    ref = cv2.GaussianBlur(img, (0, 0), sigma)
    k = int(round(sigma * factor * 2 + 1)) | 1
    same = bool(np.array_equal(cv2.GaussianBlur(img, (k, k), sigma), ref))
    smaller_same = bool(k > 1 and np.array_equal(cv2.GaussianBlur(img, (k - 2, k - 2), sigma), ref))
    return {"ksize_du_doan": k, "trung_voi_auto": same, "ksize_nho_hon_van_trung": smaller_same}

r["opencv_ksize_tu_dong"] = {}
for s in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    c8, c32 = ksize_check(s, np.uint8, 3), ksize_check(s, np.float32, 4)
    r["opencv_ksize_tu_dong"][str(s)] = {"uint8": c8, "float32": c32}
    print(f"sigma={s}: uint8 {c8}  float32 {c32}")

# 1b. Độ lệch chuẩn nhiễu trắng sau Gaussian 2D: sigma_n / (2 sqrt(pi) sigma)
r["nhieu_sau_gaussian"] = {}
noise = RNG.normal(0, 1.0, (1024, 1024))
for s in [1.0, 2.0, 3.0, 5.0]:
    measured = float(ndi.gaussian_filter(noise, s)[50:-50, 50:-50].std())
    theory = 1 / (2 * np.sqrt(np.pi) * s)
    r["nhieu_sau_gaussian"][str(s)] = {"do_duoc": measured, "ly_thuyet": theory}
    print(f"sigma={s}: std nhiễu đo={measured:.4f}, lý thuyết={theory:.4f}")

# ------------------------------------------------------------ 2. Gradient
r = section("2_gradient")

# 2a. Hệ số khuếch đại Sobel: dốc đơn vị -> 8, bậc thang cao h -> đỉnh 4h
x = np.tile(np.arange(64, dtype=np.float64), (64, 1))
gx_ramp = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)[10:-10, 10:-10]
step = np.zeros((64, 64)); step[:, 32:] = 1.0
gx_step = cv2.Sobel(step, cv2.CV_64F, 1, 0, ksize=3)
r["sobel_gain_doc_don_vi"] = float(gx_ramp.mean())
r["sobel_dinh_bac_thang_h1"] = float(gx_step.max())
print(f"Sobel trên dốc đơn vị = {gx_ramp.mean():.3f} (lý thuyết 8); "
      f"đỉnh trên bậc thang h=1 = {gx_step.max():.3f} (lý thuyết 4)")

# 2b. Đỉnh gradient sau Gaussian: 8h / (sqrt(2 pi) sigma)  (xấp xỉ liên tục)
r["dinh_gradient_sau_gaussian"] = {}
wide = np.zeros((64, 257)); wide[:, 128:] = 1.0
for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
    sm = ndi.gaussian_filter(wide, s, mode="nearest")
    peak = float(cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3)[32].max())
    theory = 8 / (np.sqrt(2 * np.pi) * s)
    # công thức rời rạc chính xác: 4*[Phi(1.5/s) - Phi(-0.5/s)]... dùng tổng Gauss rời rạc thay vì đoán
    r["dinh_gradient_sau_gaussian"][str(s)] = {"do_duoc": peak, "xap_xi_lien_tuc": theory}
    print(f"sigma={s}: đỉnh Sobel đo={peak:.3f}, xấp xỉ 8/(sqrt(2pi)s)={theory:.3f}, cận trên 4")

# 2c. scipy.ndimage.sobel trùng cv2.Sobel(ksize=3) ở phần trong ảnh
img = data.camera().astype(np.float64)
d_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
d_nd = ndi.sobel(img, axis=1)
r["ndi_sobel_bang_cv_sobel_max_lech_phan_trong"] = float(np.abs(d_cv - d_nd)[1:-1, 1:-1].max())
print(f"max |cv2.Sobel - ndi.sobel| phần trong = {r['ndi_sobel_bang_cv_sobel_max_lech_phan_trong']}")

# 2d. Bất đẳng thức chuẩn: L2 <= L1 <= sqrt(2) L2
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
l2 = np.hypot(d_cv, gy); l1 = np.abs(d_cv) + np.abs(gy)
nz = l2 > 0
ratio = l1[nz] / l2[nz]
r["ti_le_L1_tren_L2"] = {"min": float(ratio.min()), "max": float(ratio.max()), "trung_binh": float(ratio.mean())}
print(f"L1/L2: min={ratio.min():.4f} max={ratio.max():.4f} (cận 1 và {np.sqrt(2):.4f}), tb={ratio.mean():.4f}")

# 2e. Độ khuếch đại nhiễu: Sobel (tổng bình phương hệ số 12) và Laplacian 4-lân cận (20)
w = RNG.normal(0, 1.0, (1024, 1024))
r["std_nhieu_sobel_x"] = float(cv2.Sobel(w, cv2.CV_64F, 1, 0, ksize=3)[5:-5, 5:-5].std())
r["std_nhieu_laplacian"] = float(cv2.Laplacian(w, cv2.CV_64F, ksize=1)[5:-5, 5:-5].std())
print(f"std nhiễu sau Sobel_x = {r['std_nhieu_sobel_x']:.3f} (lý thuyết sqrt(12)={np.sqrt(12):.3f}); "
      f"sau Laplacian = {r['std_nhieu_laplacian']:.3f} (lý thuyết sqrt(20)={np.sqrt(20):.3f})")

# ----------------------------------------------- 3. Quy ước ngưỡng của OpenCV
r = section("3_nguong_opencv")
# Dùng overload Canny(dx, dy): một đường dọc có |dx| = 100, dy = 0.
dx = np.zeros((32, 32), np.int16); dx[:, 16] = 100
dy = np.zeros_like(dx)
for norm, l2flag in [("L1", False), ("L2", True)]:
    at = int((cv2.Canny(dx, dy, 50, 100, L2gradient=l2flag) > 0).sum())
    above = int((cv2.Canny(dx, dy, 50, 99, L2gradient=l2flag) > 0).sum())
    r[f"{norm}_M_bang_T_high"] = at
    r[f"{norm}_M_lon_hon_T_high"] = above
    print(f"{norm}: M=100, T_high=100 -> {at} pixel; T_high=99 -> {above} pixel")
# Ngưỡng thấp: đường mạnh nối với đoạn có M = T_low
dx2 = np.zeros((32, 32), np.int16); dx2[:16, 16] = 200; dx2[16:, 16] = 50
for tl in [50, 49]:
    n = int((cv2.Canny(dx2, np.zeros_like(dx2), tl, 150) > 0).sum())
    r[f"doan_yeu_M50_T_low_{tl}"] = n
    print(f"đoạn mạnh 16 px + đoạn yếu M=50: T_low={tl} -> {n} pixel")
# cv2.Canny(ảnh) == cv2.Canny(Sobel dx, Sobel dy) với biên replicate -> mô tả đầy đủ bước gradient
cam8 = data.camera()
for name, border in [("replicate", cv2.BORDER_REPLICATE), ("reflect101", cv2.BORDER_REFLECT_101)]:
    gx16 = cv2.Sobel(cam8, cv2.CV_16S, 1, 0, ksize=3, borderType=border)
    gy16 = cv2.Sobel(cam8, cv2.CV_16S, 0, 1, ksize=3, borderType=border)
    for l2 in (False, True):
        diff = int((cv2.Canny(cam8, 100, 200, L2gradient=l2) != cv2.Canny(gx16, gy16, 100, 200, L2gradient=l2)).sum())
        r[f"canny_anh_vs_canny_dxdy_{name}_{'L2' if l2 else 'L1'}_pixel_khac"] = diff
        print(f"Canny(ảnh) vs Canny(dx,dy) biên {name}, {'L2' if l2 else 'L1'}: {diff} pixel khác")
# Hoán đổi ngưỡng
r["hoan_doi_nguong_giong_nhau"] = bool(np.array_equal(cv2.Canny(data.camera(), 200, 100),
                                                      cv2.Canny(data.camera(), 100, 200)))
print(f"cv2.Canny(img,200,100) == cv2.Canny(img,100,200): {r['hoan_doi_nguong_giong_nhau']}")

# ----------------------------------------------- 4. Quy ước ngưỡng của skimage
r = section("4_nguong_skimage")
# Gọi thẳng hàm NMS nội bộ với độ lớn chính xác (tránh sai số sqrt): giá trị bằng T_low có được giữ?
from skimage.feature import _canny_cy
step_img = np.zeros((32, 32)); step_img[:, 16:] = 1.0
gxs_, gys_ = ndi.sobel(step_img, axis=1), ndi.sobel(step_img, axis=0)
mag_exact = np.abs(gxs_)  # gy = 0 nên |gx| là độ lớn chính xác = 4
mask_ = np.ones_like(step_img, bool); mask_[[0, -1], :] = False; mask_[:, [0, -1]] = False
nms = np.asarray(_canny_cy._nonmaximum_suppression_bilinear(gys_, gxs_, mag_exact, mask_, 4.0))
r["nms_giu_M_bang_T_low"] = bool(nms[5, 15] == 4.0)
print(f"skimage NMS: M=4, T_low=4 -> giữ: {r['nms_giu_M_bang_T_low']} "
      f"(mã nguồn: high_mask = low_masked >= high_threshold)")
# Qua canny() đầy đủ, bước chuẩn hóa bleed-over (chia cho Gaussian của mask, mode='constant')
# làm ảnh lệch ~1e-16 nên M lệch ~1e-15 và đặt ngưỡng đúng bằng 4 sẽ mất cạnh:
from skimage.feature._canny import _preprocess
sm_, _ = _preprocess(step_img, None, 0, "constant", 0.0)
r["canny_high_4_pixel"] = int(feature.canny(step_img, sigma=0, high_threshold=4.0).sum())
r["canny_do_lon_sau_preprocess_tru_4"] = float(np.hypot(ndi.sobel(sm_, axis=1), ndi.sobel(sm_, axis=0))[5, 15] - 4)
print(f"canny(high=4) -> {r['canny_high_4_pixel']} pixel; M sau _preprocess - 4 = "
      f"{r['canny_do_lon_sau_preprocess_tru_4']:.2e}")

# Xử lý điểm hòa (tie) trong NMS trên bậc thang lý tưởng: 2 cột 15, 16 cùng M = 4
r["tie_skimage_cot"] = np.unique(np.where(feature.canny(step_img, sigma=0, high_threshold=1.0))[1]).tolist()
s8 = (step_img * 255).astype(np.uint8)
r["tie_opencv_cot"] = np.unique(np.where(cv2.Canny(s8, 100, 200))[1]).tolist()
r["tie_opencv_hang_canh_ngang"] = np.unique(np.where(cv2.Canny(s8.T.copy(), 100, 200))[0]).tolist()
print(f"Tie: skimage giữ cột {r['tie_skimage_cot']}, OpenCV giữ cột {r['tie_opencv_cot']}, "
      f"cạnh ngang OpenCV giữ hàng {r['tie_opencv_hang_canh_ngang']}")
r["uint8_va_float_cung_ket_qua"] = bool(np.array_equal(
    feature.canny(data.camera()), feature.canny(skimage.img_as_float(data.camera()))))
print(f"skimage uint8 == float: {r['uint8_va_float_cung_ket_qua']}")

# ------------------------------------- 5. Quy đổi ngưỡng OpenCV <-> skimage
r = section("5_quy_doi_nguong")
cam = data.camera()
cases = {}
for sigma, lo, hi in [(1.0, 25.5, 51.0), (2.0, 15.0, 40.0), (1.0, 50.0, 100.0), (3.0, 10.0, 30.0)]:
    sk = feature.canny(cam, sigma=sigma, low_threshold=lo, high_threshold=hi)
    # (a) cách Lab3: GaussianBlur trên uint8 rồi cv2.Canny L1, cùng con số ngưỡng
    naive = cv2.Canny(cv2.GaussianBlur(cam, (0, 0), sigma), lo, hi) > 0
    # (b) quy đổi đúng: làm mờ float giống skimage, Sobel float, nhân thang 255 (vì skimage
    #     chuẩn hóa ảnh về [0,1]), làm tròn int16, Canny(dx,dy) với L2 và cùng ngưỡng.
    camf = skimage.img_as_float(cam)
    sm = skimage.filters.gaussian(camf, sigma=sigma, mode="constant", preserve_range=False)
    bleed = skimage.filters.gaussian(np.ones_like(camf), sigma=sigma, mode="constant")
    sm = sm / bleed
    gxs = np.round(ndi.sobel(sm, axis=1) * 255).astype(np.int16)
    gys = np.round(ndi.sobel(sm, axis=0) * 255).astype(np.int16)
    conv = cv2.Canny(gxs, gys, lo, hi, L2gradient=True) > 0
    conv[[0, -1], :] = False; conv[:, [0, -1]] = False  # skimage bỏ viền 1 px
    key = f"sigma{sigma}_{lo}_{hi}"
    cases[key] = {"iou_naive_L1": iou(sk, naive), "iou_quy_doi_L2": iou(sk, conv),
                  "pixel_skimage": int(sk.sum()), "pixel_naive": int(naive.sum()),
                  "pixel_quy_doi": int(conv.sum())}
    print(f"{key}: IoU cách Lab3={cases[key]['iou_naive_L1']:.3f}  "
          f"IoU quy đổi={cases[key]['iou_quy_doi_L2']:.3f}  "
          f"(px sk={sk.sum()}, naive={naive.sum()}, quy đổi={conv.sum()})")
r.update(cases)

# ------------------------------------------- 6. Tính đơn điệu theo ngưỡng
r = section("6_don_dieu")
violations = 0; checks = 0
mags = [10, 20, 30, 40, 60, 80]
for i, lo in enumerate(mags):
    for hi in mags[i:]:
        e = feature.canny(cam, sigma=1.5, low_threshold=lo, high_threshold=hi)
        for lo2 in [m for m in mags if lo <= m <= hi]:
            e2 = feature.canny(cam, sigma=1.5, low_threshold=lo2, high_threshold=hi)
            checks += 1; violations += int((e2 & ~e).any())
        for hi2 in [m for m in mags if m >= hi]:
            e2 = feature.canny(cam, sigma=1.5, low_threshold=lo, high_threshold=hi2)
            checks += 1; violations += int((e2 & ~e).any())
r["so_cap_kiem_tra"] = checks; r["so_vi_pham"] = violations
print(f"Đơn điệu theo ngưỡng (skimage): {checks} cặp, {violations} vi phạm")
sig_counts = {str(s): int(feature.canny(cam, sigma=s).sum()) for s in [0.5, 1, 1.5, 2, 3, 5]}
r["so_pixel_theo_sigma"] = sig_counts
e1, e2 = feature.canny(cam, sigma=1.0), feature.canny(cam, sigma=2.0)
r["sigma2_khong_la_tap_con_sigma1_pixel"] = int((e2 & ~e1).sum())
print(f"Số pixel theo sigma: {sig_counts}; pixel có ở sigma=2 nhưng không có ở sigma=1: "
      f"{r['sigma2_khong_la_tap_con_sigma1_pixel']}")

# ------------------------------------------- 7. SNR của gradient theo sigma
r = section("7_snr_gradient")
# Lý thuyết (xấp xỉ liên tục, sigma đủ lớn): std nhiễu của Sobel_x(G_sigma * n) ~ 8 sigma_n / (2 sqrt(2 pi) sigma^2)
w = RNG.normal(0, 1.0, (1024, 1024))
for s in [1.0, 2.0, 3.0, 5.0]:
    measured = float(cv2.Sobel(ndi.gaussian_filter(w, s), cv2.CV_64F, 1, 0, ksize=3)[60:-60, 60:-60].std())
    theory = 8 / (2 * np.sqrt(2 * np.pi) * s**2)
    peak = results["2_gradient"]["dinh_gradient_sau_gaussian"][str(s)]["do_duoc"]
    r[str(s)] = {"std_nhieu_gradient_do": measured, "ly_thuyet": theory,
                 "snr_do_h1_sigman1": peak / measured, "snr_ly_thuyet_2sigma": 2 * s}
    print(f"sigma={s}: std nhiễu gradient đo={measured:.4f}, lý thuyết={theory:.4f}; "
          f"SNR đo (h=1, sigma_n=1)={peak/measured:.3f}, lý thuyết 2*sigma={2*s:.1f}")

# ------------------------------------------- 8. Ảnh màu
r = section("8_anh_mau")
# Hai màu đẳng sáng: Y = 0.299R + 0.587G + 0.114B bằng nhau -> ảnh xám không còn cạnh
left, right = (200, 100, 50), (55, 170, 70)  # (R, G, B): cam đất và xanh lá, Y ~ 124.2
yl = 0.299 * left[0] + 0.587 * left[1] + 0.114 * left[2]
yr = 0.299 * right[0] + 0.587 * right[1] + 0.114 * right[2]
rgb = np.zeros((64, 64, 3), np.uint8); rgb[:, :32] = left; rgb[:, 32:] = right
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
r["Y_trai_phai"] = [yl, yr]
r["canh_tren_anh_xam"] = int((cv2.Canny(gray, 20, 40) > 0).sum())
r["canh_tren_anh_3_kenh"] = int((cv2.Canny(rgb, 20, 40) > 0).sum())
print(f"Y trái={yl:.2f}, phải={yr:.2f}; cạnh trên ảnh xám={r['canh_tren_anh_xam']}, "
      f"cv2.Canny nhận ảnh 3 kênh={r['canh_tren_anh_3_kenh']}")
# cv2.Canny 3 kênh: kiểm tra giả thuyết "lấy kênh có gradient lớn nhất"
astro = data.astronaut()
e3 = cv2.Canny(astro, 100, 200) > 0
per = [cv2.Canny(np.ascontiguousarray(astro[..., c]), 100, 200) > 0 for c in range(3)]
r["iou_3kenh_vs_OR_tung_kenh"] = iou(e3, per[0] | per[1] | per[2])
r["iou_3kenh_vs_xam"] = iou(e3, cv2.Canny(cv2.cvtColor(astro, cv2.COLOR_RGB2GRAY), 100, 200) > 0)
print(f"astronaut: IoU(Canny 3 kênh, OR từng kênh)={r['iou_3kenh_vs_OR_tung_kenh']:.3f}, "
      f"IoU(Canny 3 kênh, Canny ảnh xám)={r['iou_3kenh_vs_xam']:.3f}")
# Giả thuyết: tại mỗi pixel OpenCV chọn kênh có độ lớn L1 lớn nhất rồi chạy Canny 1 kênh trên (dx, dy) đó
#   (Sobel phải dùng BORDER_REPLICATE như Canny bên trong; mặc định REFLECT_101 lệch 98 px ở viền.)
REP = cv2.BORDER_REPLICATE
dxs = np.stack([cv2.Sobel(astro[..., c], cv2.CV_16S, 1, 0, ksize=3, borderType=REP) for c in range(3)], -1)
dys = np.stack([cv2.Sobel(astro[..., c], cv2.CV_16S, 0, 1, ksize=3, borderType=REP) for c in range(3)], -1)
best = np.argmax(np.abs(dxs.astype(np.int32)) + np.abs(dys.astype(np.int32)), axis=-1)[..., None]
dx1 = np.take_along_axis(dxs, best, -1)[..., 0].copy()
dy1 = np.take_along_axis(dys, best, -1)[..., 0].copy()
e_max = cv2.Canny(dx1, dy1, 100, 200) > 0
r["iou_3kenh_vs_chon_kenh_max_L1"] = iou(e3, e_max)
r["so_pixel_khac_3kenh_vs_chon_kenh_max_L1"] = int((e3 ^ e_max).sum())
print(f"IoU(Canny 3 kênh, chọn kênh max L1)={r['iou_3kenh_vs_chon_kenh_max_L1']:.4f}, "
      f"pixel khác={r['so_pixel_khac_3kenh_vs_chon_kenh_max_L1']}")

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nĐã ghi {OUT}")
