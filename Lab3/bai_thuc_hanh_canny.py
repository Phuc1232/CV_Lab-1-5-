"""Bài thực hành chương 2 (phần 2) — Canny Edge Detector.

Cùng nội dung, tham số với notebook Lab3.ipynb, chạy độc lập:
    .venv/bin/python bai_thuc_hanh_canny.py

Mỗi ảnh kết quả được lưu thành một file riêng trong ket_qua/<tên bài>/.
"""
from pathlib import Path

import cv2
import numpy as np
from skimage import data, feature
from skimage.util import img_as_ubyte

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "ket_qua"
INPUT_DIR = BASE_DIR / "du_lieu"
MIN_AREA = 300  # vùng nhỏ hơn số pixel này là mẩu cạnh vụn, không phải đối tượng


def save(folder, name, image):
    """Lưu một ảnh thành file riêng: ket_qua/<folder>/<name>.png (ảnh bool đổi thành 0/255)."""
    path = OUTPUT_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    cv2.imwrite(str(path / f"{name}.png"), image)


def load_gray(source):
    """Nhận đường dẫn file ảnh hoặc mảng NumPy, trả về ảnh xám uint8 (2 chiều) để đưa vào Canny (ngoài slide)."""
    if isinstance(source, (str, Path)):
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)   # đọc thẳng thành ảnh xám
        if gray is None:
            raise FileNotFoundError(f"Không đọc được ảnh '{source}': sai đường dẫn hoặc file không phải ảnh")
        return gray
    image = img_as_ubyte(source)   # đưa về uint8 0–255 (cv2.Canny chỉ nhận uint8)
    if image.ndim == 3:            # ảnh màu: skimage dùng thứ tự RGB; [:, :, :3] bỏ kênh alpha nếu có
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    return image


def edge_ratio(edges):
    """Phần trăm pixel cạnh — dùng để so sánh định lượng giữa các lần chạy (ngoài slide)."""
    return 100 * np.count_nonzero(edges) / edges.size


def canny_cv(image, low, high, sigma=0):
    """cv2.Canny không có bước Gaussian (bước 1 của Canny) nên tự làm mờ trước nếu sigma > 0."""
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.Canny(image, low, high)


# ---------------- Bài 1 ----------------
def bai1(img):
    """Canny bằng OpenCV (ngưỡng 100/200) và scikit-image (mặc định sigma=1, ngưỡng 10%/20%)."""
    print("\n=== Bài 1. OpenCV và scikit-image ===")
    edges_cv = cv2.Canny(img, 100, 200)   # cv2.Canny(ảnh, ngưỡng thấp = 100, ngưỡng cao = 200), so với độ lớn gradient Sobel
    edges_sk = feature.canny(img)         # mặc định: sigma = 1, ngưỡng thấp = 10% × 255 = 25.5, ngưỡng cao = 20% × 255 = 51
    print(f"OpenCV 100/200     : {edge_ratio(edges_cv):.2f}% pixel cạnh")
    print(f"scikit-image mặc định: {edge_ratio(edges_sk):.2f}% pixel cạnh")
    save("bai1_opencv_vs_skimage", "01_anh_goc", img)
    save("bai1_opencv_vs_skimage", "02_cv2_canny_100_200", edges_cv)
    save("bai1_opencv_vs_skimage", "03_skimage_canny_mac_dinh", edges_sk)


# ---------------- Bài 2 ----------------
def bai2(img):
    """Thay đổi sigma, ngưỡng thấp, ngưỡng cao và so sánh với mặc định."""
    print("\n=== Bài 2.1. Thay đổi sigma ===")
    # sigma = độ lệch chuẩn của lọc Gaussian (bước 1): sigma càng lớn → ảnh càng mờ → càng ít cạnh vụn
    for i, s in enumerate([0.5, 1, 2, 3, 5], start=1):
        e_sk = feature.canny(img, sigma=s)        # skimage: chỉ đổi sigma, ngưỡng giữ mặc định 25.5 / 51
        e_cv = canny_cv(img, 100, 200, sigma=s)   # OpenCV: GaussianBlur với sigma rồi Canny, ngưỡng giữ cố định 100 / 200
        print(f"sigma={s:<4} skimage: {edge_ratio(e_sk):5.2f}%   OpenCV: {edge_ratio(e_cv):5.2f}%")
        tag = str(s).replace(".", "_")
        save("bai2_sigma", f"{i:02d}_skimage_sigma_{tag}", e_sk)
        save("bai2_sigma", f"{i + 5:02d}_opencv_gauss_sigma_{tag}", e_cv)

    print("\n=== Bài 2.2. Thay đổi ngưỡng thấp (giữ ngưỡng cao) ===")
    # Ngưỡng thấp OpenCV: 100 = mốc bài 1, 200 = bằng ngưỡng cao
    # Ngưỡng thấp skimage (thang 0–255): 25.5 = mặc định 10%, 51 = bằng ngưỡng cao
    for i, (cv_low, sk_low) in enumerate(zip([20, 50, 100, 150, 200], [5, 15, 25.5, 40, 51]), start=1):
        e_cv = cv2.Canny(img, cv_low, 200)   # ngưỡng cao giữ cố định 200, chỉ đổi ngưỡng thấp
        e_sk = feature.canny(img, low_threshold=sk_low, high_threshold=51)   # ngưỡng cao cố định 51 (mặc định 20%), sigma mặc định 1
        print(f"OpenCV low={cv_low:<3} high=200: {edge_ratio(e_cv):5.2f}%   "
              f"skimage low={sk_low:<4} high=51: {edge_ratio(e_sk):5.2f}%")
        save("bai2_nguong_thap", f"{i:02d}_opencv_low_{cv_low}_high_200", e_cv)
        save("bai2_nguong_thap", f"{i + 5:02d}_skimage_low_{str(sk_low).replace('.', '_')}_high_51", e_sk)

    print("\n=== Bài 2.3. Thay đổi ngưỡng cao (giữ ngưỡng thấp) ===")
    # Ngưỡng cao OpenCV: 200 = mốc bài 1 | ngưỡng cao skimage (thang 0–255): 51 = mặc định 20%
    for i, (cv_high, sk_high) in enumerate(zip([100, 150, 200, 300, 400], [30, 51, 80, 120, 160]), start=1):
        e_cv = cv2.Canny(img, 50, cv_high)   # ngưỡng thấp cố định 50 (luôn nhỏ hơn ngưỡng cao nhỏ nhất là 100)
        e_sk = feature.canny(img, low_threshold=25.5, high_threshold=sk_high)   # ngưỡng thấp cố định 25.5 (mặc định 10%)
        print(f"OpenCV low=50 high={cv_high:<3}: {edge_ratio(e_cv):5.2f}%   "
              f"skimage low=25.5 high={sk_high:<3}: {edge_ratio(e_sk):5.2f}%")
        save("bai2_nguong_cao", f"{i:02d}_opencv_low_50_high_{cv_high}", e_cv)
        save("bai2_nguong_cao", f"{i + 5:02d}_skimage_low_25_5_high_{sk_high}", e_sk)

    print("\n=== Bài 2.4. So sánh với giá trị mặc định ===")
    results = {
        "01_skimage_mac_dinh": feature.canny(img),                                                     # sigma = 1, ngưỡng 25.5 / 51
        "02_skimage_sigma_2_15_40": feature.canny(img, sigma=2, low_threshold=15, high_threshold=40),  # tỉ lệ ≈ 1 : 2.7
        "03_opencv_100_200_moc": cv2.Canny(img, 100, 200),                                             # mốc: không làm mờ
        "04_opencv_gauss_sigma_2_30_90": canny_cv(img, 30, 90, sigma=2),                               # tỉ lệ 1 : 3
    }
    for name, edges in results.items():
        print(f"{name:32s}: {edge_ratio(edges):5.2f}%")
        save("bai2_so_sanh_mac_dinh", name, edges)


# ---------------- Bài 3 ----------------
def bai3(img):
    """Canny trên ảnh nhiều nhiễu, tương phản thấp, nhiều chi tiết: mặc định và đã chỉnh tham số."""
    print("\n=== Bài 3. Các loại ảnh khác nhau ===")
    # Tạo ảnh test có điều kiện biết trước (ngoài slide); seed cố định để mỗi lần chạy ra cùng ảnh nhiễu
    rng = np.random.default_rng(0)
    noisy = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)   # nhiễu Gaussian: trung bình 0, độ lệch chuẩn 25
    low_contrast = (100 + img.astype(np.float64) / 255 * 40).astype(np.uint8)        # nén dải sáng 0–255 → 100–140
    detailed = data.grass()   # ảnh cỏ có sẵn trong skimage, toàn chi tiết nhỏ

    # (tên ảnh, thư mục, ảnh, tham số skimage chỉnh, (sigma, ngưỡng thấp, ngưỡng cao) OpenCV chỉnh); sigma = 0 là không làm mờ
    cases = [
        ("nhieu_nhieu", "nhieu", noisy, dict(sigma=3), (3, 30, 90)),   # sigma = 3 làm mờ mạnh để xoá nhiễu
        # use_quantiles=True: ngưỡng theo phân vị gradient của chính ảnh, 0.8 / 0.9 = top 20% / top 10%; OpenCV hạ ngưỡng 15 / 40
        ("tuong_phan_thap", "tuong_phan_thap", low_contrast,
         dict(sigma=1, low_threshold=0.8, high_threshold=0.9, use_quantiles=True), (0, 15, 40)),
        ("nhieu_chi_tiet", "chi_tiet", detailed, dict(sigma=3, low_threshold=20, high_threshold=40), (3, 30, 90)),  # sigma = 3 làm mờ texture, hạ ngưỡng vì gradient yếu đi
    ]
    print(f"{'Ảnh':16s} {'sk mặc định':>12s} {'sk chỉnh':>10s} {'cv 100/200':>11s} {'cv chỉnh':>10s}")
    for i, (name, folder, image, sk_params, (sigma, low, high)) in enumerate(cases, start=1):
        results = {
            "02_skimage_mac_dinh": feature.canny(image),
            "03_skimage_chinh": feature.canny(image, **sk_params),
            "04_opencv_100_200": cv2.Canny(image, 100, 200),
            "05_opencv_chinh": canny_cv(image, low, high, sigma=sigma),
        }
        print(f"{name:16s} " + " ".join(f"{edge_ratio(e):>10.2f}" for e in results.values()))
        save("bai3_anh_dau_vao", f"{i:02d}_{name}", image)
        save(f"bai3_{folder}", f"01_{name}", image)
        for file_name, edges in results.items():
            save(f"bai3_{folder}", file_name, edges)


# ---------------- Bài 4 ----------------
def find_regions(edges, kernel):
    """Closing nối chỗ biên đứt → lấy đường bao ngoài → bỏ vùng nhỏ hơn MIN_AREA."""
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)   # closing = dilate (giãn nở) rồi erode (xói mòn)
    # RETR_EXTERNAL: chỉ lấy đường bao ngoài cùng; CHAIN_APPROX_SIMPLE: chỉ lưu điểm đầu/cuối mỗi đoạn thẳng
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return closed, [c for c in contours if cv2.contourArea(c) >= MIN_AREA]   # contourArea = diện tích bên trong đường bao


def shape_name(contour):
    """Đặt tên hình theo số đỉnh của đa giác xấp xỉ (sai lệch tối đa 4% chu vi)."""
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)   # True = đường bao khép kín
    if len(approx) == 3:
        return "Tam giac"
    if len(approx) == 4:
        _, _, w, h = cv2.boundingRect(approx)
        return "Vuong" if 0.9 <= w / h <= 1.1 else "Chu nhat"   # rộng/cao của khung bao ≈ 1 (lệch ≤ 10%) → vuông
    if len(approx) == 5:
        return "Ngu giac"
    return "Tron"


def bai4_phan_doan(kernel):
    """Canny + closing + tô kín đường bao → mỗi đồng xu là một vùng."""
    print("\n=== Bài 4.1. Canny + phân đoạn ===")
    coins = data.coins()
    edges = canny_cv(coins, 30, 90, sigma=1.5)   # làm mờ sigma = 1.5, ngưỡng thấp 30, ngưỡng cao 90 (1 : 3)
    closed, regions = find_regions(edges, kernel)

    mask = np.zeros_like(coins)
    cv2.drawContours(mask, regions, -1, 255, thickness=-1)   # -1 = vẽ mọi đường bao, 255 = màu trắng, thickness=-1 = tô kín
    overlay = cv2.cvtColor(coins, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, regions, -1, (0, 0, 255), 2)   # (0, 0, 255) = đỏ theo BGR của OpenCV, nét dày 2 px
    print(f"Tìm được {len(regions)} vùng (đồng xu) có diện tích ≥ {MIN_AREA} px")

    for name, image in [("01_anh_goc", coins), ("02_1_canny_sigma_1_5_30_90", edges),
                        ("03_2_closing_55", closed), ("04_34_vung_da_to_kin", mask),
                        ("05_vung_tren_anh_goc", overlay)]:
        save("bai4_phan_doan", name, image)
    return regions


def bai4_nhan_dang(kernel, coin_regions):
    """Canny + closing + approxPolyDP để nhận dạng hình dạng các đối tượng."""
    print("\n=== Bài 4.2. Canny + nhận dạng hình dạng ===")
    # Ảnh test vẽ sẵn để biết trước đáp án (ngoài slide). Làm khó: nền sáng không đều (tối trái → sáng phải),
    # hình bị xoay, có hình tối trên nền sáng, có hình tương phản thấp, nhiễu mạnh (độ lệch chuẩn 20)
    rng = np.random.default_rng(0)
    shapes = np.tile(np.linspace(50, 150, 700), (400, 1)).astype(np.uint8)   # nền 400x700, độ sáng 50 → 150

    def rotated_rect(cx, cy, w, h, angle):
        """4 đỉnh của hình chữ nhật tâm (cx, cy), kích thước w x h, xoay angle độ."""
        return cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)

    # fillPoly(ảnh, [danh sách đỉnh (x, y)], độ sáng tô 0–255) | circle(ảnh, tâm (x, y), bán kính, độ sáng, -1 = tô kín)
    cv2.fillPoly(shapes, [np.array([[40, 170], [150, 170], [95, 75]])], 210)                        # tam giác sáng
    cv2.fillPoly(shapes, [np.array([[60, 370], [200, 330], [90, 250]])], 15)                         # tam giác lệch, tối
    cv2.fillPoly(shapes, [rotated_rect(270, 110, 90, 90, 30)], 190)                                  # vuông xoay 30°
    cv2.fillPoly(shapes, [rotated_rect(300, 300, 150, 60, -20)], 140)                                # chữ nhật xoay, tương phản thấp
    cv2.fillPoly(shapes, [np.array([[470, 40], [520, 78], [502, 138], [438, 138], [420, 78]])], 30)  # ngũ giác tối
    cv2.circle(shapes, (490, 280), 60, 200, -1)                                                      # tròn lớn
    cv2.circle(shapes, (630, 110), 28, 75, -1)                                                       # tròn nhỏ, tối
    cv2.fillPoly(shapes, [rotated_rect(635, 300, 40, 110, 10)], 230)                                 # chữ nhật đứng, xoay 10°
    shapes = np.clip(shapes + rng.normal(0, 20, shapes.shape), 0, 255).astype(np.uint8)
    INPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(INPUT_DIR / "hinh_dang.png"), shapes)

    edges = canny_cv(shapes, 50, 150, sigma=1.5)   # sigma = 1.5, ngưỡng thấp 50, ngưỡng cao 150 (1 : 3)
    _, contours = find_regions(edges, kernel)

    labeled = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
    found = []
    for c in contours:
        name = shape_name(c)
        x, y, _, _ = cv2.boundingRect(c)
        found.append((x, name))
        cv2.drawContours(labeled, [c], -1, (0, 0, 255), 2)
        cv2.putText(labeled, name, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    print("Đối tượng tìm được (trái → phải):", [n for _, n in sorted(found)])

    save("bai4_nhan_dang_hinh", "01_anh_test", shapes)
    save("bai4_nhan_dang_hinh", "02_canny_sigma_1_5_50_150", edges)
    save("bai4_nhan_dang_hinh", "03_ket_qua_nhan_dang", labeled)

    coin_names = [shape_name(c) for c in coin_regions]
    print("Nhận dạng các vùng đồng xu:", {n: coin_names.count(n) for n in set(coin_names)})


def main():
    image_source = data.camera()   # đổi ảnh ở đây, vd: data.astronaut() hoặc INPUT_DIR / "ten_anh.jpg"
    img = load_gray(image_source)
    bai1(img)
    bai2(img)
    bai3(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # phần tử cấu trúc hình tròn 5×5 px cho closing
    coin_regions = bai4_phan_doan(kernel)
    bai4_nhan_dang(kernel, coin_regions)
    print(f"\nĐã lưu ảnh kết quả vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
