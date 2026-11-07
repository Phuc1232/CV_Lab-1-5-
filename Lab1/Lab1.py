"""
=============================================================================
BÀI THỰC HÀNH CHƯƠNG 1: CÁC THAO TÁC CƠ BẢN VỚI HÌNH ẢNH TRONG OPENCV & PILLOW
=============================================================================
Nội dung thực hiện:
  1. Cài đặt và sử dụng thư viện OpenCV và Pillow
  2. Đọc, hiển thị và lưu ảnh với các định dạng khác nhau
  3. Chuyển đổi không gian màu: BGR sang Grayscale, HSV, LAB
  4. Cắt xén (Crop) và Thay đổi kích thước theo tỷ lệ (Resize preserving Aspect Ratio)
  5. Vẽ các hình học cơ bản (đường thẳng, hình chữ nhật, hình tròn) và thêm văn bản
=============================================================================
"""

import os
import sys
import cv2
from PIL import Image

# Đảm bảo in tiếng Việt không bị lỗi charmap trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thư mục hiện tại của file mã nguồn (tránh lỗi hardcode đường dẫn)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. ĐỌC VÀ HIỂN THỊ ẢNH (OPENCV & PILLOW)
# =============================================================================
def doc_va_hien_thi_anh(duong_dan_anh):
    """
    Đọc ảnh từ đường dẫn, in thông tin kích thước và lưu sang định dạng khác.
    Sử dụng cả OpenCV và Pillow.
    """
    print("\n--- PHẦN 1: ĐỌC VÀ HIỂN THỊ ẢNH ---")
    
    # 1.1. Đọc và xử lý bằng OpenCV
    anh_cv = cv2.imread(duong_dan_anh)
    if anh_cv is None:
        print(f"[LỖI] Không thể đọc ảnh từ: {duong_dan_anh}")
        return None

    cao, rong = anh_cv.shape[:2]
    so_kenh = anh_cv.shape[2] if len(anh_cv.shape) > 2 else 1
    
    print("[OpenCV] Đọc ảnh thành công:")
    print(f"  - Chiều cao (Height): {cao} px")
    print(f"  - Chiều rộng (Width): {rong} px")
    print(f"  - Số kênh màu: {so_kenh} (OpenCV đọc mặc định theo thứ tự BGR)")

    # Lưu ảnh sang định dạng PNG trong thư mục hiện tại
    duong_dan_luu_png = os.path.join(CURRENT_DIR, "anh_luu.png")
    cv2.imwrite(duong_dan_luu_png, anh_cv)
    print(f"  -> Đã lưu ảnh OpenCV sang định dạng PNG: {duong_dan_luu_png}")

    # Hiển thị ảnh OpenCV
    cv2.imshow("1. Anh goc (OpenCV)", anh_cv)

    # 1.2. Đọc và xử lý bằng Pillow
    try:
        anh_pil = Image.open(duong_dan_anh)
        print(f"[Pillow] Kích thước ảnh (Width, Height): {anh_pil.size}")
        print(f"  - Hệ màu: {anh_pil.mode} (Pillow đọc theo chuẩn RGB)")

        # Lưu ảnh sang định dạng JPEG bằng Pillow
        duong_dan_luu_pil = os.path.join(CURRENT_DIR, "anh_luu_pillow.jpg")
        anh_pil.save(duong_dan_luu_pil, format="JPEG")
        print(f"  -> Đã lưu ảnh Pillow sang định dạng JPEG: {duong_dan_luu_pil}")

        # Hiển thị ảnh qua trình xem ảnh mặc định của hệ thống
        # anh_pil.show(title="Anh mo bang Pillow")
    except Exception as e:
        print(f"[Pillow] Lỗi khi xử lý ảnh bằng Pillow: {e}")

    print("\n>> Nhấn phím bất kỳ trên cửa sổ ảnh để chuyển sang Phần 2...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return anh_cv


# =============================================================================
# 2. CHUYỂN ĐỔI KHÔNG GIAN MÀU
# =============================================================================
def chuyen_doi_khong_gian_mau(anh):
    """
    Chuyển đổi ảnh sang các không gian màu: Grayscale, HSV, LAB.
    Lưu ý: Ảnh đọc bằng cv2.imread có hệ màu gốc là BGR.
    """
    print("\n--- PHẦN 2: CHUYỂN ĐỔI KHÔNG GIAN MÀU ---")

    # 2.1. Chuyển đổi sang Grayscale
    # Công thức BT.601: Y = 0.299*R + 0.587*G + 0.114*B
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    path_xam = os.path.join(CURRENT_DIR, "anh_grayscale.jpg")
    cv2.imwrite(path_xam, anh_xam)
    print(f"  -> Đã chuyển và lưu ảnh Grayscale: {path_xam}")

    # 2.2. Chuyển đổi sang HSV (Hue - Saturation - Value)
    # H: 0-179 (sắc thái màu), S: 0-255 (độ đậm), V: 0-255 (độ sáng)
    anh_hsv = cv2.cvtColor(anh, cv2.COLOR_BGR2HSV)
    path_hsv = os.path.join(CURRENT_DIR, "anh_hsv.jpg")
    cv2.imwrite(path_hsv, anh_hsv)
    print(f"  -> Đã chuyển và lưu ảnh HSV: {path_hsv}")

    # 2.3. Chuyển đổi sang LAB (Luminance, a: Green-Red, b: Blue-Yellow)
    anh_lab = cv2.cvtColor(anh, cv2.COLOR_BGR2LAB)
    path_lab = os.path.join(CURRENT_DIR, "anh_lab.jpg")
    cv2.imwrite(path_lab, anh_lab)
    print(f"  -> Đã chuyển và lưu ảnh LAB: {path_lab}")

    # Hiển thị kết quả
    cv2.imshow("2.1. Anh Grayscale", anh_xam)
    cv2.imshow("2.2. Anh HSV (hien thi qua bang mau BGR)", anh_hsv)
    cv2.imshow("2.3. Anh LAB (hien thi qua bang mau BGR)", anh_lab)

    print("\n>> Nhấn phím bất kỳ trên cửa sổ ảnh để chuyển sang Phần 3...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# 3. CẮT XÉN VÀ THAY ĐỔI KÍCH THƯỚC THEO TỶ LỆ
# =============================================================================
def cat_xen_va_thay_doi_kich_thuoc(anh, ti_le=0.5):
    """
    - Cắt một vùng ảnh bất kỳ (vùng trung tâm 50%).
    - Thay đổi kích thước theo tỷ lệ (Scale Factor) để bảo toàn tỷ lệ khung hình,
      tránh hiện tượng ảnh bị méo/biến dạng khi ép kích thước cố định.
    """
    print("\n--- PHẦN 3: CẮT XÉN VÀ THAY ĐỔI KÍCH THƯỚC ---")
    cao, rong = anh.shape[:2]

    # 3.1. Cắt xén (Crop) vùng trung tâm ảnh
    y1, y2 = cao // 4, 3 * cao // 4
    x1, x2 = rong // 4, 3 * rong // 4
    anh_cat = anh[y1:y2, x1:x2]

    path_cat = os.path.join(CURRENT_DIR, "anh_cat.jpg")
    cv2.imwrite(path_cat, anh_cat)
    print(f"  -> Cắt vùng trung tâm: từ ({x1}, {y1}) đến ({x2}, {y2})")
    print(f"  -> Kích thước ảnh sau cắt: {anh_cat.shape[1]}x{anh_cat.shape[0]} px")
    print(f"  -> Đã lưu ảnh cắt: {path_cat}")

    # 3.2. Thay đổi kích thước theo tỷ lệ (Resize theo Scale Factor)
    # Phương pháp này bảo toàn tỷ lệ khung hình (Aspect Ratio), không làm biến dạng vật thể
    anh_resize = cv2.resize(
        anh,
        None,
        fx=ti_le,
        fy=ti_le,
        interpolation=cv2.INTER_AREA if ti_le < 1.0 else cv2.INTER_CUBIC
    )

    path_resize = os.path.join(CURRENT_DIR, "anh_resize.jpg")
    cv2.imwrite(path_resize, anh_resize)
    print(f"  -> Thay đổi kích thước theo tỷ lệ: {ti_le * 100:.0f}%")
    print(f"  -> Kích thước mới: {anh_resize.shape[1]}x{anh_resize.shape[0]} px (Gốc: {rong}x{cao} px)")
    print(f"  -> Đã lưu ảnh resize: {path_resize}")

    # Hiển thị
    cv2.imshow("3.1. Anh da cat (Crop)", anh_cat)
    cv2.imshow(f"3.2. Anh Resize {ti_le * 100:.0f}%", anh_resize)

    print("\n>> Nhấn phím bất kỳ trên cửa sổ ảnh để chuyển sang Phần 4...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# 4. VẼ HÌNH CƠ BẢN VÀ THÊM VĂN BẢN
# =============================================================================
def ve_hinh_va_them_van_ban(anh):
    """
    Vẽ các hình cơ bản: đường thẳng, hình tròn, hình chữ nhật và thêm văn bản.
    Tọa độ và kích thước được tính theo tỷ lệ kích thước ảnh (Relative Coordinates)
    để đảm bảo không bị tràn mép ảnh bất kể kích thước ảnh đầu vào.
    """
    print("\n--- PHẦN 4: VẼ HÌNH CƠ BẢN VÀ THÊM VĂN BẢN ---")
    anh_ve = anh.copy()
    cao, rong = anh_ve.shape[:2]

    # Độ dày nét vẽ tỷ lệ theo kích thước ảnh (tối thiểu 1px)
    do_day = max(2, int(min(cao, rong) * 0.005))
    font_scale = max(0.6, min(cao, rong) * 0.0018)

    # 4.1. Vẽ đường thẳng (Màu xanh dương - BGR: (255, 0, 0))
    pt1_line = (int(rong * 0.1), int(cao * 0.1))
    pt2_line = (int(rong * 0.9), int(cao * 0.1))
    cv2.line(anh_ve, pt1_line, pt2_line, (255, 0, 0), do_day)

    # 4.2. Vẽ hình chữ nhật (Màu xanh lá - BGR: (0, 255, 0))
    pt1_rect = (int(rong * 0.15), int(cao * 0.2))
    pt2_rect = (int(rong * 0.85), int(cao * 0.75))
    cv2.rectangle(anh_ve, pt1_rect, pt2_rect, (0, 255, 0), do_day)

    # 4.3. Vẽ hình tròn (Màu đỏ - BGR: (0, 0, 255))
    tam_tron = (int(rong * 0.5), int(cao * 0.475))
    ban_kinh = int(min(cao, rong) * 0.18)
    cv2.circle(anh_ve, tam_tron, ban_kinh, (0, 0, 255), do_day)

    # 4.4. Thêm văn bản (Màu trắng - BGR: (255, 255, 255))
    vi_tri_text = (int(rong * 0.2), int(cao * 0.6))
    cv2.putText(
        anh_ve,
        "OpenCV & Python",
        vi_tri_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        do_day,
        cv2.LINE_AA
    )

    path_ve = os.path.join(CURRENT_DIR, "anh_ve.jpg")
    cv2.imwrite(path_ve, anh_ve)
    print(f"  -> Đã vẽ các hình cơ bản và thêm chữ.")
    print(f"  -> Đã lưu ảnh kết quả: {path_ve}")

    # Hiển thị
    cv2.imshow("4. Anh ve hinh hoc va van ban", anh_ve)

    print("\n>> Nhấn phím bất kỳ trên cửa sổ ảnh để hoàn thành bài thực hành...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN)
# =============================================================================
def main():
    print("=" * 60)
    print(" BÀI THỰC HÀNH CHƯƠNG 1 - XỬ LÝ ẢNH CƠ BẢN VỚI OPENCV & PILLOW")
    print("=" * 60)

    # Đường dẫn ảnh mặc định
    duong_dan_mac_dinh = os.path.join(CURRENT_DIR, "anh.jpg")

    prompt = f"Nhập đường dẫn ảnh [Nhấn Enter để dùng mặc định '{os.path.basename(duong_dan_mac_dinh)}']: "
    duong_dan = input(prompt).strip().strip('"').strip("'")

    if not duong_dan:
        duong_dan = duong_dan_mac_dinh

    if not os.path.exists(duong_dan):
        print(f"[LỖI] Không tìm thấy file tại đường dẫn: {duong_dan}")
        sys.exit(1)

    # 1. Đọc và hiển thị ảnh
    anh = doc_va_hien_thi_anh(duong_dan)
    if anh is None:
        sys.exit(1)

    # 2. Chuyển đổi không gian màu
    chuyen_doi_khong_gian_mau(anh)

    # 3. Cắt xén và thay đổi kích thước theo tỷ lệ
    cat_xen_va_thay_doi_kich_thuoc(anh, ti_le=0.5)

    # 4. Vẽ hình cơ bản và văn bản
    ve_hinh_va_them_van_ban(anh)

    print("\n" + "=" * 60)
    print(" HOÀN THÀNH TOÀN BỘ BÀI THỰC HÀNH CHƯƠNG 1!")
    print(f" Tất cả ảnh kết quả đã được lưu tại: {CURRENT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()