import cv2
from PIL import Image


# =========================================================
# 1. ĐỌC VÀ HIỂN THỊ ẢNH
# =========================================================

# Đường dẫn đến ảnh
duong_dan = r"C:\lab01\CV_Lab-1-5-\Lab1\anh.jpg"

# Đọc ảnh bằng OpenCV
anh = cv2.imread(duong_dan)

# Kiểm tra ảnh có tồn tại không
if anh is None:
    print("Không tìm thấy ảnh!")
    exit()

print("Đọc ảnh thành công!")

# Hiển thị kích thước ảnh
print("Chiều cao:", anh.shape[0])
print("Chiều rộng:", anh.shape[1])
print("Số kênh màu:", anh.shape[2])

# Hiển thị ảnh
cv2.imshow("Anh goc", anh)

# Lưu ảnh sang định dạng PNG
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_luu.png", anh)

# =========================================================
# 2. SỬ DỤNG PILLOW ĐỂ ĐỌC VÀ LƯU ẢNH
# =========================================================

anh_pil = Image.open(duong_dan)

print("Kích thước ảnh bằng Pillow:", anh_pil.size)

# Lưu ảnh sang định dạng JPEG
anh_pil.save(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_luu_pillow.jpg")


# =========================================================
# 3. CHUYỂN ĐỔI KHÔNG GIAN MÀU
# =========================================================

# RGB -> Grayscale
anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)

# Hiển thị ảnh grayscale
cv2.imshow("Anh Grayscale", anh_xam)

# Lưu ảnh grayscale
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_grayscale.jpg", anh_xam)


# ---------------------------------------------------------
# Chuyển sang HSV
# ---------------------------------------------------------

anh_hsv = cv2.cvtColor(anh, cv2.COLOR_BGR2HSV)

cv2.imshow("Anh HSV", anh_hsv)

# Lưu ảnh HSV
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_hsv.jpg", anh_hsv)


# ---------------------------------------------------------
# Chuyển sang LAB
# ---------------------------------------------------------

anh_lab = cv2.cvtColor(anh, cv2.COLOR_BGR2LAB)

cv2.imshow("Anh LAB", anh_lab)

# Lưu ảnh LAB
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_lab.jpg", anh_lab)


# =========================================================
# 4. CẮT XÉN ẢNH
# =========================================================

# Lấy kích thước ảnh
cao, rong = anh.shape[:2]

# Cắt một vùng ở giữa ảnh
x1 = rong // 4
y1 = cao // 4

x2 = 3 * rong // 4
y2 = 3 * cao // 4

anh_cat = anh[y1:y2, x1:x2]

# Hiển thị ảnh đã cắt
cv2.imshow("Anh da cat", anh_cat)

# Lưu ảnh
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_cat.jpg", anh_cat)


# =========================================================
# 5. THAY ĐỔI KÍCH THƯỚC
# =========================================================

# Resize về kích thước cố định 500 x 400
anh_resize = cv2.resize(anh, (500, 400))

# Hiển thị
cv2.imshow("Anh Resize 500x400", anh_resize)

# Lưu
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_resize.jpg", anh_resize)


# ---------------------------------------------------------
# Resize theo tỷ lệ
# ---------------------------------------------------------

ti_le = 0.5

kich_thuoc_moi = (
    int(rong * ti_le),
    int(cao * ti_le)
)

anh_resize_50 = cv2.resize(anh, kich_thuoc_moi)

cv2.imshow("Anh Resize 50%", anh_resize_50)

cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_resize_50.jpg", anh_resize_50)


# =========================================================
# 6. VẼ HÌNH CƠ BẢN
# =========================================================

# Tạo bản sao của ảnh gốc
anh_ve = anh.copy()


# ---------------------------------------------------------
# Vẽ đường thẳng
# ---------------------------------------------------------

cv2.line(
    anh_ve,
    (50, 50),
    (400, 50),
    (255, 0, 0),
    3
)


# ---------------------------------------------------------
# Vẽ hình chữ nhật
# ---------------------------------------------------------

cv2.rectangle(
    anh_ve,
    (100, 100),
    (400, 300),
    (0, 255, 0),
    3
)


# ---------------------------------------------------------
# Vẽ hình tròn
# ---------------------------------------------------------

cv2.circle(
    anh_ve,
    (250, 200),
    70,
    (0, 0, 255),
    3
)


# =========================================================
# 7. THÊM VĂN BẢN
# =========================================================

cv2.putText(
    anh_ve,
    "OpenCV",
    (100, 350),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)


# Hiển thị ảnh cuối cùng
cv2.imshow("Anh ve hinh va van ban", anh_ve)

# Lưu ảnh
cv2.imwrite(r"C:\lab01\CV_Lab-1-5-\Lab1\anh_ve.jpg", anh_ve)


# =========================================================
# KẾT THÚC
# =========================================================

print("===================================")
print("ĐÃ HOÀN THÀNH BÀI THỰC HÀNH!")
print("Các ảnh đã được lưu trong thư mục.")
print("Nhấn phím bất kỳ trên cửa sổ ảnh để thoát.")
print("===================================")

cv2.waitKey(0)
cv2.destroyAllWindows()