import cv2
import os

# Lay duong dan thu muc dataset hien tai
thu_muc_hien_tai = os.path.dirname(__file__)
thu_muc_data = os.path.join(thu_muc_hien_tai, "dataset")

# 2 file anh goc de nhan ban
danh_sach_anh = ["anh1.jpg", "anh2.jpg"]

print("Dang tao them anh xoay va nhieu...")

for ten_anh in danh_sach_anh:
    duong_dan_anh = os.path.join(thu_muc_data, ten_anh)
    img = cv2.imread(duong_dan_anh)
    
    if img is not None:
        # Tao anh xoay 90 do
        anh_xoay_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(thu_muc_data, ten_anh.replace(".jpg", "_xoay90.jpg")), anh_xoay_90)
        
        # Tao anh lam mo (Blur)
        anh_mo = cv2.GaussianBlur(img, (25, 25), 0)
        cv2.imwrite(os.path.join(thu_muc_data, ten_anh.replace(".jpg", "_mo.jpg")), anh_mo)

print("Tao anh hoan tat!")