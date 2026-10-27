import cv2
import numpy as np
import pywt
import os
import glob

def bam_wavelet(duong_dan_anh, wavelet='db4'):
    # doc anh muc xam va resize cho dong nhat
    img = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
    img = cv2.resize(img, (256, 256))
    
    # bien doi wavelet va lay dai tan thap LL
    coeffs = pywt.wavedec2(img, wavelet, level=3)
    median = np.median(coeffs[0])
    
    # tao ma bam nhi phan dua tren trung vi
    return (coeffs[0] > median).flatten().astype(int)

if __name__ == "__main__":
    thu_muc_hien_tai = os.path.dirname(__file__)
    thu_muc_data = os.path.join(thu_muc_hien_tai, "dataset")
    danh_sach_anh = glob.glob(os.path.join(thu_muc_data, "*.jpg"))
    
    print("\n--- ỨNG DỤNG TÌM KIẾM ẢNH BẰNG WAVELET HASH ---")
    anh_can_tim = input("Nhập tên ảnh bạn muốn tìm (ví dụ: anh1.jpg): ")
    duong_dan_tim = os.path.join(thu_muc_data, anh_can_tim)
    
    hash_goc = bam_wavelet(duong_dan_tim)
    
    if hash_goc is not None:
        ket_qua = []
        print("Đang quét cơ sở dữ liệu...")
        for duong_dan in danh_sach_anh:
            if duong_dan != duong_dan_tim:
                hash_ss = bam_wavelet(duong_dan)
                if hash_ss is not None:
                    # tinh khoang cach hamming giua 2 ma bam
                    khoang_cach = np.sum(hash_goc != hash_ss)
                    ket_qua.append((khoang_cach, os.path.basename(duong_dan)))
        
        # sap xep de lay anh co khoang cach thap nhat
        ket_qua.sort()
        
        print(f"\n[KẾT QUẢ] Top 3 ảnh giống '{anh_can_tim}' nhất:")
        for kc, ten in ket_qua[:3]:
            print(f" -> {ten} (Khoảng cách Hamming: {kc})")
    else:
        print("Lỗi: Không tìm thấy ảnh đầu vào!")