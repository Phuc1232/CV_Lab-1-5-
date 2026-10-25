import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import os

def wavelet_hash(image_path, wavelet='db4', level=3):
    # Doc anh muc xam
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize de cac anh co cung kich thuoc truoc khi bam
    img = cv2.resize(img, (256, 256))
    
    # Bien doi Wavelet va lay dai tan so thap LL
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LL = coeffs[0]
    
    # Luong tu hoa bang trung vi
    median = np.median(LL)
    hash_code = (LL > median).flatten().astype(int)
    
    return hash_code

def tinh_khoang_cach_hamming(hash1, hash2):
    # Dem so bit khac nhau
    return np.sum(hash1 != hash2)

def agent_rule(distance, threshold):
    # Luat tac tu: quyet dinh dua tren nguong khoang cach
    if distance <= threshold:
        return 1  # 1 la phat hien tuong dong
    else:
        return 0  # 0 la khac biet

if __name__ == "__main__":
    # Lay duong dan thu muc hien tai de code chay duoc tren moi may
    thu_muc_hien_tai = os.path.dirname(__file__)
    
    # Danh sach cac cap anh de test (kem nhan thuc te)
    # Nhan 1: Giong nhau, Nhan 0: Khac nhau
    danh_sach_test = [
        (os.path.join(thu_muc_hien_tai, "dataset", "anh1.jpg"), os.path.join(thu_muc_hien_tai, "dataset", "anh1_challenge.jpg"), 1),
        (os.path.join(thu_muc_hien_tai, "dataset", "anh2.jpg"), os.path.join(thu_muc_hien_tai, "dataset", "anh2_challenge.jpg"), 1),
        (os.path.join(thu_muc_hien_tai, "dataset", "anh1.jpg"), os.path.join(thu_muc_hien_tai, "dataset", "anh2.jpg"), 0),
        (os.path.join(thu_muc_hien_tai, "dataset", "anh1_challenge.jpg"), os.path.join(thu_muc_hien_tai, "dataset", "anh2.jpg"), 0)
    ]
    
    khoang_cach_list = []
    nhan_thuc_te = []
    
    print("DANG XU LY...")
    for img1_path, img2_path, nhan in danh_sach_test:
        hash1 = wavelet_hash(img1_path)
        hash2 = wavelet_hash(img2_path)
        
        if hash1 is not None and hash2 is not None:
            dist = tinh_khoang_cach_hamming(hash1, hash2)
            khoang_cach_list.append(dist)
            nhan_thuc_te.append(nhan)
            print(f"Khoang cach Hamming: {dist}")

    # Ap dung Agent Rule
    THRESHOLD = 150 
    nhan_du_doan = [agent_rule(d, THRESHOLD) for d in khoang_cach_list]
    
    # Tinh toan chi so danh gia
    if len(nhan_thuc_te) > 0:
        acc = accuracy_score(nhan_thuc_te, nhan_du_doan)
        tn, fp, fn, tp = confusion_matrix(nhan_thuc_te, nhan_du_doan, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\n--- KET QUA ---")
        print(f"Do chinh xac: {acc * 100:.2f}%")
        print(f"Do nhay (Recall): {sensitivity * 100:.2f}%")
        print(f"Do dac hieu: {specificity * 100:.2f}%")
        
        # Ve bieu do ROC
        diem_tuong_dong = [-d for d in khoang_cach_list] 
        fpr, tpr, thresholds = roc_curve(nhan_thuc_te, diem_tuong_dong)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Bieu do ROC - Wavelet Hash')
        plt.legend(loc="lower right")
        plt.show()