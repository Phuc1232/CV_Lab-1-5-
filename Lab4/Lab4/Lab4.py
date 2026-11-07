import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import os

def bam_wavelet(duong_dan_anh, wavelet='db4', level=3):
    img = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (256, 256))
    
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LL = coeffs[0]
    
    median = np.median(LL)
    hash_code = (LL > median).flatten().astype(int)
    
    return hash_code

def tinh_khoang_cach_hamming(hash1, hash2):
    return np.sum(hash1 != hash2)

def agent_rule(distance, threshold):
    if distance <= threshold:
        return 1
    else:
        return 0

def khao_sat_cac_wavelet(data_test, nguong=150):
    # Khao sat va in bang so sanh hieu suat cac ho Wavelet
    print("\n" + "="*70)
    print(" BẢNG SO SÁNH HIỆU SUẤT CÁC PHƯƠNG PHÁP WAVELET (BÀI NÂNG CAO 1)")
    print("="*70)
    print(f"{'Ho Wavelet':<15} | {'Do Chinh Xac':<15} | {'Do Nhay':<10} | {'Do Dac Hieu':<12}")
    print("-" * 70)
    
    cac_loai_wavelet = ['haar', 'db4', 'sym4', 'coif1']
    
    for w in cac_loai_wavelet:
        nhan_tt = []
        nhan_dd = []
        for img1, img2, nhan in data_test:
            h1 = bam_wavelet(img1, wavelet=w)
            h2 = bam_wavelet(img2, wavelet=w)
            if h1 is not None and h2 is not None:
                kc = tinh_khoang_cach_hamming(h1, h2)
                nhan_tt.append(nhan)
                nhan_dd.append(agent_rule(kc, nguong))
        
        if len(nhan_tt) > 0:
            acc = accuracy_score(nhan_tt, nhan_dd)
            tn, fp, fn, tp = confusion_matrix(nhan_tt, nhan_dd, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"{w:<15} | {acc*100:>14.2f}% | {sens*100:>9.2f}% | {spec*100:>11.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    thu_muc_hien_tai = os.path.dirname(__file__)
    thu_muc_data = os.path.join(thu_muc_hien_tai, "dataset")
    
    # Gop 12 cap anh 
    data_test = [
        # 6 cap tuong dong (Nhan 1)
        (os.path.join(thu_muc_data, "anh1.jpg"), os.path.join(thu_muc_data, "anh1_challenge.jpg"), 1),
        (os.path.join(thu_muc_data, "anh2.jpg"), os.path.join(thu_muc_data, "anh2_challenge.jpg"), 1),
        (os.path.join(thu_muc_data, "anh1.jpg"), os.path.join(thu_muc_data, "anh1_mo.jpg"), 1),
        (os.path.join(thu_muc_data, "anh2.jpg"), os.path.join(thu_muc_data, "anh2_mo.jpg"), 1),
        (os.path.join(thu_muc_data, "anh1.jpg"), os.path.join(thu_muc_data, "anh1_xoay90.jpg"), 1),
        (os.path.join(thu_muc_data, "anh2.jpg"), os.path.join(thu_muc_data, "anh2_xoay90.jpg"), 1),
        
        # 6 cap khac biet (Nhan 0)
        (os.path.join(thu_muc_data, "anh1.jpg"), os.path.join(thu_muc_data, "anh2.jpg"), 0),
        (os.path.join(thu_muc_data, "anh1_challenge.jpg"), os.path.join(thu_muc_data, "anh2.jpg"), 0),
        (os.path.join(thu_muc_data, "anh1.jpg"), os.path.join(thu_muc_data, "anh2_challenge.jpg"), 0),
        (os.path.join(thu_muc_data, "anh1_mo.jpg"), os.path.join(thu_muc_data, "anh2_mo.jpg"), 0),
        (os.path.join(thu_muc_data, "anh1_xoay90.jpg"), os.path.join(thu_muc_data, "anh2_xoay90.jpg"), 0),
        (os.path.join(thu_muc_data, "anh1_mo.jpg"), os.path.join(thu_muc_data, "anh2_challenge.jpg"), 0)
    ]
    
    # 1. 
    print("DANG QUET KHAO SAT CAC PHUONG PHAP WAVELET...")
    khao_sat_cac_wavelet(data_test, nguong=150)
    
    # 2. 
    print("DANG TINH TOAN CHI TIET CHO HO WAVELET CHINH (db4)...")
    khoang_cach_list = []
    nhan_thuc_te = []
    
    for img1_path, img2_path, nhan in data_test:
        h1 = bam_wavelet(img1_path, wavelet='db4')
        h2 = bam_wavelet(img2_path, wavelet='db4')
        if h1 is not None and h2 is not None:
            dist = tinh_khoang_cach_hamming(h1, h2)
            khoang_cach_list.append(dist)
            nhan_thuc_te.append(nhan)

    # Ve bieu do ROC
    if len(nhan_thuc_te) > 0:
        diem_tuong_dong = [-d for d in khoang_cach_list] 
        fpr, tpr, thresholds = roc_curve(nhan_thuc_te, diem_tuong_dong)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Bieu do ROC - Wavelet Hash (db4)')
        plt.legend(loc="lower right")
        plt.show()