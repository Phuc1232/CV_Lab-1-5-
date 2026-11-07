name: cv-lab4-assistant
description: Trợ lý AI chuyên hỗ trợ xử lý ảnh bằng kỹ thuật Wavelet Hash và Luật Tác tử (Agent Rule) cho Lab 4.
---

# CHEATSHEET & MULTI-AGENT WORKFLOW: HỆ THỐNG BĂM ẢNH WAVELET HASH (LAB 4)

Tài liệu đặc tả kiến trúc điều phối Multi-Agent, sơ đồ luồng dữ liệu và giải thích chi tiết cơ chế toán học của hệ thống đánh giá tương đồng hình ảnh bằng **Wavelet Hash** kết hợp **Luật Tác Tử (Agent Rule)**.

---

## 1. CẤU TRÚC ĐIỀU PHỐI ĐA TÁC TỬ 
Hệ thống vận hành Lab 4 được quy định theo cơ chế phân quyền nghiêm ngặt nhằm đảm bảo chất lượng mã nguồn:

| Sub-Agent | Vai trò (Role) | Nhiệm vụ cốt lõi trong Lab 4 |
|:---:|:---|:---|
| **Sub-1** | **Architect** | Thiết kế luồng thuật toán Wavelet, quyết định sử dụng họ `db4` mức 3. Đặt ra ngưỡng Tác tử `threshold = 150`. |
| **Sub-2** | **Core Coder** | Hiện thực hóa mã nguồn Python (`Lab4.py`). Bắt buộc code theo mô hình thủ tục, không dùng OOP. Đặt tên biến 100% tiếng Việt `snake_case`. |
| **Sub-3** | **Data & CI Guardian** | Quản lý `dataset`. Chịu trách nhiệm **mở rộng dữ liệu (to lên) thành 12 cặp test** (bao gồm ảnh xoay, mờ, nhiễu) để kiểm thử thuật toán thay vì 4 cặp ban đầu. |
| **Sub-4** | **QA Auditor** | Kiểm tra đường cong ROC, đánh giá các chỉ số Accuracy, Sensitivity, Specificity. Đảm bảo AUC phản ánh đúng hiệu suất. |

---

## 2. SƠ ĐỒ LUỒNG DỮ LIỆU TỔNG THỂ 

    [ Đầu vào: Cặp Hình Ảnh (Từ Dataset 12 cặp) ]
                │
                ▼ (cv2.IMREAD_GRAYSCALE)
    [ Tiền xử lý: Chuẩn hóa kích thước 256x256 px ]
                │
                ▼ (pywt.wavedec2)
   ┌────────────────────────────────────────────────────────┐
   │         PHÂN TÍCH TẦN SỐ WAVELET TRANSFORM             │
   │  ├── Áp dụng họ Wavelet Daubechies (db4), level=3      │
   │  └── Trích xuất dải xấp xỉ tần số thấp (LL Band)       │
   └────────────────────────────┬───────────────────────────┘
                                │
                                ▼ (np.median)
   ┌────────────────────────────────────────────────────────┐
   │          LƯỢNG TỬ HÓA VÀ BĂM (QUANTIZATION)            │
   │  ├── Tính giá trị Trung vị (Median) của dải LL         │
   │  ├── Pixel > Median  -> Gán bit 1                      │
   │  └── Pixel <= Median -> Gán bit 0                      │
   │  └── Flatten thành mảng Vector Hash nhị phân           │
   └────────────────────────────┬───────────────────────────┘
                                │
                                ▼ 
   ┌────────────────────────────────────────────────────────┐
   │             ĐỐI SÁNH KHOẢNG CÁCH HAMMING               │
   │  └── Distance = ∑ (Hash1_i XOR Hash2_i)                │
   └────────────────────────────┬───────────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼ (Distance <= 150)         ▼ (Distance > 150)
        ┌───────────────────┐       ┌───────────────────┐
        │   LUẬT TÁC TỬ:    │       │   LUẬT TÁC TỬ:    │
        │   Trạng thái 1    │       │   Trạng thái 0    │
        │   (Tương đồng)    │       │    (Khác biệt)    │
        └───────────────────┘       └───────────────────┘

---

## 3. BẢNG TRA CỨU THÔNG SỐ HỆ THỐNG 

| Tham số / Biến | Giá trị | Ý nghĩa kỹ thuật & Lý do thiết lập |
|---|:---:|---|
| **`image_size`** | `256x256` | Kích thước chuẩn hóa đầu vào. Đảm bảo 2 ảnh khác kích thước vẫn sinh ra mảng Hash cùng độ dài để tính Hamming. |
| **`wavelet`** | `'db4'` | Họ Daubechies 4. Cân bằng hoàn hảo giữa khả năng nén cấu trúc không gian và khả năng kháng nhiễu cực tốt so với họ Haar. |
| **`level`** | `3` | Mức độ phân giải. Mức 3 giúp loại bỏ các nhiễu chi tiết (tần số cao) nhưng vẫn giữ được hình khối nhận dạng cốt lõi. |
| **`threshold`** | `150` | Ngưỡng ra quyết định của Luật Tác Tử (Agent Rule). Được cấu hình cứng để phân tách nhãn 0 và 1. |
| **`data_test`** | `12 cặp` | **Dataset đã được scale to lên 12 cặp** (6 tương đồng, 6 khác biệt) gồm ảnh gốc, xoay 90 độ, vẽ bậy, làm mờ để đánh giá ROC chính xác nhất. |

---

## 4. GIẢI THÍCH CHI TIẾT CÁC CỤM CHỨC NĂNG CỐT LÕI

### Cụm 1: Băm Wavelet & Lượng tử hóa trung vị
Khác với tính trung bình (Mean), hệ thống dùng **Trung vị (Median)** để băm ảnh. Về mặt toán học, giá trị trung vị có khả năng kháng các nhiễu biên độ lớn (outliers) như các vệt vẽ bậy (`challenge`) cực kỳ hiệu quả, giúp mã băm sinh ra ổn định.

### Cụm 2: So sánh Hamming & Cổng phân loại
Hệ thống tính tổng số bit sai biệt giữa hai mảng nhị phân phẳng. Nếu số lượng bit sai lệch vượt qua ngưỡng an toàn `150`, Tác tử sẽ đưa ra quyết định từ chối độ tương đồng.

### Cụm 3: Benchmarking (Bài Nâng Cao 1)
Hệ thống tích hợp cụm mã quét đa phương pháp (`haar`, `db4`, `sym4`, `coif1`) và xuất ra bảng Terminal (Tabular) để chứng minh tính ưu việt của mô hình thông qua 3 chỉ số:
- **Accuracy (Độ chính xác)**
- **Sensitivity (Độ nhạy - True Positive Rate)**
- **Specificity (Độ đặc hiệu - True Negative Rate)**

### Cụm 4: Phân tích đường cong ROC
Được nuôi bằng bộ 12 cặp dataset mở rộng, hệ thống sử dụng `sklearn.metrics` để vẽ đường cong đặc trưng hoạt động (ROC) và diện tích dưới đường cong (AUC), chứng minh hiệu suất phân loại của Tác tử vượt trội so với phán đoán ngẫu nhiên.