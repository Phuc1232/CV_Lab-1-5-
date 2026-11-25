# 03. Non-Maximum Suppression (bước 3 của Canny)

Số liệu: [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), mục `4_nguong_skimage`.

## WHAT
Chỉ giữ những pixel có độ lớn gradient là **cực đại địa phương theo hướng gradient**. Các pixel còn lại bị đặt về 0.

## WHY
Sau bước 2, mỗi cạnh tạo ra một "sườn đồi" độ lớn rộng vài pixel (rộng cỡ σ). Tiêu chí *đáp ứng đơn* của Canny đòi mỗi cạnh chỉ cho một đường. NMS tìm đỉnh của sườn đồi theo mặt cắt vuông góc với cạnh, tức theo hướng gradient.

## HOW / MATHEMATICS

Gọi $\mathbf u=(\cos\theta,\sin\theta)$ là hướng gradient tại $p$. Hai giá trị lân cận là $M^\pm=M(p\pm\mathbf u)$. Giữ $p$ khi
$$M(p)\ \succeq\ M^+\quad\text{và}\quad M(p)\ \succeq\ M^-$$
trong đó $\succeq$ là `>` hoặc `>=` tùy quy ước xử lý điểm hòa (xem bên dưới). Vì $p\pm\mathbf u$ thường không rơi đúng vào lưới pixel nên có hai cách lấy $M^\pm$.

### (A) Lượng tử hóa 4 hướng (Canny gốc, OpenCV)
Chia hướng gradient (mod 180°) thành 4 cung:

| Cung của θ | Hướng gradient | So với |
|---|---|---|
| $[-22.5°,22.5°)$ | ngang | trái, phải |
| $[22.5°,67.5°)$ | chéo | 2 lân cận chéo |
| $[67.5°,112.5°)$ | dọc | trên, dưới |
| $[112.5°,157.5°)$ | chéo ngược | 2 lân cận chéo còn lại |

**Thủ thuật của OpenCV (không cần `atan2`):** gradient gần ngang khi $|G_y|<\tan22.5°\cdot|G_x|$, gần dọc khi $|G_y|>\tan67.5°\cdot|G_x|$, với $\tan22.5°=\sqrt2-1\approx0.4142$ và $\tan67.5°=\sqrt2+1$. Các trường hợp còn lại là chéo, và hướng chéo nào được xác định bằng dấu của $G_xG_y$. OpenCV làm phép so sánh này bằng số nguyên dịch 15 bit, nên không có sai số dấu phẩy động.

### (B) Nội suy song tuyến tính (skimage)
Xét trường hợp $|G_y|\le|G_x|$ (gradient gần ngang hơn dọc), với $w=|G_y|/|G_x|\in[0,1]$:
$$M^+=(1-w)\,M(x+1,y)+w\,M(x+1,y+s),\qquad M^-=(1-w)\,M(x-1,y)+w\,M(x-1,y-s)$$
với $s=\operatorname{sign}(G_xG_y)$. Trường hợp $|G_y|>|G_x|$ làm tương tự, đổi vai trò hai trục. Cách này dùng hướng thật thay vì hướng đã lượng tử hóa, nên chính xác hơn với cạnh có góc lệch khỏi 0/45/90°.

skimage gộp ngưỡng thấp vào bước này: chỉ ghi lại pixel nào vừa là cực đại vừa có $M\ge T_l$.

### Xử lý điểm hòa (tie): khác biệt đã kiểm chứng
Cạnh bậc thang lý tưởng giữa cột 15 và 16 cho **hai** pixel liền nhau cùng $M=4h$:

| | Kết quả | Quy ước suy ra |
|---|---|---|
| skimage | giữ **cả cột 15 và 16** (cạnh dày 2 px) | `>=` ở cả hai phía |
| OpenCV (cạnh dọc) | chỉ giữ **cột 15** | một phía `>`, phía kia `>=` (bất đối xứng) nên cao nguyên rộng 2 chỉ giữ 1 |
| OpenCV (cạnh ngang) | chỉ giữ **hàng 15** | như trên |

**Hệ quả:**
1. NMS của skimage **không** đảm bảo cạnh dày đúng 1 pixel trên ảnh tổng hợp có cạnh rơi đúng giữa hai pixel. Đây đúng là loại ảnh sẽ dùng làm ground truth. Chỉ số đánh giá phải chịu được trường hợp này ([06](06_chi_so_danh_gia.md)).
2. Bản tự cài đặt phải tham số hóa quy ước xử lý điểm hòa thì mới tái tạo được từng thư viện.

### Tính chất
- Đầu ra là tập con của tập cực đại địa phương, nên $|N|\le|M>0|$.
- NMS không phụ thuộc thang của $M$: nhân $M$ với hằng số dương không đổi kết quả. Nhờ vậy, ngưỡng có thể quy đổi giữa các thang mà không ảnh hưởng đến bước 3.
- Kết quả phụ thuộc chuẩn L1/L2 chỉ qua giá trị $M$ được so sánh. Hướng được tính từ $(G_x,G_y)$ như nhau.

## PSEUDOCODE (bản lượng tử hóa)
```
cho mỗi pixel p không nằm trên viền:
    if M[p] == 0: N[p] = 0; tiếp tục
    chọn (q1, q2) = cặp lân cận theo cung của (Gx[p], Gy[p])
    N[p] = M[p] nếu M[p] > M[q1] và M[p] >= M[q2] (quy ước OpenCV)
           ngược lại 0
```
Bản vector hóa bằng NumPy: tính 4 mặt nạ cung, 8 mảng dịch chuyển của $M$, rồi kết hợp bằng `np.where`. Không cần vòng lặp Python.

## PARAMETERS
Phương pháp lấy lân cận (lượng tử hóa / nội suy), quy ước xử lý điểm hòa, xử lý viền (skimage loại hẳn viền 1 px).

## COMPLEXITY
$O(N)$ thời gian. Bản vector hóa cần thêm khoảng 8 mảng tạm cỡ $N$; bản vòng lặp (Numba/C) chỉ cần $O(1)$ bộ nhớ thêm.

## LIMITATIONS
- Ở **góc và chỗ giao nhau** (chữ T, chữ Y), hướng gradient không xác định rõ. NMS thường làm đứt cạnh ngay tại góc, khiến đường bao không khép kín. Đây là lý do P4a cần phép **closing** hình thái học.
- Lượng tử hóa 4 hướng gây răng cưa với cạnh xiên.
