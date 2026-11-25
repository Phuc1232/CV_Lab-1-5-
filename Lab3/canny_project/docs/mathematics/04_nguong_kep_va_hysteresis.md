# 04. Ngưỡng kép và nối cạnh bằng hysteresis (bước 4–5 của Canny)

Số liệu: [kiem_chung/ket_qua_phase2.txt](kiem_chung/ket_qua_phase2.txt), mục `3_nguong_opencv`, `4_nguong_skimage`, `6_don_dieu`.

## WHAT
Phân loại các pixel còn lại sau NMS thành **mạnh**, **yếu** và **loại bỏ**, rồi chỉ giữ cạnh yếu nào liên thông được với cạnh mạnh.

## WHY
Chỉ dùng một ngưỡng thì gặp thế khó:
- Ngưỡng cao làm cạnh thật bị đứt ở những đoạn gradient tạm yếu.
- Ngưỡng thấp làm nhiễu lọt vào.

Hysteresis dựa trên giả thiết **cạnh thật là đường liên tục, còn nhiễu là các đốm rời**. Ngưỡng cao chọn ra "hạt giống" chắc chắn đúng, ngưỡng thấp cho phép đường cạnh kéo dài qua các đoạn yếu.

## HOW / MATHEMATICS

### Ngưỡng kép
Với $N$ là đầu ra NMS và $T_l\le T_h$:

| Thư viện | Mạnh $S$ | Ứng viên $C=S\cup W$ | Kiểm chứng |
|---|---|---|---|
| OpenCV | $N>T_h$ | $N>T_l$ | $M=T_h=100$ thì 0 pixel, $T_h=99$ thì 32 pixel. Đoạn yếu có $M=T_l=50$ **không** được nối; $T_l=49$ thì được nối |
| skimage | $N\ge T_h$ | $N\ge T_l$ | Gọi trực tiếp hàm NMS: $M=T_l=4$ được giữ |

Trên thực tế, với dữ liệu `float`, quy ước `>=` của skimage hiếm khi có ý nghĩa. Trên bậc thang lý tưởng, bước chuẩn hóa bleed-over làm $M$ lệch $-8.9\times10^{-16}$, nên `canny(high_threshold=4)` trả về 0 pixel. **Bài học cho kiểm thử:** không bao giờ đặt ngưỡng đúng bằng giá trị dự kiến của dữ liệu; luôn đặt cách một khoảng an toàn.

OpenCV tự hoán đổi nếu $T_l>T_h$ (đã kiểm: `Canny(img,200,100) == Canny(img,100,200)`). skimage báo `ValueError`.

### Hysteresis là bài toán thành phần liên thông
Xét đồ thị $\mathcal G=(C,\mathcal E)$, trong đó hai pixel kề nhau khi là lân cận 8. Khi đó
$$E=\bigcup\{K:\ K\text{ là thành phần liên thông của }\mathcal G,\ K\cap S\neq\varnothing\}$$

**Mệnh đề (hai cách cài đặt cho cùng kết quả).** Gọi $E_1$ là tập pixel đến được từ $S$ bằng BFS/DFS trong $\mathcal G$, và $E_2$ là hợp các thành phần liên thông (gán nhãn) có chứa điểm mạnh. Khi đó $E_1=E_2$.

*Chứng minh.* Nếu $p\in E_1$ thì có đường đi trong $\mathcal G$ từ một $s\in S$ đến $p$, nên $p$ và $s$ thuộc cùng một thành phần $K$, và $K\cap S\ni s$; vậy $p\in E_2$. Ngược lại, nếu $p\in K$ với $K\cap S\ni s$ thì vì $K$ liên thông nên có đường đi từ $s$ đến $p$, và BFS xuất phát từ $s$ sẽ đến được $p$; vậy $p\in E_1$. ∎

Mệnh đề này là cơ sở cho cặp **Manual** (BFS bằng hàng đợi, dễ giải thích) và **Optimized** (`scipy.ndimage.label` với cấu trúc 3×3 toàn 1, đúng cách skimage làm). Hai bản phải cho kết quả **trùng từng pixel**, và có thể kiểm thử điều này bằng `array_equal`.

### Tính đơn điệu theo ngưỡng (dùng làm thuộc tính kiểm thử)
**Mệnh đề.** Cố định σ. Với $T_l\le T_l'$ và $T_h\le T_h'$ (luôn giữ $T_l\le T_h$), ta có $E(T_l',T_h)\subseteq E(T_l,T_h)$ và $E(T_l,T_h')\subseteq E(T_l,T_h)$.

*Chứng minh.*
- Tăng $T_h$ làm $S$ nhỏ đi trong khi $\mathcal G$ không đổi, nên tập thành phần chứa điểm mạnh nhỏ đi.
- Tăng $T_l$ làm $C'\subseteq C$, $S$ không đổi. Mỗi thành phần $K'$ của $\mathcal G'$ là đồ thị con liên thông của $\mathcal G$, nên nằm gọn trong một thành phần $K$ của $\mathcal G$. Nếu $K'$ chứa điểm mạnh thì $K$ cũng chứa điểm mạnh đó, nên $K'\subseteq K\subseteq E$. ∎

Kiểm chứng trên `camera` (σ = 1.5, lưới ngưỡng 10–80): **112 cặp, 0 vi phạm.**

Đơn điệu **không** đúng với σ: 2562 pixel có ở σ = 2 mà không có ở σ = 1, vì làm trơn dịch chuyển vị trí đỉnh gradient. Kiểm thử chỉ được dùng tính đơn điệu theo ngưỡng.

### Chọn ngưỡng
1. **Tỉ lệ:** Canny khuyến nghị $T_h/T_l\in[2,3]$. Lab3 dùng 1:2 (mặc định skimage) và 1:3.
2. **Theo nhiễu (có cơ sở toán học), bản cuối sau Phase 9.** Gọi $s=\sigma_n\sqrt{\sum(K_x*G_\sigma)^2}$ là độ lệch chuẩn **chính xác** của một thành phần gradient khi nhiễu trắng (không dùng xấp xỉ $1.6\sigma_n/\sigma^2$, vốn sai khi σ < 2). Trên nhiễu thuần, $M$ có phân phối Rayleigh nên $P(M>ks)=e^{-k^2/2}$.
   - **Ngưỡng cao = ngưỡng phổ quát** $k=\sqrt{2\ln N}$ (Donoho–Johnstone): số hạt giống nhiễu kỳ vọng là $N e^{-k^2/2}=1$. Ảnh 256² cho $k\approx4.71$.
   - **Ngưỡng thấp** $T_l=2s$: mật độ ứng viên nhiễu khoảng 10% và cụm nhỏ (đo bên dưới).
   - $\hat\sigma_n$ dùng Immerkær (1996) **có lặp loại pixel cạnh**: bỏ $|\nabla I|>4\sqrt{12}\hat\sigma$. Bản gốc ước lượng 0.018 thay vì 0.01 trên ảnh có cạnh; bản có loại cạnh cho 0.0099.

   **Lịch sử (ghi lại vì là kết quả âm có giá trị).** Mặc định ban đầu $T_h=3.5s$, $T_l=0.4T_h$ cho Precision chỉ 0.2–0.7 (X2, X6). Giả thuyết đầu tiên là *thấm*: $P(M>1.4s)=37\%$ gần ngưỡng thấm của lưới 8 lân cận (≈ 0.407), nên tập ứng viên nhiễu nối thành cụm khổng lồ. **Đã đo và bác bỏ:**

   | σ | $T_l/s$ | mật độ ứng viên (sau NMS) | cụm lớn nhất / ảnh |
   |---|---|---|---|
   | 1 | 0.8 | 0.372 | 0.0697 |
   | 1 | 1.4 | 0.262 | **0.0018** |
   | 1 | 2.2 | 0.078 | 0.0005 |
   | 2 | 1.4 | 0.168 | 0.0021 |

   Ở $T_l=1.4s$ không có cụm khổng lồ. Nguyên nhân thật là **so sánh bội**: $N e^{-3.5^2/2}\approx0.2\%\,N\approx140$ hạt giống giả trên ảnh 256², mỗi hạt kéo theo một cụm nhỏ. Ngưỡng phổ quát xử lý đúng nguyên nhân này: trên tập chọn tham số riêng (seed 100), F1 tăng từ 0.45 lên 0.77; trên X6 với σₙ = 0.15, F1 tăng từ 0.548 lên 0.988.
   - **Giới hạn đã đo:** thất bại với nhiễu muối tiêu (X6: F1 0.15) và texture (X3: 0.115), vì cả hai vi phạm giả thiết nhiễu trắng Gauss.
3. **Theo phân vị (skimage `use_quantiles`):** $T_h=$ phân vị $q_h$ của $M$. Bất biến với tương phản toàn cục, nên phù hợp ảnh tương phản thấp (P3). Nhược điểm: ảnh gần như không có cạnh vẫn bị ép phải có một tỉ lệ pixel cạnh nhất định.
4. **Otsu trên độ lớn gradient:** chọn $T_h$ tách histogram của $N>0$ thành 2 lớp, đặt $T_l=T_h/2$. Có cơ sở thống kê (tối đa phương sai giữa lớp), nhưng giả thiết histogram hai đỉnh thường không đúng.
5. **"Median ± 33%" phổ biến trên mạng:** $T_{l,h}=(1\mp0.33)\cdot\operatorname{median}(I)$. Công thức này dựa trên **cường độ ảnh** chứ không dựa trên gradient, nên thiếu cơ sở lý thuyết. Sẽ chỉ đưa vào làm baseline để đối chứng.

Các phương án 2–5 đã được so sánh trong thực nghiệm X6 ([../experiments/README.md](../experiments/README.md)). Otsu và phân vị bền nhất trên mọi loại suy giảm; ngưỡng theo nhiễu tốt nhất khi nhiễu đúng là Gauss; heuristic median cho F1 = 0 trên ảnh tương phản thấp.

## PSEUDOCODE (bản Manual)
```
strong = N > T_h ; cand = N > T_l               # quy ước OpenCV
out = zeros_like(strong); queue = toạ độ của strong
out[strong] = True
while queue không rỗng:
    p = queue.pop()
    for q in 8 lân cận của p:
        if q trong ảnh và cand[q] và not out[q]:
            out[q] = True; queue.push(q)
return out
```

## COMPLEXITY
- BFS: mỗi pixel vào hàng đợi tối đa 1 lần và xét 8 lân cận, nên $O(N)$ thời gian và $O(N)$ bộ nhớ hàng đợi trong trường hợp xấu nhất. Viết bằng Python thuần thì hằng số rất lớn (vài giây với ảnh 512²); đây chính là điểm cần so sánh với bản tối ưu.
- `ndi.label` (two-pass + union-find): $O(N\,\alpha(N))$, chạy bằng C.

## LIMITATIONS
- Nhiễu dạng đường hoặc vệt texture dài vẫn lọt qua nếu chạm một điểm mạnh.
- Một cạnh thật dài nhưng toàn bộ đều yếu (ảnh tương phản thấp) bị loại hoàn toàn.
- Ngưỡng tuyệt đối không bất biến với tương phản. Nhân ảnh với $a$ thì $M$ nhân với $a$.
