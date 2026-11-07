import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. HÀM HIỂN THỊ VÀ TRỰC QUAN HÓA (DISPLAY)
# ==========================================

def displayIMG(titles_list, images_list, w = 10, h = 6, histogram = False, forceGray = False, nameWindown = ""):
    """
    Hàm hiển thị danh sách nhiều ảnh cạnh nhau một cách linh hoạt.
    
    Args:
        titles (list): Danh sách tiêu đề cho từng ảnh.
        images (list): Danh sách các mảng numpy chứa dữ liệu ảnh.
        w (int): Chiều rộng khung hình chung.
        h (int): Chiều cao khung hình chung.
        histogram (bool): Hiển thị thêm histogram (nếu cần).
        force_gray (bool): Buộc hiển thị tất cả dưới dạng ảnh xám (nếu cần thiết).
    """
    colum = len(images_list)
    # Chặn và cảnh báo 2 trường hợp đầu vào rỗng và số lượng ảnh và title không khớp nhau 
    if colum == 0:
        print("Không có hình ảnh đầu vào!")
        return
    elif len(titles_list) != colum:
        print("Số lượng title và ảnh không khớp nhau!")
        return
    
    row = 2 if histogram else 1
    fig, axes = plt.subplots(row, colum, figsize=(w, h), squeeze=False)
    # --- ĐẶT TIÊU ĐỀ CHO CỬA SỔ HIỂN THỊ (WINDOW TITLE) ---
    try:
        fig.canvas.manager.set_window_title(nameWindown)
    except AttributeError:
        pass # Phòng hờ trường hợp môi trường chạy không hỗ trợ window manager (như một số notebook online)
        
    colors = ('r', 'g', 'b') # Định nghĩa màu vẽ histogram theo chuẩn ảnh RGB của plt

    for i in range(colum):
        current_img = images_list[i]
        # Xử lý None trong list
        if current_img is None:
            # Xử lý khi gặp None: Hiển thị khung trống kèm thông báo lỗi
            axes[0, i].text(0.5, 0.5, "Lỗi: đọc lại vui lòng đọc lại [Cảnh báo]", 
                            color='red', fontsize=10, ha='center', va='center', 
                            transform=axes[0, i].transAxes)
            axes[0, i].set_title(titles_list[i], fontsize=12, fontweight='bold')
            axes[0, i].axis("off")
            
            # Nếu có hàng histogram ở dưới, cũng tắt luôn để tránh lỗi vẽ
            if histogram:
                axes[1, i].text(0.5, 0.5, "N/A", color='gray', fontsize=10, ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis("off")
            continue # Chuyển sang cột tiếp theo
        # --- HÀNG 1: HIỂN THỊ ẢNH ---
        if forceGray:
            # hiển thị ảnh xám
            axes[0, i].imshow(current_img, cmap="gray")
        else:
            # hiển thị ảnh màu 3 kênh RGB tự động hiển thị chuẩn
            axes[0, i].imshow(current_img)
        axes[0, i].set_title(titles_list[i], fontsize=12, fontweight='bold')
        axes[0, i].axis("off")

        if histogram:
            # --- HÀNG 2: HIỂN THỊ HISTOGRAM TƯƠNG ỨNG ---
            # Tách các kênh màu của ảnh hiện tại để vẽ histogram
            if forceGray:
                hist = cv2.calcHist([current_img], [0], None, [256], [0, 256])
                axes[1, i].plot(hist, color="gray", alpha=0.8)
            else:
                for c, col in enumerate(colors):
                    # Tính toán histogram cho từng kênh màu R, G, B
                    hist = cv2.calcHist([current_img], [c], None, [256], [0, 256])
                    axes[1, i].plot(hist, color=col, alpha=0.8)
                
            axes[1, i].set_xlim([0, 256])
            axes[1, i].grid(True, linestyle='--', alpha=0.5)
            
            # Gắn nhãn cho trục của cột đầu tiên cho gọn đồ thị
            if i == 0:
                axes[1, i].set_ylabel("Tần suất (Số lượng)")
            axes[1, i].set_xlabel("Giá trị Pixel")
            axes[1, i].set_title(f"Histogram {titles_list[i]}", fontsize=10)

    plt.tight_layout()
    plt.show()

# ==========================================
# 2. BỘ XỬ LÝ ĐIỂM ẢNH (POINT PROCESSING)
# ==========================================

def lightAndContrast(img, light=0, contrast=1.0):
    """
      Điều chỉnh độ sáng và độ tương phản của ảnh sử dụng phép toán vector hóa.
      
      Args:
          img (np.ndarray): Ảnh đầu vào (có thể là ảnh xám 2D hoặc ảnh màu RGB/BGR 3D).
          light (int/float): Giá trị cộng thêm để thay đổi độ sáng (-255 đến 255).
          contrast (float): Hệ số nhân để thay đổi độ tương phản (> 0).
          
      Returns:
          np.ndarray: Ảnh sau khi biến đổi (kiểu uint8).
      """
    img_processed = contrast * img.astype(np.float32) + light
    result = np.clip(img_processed, 0, 255)
    return result.astype(np.uint8)


def imgNative(img):
    """
      Thực hiện biến đổi âm bản cho ảnh xám hoặc ảnh màu.
          s = 255 - r
      Trong đó:
          r: Giá trị cường độ sáng của pixel ở ảnh gốc.
          s: Giá trị cường độ sáng của pixel sau khi biến đổi.
      
      Args:
          img (np.ndarray): Ảnh đầu vào (có thể là ảnh xám 2D hoặc ảnh màu RGB/BGR 3D).
          
      Returns:
          np.ndarray: Ảnh âm bản sau khi đảo ngược giá trị pixel (kiểu uint8).
    """
    return 255 - img


def imgBinary(img, threshold=128):
    """
      Thực hiện cắt ngưỡng nhị phân hóa (Binary Thresholding) trên ảnh xám.
      
      Lý do cần đầu vào Grayscale:
          Phép cắt ngưỡng chỉ dựa trên một chiều cường độ sáng (intensity) duy nhất. 
          Nếu dùng ảnh màu 3 kênh (RGB), việc cắt ngưỡng độc lập trên từng kênh sẽ 
          làm biến dạng màu sắc và không tạo ra ảnh nhị phân đen/trắng thuần túy.
      
      Args:
          imgGray (np.ndarray): Ảnh đầu vào (Bắt buộc phải là ảnh xám 2D - Grayscale).
          threshold (int): Giá trị ngưỡng phân chia, từ 0 đến 255 (mặc định là 128).
          
      Returns:
          np.ndarray: Ảnh nhị phân kết quả, chỉ chứa đúng 2 giá trị 0 (Đen) và 255 (Trắng).
    """
    if len(img.shape) != 2:
        print("[Cảnh báo] Đầu vào ảnh nên là kênh màu Grayscale vì kênh màu khác có thể làm biến đổi màu sắc và không phù hợp với mục đích ảnh nhị phân!")
        return None
    elif threshold <= 0 or threshold >= 255:
        print("Giá trị cắt ngưỡng cần nằm trong [0,255] để được ảnh nhị phân")
        
    img_binary = img.copy()
    img_binary[img_binary < threshold] = 0
    img_binary[img_binary >= threshold] = 255
    return img_binary

# ==========================================
# 3. BỘ LỌC KHÔNG GIAN VÀ KHỬ NHIỄU (FILTERS)
# ==========================================

def filterMean(img, k, border=cv2.BORDER_DEFAULT):
    """
    Thực hiện lọc trung bình (Mean Filter) để làm mờ và khử nhiễu.
        
        Args:
            img (np.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).
            k: kích thước kernel
            border: cách xử lý pixel biên của ảnh
            
        Returns:
            np.ndarray: Ảnh sau khi được làm mờ qua bộ lọc trung bình kích thước k.
    """
     #Kiểm tra k <=0 vì kích thước 1 kernel không <=0
    if k <= 0:
        print("[Cảnh báo] kích thước cần > 0!")
        return None 
    # Tránh hiện tượng dịch chuyển không gian với k chẵn
    elif k%2 == 0:
        print("Kích thước kernel k phải là số nguyên dương lớn hơn 0.")
    return cv2.blur(img, (k, k), borderType=border)


def filterGaussian(img, k, sigma, border=cv2.BORDER_DEFAULT):
    """
    Thực hiện lọc Gaussian để làm mờ và khử nhiễu tự nhiên.
        
        Args:
            img (np.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).
            k (int): Kích thước kernel (khuyến nghị số lẻ, vd: 3, 5, 7).
            sigma (float): Độ lệch chuẩn của phân phối Gaussian.
            
        Returns:
            np.ndarray: Ảnh sau khi qua bộ lọc Gaussian.
    """
    if k != 0:
        if k%2==0:
            print("[Cảnh báo] Kích thước kernel cần lẻ và lớn hơn 0!")
            return None
       
    return cv2.GaussianBlur(img, (k, k), sigma, borderType=border)


def filterSharpen(img, kernel=None):
    """
    Thực hiện làm nét ảnh bằng cách tự thiết kế Custom Kernel.
        
        Args:
            img (np.ndarray): Ảnh đầu vào (ảnh xám hoặc ảnh màu).
            kernel (np.ndarray, optional): Ma trận kernel tùy chỉnh dùng để làm nét. 
                                           Mặc định là None (sẽ tự động dùng kernel chuẩn 3x3 với trọng số làm nét tiêu chuẩn).
            
        Returns:
            np.ndarray: Ảnh sau khi được làm sắc nét.
    """
    if kernel is None:
        kernel = np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ], dtype = np.float32)
    else:

        # KIỂM TRA TÍNH HỢP LỆ (INPUT VALIDATION)
        # Kiểm tra xem kernel có phải là mảng NumPy không
        if not isinstance(kernel, np.ndarray):
            print("[Cảnh báo] Kernel phải là một đối tượng numpy.ndarray!")
            return None

        # Kiểm tra số chiều (phải là ma trận 2D)
        if kernel.ndim != 2:
            print(f"[Cảnh báo] Kernel phải có 2 chiều (ndim=2), hiện tại là {kernel.ndim} chiều.")
            return None

        # Kiểm tra kích thước hàng và cột có phải là số lẻ không
        rows, cols = kernel.shape
        if rows % 2 == 0 or cols % 2 == 0:
            print(f"[Cảnh báo] Kích thước kernel phải là số lẻ (ví dụ 3x3, 5x5), hiện tại là {rows}x{cols}.")
            return None

        # Kiểm tra tổng các phần tử (Khuyến nghị tổng bằng 1.0 để bảo toàn độ sáng tổng thể của ảnh)
        kernel_sum = np.sum(kernel)
        if not np.isclose(kernel_sum, 1.0, atol=1e-5):
            print(f"[Cảnh báo] Tổng các phần tử của kernel hiện tại bằng {kernel_sum:.2f} (Khuyến nghị bằng 1.0 để giữ nguyên độ sáng ảnh).")
            return None
    return cv2.filter2D(img, -1, kernel)


def detectEdges(imgGray, method="sobel", ksize=3):
    """ 
    Phát hiện biên sử dụng toán tử Sobel hoặc Prewitt.
    
    Args:
        imgGray (np.ndarray): Ảnh đầu vào (bắt buộc là ảnh xám 2D).
        method (str): Phương pháp chọn ("sobel" hoặc "prewitt").
        ksize (int): Kích thước kernel (thường là 3).
        
    Returns:
        np.ndarray: Ảnh biên tổng hợp sau khi xử lý.
    """
    if imgGray.ndim != 2:
        print("[Cảnh báo] Đầu vào imgGray phải là ảnh xám 2D (Grayscale)!")
        return None
        
    method_lower = method.lower()
    if method_lower not in ["sobel", "prewitt"]:
        print(f"[Cảnh báo] Phương pháp '{method}' không hợp lệ! Chỉ hỗ trợ 'sobel' hoặc 'prewitt'.")
        return None
        
    # Kiểm tra ksize chung phải là số lẻ (hoặc -1 riêng cho Scharr)
    if ksize != -1 and (ksize <= 0 or ksize % 2 == 0):
        print(f"[Cảnh báo] Kích thước ksize = {ksize} không hợp lệ! Bắt buộc phải là số lẻ dương hoặc -1.")
        return None
    
    if method_lower == "sobel":
        grad_x = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=ksize)
    elif method_lower == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(imgGray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(imgGray, cv2.CV_64F, kernel_y)
    else:
        raise ValueError("Phương pháp không hợp lệ! Hãy chọn 'sobel' hoặc 'prewitt'.")

    # Tính độ lớn Gradient tổng hợp (Magnitude)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def designKernelAndFilter(imgGray):
    """
        Hàm thiết kế các Custom Kernel tùy chỉnh và trực quan hóa kết quả qua lưới 2x2.
        
        Args:
            img (np.ndarray): Ảnh đầu vào (khuyến nghị ảnh xám 2D).
    """
    # 1. Blur Gaussian 3x3 thủ công
    kernel1 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    # 2. Blur chuyển động chéo (Motion Blur)
    kernel2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=np.float32) / 5.0

    # 3. Làm nét tiêu chuẩn
    kernel3 = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    # 4. Làm nét mạnh (High-pass filter)
    kernel4 = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    images_list = [
        imgGray,
        cv2.filter2D(imgGray, -1, kernel1),
        cv2.filter2D(imgGray, -1, kernel2),
        cv2.filter2D(imgGray, -1, kernel3),
        cv2.filter2D(imgGray, -1, kernel4)
    ]
    titles_list = [
        "Ảnh gốc xám",
        "Làm mờ Gauss (Custom 3x3)",
        "Làm mờ chuyển động chéo",
        "Làm nét tiêu chuẩn",
        "Làm nét mạnh"
    ]

    displayIMG(titles_list, images_list, 20, 8, forceGray=True, nameWindown="PIII 2. Thiết kế ma trận Kernel")

# ==========================================
# 4. TẠO NHIỄU VÀ SO SÁNH KHỬ NHIỄU
# ==========================================

def addGaussianNoise(img, mean=0, var=25):
    """
        Thêm nhiễu Gauss (Additive Gaussian Noise) vào ảnh.
        
        Args:
            img (np.ndarray): Ảnh đầu vào.
            mean (float): Giá trị trung bình của nhiễu.
            var (float): Phương sai (variance) của nhiễu.
        Returns:
                np.ndarray: Ảnh sau khi đã được gán nhiễu.
    """
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    imgNoisy = img.astype(np.float32) + gaussian_noise
    return np.clip(imgNoisy, 0, 255).astype(np.uint8)


def addSaltNoise(img, ratio = 0.01):
    """
    Thêm nhiễu muối tiêu (Salt-and-Pepper Noise) vào ảnh với tỷ lệ tùy chỉnh.
    
    Args:
        img (np.ndarray): Ảnh đầu vào.
        ratio (float): Tỷ lệ pixel bị nhiễu so với tổng số pixel (mặc định là 0.01 tương đương 1%).
                       Ví dụ: 0.02 là 2%, 0.05 là 5%.
        
    Returns:
        np.ndarray: Ảnh sau khi đã được gán nhiễu.
    """
    numNoise = int(img.size * ratio)
    imgNoisy = img.copy()
    for _ in range(numNoise):
        yCoord = np.random.randint(0, img.shape[0])
        xCoord = np.random.randint(0, img.shape[1])
        imgNoisy[yCoord, xCoord] = 255 if np.random.rand() > 0.5 else 0
    return imgNoisy


def compareFilter(imgGray, typeNoise="salt"):
    """
       So sánh trực quan 4 bộ lọc không gian trên ảnh bị gán nhiễu nhân tạo.
       
       Args:
           img (np.ndarray): Ảnh đầu vào (khuyến nghị ảnh xám 2D).
           typeNoise (str): Loại nhiễu ("salt" hoặc "gaussian").
    """ 
    if typeNoise.lower() == "salt":
        imgNoisy = addSaltNoise(imgGray)
        noise_name = "Nhiễu muối tiêu"
    else:
        imgNoisy = addGaussianNoise(imgGray, mean=0, var=25)
        noise_name = "Nhiễu Gaussian"

    images_list = [
        imgGray,
        imgNoisy,
        cv2.blur(imgNoisy, (5, 5)),
        cv2.GaussianBlur(imgNoisy, (5, 5), 0.85),
        cv2.medianBlur(imgNoisy, 5),
        cv2.bilateralFilter(imgNoisy, 9, 75, 75)
    ]

    titles_list = [
        "Ảnh gốc xám",
        f"Ảnh bị {noise_name}",
        "Mean Filter",
        "Gaussian Filter",
        "Median Filter",
        "Bilateral Filter"
    ]

    displayIMG(titles_list, images_list, 22, 6, forceGray=True, nameWindown="PIII 3. Khao sát so sánh hiệu quả khử nhiễu")

# ==========================================
# 5. HÀM ĐIỀU KHIỂN CHÍNH (MAIN EXECUTION)
# ==========================================

def main():
    # 1. Nhập đường dẫn ảnh từ bàn phím và kiểm tra tính hợp lệ
    path = input("Nhập đường dẫn tới file ảnh của bạn: ").strip()
    
    if not os.path.exists(path):
        print(f" Lỗi: File không tồn tại tại đường dẫn '{path}'")
        return

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(" Lỗi: Không thể đọc định dạng ảnh này. Hãy kiểm tra lại file!")
        return

    print(f" Đọc ảnh thành công từ: {path}")

    # Chuyển đổi không gian màu chuẩn
    imgRGB = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------------------
    #  Thay đổi độ sáng và độ tương phản (có Histogram)
    # -------------------------------------------------------------
    print("\n--- PI 1&2 Thử nghiệm Độ sáng & Độ tương phản ---")
    list_bright_contrast = [
        imgRGB,
        lightAndContrast(imgRGB, light=50),
        lightAndContrast(imgRGB, light=-50),
        lightAndContrast(imgRGB, contrast=1.5),
        lightAndContrast(imgRGB, contrast=0.95)
    ]
    titles_bright_contrast = [
        "Ảnh gốc", "Tăng sáng (+50)", "Giảm sáng (-50)",
        "Tăng tương phản (x1.5)", "Giảm tương phản (x0.95)"
    ]
    displayIMG(titles_bright_contrast, list_bright_contrast, 20, 8, histogram=True, nameWindown="PI 1&2 Thử nghiệm Độ sáng & Độ tương phản")

    # -------------------------------------------------------------
    #  Ảnh Âm bản (Negative)
    # -------------------------------------------------------------
    print("--- PI 3. Thử nghiệm Ảnh Âm bản ---")
    displayIMG(["Ảnh gốc", "Ảnh âm bản"], [imgRGB, imgNative(imgRGB)], 10, 5, histogram=True, nameWindown="PI 3. Thử nghiệm Ảnh Âm bản")

    # -------------------------------------------------------------
    # Cắt ngưỡng Nhị phân hóa (Binary Thresholding)
    # -------------------------------------------------------------
    print("--- PI 4. Thử nghiệm Cắt ngưỡng Nhị phân ---")
    list_binary = [
        imgGray,
        imgBinary(imgGray, threshold=64),
        imgBinary(imgGray, threshold=128),
        imgBinary(imgGray, threshold=192)
    ]
    titles_binary = ["Ảnh gốc", "Cắt ngưỡng (T=64)", "Cắt ngưỡng (T=128)", "Cắt ngưỡng (T=192)"]
    displayIMG(titles_binary, list_binary, 16, 4, forceGray=True, nameWindown="PI 4. Thử nghiệm Cắt ngưỡng Nhị phân")

    # -------------------------------------------------------------
    # Lọc Trung bình (Mean Filter) với các kiểu Border
    # -------------------------------------------------------------
    print("--- PII 1. Thử nghiệm Lọc Trung bình & Kiểu biên ---")
    list_mean = [
        imgGray,
        filterMean(imgGray, 9, cv2.BORDER_DEFAULT),
        filterMean(imgGray, 5, cv2.BORDER_DEFAULT),
        filterMean(imgGray, 5, cv2.BORDER_CONSTANT),
        filterMean(imgGray, 5, cv2.BORDER_REPLICATE),
        filterMean(imgGray, 5, cv2.BORDER_REFLECT)
    ]
    titles_mean = [
        "Ảnh gốc xám", "Lọc tb (k=9)", "Lọc tb (k=5)",
        "Border Constant", "Border Replicate", "Border Reflect"
    ]
    displayIMG(titles_mean, list_mean, 20, 8, forceGray=True, nameWindown="PII 1. Thử nghiệm Lọc Trung bình & Kiểu biên")

    # -------------------------------------------------------------
    # Lọc Gaussian với Sigma khác nhau
    # -------------------------------------------------------------
    print("--- PII 2. Thử nghiệm Lọc Gaussian ---")
    list_gaussian = [
        imgGray,
        filterGaussian(imgGray, 0, 1.25),
        filterGaussian(imgGray, 5, 0.25),
        filterGaussian(imgGray, 5, 0.5),
        filterGaussian(imgGray, 5, 0.75),
        filterGaussian(imgGray, 5, 1.25)
    ]
    titles_gaussian = [
        "Ảnh gốc xám", "Gaussian (k=0, sigma=1.25)", "Gaussian (sigma=0.25, k=5)",
        "Gaussian (sigma=0.5)", "Gaussian (sigma=0.75)", "Gaussian (sigma=1.25)"
    ]
    displayIMG(titles_gaussian, list_gaussian, 20, 8, forceGray=True, nameWindown="PII 2. Thử nghiệm Lọc Gaussian")

    # -------------------------------------------------------------
    # Làm sắc nét ảnh (Sharpening)
    # -------------------------------------------------------------
    print("--- PII 3. Thử nghiệm Làm nét ---")
    displayIMG(["Ảnh gốc", "Làm nét tiêu chuẩn"], [imgRGB, filterSharpen(imgRGB)], 10, 5)

    # -------------------------------------------------------------
    # Phát hiện Biên (Prewitt vs Sobel)
    # -------------------------------------------------------------
    print("--- PIII 1. Thử nghiệm Phát hiện biên cạnh ---")
    list_edges = [
        imgGray,
        detectEdges(imgGray, method="prewitt"),
        detectEdges(imgGray, method="sobel")
    ]
    titles_edges = ["Ảnh xám gốc", "Phát hiện biên (Prewitt)", "Phát hiện biên (Sobel)"]
    displayIMG(titles_edges, list_edges, 15, 5, forceGray=True, nameWindown="PII 3. Thử nghiệm Làm nét")

    # -------------------------------------------------------------
    # Thiết kế Custom Kernels
    # -------------------------------------------------------------
    print("--- PIII 2. Thiết kế ma trận Kernel  ---")
    designKernelAndFilter(imgGray)

    # -------------------------------------------------------------
    # Đánh giá & So sánh Khử nhiễu
    # -------------------------------------------------------------
    print("--- PIII 3&4. Khảo sát So sánh Hiệu quả Khử nhiễu ---")
    compareFilter(imgGray)      # Nhiễu Muối tiêu
    compareFilter(imgGray, typeNoise="gaussian")  # Nhiễu Gauss

    print("\n🎉 Hoàn thành toàn bộ quá trình thực thi!")

if __name__ == "__main__":
    main()