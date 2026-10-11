import cv2
import numpy as np
import matplotlib.pyplot as plt


#PI 1. thay đổi độ sáng
#PI 2. Thay đổi độ tương phản
def lightAndContrast(img, light=0, contrast=1.0):
  # nếu để nguyên thì kiểu dữ liệu img sẽ là uint8 [0,255] và cộng vào vượt 256 sẽ tự động mod 256 và gán lại
  # vậy để co thể điều chỉnh mức độ ta cần chuyển về dãi rộng hơn
  imgLightAndContrast = contrast * img.astype(np.float32) + light

  # đưa các pixel < 0 về 0 và >255 về 255
  result = np.clip(imgLightAndContrast, 0, 255)

  return result.astype(np.uint8)


#PI 3. Biến đổi âm bản
def imgNegative(img):
  return 255 - img

#PI 4. Cắt ngưỡng
def imgBinary(img):
  imgBinary = img.copy()
  imgBinary[imgBinary < 128] = 0
  imgBinary[imgBinary >= 128] = 255
  return imgBinary

#PII 1. Lọc trung bình
def filterMean(img):
  return cv2.blur(img, (5,5))

#PII 2. Lọc Gaussian
def filterGaussian(img):
  return cv2.GaussianBlur(img, (5,5), 0)

#PII 3. Làm nét
def filterSharpen(img):
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ], dtype = np.float32)
  return cv2.filter2D(img, -1, kernel)

#PIII 1. Phát hiện cạnh
def detectionEdge(img):
  # Lọc theo từng hướng 
  imgSobelx = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
  imgSobely = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))

  # gộp lại
  imgSobel = cv2.convertScaleAbs(cv2.addWeighted(np.abs(imgSobelx), 0.5, np.abs(imgSobely), 0.5, 0))

  kernelPrewittx = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
  ])

  kernelPrewitty = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
  ])
  # Lọc theo từng hường 
  imgPrewittx = cv2.filter2D(img, cv2.CV_64F, kernelPrewittx)
  imgPrewitty = cv2.filter2D(img, cv2.CV_64F, kernelPrewitty)

  #gộp lại
  imgPrewitt = cv2.convertScaleAbs(cv2.addWeighted(np.abs(imgPrewittx), 0.5, np.abs(imgPrewitty), 0.5, 0))

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

  axes[0].imshow(imgSobel, cmap='gray')
  axes[0].set_title("Phát hiện cạnh với Sobel")
  axes[0].axis("off")

  axes[1].imshow(imgPrewitt, cmap='gray')
  axes[1].set_title("Phát hiện cạnh với Prewitt")
  axes[1].axis("off")

  plt.show()

#PIII 2. tự thiết kế kernel
def designKernelAndFilter(img):
  kernel1 = np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
  ], dtype = np.float32)/16

  kernel2 = np.array([
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
  ], dtype = np.float32)/5

  kernel3 = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ], dtype = np.float32)

  kernel4 = np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
  ], dtype = np.float32)

  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

  axes[0,0].imshow(cv2.filter2D(img, -1, kernel1), cmap='gray')
  axes[0,0].set_title("Làm mờ tự nhiên")
  axes[0,0].axis("off")

  axes[0,1].imshow(cv2.filter2D(img, -1, kernel2), cmap='gray')
  axes[0,1].set_title("làm mờ theo chuyển động chéo")
  axes[0,1].axis("off")

  axes[1,0].imshow(cv2.filter2D(img, -1, kernel3), cmap='gray')
  axes[1,0].set_title("Làm sắc nét hình tiêu chuẩn")
  axes[1,0].axis("off")

  axes[1,1].imshow(cv2.filter2D(img, -1, kernel4), cmap='gray')
  axes[1,1].set_title("Làm nét hình mạnh")
  axes[1,1].axis("off")

  plt.show()

#PIII 3. So sánh các bộ lọc
#PIII 4. Lọc với các vộ lọc phi tuyến tính
def compareFilter(img, numNoise):
  imgNoisy = img.copy()
  # Thêm ngẫu nhiên các điểm nhiễu trắng/đen
  for i in range(numNoise):
      yCoord = np.random.randint(0, img.shape[0])
      xCoord = np.random.randint(0, img.shape[1])
      imgNoisy[yCoord, xCoord] = 255 if np.random.rand() > 0.5 else 0

  # Thực thi 4 loại lọc trên cùng một ảnh có nhiễu
  mean = cv2.blur(imgNoisy, (5,5))
  gaussian = cv2.GaussianBlur(imgNoisy, (5,5), 0)
  median = cv2.medianBlur(imgNoisy, 5)
  bilateral = cv2.bilateralFilter(imgNoisy, 9, 75, 75)

  # Hiển thị so sánh 
  fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
  
  axes[0].imshow(imgNoisy, cmap='gray' )
  axes[0].set_title("1. Ảnh có nhiễu/biên")
  axes[0].axis("off")

  axes[1].imshow(mean, cmap='gray' )
  axes[1].set_title("2. Lọc trung bình")
  axes[1].axis("off")

  axes[2].imshow(gaussian, cmap='gray')
  axes[2].set_title("3. Lọc Gaussian")
  axes[2].axis("off")

  axes[3].imshow(median, cmap='gray')
  axes[3].set_title("4. Lọc Median")
  axes[3].axis("off")

  axes[4].imshow(bilateral, cmap='gray')
  axes[4].set_title("5. Lọc Bilateral")
  axes[4].axis("off")

  plt.show()

def main():
  # Đọc ảnh và chuyển không gian màu
  path = input("Nhập đường dẫn đến hình")
  imgRGB = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

  # thực thi phần 1
  print("I Toán tử điểm ảnh\n")
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    
  axes[0,0].imshow(lightAndContrast(imgRGB, 50))
  axes[0,0].set_title("Tăng độ sáng")
  axes[0,0].axis("off")

  axes[0,1].imshow(lightAndContrast(imgRGB, -50))
  axes[0,1].set_title("Giảm độ sáng")
  axes[0,1].axis("off")

  axes[1,0].imshow(lightAndContrast(imgRGB, 0, 1.5))
  axes[1,0].set_title("Tăng độ tương phản")
  axes[1,0].axis("off")

  axes[1,1].imshow(lightAndContrast(imgRGB, 0, 0.95))
  axes[1,1].set_title("Giảm độ tương phản")
  axes[1,1].axis("off")

  axes[2,0].imshow(imgNegative(imgGray), cmap='gray')
  axes[2,0].set_title("Ảnh âm bản")
  axes[2,0].axis("off")

  axes[2,1].imshow(imgBinary(imgGray), cmap='gray')
  axes[2,1].set_title("Ảnh cắt ngưỡng")
  axes[2,1].axis("off")

  plt.show()

  # thực thi phần 2
  print("II Lọc tuyến tính\n")
  fig1, axes1 = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))

  axes1[0].imshow(imgGray, cmap='gray')
  axes1[0].set_title("Ảnh gốc")
  axes1[0].axis("off")

  axes1[1].imshow(filterMean(imgGray), cmap='gray')
  axes1[1].set_title("Sau khi lọc trung bình")
  axes1[1].axis("off")

  axes1[2].imshow(filterGaussian(imgGray), cmap='gray')
  axes1[2].set_title("Sau khi lọc Gaussian")
  axes1[2].axis("off")

  axes1[3].imshow(filterSharpen(imgGray), cmap='gray')
  axes1[3].set_title("Sau khi làm nét")
  axes1[3].axis("off")

  plt.show()


  # thực thi phần 3
  print("III Bài tập nâng cao\n")
  detectionEdge(imgGray)
  designKernelAndFilter(imgGray)
  numNoise = imgGray.size //100
  compareFilter(imgGray, numNoise)


main()
