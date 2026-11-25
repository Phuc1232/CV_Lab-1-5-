"""
Main Program - Digital Image Processing Lab 02.

Điều khiển chương trình thông qua menu:
1. Toán tử điểm ảnh.
2. Lọc tuyến tính.
3. Bài tập nâng cao.

Chương trình chỉ xử lý và hiển thị ảnh, không lưu kết quả.
"""

from pathlib import Path

import cv2
import numpy as np


from .utils import *
from .imageProcessing import *

# ==========================================================
# 1. CẤU HÌNH
# ==========================================================

INPUT_PATH = Path("data/input/lena.png")


# ==========================================================
# 2. HÀM HỖ TRỢ
# ==========================================================

def input_number(prompt, default, data_type=float):
    """Nhận tham số số từ người dùng."""

    while True:
        value = input(
            f"{prompt} [Mặc định: {default}]: "
        ).strip()

        if not value:
            return default

        try:
            number = data_type(value)

            if isinstance(number, float) and not np.isfinite(number):
                print("Vui lòng nhập số hữu hạn.")
                continue

            return number

        except ValueError:
            print("Giá trị không hợp lệ. Vui lòng nhập lại.")


def show_result(original, result, title, mode="RGB"):
    """Hiển thị ảnh gốc và ảnh sau xử lý."""

    display_images(
        {
            "Original": (original, mode),
            "Result": (result, mode)
        },
        title=title
    )


# ==========================================================
# 3. I. TOÁN TỬ ĐIỂM ẢNH
# ==========================================================

def run_point_operations(image):
    """Thực hiện các toán tử điểm ảnh."""

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    while True:

        choice = str(show_point_operations_menu()).strip()

        if choice == "0":
            return

        try:
            if choice == "1":
                beta = input_number(
                    "Nhập giá trị tăng/giảm độ sáng(50/-50)",
                    0
                )

                result = adjust_brightness_contrast(image, beta=beta)

                show_result(
                    image, result, "Brightness Adjustment"
                )

            elif choice == "2":
                alpha = input_number(
                    "Nhập hệ số tương phản",
                    1.5
                )

                result = adjust_brightness_contrast(image, alpha=alpha)

                show_result(
                    image, result, "Contrast Adjustment"
                )

            elif choice == "3":
                result = negative_image(image)

                show_result(
                    image, result, "Negative Image"
                )

            elif choice == "4":
                threshold = input_number(
                    "Nhập giá trị ngưỡng",
                    128,
                    int
                )

                result = threshold_image(
                    gray_image,
                    threshold=threshold
                )

                show_result(
                    gray_image,
                    result,
                    "Binary Thresholding",
                    mode="GRAY"
                )

            else:
                print("Lựa chọn không hợp lệ.")

        except (TypeError, ValueError, cv2.error) as error:
            print(f"Lỗi: {error}")


# ==========================================================
# 4. II. LỌC TUYẾN TÍNH
# ==========================================================

def run_linear_filters(image):
    """Thực hiện các bộ lọc tuyến tính."""

    while True:

        choice = str(show_linear_filters_menu()).strip()

        if choice == "0":
            return

        try:
            if choice == "1":
                k = input_number(
                    "Nhập kích thước kernel Mean",
                    3,
                    int
                )

                result = mean_filter(image, kernel_size=k)

                show_result(image, result, "Mean Filter")

            elif choice == "2":
                k = input_number(
                    "Nhập kích thước kernel Gaussian",
                    5,
                    int
                )

                sigma = input_number(
                    "Nhập sigma",
                    1.0
                )

                result = gaussian_filter(
                    image,
                    kernel_size=k,
                    sigma=sigma
                )

                show_result(image, result, "Gaussian Filter")

            elif choice == "3":
                result = sharpen_filter(image)

                show_result(image, result, "Sharpen Filter")

            else:
                print("Lựa chọn không hợp lệ.")

        except (TypeError, ValueError, cv2.error) as error:
            print(f"Lỗi: {error}")


# ==========================================================
# 5. III.1. PHÁT HIỆN CẠNH
# ==========================================================

def run_edge_detection(gray_image):
    """Thực hiện phát hiện cạnh bằng Sobel và Prewitt."""


    try:
        k = input_number(
            "Nhập kích thước kernel Sobel (1, 3, 5, 7)",
            3,
            int
        )

        sobel = sobel_edge_detection(
            gray_image,
            kernel_size=k
        )

        prewitt = prewitt_edge_detection(gray_image)

        display_images(
            {
                "Original": (gray_image, "GRAY"),
                "Sobel": (sobel, "GRAY"),
                "Prewitt": (prewitt, "GRAY")
            },
            title="Edge Detection"
        )

    except (TypeError, ValueError, cv2.error) as error:
        print(f"Lỗi: {error}")


# ==========================================================
# 6. III.2. CUSTOM KERNEL
# ==========================================================

def run_custom_kernels(gray_image):
    """Thực hiện các kernel mẫu và kernel tùy chỉnh."""
    print("\n===== CUSTOM KERNEL =====")
    print("0. De quay lai")
    print("1. Su dung kernel thiet ke san")
    print("2. ap kernel tu thiet ke")

    choice = input("Chọn: ").strip()

    try:
        if choice == "1":
            result = apply_preset_kernels(gray_image)
            display_images(
                {
                    "Original": (gray_image, "GRAY"),
                    "Box Blur": (result[0], "GRAY"),
                    "Sharpen": (result[1], "GRAY"),
                    "Laplacian": (result[2], "GRAY"),
                    "Emboss": (result[3], "GRAY")
                },
                title="Custom Kernel",
                columns= 5
            )
        elif choice == "2":
            kernel = input("Nhap kernel:")
            result = apply_custom_kernel(gray_image, kernel)
            display_images(
                        {
                            "Original": (gray_image, "GRAY"),
                            "your kernel": (result, "GRAY")
                        },
                        title="Custom Kernel"
                    )
        else:
            return

   

    except (TypeError, ValueError, cv2.error) as error:
        print(f"Lỗi: {error}")


# ==========================================================
# 7. III.3. SO SÁNH BỘ LỌC
# ==========================================================


def run_comparison(gray_image):
    """So sánh bốn bộ lọc trên ảnh nhiễu và đánh giá với ảnh gốc."""

    print("\n===== FILTER COMPARISON =====")

    try:
        # 1. Đọc ảnh nhiễu
        noisy_path = input(
            "Nhập đường dẫn ảnh nhiễu: "
        ).strip().strip('"').strip("'")

        if not noisy_path:
            raise ValueError("Đường dẫn ảnh không được để trống.")

        noisy_image = read_image(
            noisy_path,
            mode="GRAY"
        )

        # 2. Kiểm tra kích thước ảnh
        if noisy_image.shape != gray_image.shape:
            raise ValueError(
                "Ảnh gốc và ảnh nhiễu phải cùng kích thước."
            )

        # 3. Áp dụng bốn bộ lọc
        filtered_images, benchmark_results = compare_filters(
            noisy_image,
            repeat=5
        )

        filter_names = [
            "Mean",
            "Gaussian",
            "Median",
            "Bilateral"
        ]

        if len(filtered_images) != 4 or len(benchmark_results) != 4:
            raise RuntimeError(
                "Số lượng kết quả trả về không hợp lệ."
            )

        # 4. Đánh giá ảnh sau lọc với ảnh gốc sạch
        quality_results = evaluate_filter_results(
            gray_image,
            filtered_images
        )

        # 5. Tính chỉ số của ảnh nhiễu ban đầu
        noisy_metrics = calculate_metrics(
            gray_image,
            noisy_image
        )

        # 6. Tổng hợp kết quả đánh giá
        evaluation = [
            {
                "Filter": "Noisy Input",
                **noisy_metrics,
                "Execution Time (ms)": None
            }
        ]

        for quality, benchmark in zip(
            quality_results,
            benchmark_results
        ):
            evaluation.append({
                "Filter": quality["Filter"],
                "MSE": quality["MSE"],
                "MAE": quality["MAE"],
                "PSNR": quality["PSNR"],
                "Execution Time (ms)": benchmark[
                    "Execution Time (ms)"
                ]
            })

        # 7. Chuẩn bị ảnh để hiển thị
        display_results = {
            "Original": (gray_image, "GRAY"),
            "Noisy Input": (noisy_image, "GRAY")
        }

        for name, result in zip(
            filter_names,
            filtered_images
        ):
            display_results[name] = (
                result,
                "GRAY"
            )

        display_images(
            display_results,
            title="Filter Comparison",
            columns=3
        )

        # 8. Hiển thị bảng đánh giá
        print("\n================ COMPARISON RESULTS ================")

        print(
            f"{'Filter':<15}"
            f"{'Time (ms)':>12}"
            f"{'MSE':>12}"
            f"{'MAE':>12}"
            f"{'PSNR (dB)':>14}"
        )

        print("-" * 65)

        for metric in evaluation:

            execution_time = metric["Execution Time (ms)"]

            time_text = (
                "-"
                if execution_time is None
                else f"{execution_time:.3f}"
            )

            print(
                f"{metric['Filter']:<15}"
                f"{time_text:>12}"
                f"{metric['MSE']:>12.3f}"
                f"{metric['MAE']:>12.3f}"
                f"{metric['PSNR']:>14.3f}"
            )

        print("=" * 65)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        cv2.error
    ) as error:

        print(f"Lỗi so sánh bộ lọc: {error}")



# ==========================================================
# 8. III.4. LỌC PHI TUYẾN TÍNH
# ==========================================================

def run_nonlinear_filters(image):
    """Thực hiện Median Filter và Bilateral Filter."""

    print("\n===== NONLINEAR FILTERS =====")
    print("1. Median Filter")
    print("2. Bilateral Filter")
    print("0. Quay lại")

    choice = input("Chọn bộ lọc: ").strip()

    try:
        if choice == "1":
            k = input_number(
                "Nhập kích thước kernel Median",
                3,
                int
            )

            result = median_filter(
                image,
                kernel_size=k
            )

            show_result(image, result, "Median Filter")

        elif choice == "2":
            diameter = input_number(
                "Nhập diameter",
                9,
                int
            )

            sigma_color = input_number(
                "Nhập sigma color",
                75.0
            )

            sigma_space = input_number(
                "Nhập sigma space",
                75.0
            )

            result = bilateral_filter(
                image,
                diameter=diameter,
                sigma_color=sigma_color,
                sigma_space=sigma_space
            )

            show_result(image, result, "Bilateral Filter")

        elif choice != "0":
            print("Lựa chọn không hợp lệ.")

    except (TypeError, ValueError, cv2.error) as error:
        print(f"Lỗi: {error}")


# ==========================================================
# 9. III. BÀI TẬP NÂNG CAO
# ==========================================================

def run_advanced(image, image_path):
    """Điều phối các bài tập nâng cao."""
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    while True:

        choice = str(show_advanced_menu()).strip()

        if choice == "0":
            return

        elif choice == "1":
            run_edge_detection(imageGray)

        elif choice == "2":
            run_custom_kernels(imageGray)

        elif choice == "3":
            run_comparison(imageGray)

        elif choice == "4":
            run_nonlinear_filters(image)

        else:
            print("Lựa chọn không hợp lệ.")


# ==========================================================
# 10. MAIN
# ==========================================================

def main():
    """Điểm bắt đầu chương trình Lab 02."""

    image_path = INPUT_PATH

    while True:

        try:
            image_path = input("nhap duong dan den anh:")

            if image_path is None:
                print("Đã thoát chương trình.")
                return

            image = read_image(
                image_path,
                mode="BGR"
            )

            break

        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"Lỗi: {error}")

    while True:

        choice = str(show_main_menu()).strip()

        try:
            if choice == "0":
                print("Đã thoát chương trình.")
                break

            elif choice == "1":
                run_point_operations(image)

            elif choice == "2":
                run_linear_filters(image)

            elif choice == "3":
                run_advanced(image, image_path)

            else:
                print("Lựa chọn không hợp lệ.")

        except (TypeError, ValueError, OSError, RuntimeError, cv2.error) as error:
            print(f"Lỗi: {error}")


if __name__ == "__main__":
    main()
