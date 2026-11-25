"""
Module so sánh và đánh giá các bộ lọc ảnh.

Cung cấp chức năng áp dụng và so sánh bốn thuật toán lọc ảnh:

- Mean Filter: Bộ lọc trung bình.
- Gaussian Filter: Bộ lọc Gaussian.
- Median Filter: Bộ lọc trung vị.
- Bilateral Filter: Bộ lọc song phương.

Các tiêu chí đánh giá bao gồm:

1. MSE (Mean Squared Error) - Sai số bình phương trung bình:
   Đo lường mức độ sai khác giữa ảnh tham chiếu và ảnh sau lọc.
   Giá trị càng nhỏ, chất lượng ảnh sau lọc càng tốt.
   Giá trị tốt nhất: 0.

2. MAE (Mean Absolute Error) - Sai số tuyệt đối trung bình:
   Đo lường độ sai lệch tuyệt đối trung bình giữa các pixel
   của ảnh tham chiếu và ảnh sau lọc.
   Giá trị càng nhỏ, độ sai lệch càng thấp.
   Giá trị tốt nhất: 0.

3. PSNR (Peak Signal-to-Noise Ratio) - Tỷ số tín hiệu
   trên nhiễu cực đại:
   Đánh giá chất lượng ảnh sau lọc thông qua tỷ số giữa
   mức tín hiệu cực đại và sai số nhiễu.
   Giá trị càng lớn, chất lượng ảnh càng gần ảnh tham chiếu.
   Giá trị tốt nhất: +inf dB (khi hai ảnh giống hệt nhau).

4. Execution Time - Thời gian thực thi:
   Đo lường thời gian trung bình để bộ lọc xử lý một ảnh,
   tính bằng mili giây (ms).
   Giá trị càng nhỏ, tốc độ xử lý càng nhanh.
   Giá trị tốt nhất: Càng gần 0 ms càng tốt.

Nguyên tắc đánh giá:
- MSE và MAE: Càng thấp càng tốt.
- PSNR: Càng cao càng tốt.
- Execution Time: Càng thấp càng tốt.

Các bộ lọc được áp dụng trên cùng một ảnh đầu vào
để đảm bảo tính nhất quán trong quá trình so sánh.

Ảnh tham chiếu sạch được sử dụng để đánh giá chất lượng
khử nhiễu thông qua MSE, MAE và PSNR.
"""

from collections.abc import Callable
from time import perf_counter

import cv2
import numpy as np

from numpy.typing import NDArray

from src.utils import (
    validate_image,
    validate_non_negative_int
)

from .linearFilters import (
    mean_filter,
    gaussian_filter
)

from .nonlinearFilters import (
    median_filter,
    bilateral_filter
)


ImageArray = NDArray[np.uint8]
MetricResult = dict[str, str | float]


# ==========================================================
# 1. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ==========================================================

def calculate_metrics(
    noisy_image: ImageArray,
    filtered_image: ImageArray
) -> dict[str, float]:
    """
    Tính toán các chỉ số đánh giá giữa ảnh nhiễu và ảnh sau lọc.

    Args:
        noisy_image: Ảnh nhiễu ban đầu được sử dụng làm ảnh tham chiếu.
        filtered_image: Ảnh thu được sau khi áp dụng bộ lọc.
            Hai ảnh phải có cùng kích thước và số kênh.

    Returns:
        Dictionary chứa ba chỉ số đánh giá:
            - MSE: Sai số bình phương trung bình.
            - MAE: Sai số tuyệt đối trung bình.
            - PSNR: Tỷ số tín hiệu trên nhiễu cực đại (dB).

    Raises:
        TypeError: Nếu dữ liệu ảnh không đúng kiểu dữ liệu yêu cầu.
        ValueError: Nếu ảnh không hợp lệ hoặc hai ảnh khác kích thước,
            số kênh.

    Notes:
        Gọi N là tổng số pixel trên tất cả các kênh ảnh.
        I(i) là giá trị pixel thứ i của ảnh tham chiếu.
        K(i) là giá trị pixel thứ i của ảnh sau lọc.

        1. Sai số bình phương trung bình (MSE):

            MSE = (1/N) * SUM((I(i) - K(i))^2)

        2. Sai số tuyệt đối trung bình (MAE):

            MAE = (1/N) * SUM(|I(i) - K(i)|)

        3. Tỷ số tín hiệu trên nhiễu cực đại (PSNR):

            PSNR = 10 * log10(MAX^2 / MSE)

        Trong đó:
            - MAX = 255 đối với ảnh uint8.
            - SUM biểu thị tổng sai số trên toàn bộ pixel.
            - log10 là logarit cơ số 10.

        Nếu MSE = 0, hai ảnh giống hệt nhau và PSNR = vô cực.

        Ảnh được chuyển sang kiểu float64 trước khi tính toán
        nhằm tránh hiện tượng tràn số trong phép tính.
    """

    validate_image(noisy_image)
    validate_image(filtered_image)

    if noisy_image.shape != filtered_image.shape:
        raise ValueError(
            "Hai ảnh phải có cùng kích thước và số kênh."
        )

    # Chuyển sang float64 để tránh tràn số
    noisy = noisy_image.astype(np.float64)
    filtered = filtered_image.astype(np.float64)

    difference = noisy - filtered

    # Mean Squared Error
    mse = float(np.mean(difference ** 2))

    # Mean Absolute Error
    mae = float(np.mean(np.abs(difference)))

    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            10 * np.log10((255 ** 2) / mse)
        )

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr
    }


# ==========================================================
# 2. ĐÁNH GIÁ DANH SÁCH ẢNH SAU KHI LỌC
# ==========================================================

def evaluate_filter_results(
    noisy_image: ImageArray,
    filtered_images: list[ImageArray],
    filter_names: tuple[str, ...] = (
        "Mean",
        "Gaussian",
        "Median",
        "Bilateral"
    )
) -> list[MetricResult]:
    """
    Đánh giá và tổng hợp các chỉ số của nhiều ảnh sau khi lọc.

    Args:
        noisy_image: Ảnh nhiễu ban đầu được sử dụng làm ảnh tham chiếu.
        filtered_images: Danh sách các ảnh thu được sau khi áp dụng
            các bộ lọc.
        filter_names: Bộ tên tương ứng với từng bộ lọc.
            Giá trị mặc định gồm Mean, Gaussian, Median và Bilateral.

    Returns:
        Danh sách dictionary chứa các thông tin đánh giá
        tương ứng với từng bộ lọc:
            - Filter: Tên bộ lọc.
            - MSE: Sai số bình phương trung bình.
            - MAE: Sai số tuyệt đối trung bình.
            - PSNR: Tỷ số tín hiệu trên nhiễu cực đại (dB).

    Raises:
        TypeError: Nếu filtered_images không phải kiểu list
            hoặc dữ liệu ảnh không đúng kiểu yêu cầu.
        ValueError: Nếu số lượng ảnh không tương ứng với số tên
            bộ lọc, hoặc ảnh không hợp lệ hay khác kích thước
            so với ảnh tham chiếu.

    Notes:
        Các chỉ số được tính bằng hàm calculate_metrics().

        Mỗi ảnh sau lọc được so sánh độc lập với ảnh nhiễu ban đầu.

        Kết quả được lưu theo đúng thứ tự của danh sách đầu vào.
    """

    validate_image(noisy_image)

    if not isinstance(filtered_images, list):
        raise TypeError(
            "filtered_images phải là một list."
        )

    if len(filtered_images) != len(filter_names):
        raise ValueError(
            "Số lượng ảnh phải tương ứng với số bộ lọc."
        )

    evaluation = []

    for name, filtered_image in zip(
        filter_names,
        filtered_images
    ):

        metrics = calculate_metrics(
            noisy_image,
            filtered_image
        )

        evaluation.append({
            "Filter": name,
            **metrics
        })

    return evaluation


# ==========================================================
# 3. SO SÁNH BỐN BỘ LỌC
# ==========================================================

def compare_filters(
    noisy_image: ImageArray,
    kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    bilateral_diameter: int = 5,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    border_type: int = cv2.BORDER_DEFAULT,
    repeat: int = 5
) -> tuple[list[ImageArray], list[MetricResult]]:
    """
    Áp dụng và so sánh bốn bộ lọc trên cùng một ảnh nhiễu.

    Args:
        noisy_image: Ảnh đầu vào đã được tạo nhiễu.
            Yêu cầu ảnh grayscale hoặc ảnh màu 3 kênh, kiểu uint8.
        kernel_size: Kích thước kernel dùng cho Mean, Gaussian
            và Median Filter. Mặc định là 5.
        gaussian_sigma: Độ lệch chuẩn của Gaussian Filter.
            Mặc định là 1.0.
        bilateral_diameter: Đường kính vùng lân cận của
            Bilateral Filter. Mặc định là 5.
        bilateral_sigma_color: Độ lệch chuẩn cường độ màu của
            Bilateral Filter. Mặc định là 75.0.
        bilateral_sigma_space: Độ lệch chuẩn không gian của
            Bilateral Filter. Mặc định là 75.0.
        border_type: Phương pháp xử lý pixel tại biên ảnh.
            Mặc định là cv2.BORDER_DEFAULT.
        repeat: Số lần thực hiện mỗi bộ lọc để tính thời gian
            thực thi trung bình. Mặc định là 5.

    Returns:
        Tuple gồm hai thành phần:

            - filtered_images: Danh sách bốn ảnh sau khi lọc,
                theo thứ tự Mean, Gaussian, Median và Bilateral.

            - evaluation: Danh sách dictionary chứa các chỉ số
                đánh giá của từng bộ lọc:
                    + Filter: Tên bộ lọc.
                    + MSE: Sai số bình phương trung bình.
                    + MAE: Sai số tuyệt đối trung bình.
                    + PSNR: Tỷ số tín hiệu trên nhiễu cực đại (dB).
                    + Execution Time (ms): Thời gian thực thi
                    trung bình tính bằng mili giây.

    Raises:
        TypeError: Nếu ảnh đầu vào hoặc repeat không đúng kiểu
            dữ liệu yêu cầu.
        ValueError: Nếu ảnh không hợp lệ hoặc repeat nhỏ hơn 1.

    Notes:
        Bốn bộ lọc được áp dụng độc lập trên cùng một ảnh nhiễu.

        Mỗi bộ lọc được thực hiện một lần khởi động (warm-up)
        trước khi tiến hành đo thời gian.

        Thời gian thực thi trung bình được tính theo công thức:

            T_avg = (T_total / repeat) * 1000

        Trong đó:
            - T_avg: Thời gian thực thi trung bình (ms).
            - T_total: Tổng thời gian thực hiện repeat lần (s).
            - repeat: Số lần thực hiện bộ lọc.
            - 1000: Hệ số chuyển đổi từ giây sang mili giây.

        Các chỉ số MSE, MAE và PSNR được tính bằng hàm
        evaluate_filter_results(), sử dụng ảnh nhiễu ban đầu
        làm ảnh tham chiếu.
    """

    # Kiểm tra ảnh đầu vào
    validate_image(noisy_image)

    # Kiểm tra số lần lặp
    repeat = validate_non_negative_int(
        repeat,
        "repeat"
    )

    if repeat < 1:
        raise ValueError(
            "repeat phải lớn hơn hoặc bằng 1."
        )

    # Khai báo các bộ lọc
    filters: list[
        tuple[str, Callable[[], ImageArray]]
    ] = [

        (
            "Mean",
            lambda: mean_filter(
                noisy_image,
                kernel_size=kernel_size,
                border_type=border_type
            )
        ),

        (
            "Gaussian",
            lambda: gaussian_filter(
                noisy_image,
                kernel_size=kernel_size,
                sigma=gaussian_sigma,
                border_type=border_type
            )
        ),

        (
            "Median",
            lambda: median_filter(
                noisy_image,
                kernel_size=kernel_size
            )
        ),

        (
            "Bilateral",
            lambda: bilateral_filter(
                noisy_image,
                diameter=bilateral_diameter,
                sigma_color=bilateral_sigma_color,
                sigma_space=bilateral_sigma_space,
                border_type=border_type
            )
        )
    ]

    filtered_images: list[ImageArray] = []
    execution_times: list[float] = []

    # Thực hiện lần lượt bốn bộ lọc
    for name, filter_function in filters:

        # Warm-up: chạy trước khi đo
        result = filter_function()

        start_time = perf_counter()

        # Đo thời gian thực thi
        for _ in range(repeat):
            result = filter_function()

        end_time = perf_counter()

        # Thời gian trung bình (ms)
        average_time = (
            (end_time - start_time) / repeat
        ) * 1000

        filtered_images.append(result)
        execution_times.append(average_time)

    # Tính các chỉ số đánh giá
    evaluation = evaluate_filter_results(
        noisy_image,
        filtered_images
    )

    # Bổ sung thời gian thực thi
    for index in range(len(evaluation)):
        evaluation[index]["Execution Time (ms)"] = (
            execution_times[index]
        )

    return filtered_images, evaluation
