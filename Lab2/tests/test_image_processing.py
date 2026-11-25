
from pathlib import Path

import cv2
import numpy as np
import pytest



from src import *


# ==========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def save_result(
    image: np.ndarray,
    folder: str,
    filename: str
) -> Path:
    """
    Lưu ảnh kết quả vào thư mục tương ứng.

    Args:
        image: Ảnh cần lưu.
        folder: Thư mục con bên trong output.
        filename: Tên file ảnh đầu ra.

    Returns:
        Đường dẫn file ảnh đã lưu.
    """

    output_path = OUTPUT_DIR / folder / filename

    color_order = (
        "RGB" if image.ndim == 3 else "GRAY"
    )

    saved_path = write_image(
        output_path,
        image,
        color_order=color_order
    )

    assert saved_path.is_file()

    return saved_path


# ==========================================================
# 2. FIXTURE - ĐỌC ẢNH MỘT LẦN
# ==========================================================

@pytest.fixture(scope="session")
def source_images():
    """
    Nhập đường dẫn và đọc ảnh một lần cho toàn bộ test.

    Tạo ảnh RGB và Grayscale từ ảnh đã đọc.
    """

    image_path = r"C:\Users\TAN LOC\Desktop\CV_Lab-1-5-\Lab2\data\input\images.jpg"

    if not image_path:
        pytest.fail("Đường dẫn ảnh không được để trống.")

    # Đọc ảnh duy nhất một lần
    imgRGB = read_image(
        image_path,
        mode="RGB"
    )

    # Tạo ảnh grayscale từ ảnh RGB
    gray = cv2.cvtColor(
        imgRGB,
        cv2.COLOR_RGB2GRAY
    )

    print("\n========== THONG TIN ANH ==========")
    print(get_image_info(imgRGB))
    print("===================================")

    return {
        "rgb": imgRGB,
        "gray": gray
    }


# ==========================================================
# 3. POINT OPERATIONS - BRIGHTNESS & CONTRAST
# ==========================================================

@pytest.mark.parametrize(
    "beta, alpha, filename",
    [
        (50, 1.0, "brightness_50.png"),
        (-50, 1.0, "brightness_minus_50.png"),
        (0, 1.5, "contrast_1.5.png"),
        (0, 0.95, "contrast_0.95.png"),
        (50, 1.5, "brightness_contrast.png")
    ]
)
def test_light_and_contrast(
    source_images,
    beta,
    alpha,
    filename
):
    """Kiểm tra điều chỉnh độ sáng và tương phản."""

    image = source_images["rgb"]

    result = adjust_brightness_contrast(
        image,
        beta=beta,
        alpha=alpha
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "pointOperations",
        filename
    )


# ==========================================================
# 4. POINT OPERATIONS - NEGATIVE
# ==========================================================

def test_negative_image(source_images):
    """Kiểm tra chuyển đổi ảnh sang ảnh âm bản."""

    image = source_images["rgb"]

    result = negative_image(image)

    expected = 255 - image

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    np.testing.assert_array_equal(
        result,
        expected
    )

    save_result(
        result,
        "pointOperations",
        "negative.png"
    )


# ==========================================================
# 5. POINT OPERATIONS - BINARY THRESHOLD
# ==========================================================

@pytest.mark.parametrize(
    "threshold",
    [64, 128, 192]
)
def test_binary_threshold(
    source_images,
    threshold
):
    """Kiểm tra nhị phân hóa ảnh với các ngưỡng khác nhau."""

    image = source_images["gray"]

    result = threshold_image(
        image,
        threshold=threshold
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    assert set(np.unique(result)).issubset({0, 255})

    save_result(
        result,
        "pointOperations",
        f"binary_{threshold}.png"
    )


def test_binary_threshold_default(source_images):
    """Kiểm tra nhị phân hóa ảnh với ngưỡng mặc định."""

    image = source_images["gray"]

    result = threshold_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    assert set(np.unique(result)).issubset({0, 255})

    save_result(
        result,
        "pointOperations",
        "binary_default.png"
    )


# ==========================================================
# 6. LINEAR FILTERS - MEAN
# ==========================================================

@pytest.mark.parametrize(
    "kernel_size, border_type, filename",
    [
        (3, cv2.BORDER_DEFAULT, "mean_k2.png"),
        (5, cv2.BORDER_DEFAULT, "mean_k5.png"),
        (5, cv2.BORDER_CONSTANT, "mean_constant.png"),
        (5, cv2.BORDER_REPLICATE, "mean_replicate.png"),
        (5, cv2.BORDER_REFLECT, "mean_reflect.png")
    ]
)
def test_mean_filter(
    source_images,
    kernel_size,
    border_type,
    filename
):
    """Kiểm tra Mean Filter với các kernel và chế độ biên."""

    image = source_images["gray"]

    result = mean_filter(
        image,
        kernel_size,
        border_type
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "linearFilters",
        filename
    )


# ==========================================================
# 7. LINEAR FILTERS - GAUSSIAN
# ==========================================================

@pytest.mark.parametrize(
    "kernel_size, sigma, filename",
    [
        (0, 1.25, "gaussian_auto.png"),
        (5, 0.25, "gaussian_sigma_025.png"),
        (5, 0.5, "gaussian_sigma_05.png"),
        (5, 0.75, "gaussian_sigma_075.png"),
        (5, 1.25, "gaussian_sigma_125.png")
    ]
)
def test_gaussian_filter(
    source_images,
    kernel_size,
    sigma,
    filename
):
    """Kiểm tra Gaussian Filter với các giá trị Sigma."""

    image = source_images["gray"]

    result = gaussian_filter(
        image,
        kernel_size,
        sigma
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "linearFilters",
        filename
    )


# ==========================================================
# 8. LINEAR FILTERS - SHARPEN
# ==========================================================

def test_sharpen_filter(source_images):
    """Kiểm tra chức năng làm sắc nét ảnh."""

    image = source_images["rgb"]

    result = sharpen_filter(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "linearFilters",
        "sharpen.png"
    )


# ==========================================================
# 9. CUSTOM KERNELS
# ==========================================================

def test_preset_kernels(source_images):
    """Kiểm tra áp dụng bốn kernel mẫu."""

    image = source_images["gray"]

    results = apply_preset_kernels(image)

    kernel_names = [
        "box_blur",
        "sharpen",
        "laplacian",
        "emboss"
    ]

    assert isinstance(results, list)
    assert len(results) == 4

    for name, result in zip(kernel_names, results):

        assert result.shape == image.shape
        assert result.dtype == np.uint8

        save_result(
            result,
            "linearFilters/customKernels",
            f"{name}.png"
        )


def test_custom_kernel(source_images):
    """Kiểm tra áp dụng kernel do người dùng thiết kế."""

    image = source_images["gray"]

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    result = apply_custom_kernel(
        image,
        kernel
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "linearFilters/customKernels",
        "custom_kernel.png"
    )


def test_custom_kernel_identity(source_images):
    """Kiểm tra kernel đồng nhất."""

    image = source_images["gray"]

    kernel = np.zeros(
        (3, 3),
        dtype=np.float32
    )

    kernel[1, 1] = 1

    result = apply_custom_kernel(
        image,
        kernel
    )

    np.testing.assert_array_equal(
        result,
        image
    )


def test_custom_kernel_invalid(source_images):
    """Kiểm tra kernel có kích thước không hợp lệ."""

    image = source_images["gray"]

    kernel = np.ones(
        (2, 2),
        dtype=np.float32
    )

    with pytest.raises((TypeError, ValueError)):
        apply_custom_kernel(
            image,
            kernel
        )


# ==========================================================
# 10. EDGE DETECTION
# ==========================================================

def test_prewitt_edge_detection(source_images):
    """Kiểm tra phát hiện biên bằng Prewitt."""

    image = source_images["gray"]

    result = prewitt_edge_detection(image)

    assert result.shape[:2] == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "edgeDetection",
        "prewitt.png"
    )


def test_sobel_edge_detection(source_images):
    """Kiểm tra phát hiện biên bằng Sobel."""

    image = source_images["gray"]

    result = sobel_edge_detection(image)

    assert result.shape[:2] == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "edgeDetection",
        "sobel.png"
    )


# ==========================================================
# 11. NONLINEAR FILTERS - MEDIAN
# ==========================================================

def test_median_filter(source_images):
    """Kiểm tra Median Filter."""

    image = source_images["gray"]

    result = median_filter(
        image,
        kernel_size=5
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "nonlinearFilter",
        "median.png"
    )


# ==========================================================
# 12. NONLINEAR FILTERS - BILATERAL
# ==========================================================

def test_bilateral_filter(source_images):
    """Kiểm tra Bilateral Filter."""

    image = source_images["gray"]

    result = bilateral_filter(
        image,
        diameter=5,
        sigma_color=75.0,
        sigma_space=75.0
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8

    save_result(
        result,
        "nonlinearFilter",
        "bilateral.png"
    )


# ==========================================================
# 13. TẠO ẢNH NHIỄU MUỐI TIÊU
# ==========================================================

def create_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05
) -> np.ndarray:
    """
    Tạo nhiễu muối tiêu từ ảnh gốc.

    Args:
        image: Ảnh grayscale đầu vào.
        amount: Tỷ lệ pixel bị nhiễu. Mặc định là 0.05.

    Returns:
        Ảnh sau khi thêm nhiễu muối tiêu.
    """

    rng = np.random.default_rng(42)

    noisy = image.copy()
    flat = noisy.ravel()

    num_pixels = int(flat.size * amount)

    indices = rng.choice(
        flat.size,
        num_pixels,
        replace=False
    )

    salt_mask = rng.random(num_pixels) < 0.5

    flat[indices[salt_mask]] = 255
    flat[indices[~salt_mask]] = 0

    return noisy


# ==========================================================
# 14. TẠO ẢNH NHIỄU GAUSSIAN
# ==========================================================

def create_gaussian_noise(
    image: np.ndarray,
    sigma: float = 20.0
) -> np.ndarray:
    """
    Tạo nhiễu Gaussian từ ảnh gốc.

    Args:
        image: Ảnh grayscale đầu vào.
        sigma: Độ lệch chuẩn của nhiễu. Mặc định là 20.0.

    Returns:
        Ảnh sau khi thêm nhiễu Gaussian.
    """

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


# ==========================================================
# 15. COMPARE FILTERS
# ==========================================================

def test_compare_filters(source_images):
    """
    So sánh bốn bộ lọc trên hai loại ảnh nhiễu.

    Hai ảnh nhiễu được tạo độc lập từ cùng một ảnh gốc.

    Mỗi ảnh nhiễu được đưa vào compare_filters() một lần.
    """

    # Lấy ảnh gốc đã đọc từ fixture
    original_image = source_images["gray"]

    # Tạo hai ảnh nhiễu từ cùng ảnh gốc
    salt_pepper_image = create_salt_pepper_noise(
        original_image
    )

    gaussian_image = create_gaussian_noise(
        original_image
    )

    # Danh sách hai ảnh nhiễu
    noisy_images = [
        ("salt_pepper", salt_pepper_image),
        ("gaussian", gaussian_image)
    ]

    # Tên bốn bộ lọc
    filter_names = [
        "mean",
        "gaussian",
        "median",
        "bilateral"
    ]

    # So sánh lần lượt hai loại nhiễu
    for noise_name, noisy_image in noisy_images:

        # Lưu ảnh nhiễu đầu vào
        save_result(
            noisy_image,
            "comparisons",
            f"{noise_name}_input.png"
        )

        # Áp dụng bốn bộ lọc
        filtered_images, evaluation = compare_filters(
            noisy_image,
            repeat=3
        )

        # Kiểm tra số lượng kết quả
        assert isinstance(filtered_images, list)
        assert len(filtered_images) == 4
        assert len(evaluation) == 4

        # Kiểm tra và lưu ảnh sau lọc
        for name, result in zip(
            filter_names,
            filtered_images
        ):

            assert result.shape == original_image.shape
            assert result.dtype == np.uint8

            save_result(
                result,
                "comparisons",
                f"{noise_name}_{name}.png"
            )

        # Kiểm tra các chỉ số đánh giá
        for metrics in evaluation:

            assert metrics["MSE"] >= 0
            assert metrics["MAE"] >= 0

            assert (
                metrics["PSNR"] >= 0
                or np.isinf(metrics["PSNR"])
            )

            assert metrics["Execution Time (ms)"] >= 0

        # In kết quả so sánh
        print(f"\n========== {noise_name.upper()} NOISE ==========")

        for metrics in evaluation:
            print(metrics)


# ==========================================================
# 16. KIỂM TRA TÍNH CHÍNH XÁC CỦA METRICS
# ==========================================================

def test_calculate_metrics():
    """Kiểm tra tính chính xác của MSE, MAE và PSNR."""

    reference = np.array(
        [[10, 20], [30, 40]],
        dtype=np.uint8
    )

    filtered = np.array(
        [[10, 25], [20, 40]],
        dtype=np.uint8
    )

    metrics = calculate_metrics(
        reference,
        filtered
    )

    difference = (
        reference.astype(np.float64)
        - filtered.astype(np.float64)
    )

    expected_mse = np.mean(difference ** 2)
    expected_mae = np.mean(np.abs(difference))

    expected_psnr = 10 * np.log10(
        (255 ** 2) / expected_mse
    )

    assert metrics["MSE"] == pytest.approx(expected_mse)
    assert metrics["MAE"] == pytest.approx(expected_mae)
    assert metrics["PSNR"] == pytest.approx(expected_psnr)


def test_calculate_metrics_identical_images():
    """Kiểm tra trường hợp hai ảnh hoàn toàn giống nhau."""

    image = np.array(
        [[10, 20], [30, 40]],
        dtype=np.uint8
    )

    metrics = calculate_metrics(
        image,
        image.copy()
    )

    assert metrics["MSE"] == 0
    assert metrics["MAE"] == 0
    assert np.isinf(metrics["PSNR"])


# ==========================================================
# 17. KIỂM TRA ĐÁNH GIÁ NHIỀU BỘ LỌC
# ==========================================================

def test_evaluate_filter_results():
    """Kiểm tra tổng hợp chỉ số đánh giá nhiều ảnh."""

    reference = np.array(
        [[10, 20], [30, 40]],
        dtype=np.uint8
    )

    filtered_images = [
        reference.copy(),
        np.array(
            [[15, 25], [35, 45]],
            dtype=np.uint8
        )
    ]

    evaluation = evaluate_filter_results(
        reference,
        filtered_images,
        filter_names=("Filter A", "Filter B")
    )

    assert len(evaluation) == 2

    assert evaluation[0]["Filter"] == "Filter A"
    assert evaluation[1]["Filter"] == "Filter B"

    assert evaluation[0]["MSE"] == 0
    assert evaluation[0]["MAE"] == 0
    assert np.isinf(evaluation[0]["PSNR"])

    assert evaluation[1]["MSE"] > 0
    assert evaluation[1]["MAE"] > 0
