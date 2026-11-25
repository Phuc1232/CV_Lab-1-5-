import math
import numpy as np
import matplotlib.pyplot as plt


def display_images(
    images: dict[str, tuple[np.ndarray, str]],
    title: str = "Kết quả xử lý ảnh",
    columns: int = 3
) -> None:
    """
    Hiển thị ảnh gốc và các ảnh kết quả xử lý.

    Parameters
    ----------
    images : dict
        Dictionary chứa tên ảnh và tuple (image, color_mode).
        color_mode gồm BGR, RGB hoặc GRAY.

    title : str, optional
        Tiêu đề chung của cửa sổ hiển thị.

    columns : int, optional
        Số cột ảnh muốn hiển thị.

    Raises
    ------
    ValueError
        Nếu dữ liệu ảnh hoặc số cột không hợp lệ.
    """

    if not isinstance(images, dict) or not images:
        raise ValueError("Danh sách ảnh không được để trống.")

    if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
        raise ValueError("Số cột phải là số nguyên dương.")

    # Tính số hàng cần hiển thị
    total_images = len(images)
    rows = math.ceil(total_images / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False
    )

    # Duyệt qua từng ảnh và hiển thị
    for ax, (name, image_info) in zip(axes.flat, images.items()):

        if not isinstance(image_info, tuple) or len(image_info) != 2:
            raise ValueError(
                f"Ảnh '{name}' phải có định dạng (image, color_mode)."
            )

        image, color_mode = image_info

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(f"Ảnh '{name}' không hợp lệ.")

        color_mode = color_mode.upper()

        if image.ndim == 2:
            # Ảnh xám
            ax.imshow(image, cmap="gray")

        elif image.ndim == 3 and image.shape[2] == 3:
            # Chuyển BGR sang RGB nếu cần
            if color_mode == "BGR":
                image = image[:, :, ::-1]
            elif color_mode != "RGB":
                raise ValueError(
                    f"Chế độ màu không hợp lệ cho ảnh '{name}'."
                )

            ax.imshow(image)

        else:
            raise ValueError(f"Định dạng ảnh '{name}' không được hỗ trợ.")

        ax.set_title(name)
        ax.axis("off")

    # Ẩn các ô hiển thị còn thừa
    for ax in axes.flat[total_images:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    
def show_main_menu():

    print("\n===== DIGITAL IMAGE PROCESSING LAB 02 =====")
    print("1. Toán tử điểm ảnh")
    print("2. Lọc tuyến tính")
    print("3. Bài tập nâng cao")
    print("0. Thoát")

    return input("Lựa chọn: ").strip()


def show_point_operations_menu():

    print("\n===== TOÁN TỬ ĐIỂM ẢNH =====")
    print("1. Thay đổi độ sáng")
    print("2. Thay đổi độ tương phản")
    print("3. Biến đổi âm bản")
    print("4. Cắt ngưỡng")
    print("0. Quay lại")

    return input("Lựa chọn: ").strip()


def show_linear_filters_menu():

    print("\n===== LỌC TUYẾN TÍNH =====")
    print("1. Mean Filter")
    print("2. Gaussian Filter")
    print("3. Sharpen Filter")
    print("0. Quay lại")

    return input("Lựa chọn: ").strip()


def show_advanced_menu():

    print("\n===== BÀI TẬP NÂNG CAO =====")
    print("1. Phát hiện cạnh Sobel, Prewitt")
    print("2. Thiết kế Custom Kernel")
    print("3. So sánh các bộ lọc")
    print("4. Median và Bilateral Filter")
    print("0. Quay lại")

    return input("Lựa chọn: ").strip()