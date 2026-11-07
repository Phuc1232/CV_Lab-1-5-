"""Load an optional TOML preset for webcam recognition."""

import math
from pathlib import Path

import torch


def load_recognition_config(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as error:
            raise ValueError("Cài phụ thuộc trong requirements.txt để dùng TOML trên Python 3.10.") from error

    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    if config.keys() != {"recognition"} or not isinstance(config["recognition"], dict):
        raise ValueError("Tệp cấu hình cần một bảng [recognition].")

    settings = config["recognition"]
    unknown = settings.keys() - {"camera", "db_dir", "threshold", "frame_skip", "device"}
    if unknown:
        raise ValueError(f"Khóa cấu hình chưa được hỗ trợ: {', '.join(sorted(unknown))}")

    if "camera" in settings and type(settings["camera"]) is not int:
        raise ValueError("camera phải là số nguyên.")
    if "frame_skip" in settings:
        frame_skip = settings["frame_skip"]
        if type(frame_skip) is not int or frame_skip < 1:
            raise ValueError("frame_skip phải là số nguyên dương.")
    if "threshold" in settings:
        threshold = settings["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise ValueError("threshold phải là một số hữu hạn.")
    if "device" in settings:
        device = settings["device"]
        if not isinstance(device, str):
            raise ValueError("device phải là chuỗi.")
        try:
            torch.device(device)
        except RuntimeError as error:
            raise ValueError("device phải là tên thiết bị PyTorch, ví dụ cpu hoặc cuda:0.") from error
    if "db_dir" in settings:
        db_dir = settings["db_dir"]
        if not isinstance(db_dir, str) or not db_dir.strip():
            raise ValueError("db_dir phải là đường dẫn dạng chuỗi.")
        settings["db_dir"] = str(path.parent / db_dir)

    return settings
