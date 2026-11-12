"""Experiment Engine: cấu hình TOML -> lưới tham số -> chạy -> lưu (tái lập được) -> báo cáo."""

from .config import ExperimentConfig, load_config
from .runner import run_experiment

__all__ = ["ExperimentConfig", "load_config", "run_experiment"]
