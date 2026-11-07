"""Backward-compatible entry point for Lab 5 registration."""

from src.facelab.backends import open_webcam
from src.facelab.cli import register_main as main
from src.facelab.datasets.registration import register_from_image, register_from_webcam
from src.facelab.paths import DEFAULT_DB_DIR


if __name__ == "__main__":
    main()
