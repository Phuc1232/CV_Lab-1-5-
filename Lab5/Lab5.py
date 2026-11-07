"""Backward-compatible entry point for Lab 5 recognition."""

from src.facelab.backends import open_webcam
from src.facelab.cli import main, run_webcam
from src.facelab.core.recognition import FaceRecognitionSystem
from src.facelab.imaging.overlay import draw_hud
from src.facelab.paths import DEFAULT_DB_DIR


if __name__ == "__main__":
    main()
