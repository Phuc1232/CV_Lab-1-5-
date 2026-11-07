"""Paths shared by face recognition and registration."""

from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = str(LAB_ROOT / "known_faces")
