"""Persist a gallery of normalized face embeddings alongside its images."""

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from uuid import uuid4

import numpy as np


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DIMENSIONS = 512
_VECTOR_DTYPE = np.dtype("<f4")
_DATABASE_NAME = "embeddings.sqlite3"
_DATABASE_NAMES = {_DATABASE_NAME, *(f"{_DATABASE_NAME}-{suffix}" for suffix in ("journal", "wal", "shm"))}
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"),
}
_DELETE_SNAPSHOT = """DELETE FROM samples
                     WHERE image_path = ? AND identity = ? AND source_hash = ? AND model_signature = ?"""


def normalize_identity(name: str) -> str:
    identity = name.strip().replace(" ", "_")
    if (
        not identity
        or identity.endswith(".")
        or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in identity)
        or identity.split(".", 1)[0].rstrip().upper() in _RESERVED_NAMES
        or identity.casefold() in _DATABASE_NAMES
    ):
        raise ValueError("ID trống hoặc chứa ký tự không hợp lệ cho tên thư mục.")
    return identity


class EmbeddingStore:
    def __init__(self, directory: str | Path, model_signature: str):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / _DATABASE_NAME
        if self.path.is_symlink():
            raise ValueError("Tệp cơ sở dữ liệu embedding không được là liên kết tượng trưng.")
        self.model_signature = model_signature
        with self._connection() as connection:
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS samples (
                    image_path TEXT PRIMARY KEY,
                    identity TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    model_signature TEXT NOT NULL,
                    embedding BLOB NOT NULL CHECK (length(embedding) = {_DIMENSIONS * _VECTOR_DTYPE.itemsize})
                )
            """)

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            yield connection

    def __len__(self) -> int:
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(DISTINCT identity) FROM samples WHERE model_signature = ?",
                (self.model_signature,),
            ).fetchone()[0]

    def iter_embeddings(self) -> Iterator[tuple[str, np.ndarray]]:
        with self._connection() as connection:
            for identity, blob in connection.execute(
                "SELECT identity, embedding FROM samples WHERE model_signature = ? ORDER BY rowid",
                (self.model_signature,),
            ):
                yield identity, np.frombuffer(blob, dtype=_VECTOR_DTYPE)

    def _is_identity_directory(self, path: Path) -> bool:
        try:
            return path.is_dir() and not path.is_symlink() and path.resolve() == path
        except OSError:
            return False

    @staticmethod
    def _is_image(path: Path) -> bool:
        try:
            return path.suffix.lower() in _IMAGE_EXTENSIONS and path.is_file() and not path.is_symlink()
        except OSError:
            return False

    def _image_samples(self) -> Iterator[tuple[Path, str]]:
        for path in sorted(self.root.iterdir()):
            if self._is_image(path):
                name, separator, suffix = path.stem.rpartition("_")
                identity = name if separator and suffix.isdigit() else path.stem
                yield path, identity
            elif self._is_identity_directory(path):
                try:
                    images = sorted(path.iterdir())
                except OSError as error:
                    print(f"  [Lỗi] Không thể đọc thư mục ID {path.name}: {error}")
                    continue
                for image in images:
                    if self._is_image(image):
                        yield image, path.name

    def _indexed_sample(self, relative_path: str) -> tuple[str, str, str] | None:
        with self._connection() as connection:
            return connection.execute(
                "SELECT identity, source_hash, model_signature FROM samples WHERE image_path = ?",
                (relative_path,),
            ).fetchone()

    def upsert(self, relative_path: str, identity: str, source_hash: str, embedding: np.ndarray):
        vector = np.asarray(embedding, dtype=_VECTOR_DTYPE).reshape(-1)
        if vector.size != _DIMENSIONS or not np.isfinite(vector).all():
            raise ValueError(f"Embedding cần {_DIMENSIONS} thành phần hữu hạn.")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO samples (image_path, identity, source_hash, model_signature, embedding)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(image_path) DO UPDATE SET identity = excluded.identity,
                       source_hash = excluded.source_hash, model_signature = excluded.model_signature,
                       embedding = excluded.embedding""",
                (relative_path, identity, source_hash, self.model_signature, vector.tobytes()),
            )

    def _remove_snapshot(self, relative_path: str, snapshot: tuple[str, str, str] | None):
        if snapshot is None:
            return
        with self._connection() as connection:
            connection.execute(_DELETE_SNAPSHOT, (relative_path, *snapshot))

    def _is_indexable_image(self, relative_path: str) -> bool:
        relative = PurePosixPath(relative_path)
        if (
            relative.is_absolute()
            or len(relative.parts) not in (1, 2)
            or ".." in relative.parts
            or (os.name == "nt" and "\\" in relative_path)
        ):
            return False
        path = self.root.joinpath(*relative.parts)
        return path.is_relative_to(self.root) and self._is_image(path) and (
            path.parent == self.root or self._is_identity_directory(path.parent)
        )

    def reconcile(self, encode: Callable[[bytes], np.ndarray | None]) -> int:
        """Sync persisted samples with the images currently present on disk."""
        for image, identity in self._image_samples():
            relative_path = image.relative_to(self.root).as_posix()
            snapshot = self._indexed_sample(relative_path)
            try:
                image_bytes = image.read_bytes()
            except OSError as error:
                print(f"  [Lỗi] Không thể đọc ảnh {relative_path}: {error}")
                self._remove_snapshot(relative_path, snapshot)
                continue

            source_hash = sha256(image_bytes).hexdigest()
            if snapshot == (identity, source_hash, self.model_signature):
                continue
            try:
                embedding = encode(image_bytes)
            except Exception as error:
                print(f"  [Lỗi] Không thể xử lý ảnh {relative_path}: {error}")
                embedding = None

            try:
                current_bytes = image.read_bytes()
                if current_bytes != image_bytes:
                    if snapshot != (identity, sha256(current_bytes).hexdigest(), self.model_signature):
                        self._remove_snapshot(relative_path, snapshot)
                    continue
            except OSError as error:
                print(f"  [Lỗi] Không thể kiểm tra lại ảnh {relative_path}: {error}")
                self._remove_snapshot(relative_path, snapshot)
                continue

            if embedding is not None:
                try:
                    self.upsert(relative_path, identity, source_hash, embedding)
                except ValueError as error:
                    print(f"  [Lỗi] Embedding không hợp lệ cho ảnh {relative_path}: {error}")
                else:
                    print(f"  [+] Đã lập chỉ mục: {identity} (tệp: {relative_path})")
                    continue
            else:
                print(f"  [!] Không tìm thấy khuôn mặt hợp lệ trong ảnh: {relative_path}")
            self._remove_snapshot(relative_path, snapshot)

        with self._connection() as connection:
            indexed_samples = connection.execute(
                "SELECT image_path, identity, source_hash, model_signature FROM samples",
            ).fetchall()
        missing_samples = [sample for sample in indexed_samples if not self._is_indexable_image(sample[0])]
        with self._connection() as connection:
            connection.executemany(_DELETE_SNAPSHOT, missing_samples)
            return connection.execute(
                "SELECT COUNT(*) FROM samples WHERE model_signature = ?", (self.model_signature,),
            ).fetchone()[0]

    def save_jpeg(self, identity: str, jpeg_bytes: bytes, embedding: np.ndarray) -> Path:
        folder = self.root / normalize_identity(identity)
        folder.mkdir(exist_ok=True)
        if not self._is_identity_directory(folder):
            raise ValueError("Thư mục ID phải nằm trực tiếp trong thư mục ảnh mẫu.")
        folder = folder.resolve()
        image = folder / f"{uuid4().hex}.jpg"
        relative_path = image.relative_to(self.root).as_posix()
        descriptor, temporary_name = tempfile.mkstemp(dir=folder, prefix=".sample-", suffix=".tmp")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(jpeg_bytes)
                stream.flush()
                os.fsync(stream.fileno())

            # Publish atomically and fail if the target name already exists.
            if os.name == "nt":
                temporary.rename(image)
            else:
                os.link(temporary, image)
            try:
                self.upsert(relative_path, folder.name, sha256(jpeg_bytes).hexdigest(), embedding)
            except Exception:
                image.unlink(missing_ok=True)
                raise
        finally:
            temporary.unlink(missing_ok=True)
        return image
