"""Lớp Dataset: nạp, kiểm tra, metadata, thống kê, chia tập, lưu/nạp lại (image-data-rules.md)."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..imaging.io import IMAGE_EXTENSIONS, load_image, save_image
from ..imaging.validate import InvalidImageError, check_image
from . import synthetic

log = logging.getLogger(__name__)
SKIMAGE_NAMES = ("camera", "coins", "moon", "page", "text", "grass", "brick", "astronaut", "coffee", "chelsea")


@dataclass
class Sample:
    """Một mẫu: ảnh (+ ground truth nếu có) và metadata."""

    name: str
    image: np.ndarray
    edges_gt: np.ndarray | None = None
    labels_gt: np.ndarray | None = None
    classes_gt: dict[int, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        img = self.image
        return {"name": self.name, "height": int(img.shape[0]), "width": int(img.shape[1]),
                "channels": 1 if img.ndim == 2 else int(img.shape[2]), "dtype": str(img.dtype),
                "min": float(img.min()), "max": float(img.max()), "mean": float(img.mean()),
                "std": float(img.std()), "has_edges_gt": self.edges_gt is not None,
                "n_objects": len(self.classes_gt), **self.meta}


@dataclass
class Dataset:
    name: str
    samples: list[Sample]
    skipped: list[dict] = field(default_factory=list)  # ảnh hỏng/không hợp lệ khi nạp

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def batches(self, size: int) -> Iterator[list[Sample]]:
        it = iter(self.samples)
        while chunk := list(itertools.islice(it, size)):
            yield chunk

    # ------------------------------------------------------------ nguồn dữ liệu
    @classmethod
    def from_folder(cls, folder: str | Path, mode: str = "gray") -> "Dataset":
        """Nạp mọi ảnh trong thư mục; ảnh hỏng được ghi vào ``skipped`` thay vì làm dừng cả lô."""
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Không có thư mục: {folder}")
        samples, skipped = [], []
        for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                img = load_image(path, mode)
            except InvalidImageError as exc:
                log.warning("Bỏ qua %s: %s", path.name, exc)
                skipped.append({"path": str(path), "reason": str(exc)})
                continue
            gt_path = path.with_suffix(".edges.npy")
            edges = np.load(gt_path).astype(bool) if gt_path.exists() else None
            samples.append(Sample(path.stem, img, edges, meta={"source": str(path)}))
        return cls(folder.name, samples, skipped)

    @classmethod
    def from_skimage(cls, names: list[str] | tuple[str, ...] = ("camera", "coins"), mode: str = "gray") -> "Dataset":
        from skimage import data
        from ..imaging.convert import rgb_to_gray
        samples = []
        for n in names:
            if n not in SKIMAGE_NAMES:
                raise ValueError(f"Ảnh skimage không hỗ trợ: {n!r}; chọn {SKIMAGE_NAMES}")
            img = getattr(data, n)()
            if img.ndim == 3:
                img = img[..., :3]
                if mode == "gray":
                    img = rgb_to_gray(img)
            elif mode == "rgb":
                img = np.stack([img] * 3, -1)
            samples.append(Sample(n, check_image(img), meta={"source": f"skimage.data.{n}"}))
        return cls("skimage", samples)

    @classmethod
    def synthetic(cls, family: str = "shapes", count: int = 8, size: int = 256, seed: int = 0,
                  degradations: list[dict] | None = None, **family_kwargs) -> "Dataset":
        """Sinh ``count`` ảnh sạch thuộc ``family`` × mỗi bộ suy giảm trong ``degradations``.

        family: shapes | step | isoluminant | video. Seed cố định để tái lập.
        """
        rng = np.random.default_rng(seed)
        degradations = degradations or [{}]
        samples: list[Sample] = []
        for i in range(count):
            if family == "video":
                frames = synthetic.moving_video(size=size, rng=rng, **family_kwargs)
                for fr in frames:
                    samples.append(Sample(f"video{i:02d}_f{fr['frame']:02d}", fr["image"], fr["edges"], fr["labels"],
                                          fr["classes"], {"family": family, "video_id": i, "frame": fr["frame"],
                                                          "seed": seed, **family_kwargs}))
                continue
            if family == "shapes":
                clean = synthetic.shapes_scene(size, rng=rng, **family_kwargs)
            elif family == "step":
                angle = family_kwargs.get("angles", [0, 15, 30, 45, 60, 75, 90])[i % 7]
                clean = synthetic.step_edge(size, angle)
            elif family == "isoluminant":
                clean = synthetic.isoluminant_scene(size, rng)
            else:
                raise ValueError(f"family không hỗ trợ: {family!r}")
            for d in degradations:
                img = synthetic.degrade(clean["image"], np.random.default_rng([seed, i, len(samples)]), **d)
                tag = "_".join(f"{k}{v}" for k, v in d.items()) or "clean"
                samples.append(Sample(f"{family}{i:02d}_{tag}", img, clean["edges"], clean["labels"],
                                      clean["classes"], {"family": family, "index": i, "seed": seed,
                                                         "degradation": d}))
        return cls(f"synthetic_{family}", samples)

    # ------------------------------------------------------------ phân tích
    def stats(self) -> dict:
        desc = [s.describe() for s in self.samples]
        if not desc:
            return {"count": 0, "skipped": len(self.skipped)}
        sizes = sorted({(d["height"], d["width"]) for d in desc})
        return {"count": len(desc), "skipped": len(self.skipped), "sizes": sizes,
                "channels": sorted({d["channels"] for d in desc}), "dtypes": sorted({d["dtype"] for d in desc}),
                "mean_intensity": float(np.mean([d["mean"] for d in desc])),
                "mean_std": float(np.mean([d["std"] for d in desc])),
                "with_ground_truth": sum(d["has_edges_gt"] for d in desc),
                "objects": sum(d["n_objects"] for d in desc)}

    def split(self, test_fraction: float = 0.3, seed: int = 0) -> tuple["Dataset", "Dataset"]:
        """Chia ngẫu nhiên có seed (dùng để chỉnh tham số trên tập 'train', báo cáo trên 'test')."""
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction phải trong (0, 1)")
        idx = np.random.default_rng(seed).permutation(len(self.samples))
        n_test = max(1, int(round(test_fraction * len(idx))))
        test = [self.samples[i] for i in sorted(idx[:n_test])]
        train = [self.samples[i] for i in sorted(idx[n_test:])]
        return Dataset(f"{self.name}_train", train), Dataset(f"{self.name}_test", test)

    # ------------------------------------------------------------ lưu / chuyển định dạng
    def save(self, folder: str | Path) -> Path:
        """Ghi PNG (uint8) + ground truth .npz + metadata.json; nạp lại bằng :meth:`load`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta = []
        for s in self.samples:
            save_image(folder / f"{s.name}.png", s.image)
            np.savez_compressed(folder / f"{s.name}.gt.npz",
                                edges=s.edges_gt if s.edges_gt is not None else np.zeros(0, bool),
                                labels=s.labels_gt if s.labels_gt is not None else np.zeros(0, np.int32))
            meta.append({**s.describe(), "classes": {str(k): v for k, v in s.classes_gt.items()}})
        (folder / "metadata.json").write_text(json.dumps({"name": self.name, "samples": meta},
                                                         ensure_ascii=False, indent=2, default=str))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Dataset":
        folder = Path(folder)
        info = json.loads((folder / "metadata.json").read_text())
        samples = []
        for m in info["samples"]:
            mode = "rgb" if m["channels"] == 3 else "gray"
            img = load_image(folder / f"{m['name']}.png", mode)
            gt = np.load(folder / f"{m['name']}.gt.npz")
            samples.append(Sample(m["name"], img, gt["edges"] if gt["edges"].size else None,
                                  gt["labels"] if gt["labels"].size else None,
                                  {int(k): v for k, v in m.get("classes", {}).items()}, {"source": str(folder)}))
        return cls(info["name"], samples)
