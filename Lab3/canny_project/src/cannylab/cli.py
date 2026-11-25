"""Dòng lệnh: python -m cannylab {list | run CONFIG | verify | dataset}.

Đây là chỗ duy nhất được print (python-coding-rules.md mục 5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _cmd_list(_: argparse.Namespace) -> int:
    from . import registry
    for spec in sorted(registry.available(), key=lambda s: (s.family, s.name)):
        print(f"{spec.family:9s} {spec.implementation:8s} {spec.name:24s} {spec.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .experiments import load_config, run_experiment
    for path in args.configs:
        cfg = load_config(path)
        out = run_experiment(cfg, args.out if len(args.configs) == 1 else None)
        print(f"[{cfg.name}] -> {out}")
        print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Kiểm chứng nhanh: Canny tự cài so với cv2.Canny và skimage.feature.canny trên ảnh skimage.data."""
    import cv2
    from skimage import feature

    from .core import OPENCV, SKIMAGE, canny
    from .datasets import Dataset
    report = []
    for s in Dataset.from_skimage(args.images):
        for lo, hi, l2 in [(100, 200, False), (50, 150, True)]:
            diff = int((canny(s.image, None, lo, hi, profile=OPENCV, norm="L2" if l2 else "L1").edges
                        ^ (cv2.Canny(s.image, lo, hi, L2gradient=l2) > 0)).sum())
            report.append({"image": s.name, "profile": "opencv", "params": f"{lo}/{hi} L2={l2}", "n_diff": diff})
        for sigma in (1.0, 2.0):
            diff = int((canny(s.image, sigma, profile=SKIMAGE).edges ^ feature.canny(s.image, sigma=sigma)).sum())
            report.append({"image": s.name, "profile": "skimage", "params": f"sigma={sigma}", "n_diff": diff})
    for r in report:
        print(f"{r['image']:10s} {r['profile']:8s} {r['params']:18s} pixel lệch = {r['n_diff']}")
    bad = [r for r in report if r["n_diff"]]
    print("KẾT LUẬN:", "trùng tuyệt đối" if not bad else f"{len(bad)} cấu hình lệch")
    return 1 if bad else 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from .datasets import Dataset
    degr = json.loads(args.degradations) if args.degradations else None
    ds = Dataset.synthetic(args.family, args.count, args.size, args.seed, degr)
    path = ds.save(args.out)
    print(json.dumps(ds.stats(), ensure_ascii=False, indent=2, default=str))
    print(f"Đã lưu {len(ds)} mẫu vào {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cannylab", description="Canny: tự cài đặt, thực nghiệm, đánh giá")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="liệt kê thuật toán đã đăng ký").set_defaults(fn=_cmd_list)
    p = sub.add_parser("run", help="chạy một hoặc nhiều file cấu hình TOML")
    p.add_argument("configs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="thư mục kết quả (mặc định results/<tên>_<thời gian>)")
    p.set_defaults(fn=_cmd_run)
    p = sub.add_parser("verify", help="so khớp Canny tự cài với OpenCV/skimage")
    p.add_argument("--images", nargs="+", default=["camera", "coins", "moon", "page"])
    p.set_defaults(fn=_cmd_verify)
    p = sub.add_parser("dataset", help="sinh và lưu bộ dữ liệu tổng hợp")
    p.add_argument("out", type=Path)
    p.add_argument("--family", default="shapes")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--degradations", default=None, help='JSON, ví dụ \'[{"noise_sigma":0.05}]\'')
    p.set_defaults(fn=_cmd_dataset)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
