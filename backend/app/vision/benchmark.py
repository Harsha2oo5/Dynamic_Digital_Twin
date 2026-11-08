"""CPU benchmark: detection latency / FPS per model and input size, tracking + identity cost, peak memory.

Images are read (never written) from the original dataset. Run with nothing else using the CPU.

Usage (from backend/):
    python -m app.vision.benchmark [--models models/pretrained/yolo11n.pt models/custom/digital_twin_v2.pt]
                                   [--sizes 320 416 640] [--images 30] [--out ../experiments/benchmarks/<name>.json]
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from app.core.config import BACKEND_DIR, PROJECT_DIR, get_settings
from app.core.device import configure_torch
from app.vision.detector import Detector
from app.vision.track_manager import TrackingSession
from app.vision.tracker import load_tracker_args


def sample_images(root: Path, n: int, seed: int = 0) -> list[np.ndarray]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and
                   not p.name.startswith("._"))
    rng = np.random.default_rng(seed)
    picks = [files[i] for i in rng.choice(len(files), size=min(n, len(files)), replace=False)]
    return [img for img in (cv2.imread(str(p)) for p in picks) if img is not None]


def pan_sequence(img: np.ndarray, frames: int = 40, out=(640, 480)) -> list[np.ndarray]:
    """Synthetic camera pan across one real photo (for the tracking-cost measurement)."""
    h, w = img.shape[:2]
    cw, ch = int(w * 0.7), int(h * 0.7)
    return [cv2.resize(img[(h - ch) // 2:(h - ch) // 2 + ch, int((w - cw) * i / (frames - 1)):][:, :cw], out)
            for i in range(frames)]


def bench_model(path: Path, sizes: list[int], images: list[np.ndarray], tracker_cfg: str, conf: float) -> dict:
    proc = psutil.Process()
    det = Detector(path, "cpu")
    t0 = time.perf_counter()
    if not det.load(warmup=False):
        return {"model": str(path), "error": det.error}
    out = {"model": path.name, "load_s": round(time.perf_counter() - t0, 2), "sizes": {}}
    for size in sizes:
        det.predict(images[0], conf=conf, iou=0.45, imgsz=size)  # warm-up
        lat, n_det = [], 0
        for img in images:
            r = det.predict(img, conf=conf, iou=0.45, imgsz=size)
            lat.append(r.inference_ms)
            n_det += len(r)
        lat.sort()
        out["sizes"][size] = {"mean_ms": round(statistics.mean(lat), 1), "p50_ms": round(lat[len(lat) // 2], 1),
                              "p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1),
                              "fps": round(1000 / statistics.mean(lat), 2),
                              "detections_per_image": round(n_det / len(images), 2)}
    # tracking + lifecycle + identity on top of detections (416 px)
    seq = pan_sequence(images[0])
    sess = TrackingSession(load_tracker_args(tracker_cfg), "bench", det.names, frame_rate=8)
    track_ms = []
    for i, frame in enumerate(seq):
        r = det.predict(frame, conf=conf, iou=0.45, imgsz=416)
        s = time.perf_counter()
        sess.step(r, frame, i / 8)
        track_ms.append((time.perf_counter() - s) * 1000)
    out["tracking"] = {"frames": len(seq), "mean_ms_per_frame": round(statistics.mean(track_ms), 3),
                       "max_ms": round(max(track_ms), 3), **sess.counts()}
    out["rss_mb_after"] = round(proc.memory_info().rss / 1e6, 1)
    return out


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="CPU benchmark of detection + tracking.")
    parser.add_argument("--models", nargs="+", type=Path, default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[320, 416, 640])
    parser.add_argument("--images", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    configure_torch(s.torch_threads)
    models = args.models or [BACKEND_DIR / s.pretrained_model_path] + sorted(
        (BACKEND_DIR / "models" / "custom").glob("digital_twin_v*.pt"))[-1:]
    images = sample_images(s.resolve(s.dataset_root), args.images)
    tracker_cfg = str(s.resolve(Path(s.tracker_config)))
    proc = psutil.Process()
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "cpu": platform.processor(),
              "logical_cpus": psutil.cpu_count(), "images": len(images),
              "rss_mb_start": round(proc.memory_info().rss / 1e6, 1), "results": []}
    for m in models:
        m = m if m.is_absolute() else BACKEND_DIR / m
        print(f"Benchmarking {m.name} ...")
        report["results"].append(bench_model(m, args.sizes, images, tracker_cfg, args.conf))
    report["peak_rss_mb"] = round(max(r.get("rss_mb_after", 0) for r in report["results"]), 1)
    out = args.out or PROJECT_DIR / "experiments" / "benchmarks" / f"vision_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["results"]:
        if "error" in r:
            print(f"{r['model']}: {r['error']}")
            continue
        sizes = "  ".join(f"{k}px {v['mean_ms']} ms ({v['fps']} FPS)" for k, v in r["sizes"].items())
        print(f"{r['model']}: {sizes} | tracking {r['tracking']['mean_ms_per_frame']} ms/frame | "
              f"RSS {r['rss_mb_after']} MB")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
