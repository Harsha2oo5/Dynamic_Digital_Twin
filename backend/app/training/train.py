"""Fine-tune YOLO11n on the Digital Twin dataset (CPU). Never launched automatically.

Modes:
    --estimate   benchmark one training step on this machine and project the workload (no labels needed)
    --pilot      short smoke-test run (default 5 epochs) to measure real per-epoch time and sanity-check learning
    (default)    full run (default 30 epochs, early-stopping patience 8)

Usage:
    python -m app.training.train --estimate --train-images 380 --val-images 108
    python -m app.training.train --data dataset/digital_twin_yolo/data.yaml --pilot
    python -m app.training.train --data dataset/digital_twin_yolo/data.yaml --epochs 30 --promote
"""

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from app.core.config import BACKEND_DIR, PROJECT_DIR
from app.core.device import resolve_device
from app.training.taxonomy import names_match_taxonomy

PRETRAINED = BACKEND_DIR / "models" / "pretrained" / "yolo11n.pt"
CUSTOM_CHECKPOINT = BACKEND_DIR / "models" / "custom" / "digital_twin_best.pt"
RUNS_DIR = PROJECT_DIR / "experiments" / "training"

DEFAULTS = {"imgsz": 640, "batch": 4, "epochs": 30, "pilot_epochs": 5, "patience": 8, "workers": 2, "seed": 42}


def build_train_args(data: Path, *, epochs: int, imgsz: int, batch: int, patience: int, workers: int, seed: int,
                     name: str, project: Path = RUNS_DIR) -> dict:
    """Ultralytics train() arguments. CPU-only, reproducible, never overwrites a run."""
    return {
        "data": str(Path(data).resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "workers": workers,
        "seed": seed,
        "deterministic": True,
        "device": resolve_device("cpu"),
        "amp": False,  # mixed precision is a GPU feature
        "project": str(project),
        "name": name,
        "exist_ok": False,
        "plots": True,  # training curves, confusion matrix, PR/F1 curves, val batch predictions
        "save": True,  # best.pt and last.pt
        "pretrained": True,  # transfer learning from the COCO checkpoint
        "verbose": True,
    }


def check_training_ready(data_yaml: Path, require_review: bool = True) -> list[str]:
    """Return blocking problems; an empty list means training may start.

    `require_review`: the dataset must have been built by prepare_dataset from a human-reviewed export
    (split_report.json records the review manifest). Off only for unit tests on synthetic data."""
    from app.training.prepare_dataset import verify_prepared

    if not data_yaml.is_file():
        return [f"data.yaml not found: {data_yaml}"]
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    problems = []
    if not names_match_taxonomy(cfg.get("names", {})):
        problems.append("data.yaml class names do not match the seven-class taxonomy")
    root = Path(cfg.get("path", data_yaml.parent))
    for split in ("train", "val"):
        labels = root / "labels" / split
        if not labels.is_dir() or not any(labels.glob("*.txt")):
            problems.append(f"No labels for split '{split}' in {labels}")
    if not problems and (root / "split_manifest.csv").is_file():
        problems += verify_prepared(root)
    if require_review:
        report = root / "split_report.json"
        reviewed = report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("review_manifest")
        if not reviewed:
            problems.append("dataset was not built from a human-reviewed export (no review manifest in "
                            "split_report.json); run review_tool export + prepare_dataset")
    return problems


def _tensors(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors(v)


def benchmark_train_step(imgsz: int, batch: int, steps: int = 3, model_path: Path = PRETRAINED) -> dict:
    """Measure forward+backward time of one training batch on CPU (random input, real network).

    Excludes data loading, augmentation, loss assignment and validation, so it is a LOWER BOUND
    on real training time. The pilot run gives the real per-epoch time.
    """
    import torch
    from ultralytics import YOLO

    net = YOLO(str(model_path)).model.to("cpu").train()
    for p in net.parameters():
        p.requires_grad_(True)
    x = torch.rand(batch, 3, imgsz, imgsz)
    times = []
    for i in range(steps + 1):  # first step is warm-up
        start = time.perf_counter()
        out = net(x)
        loss = sum(t.float().mean() for t in _tensors(out))
        loss.backward()
        net.zero_grad(set_to_none=True)
        if i:
            times.append(time.perf_counter() - start)
    net.eval()
    with torch.no_grad():
        start = time.perf_counter()
        net(torch.rand(1, 3, imgsz, imgsz))
        infer = time.perf_counter() - start
    return {"train_step_s": sum(times) / len(times), "val_image_s": infer, "imgsz": imgsz, "batch": batch,
            "torch_threads": torch.get_num_threads()}


def project_workload(bench: dict, train_images: int, val_images: int, epochs: int) -> dict:
    steps = -(-train_images // bench["batch"])
    epoch_s = steps * bench["train_step_s"] + val_images * bench["val_image_s"]
    return {"steps_per_epoch": steps, "epoch_s_lower_bound": round(epoch_s, 1),
            "total_h_lower_bound": round(epoch_s * epochs / 3600, 2), "epochs": epochs}


def read_results_csv(run_dir: Path) -> dict:
    """Final and best epoch rows from Ultralytics results.csv."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [{k.strip(): float(v) for k, v in r.items() if v.strip()} for r in csv.DictReader(f)]
    if not rows:
        return {}
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: r.get(key, 0.0))
    return {"epochs_completed": len(rows), "final": rows[-1], "best": best}


def promote_checkpoint(best: Path, target: Path = CUSTOM_CHECKPOINT) -> Path:
    """Copy best.pt to the custom-model slot, backing up any existing checkpoint first.
    The pretrained checkpoint is never touched."""
    if target.resolve() == PRETRAINED.resolve():
        raise ValueError("Refusing to overwrite the pretrained checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(f"{target.stem}_{datetime.now():%Y%m%d_%H%M%S}{target.suffix}")
        shutil.copy2(target, backup)
        print(f"Backed up previous checkpoint to {backup}")
    shutil.copy2(best, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n on CPU.")
    parser.add_argument("--data", type=Path, help="Path to data.yaml")
    parser.add_argument("--model", type=Path, default=PRETRAINED)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--name", default=None, help="Run name (default: pilot_/run_ + timestamp)")
    parser.add_argument("--pilot", action="store_true", help=f"Smoke test: {DEFAULTS['pilot_epochs']} epochs")
    parser.add_argument("--estimate", action="store_true", help="Benchmark and project the workload, then exit")
    parser.add_argument("--train-images", type=int, default=None, help="For --estimate without a dataset")
    parser.add_argument("--val-images", type=int, default=None, help="For --estimate without a dataset")
    parser.add_argument("--promote", action="store_true", help="Copy best.pt to models/custom/ after training")
    args = parser.parse_args()
    epochs = DEFAULTS["pilot_epochs"] if args.pilot else args.epochs

    if args.estimate:
        n_train, n_val = args.train_images, args.val_images
        if args.data and args.data.is_file():
            root = Path(yaml.safe_load(args.data.read_text(encoding="utf-8")).get("path", args.data.parent))
            n_train = n_train or sum(1 for _ in (root / "images" / "train").iterdir())
            n_val = n_val or sum(1 for _ in (root / "images" / "val").iterdir())
        if not n_train or n_val is None:
            sys.exit("--estimate needs --data or --train-images/--val-images")
        bench = benchmark_train_step(args.imgsz, args.batch, model_path=args.model)
        print(json.dumps({"benchmark": bench, "projection": project_workload(bench, n_train, n_val, epochs)}, indent=2))
        print("Projection is a lower bound (excludes data loading, augmentation and loss computation).")
        return

    if not args.data:
        sys.exit("--data is required")
    problems = check_training_ready(args.data)
    if problems:
        print("Training blocked:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    from ultralytics import YOLO
    import torch
    import ultralytics

    name = args.name or datetime.now().strftime(f"{'pilot' if args.pilot else 'run'}_%Y%m%d_%H%M%S")
    if (RUNS_DIR / name).exists():
        sys.exit(f"Run {RUNS_DIR / name} already exists; choose another --name")
    train_args = build_train_args(args.data, epochs=epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                                  workers=args.workers, seed=args.seed, name=name)

    model = YOLO(str(args.model))  # pretrained file is only read
    started = time.time()
    model.train(**train_args)
    duration = time.time() - started

    run_dir = RUNS_DIR / name
    summary = {
        "run": name,
        "mode": "pilot" if args.pilot else "full",
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "train_args": train_args,
        "base_model": str(args.model),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "cpu": platform.processor()},
        "results": read_results_csv(run_dir),
        "weights": {"best": str(run_dir / "weights" / "best.pt"), "last": str(run_dir / "weights" / "last.pt")},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training finished in {duration / 60:.1f} min. Summary: {run_dir / 'run_summary.json'}")

    best = run_dir / "weights" / "best.pt"
    if args.promote and best.is_file():
        print(f"Promoted to {promote_checkpoint(best)}. Set DETECTION_MODEL=custom in backend/.env to use it.")


if __name__ == "__main__":
    main()
