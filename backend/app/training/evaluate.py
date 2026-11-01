"""Evaluate a detection checkpoint on a dataset split (CPU).

Reports precision, recall, F1, mAP50 and mAP50-95 (overall and per class), flags classes with
too few test instances or weak results, and saves the confusion matrix, PR/F1 curves, validation
batch predictions and annotated predictions for every image of the split under
experiments/evaluation/<run>/.

Usage:
    python -m app.training.evaluate --model models/custom/digital_twin_best.pt \
        --data dataset/digital_twin_yolo/data.yaml --split test
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from app.core.config import PROJECT_DIR
from app.core.device import resolve_device
from app.training.taxonomy import CLASS_NAMES

MIN_TEST_INSTANCES = 10  # fewer than this -> per-class metrics are not statistically meaningful
WEAK_MAP50 = 0.5


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def class_flags(per_class: dict[str, dict]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {}
    for name in CLASS_NAMES.values():
        m = per_class.get(name)
        notes = []
        if m is None or m.get("instances", 0) == 0:
            notes.append("no instances in this split")
        else:
            if m["instances"] < MIN_TEST_INSTANCES:
                notes.append(f"only {m['instances']} instances (< {MIN_TEST_INSTANCES}); metrics unreliable")
            if m["mAP50"] < WEAK_MAP50:
                notes.append(f"weak mAP50 ({m['mAP50']:.2f})")
        if notes:
            flags[name] = notes
    return flags


def evaluate(model_path: Path, data: Path, split: str, imgsz: int, batch: int, save_predictions: bool = True) -> dict:
    from ultralytics import YOLO

    project = PROJECT_DIR / "experiments" / "evaluation"
    name = datetime.now().strftime(f"{model_path.stem}_{split}_%Y%m%d_%H%M%S")
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data.resolve()), split=split, imgsz=imgsz, batch=batch,
                        device=resolve_device("cpu"), project=str(project), name=name, exist_ok=False, plots=True)
    save_dir = Path(metrics.save_dir)

    box = metrics.box
    counts = getattr(metrics, "nt_per_class", None)
    summary = {
        "model": str(model_path),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "f1": round(f1_score(float(box.mp), float(box.mr)), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "speed_ms_per_image": {k: round(v, 2) for k, v in metrics.speed.items()},
        "per_class": {},
    }
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        summary["per_class"][metrics.names[int(cls)]] = {
            "instances": int(counts[int(cls)]) if counts is not None else None,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(f1_score(float(p), float(r)), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap), 4),
        }
    summary["class_flags"] = class_flags(summary["per_class"])

    if save_predictions:
        cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
        images = Path(cfg.get("path", data.parent)) / cfg[split]
        model.predict(source=str(images), imgsz=imgsz, device=resolve_device("cpu"), conf=0.25, save=True,
                      project=str(save_dir), name=f"{split}_predictions", exist_ok=False, verbose=False, stream=False)
        summary["prediction_images"] = str(save_dir / f"{split}_predictions")

    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["save_dir"] = str(save_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on CPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--no-predictions", action="store_true", help="Skip saving annotated split images")
    args = parser.parse_args()

    s = evaluate(args.model, args.data, args.split, args.imgsz, args.batch, not args.no_predictions)
    print(json.dumps({k: s[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")}, indent=2))
    print(f"{'class':<20}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for name, m in s["per_class"].items():
        print(f"{name:<20}{m['instances'] or '-':>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['mAP50']:>8.3f}{m['mAP50-95']:>10.3f}")
    for name, notes in s["class_flags"].items():
        print(f"  ! {name}: {'; '.join(notes)}")
    print("Comparison with the pretrained COCO model is not meaningful for these classes: none of the seven "
          "components exist in COCO, so YOLO11n-pretrained scores 0 by construction.")
    print(f"Saved to {s['save_dir']}")


if __name__ == "__main__":
    main()
