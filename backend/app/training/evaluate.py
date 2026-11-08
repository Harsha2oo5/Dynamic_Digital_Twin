"""Evaluate a detection checkpoint on a dataset split (CPU).

Reports precision, recall, F1, mAP50 and mAP50-95 (overall and per class), flags classes with
too few test instances or weak results, and saves the confusion matrix, PR/F1 curves, validation
batch predictions and annotated predictions for every image of the split under
experiments/evaluation/<run>/.

On a dataset built from pseudo-labels (`prepare_dataset --pseudo`) the reference boxes are automatic labels, so
the same numbers measure AGREEMENT WITH PSEUDO-LABELS, not accuracy; metrics.json then carries
`label_source: pseudo` and `metrics_meaning`, and the CLI prints them under that heading.

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
# Look-alike pairs worth watching on BTS equipment (both directions are reported).
CONFUSION_PAIRS = [("antenna", "radio_unit"), ("router", "power_supply_unit"), ("battery", "power_supply_unit"),
                   ("equipment_cabinet", "battery"), ("equipment_cabinet", "power_supply_unit"),
                   ("equipment_cabinet", "router"), ("equipment_cabinet", "radio_unit")]


def label_provenance(data: Path) -> dict:
    """label_source of the dataset (from prepare_dataset's split_report.json) and what the metrics mean."""
    cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8")) or {}
    report = Path(cfg.get("path", Path(data).parent)) / "split_report.json"
    src = json.loads(report.read_text(encoding="utf-8")).get("label_source", "unknown") if report.is_file() else "unknown"
    meaning = {"pseudo": "agreement with automatic pseudo-labels (NOT accuracy; no human-verified reference)",
               "reviewed": "accuracy against human-reviewed labels"}.get(src, "unknown label provenance")
    return {"label_source": src, "metrics_meaning": meaning}


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


def confusion_analysis(matrix: list[list[float]], names: dict[int, str]) -> dict:
    """Read an Ultralytics detection confusion matrix (matrix[predicted][true], last index = background).

    Per class: ground-truth boxes, correctly classified, confused with another class, missed (no
    prediction), background false positives (prediction with no object). Plus the look-alike pairs."""
    nc = len(matrix) - 1
    by_name = {n: i for i, n in names.items()}
    per_class = {}
    for t in range(nc):
        gt = sum(matrix[p][t] for p in range(nc + 1))
        confused = {names[p]: int(matrix[p][t]) for p in range(nc) if p != t and matrix[p][t]}
        per_class[names[t]] = {"ground_truth": int(gt), "correct": int(matrix[t][t]),
                               "confused_as": dict(sorted(confused.items(), key=lambda kv: -kv[1])),
                               "missed": int(matrix[nc][t]), "background_false_positives": int(matrix[t][nc])}
    pairs = []
    for a, b in CONFUSION_PAIRS:
        if a in by_name and b in by_name:
            ia, ib = by_name[a], by_name[b]
            pairs.append({"pair": f"{a} <-> {b}", f"{a}_predicted_as_{b}": int(matrix[ib][ia]),
                          f"{b}_predicted_as_{a}": int(matrix[ia][ib])})
    return {"per_class": per_class, "pairs": pairs,
            "note": "Cable bundle vs individual cables is a box-granularity question the confusion matrix cannot "
                    "show; inspect the cable prediction images."}


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
    summary.update(label_provenance(data))
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        summary["confusion"] = confusion_analysis(cm.matrix.tolist(), metrics.names)

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
    print(f"Label source: {s['label_source']} -> metrics are {s['metrics_meaning']}")
    if s["label_source"] == "pseudo":
        print("=== Agreement with pseudo-labels (not accuracy) ===")
    print(json.dumps({k: s[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")}, indent=2))
    print(f"{'class':<20}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for name, m in s["per_class"].items():
        print(f"{name:<20}{m['instances'] or '-':>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['mAP50']:>8.3f}{m['mAP50-95']:>10.3f}")
    for name, notes in s["class_flags"].items():
        print(f"  ! {name}: {'; '.join(notes)}")
    if "confusion" in s:
        print("Confusions (counts from the confusion matrix):")
        for name, c in s["confusion"]["per_class"].items():
            if c["ground_truth"] or c["background_false_positives"]:
                print(f"  {name:<20} gt {c['ground_truth']:>4}  correct {c['correct']:>4}  missed {c['missed']:>4}  "
                      f"background FP {c['background_false_positives']:>4}  confused as {c['confused_as'] or '-'}")
        for pr in s["confusion"]["pairs"]:
            print("  " + "  ".join(f"{k}: {v}" for k, v in pr.items()))
    print("Comparison with the pretrained COCO model is not meaningful for these classes: none of the seven "
          "components exist in COCO, so YOLO11n-pretrained scores 0 by construction.")
    print(f"Saved to {s['save_dir']}")


if __name__ == "__main__":
    main()
