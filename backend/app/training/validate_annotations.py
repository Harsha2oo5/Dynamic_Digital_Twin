"""Validate YOLO-format annotations exported from CVAT / Label Studio / Roboflow.

Checks, against the seven-class taxonomy and the annotation manifest:
  - every label file pairs with exactly one original image (and reports images without labels)
  - class-name files in the export (classes.txt, obj.names, data.yaml, notes.json) match the taxonomy order
  - each line has 5 values, class id in range, coordinates normalised, positive size, box inside the image
  - each image contains at least one box of its folder's class (folder = the component it was collected for)
  - empty label files, tiny boxes, duplicate boxes (warnings)

Usage:
    python -m app.training.validate_annotations ../annotation/exports/my_export [--manifest ../annotation/manifest.csv]
Exit code 1 if any error is found.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.training.annotation_kit import image_key, read_manifest
from app.training.dataset_validator import IMAGE_EXTS
from app.training.taxonomy import CLASS_IDS, CLASS_NAMES, NUM_CLASSES, names_match_taxonomy

EPS = 1e-3  # tolerance for float rounding in exported coordinates
MIN_BOX_AREA = 1e-4  # boxes smaller than 0.01 % of the image are suspicious
MAX_BOX_AREA = 0.95  # boxes covering almost the whole image need a second look
DUPLICATE_IOU = 0.95  # same-class boxes this similar are an annotation mistake
CABLE_OVERLAP_IOU = 0.30  # overlapping cable boxes contradict the bundle rule
CABLE_ID = CLASS_IDS["cable"]
CANDIDATE_MARKER = "CANDIDATES_NOT_GROUND_TRUTH.txt"  # written by auto_annotate into every candidate run
VERIFIED_NEGATIVES = "verified_negatives.txt"  # reviewer-confirmed images without any target component
NON_LABEL_TXT = {"classes.txt", "train.txt", "valid.txt", "val.txt", "test.txt", "obj.names", "readme.txt",
                 "verified_negatives.txt", "candidates_not_ground_truth.txt"}


@dataclass
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class ValidationReport:
    label_files: int = 0
    paired: dict[str, Path] = field(default_factory=dict)  # manifest key -> label file
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)  # original paths without a label file
    unmatched_label_files: list[str] = field(default_factory=list)
    instances: Counter = field(default_factory=Counter)  # class id -> boxes
    images_with_class: Counter = field(default_factory=Counter)
    names_source: str | None = None
    verified_negatives: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "label_files": self.label_files,
            "paired_images": len(self.paired),
            "missing_labels": len(self.missing_labels),
            "unmatched_label_files": self.unmatched_label_files,
            "errors": self.errors,
            "warnings": self.warnings,
            "class_names_checked_against": self.names_source,
            "verified_negative_images": len(self.verified_negatives),
            "instances_per_class": {CLASS_NAMES[c]: self.instances[c] for c in CLASS_NAMES},
            "images_per_class": {CLASS_NAMES[c]: self.images_with_class[c] for c in CLASS_NAMES},
        }


def parse_line(line: str) -> Box:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 values, got {len(parts)}")
    try:
        cls_f = float(parts[0])
        x, y, w, h = map(float, parts[1:])
    except ValueError as exc:
        raise ValueError("non-numeric value") from exc
    if not cls_f.is_integer():
        raise ValueError(f"class id {parts[0]} is not an integer")
    return Box(int(cls_f), x, y, w, h)


def check_box(box: Box) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one normalised YOLO box."""
    errors, warnings = [], []
    if not 0 <= box.cls < NUM_CLASSES:
        errors.append(f"class id {box.cls} outside 0..{NUM_CLASSES - 1}")
    if not all(-EPS <= v <= 1 + EPS for v in (box.x, box.y, box.w, box.h)):
        errors.append("coordinates not normalised to [0, 1]")
    if box.w <= 0 or box.h <= 0:
        errors.append("non-positive width/height")
    elif (box.x - box.w / 2 < -EPS or box.x + box.w / 2 > 1 + EPS
          or box.y - box.h / 2 < -EPS or box.y + box.h / 2 > 1 + EPS):
        errors.append("box extends outside the image")
    elif box.w * box.h < MIN_BOX_AREA:
        warnings.append(f"tiny box ({box.w * box.h:.6f} of image area)")
    elif box.w * box.h > MAX_BOX_AREA:
        warnings.append(f"near-whole-image box ({box.w * box.h:.2f} of image area): confirm it is a real object extent")
    return errors, warnings


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x - a.w / 2, a.y - a.h / 2, a.x + a.w / 2, a.y + a.h / 2
    bx1, by1, bx2, by2 = b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def read_label_file(path: Path) -> tuple[list[Box], list[str], list[str]]:
    boxes, errors, warnings = [], [], []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            box = parse_line(raw)
        except ValueError as exc:
            errors.append(f"{path.name}:{lineno}: {exc}")
            continue
        errs, warns = check_box(box)
        errors += [f"{path.name}:{lineno}: {e}" for e in errs]
        warnings += [f"{path.name}:{lineno}: {w}" for w in warns]
        boxes.append(box)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].cls != boxes[j].cls:
                continue
            overlap = iou(boxes[i], boxes[j])
            if overlap > DUPLICATE_IOU:
                errors.append(f"{path.name}: duplicate {CLASS_NAMES.get(boxes[i].cls)} boxes (IoU {overlap:.2f})")
            elif boxes[i].cls == CABLE_ID and overlap > CABLE_OVERLAP_IOU:
                warnings.append(f"{path.name}: overlapping cable boxes (IoU {overlap:.2f}); an inseparable bundle "
                                "must be ONE box")
    return boxes, errors, warnings


def is_image_list_file(path: Path) -> bool:
    """CVAT's YOLO 1.1 export adds subset lists (train.txt, default.txt, …) whose lines are image paths."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return bool(lines) and all(Path(ln).suffix.lower() in IMAGE_EXTS for ln in lines)


def find_class_names(export_dir: Path) -> tuple[list[str] | None, str | None]:
    """Look for the class list the annotation tool exported."""
    for name in ("classes.txt", "obj.names"):
        for p in export_dir.rglob(name):
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()], str(p)
    for name in ("data.yaml", "dataset.yaml"):
        for p in export_dir.rglob(name):
            names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            return names, str(p)
    for p in export_dir.rglob("notes.json"):  # Label Studio
        cats = json.loads(p.read_text(encoding="utf-8")).get("categories", [])
        return [c["name"] for c in sorted(cats, key=lambda c: c["id"])], str(p)
    return None, None


def candidate_marker(export_dir: Path) -> Path | None:
    """Machine-generated candidate runs carry a marker file; such directories are never ground truth."""
    for d in [export_dir, *export_dir.resolve().parents]:
        if (d / CANDIDATE_MARKER).is_file():
            return d / CANDIDATE_MARKER
    return next(export_dir.rglob(CANDIDATE_MARKER), None)


def read_verified_negatives(export_dir: Path) -> set[str]:
    """Image names (one per line) that a reviewer confirmed contain none of the seven classes."""
    path = export_dir / VERIFIED_NEGATIVES
    if not path.is_file():
        return set()
    return {image_key(ln.strip()) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def validate_export(export_dir: Path, manifest: dict[str, dict], require_all: bool = False) -> ValidationReport:
    rep = ValidationReport()
    marker = candidate_marker(export_dir)
    if marker is not None:
        rep.errors.append(f"{export_dir} contains machine-generated candidates ({marker}); only human-reviewed "
                          "exports may be validated as ground truth")
        return rep
    negatives = read_verified_negatives(export_dir)
    rep.verified_negatives = sorted(negatives)
    names, source = find_class_names(export_dir)
    rep.names_source = source
    if names is None:
        rep.warnings.append("No class list found in export (classes.txt / obj.names / data.yaml / notes.json); "
                            "make sure the tool's label order equals classes.txt from the annotation kit.")
    elif not names_match_taxonomy(names):
        rep.errors.append(f"Class list in {source} does not match the taxonomy order {list(CLASS_NAMES.values())}: "
                          f"{names}. Class ids would be misinterpreted.")

    by_key: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(export_dir.rglob("*.txt")):
        if p.name.lower() in NON_LABEL_TXT or is_image_list_file(p):
            continue
        by_key[image_key(p.name)].append(p)
    rep.label_files = sum(len(v) for v in by_key.values())

    for key, files in by_key.items():
        if key not in manifest:
            rep.unmatched_label_files += [str(f.relative_to(export_dir)) for f in files]
            continue
        if len(files) > 1:
            rep.errors.append(f"{manifest[key]['original_path']}: {len(files)} label files map to this image")
            continue
        rep.paired[key] = files[0]

    if rep.unmatched_label_files:
        rep.errors.append(f"{len(rep.unmatched_label_files)} label files do not match any dataset image")

    for key, row in manifest.items():
        label = rep.paired.get(key)
        if label is None:
            rep.missing_labels.append(row["original_path"])
            continue
        boxes, errors, warnings = read_label_file(label)
        rep.errors += [f"{row['original_path']} ({e})" for e in errors]
        rep.warnings += [f"{row['original_path']} ({w})" for w in warnings]
        if key in negatives:
            if boxes:
                rep.errors.append(f"{row['original_path']}: listed as verified negative but has {len(boxes)} boxes")
            continue
        if not boxes:
            rep.errors.append(f"{row['original_path']}: label file is empty; annotate it or list it in "
                              f"{VERIFIED_NEGATIVES} if a reviewer confirmed it shows none of the seven classes")
            continue
        rep.instances.update(b.cls for b in boxes)
        rep.images_with_class.update({b.cls for b in boxes})
        folder_cls = row["folder_class_id"]
        if folder_cls is not None and folder_cls not in {b.cls for b in boxes}:
            rep.errors.append(f"{row['original_path']}: no '{CLASS_NAMES[folder_cls]}' box, but the image comes from "
                              f"the {CLASS_NAMES[folder_cls]} folder")

    if require_all and rep.missing_labels:
        rep.errors.append(f"{len(rep.missing_labels)} images have no label file")
    elif rep.missing_labels:
        rep.warnings.append(f"{len(rep.missing_labels)} images have no label file yet (excluded from the dataset)")
    if not rep.paired:
        rep.errors.append("No label files could be paired with dataset images")
    return rep


def main() -> None:
    from app.core.config import PROJECT_DIR

    parser = argparse.ArgumentParser(description="Validate YOLO annotations against the Digital Twin taxonomy.")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "annotation" / "manifest.csv")
    parser.add_argument("--require-all", action="store_true", help="Treat unlabelled images as an error")
    parser.add_argument("--json", type=Path, help="Write the full report to this file")
    args = parser.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"Manifest not found: {args.manifest} (run `python -m app.training.annotation_kit` first)")
    rep = validate_export(args.export_dir, read_manifest(args.manifest), args.require_all)
    s = rep.summary()
    print(f"Label files: {s['label_files']}  paired: {s['paired_images']}  images without labels: {s['missing_labels']}")
    print("Instances per class:", s["instances_per_class"])
    for e in rep.errors[:50]:
        print(f"  ERROR   {e}")
    for w in rep.warnings[:30]:
        print(f"  warning {w}")
    if args.json:
        args.json.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("VALID" if rep.ok else f"INVALID ({len(rep.errors)} errors)")
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
