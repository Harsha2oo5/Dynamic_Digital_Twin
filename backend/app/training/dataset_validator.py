"""Read-only dataset inspection and validation.

Never modifies, moves or deletes anything under the dataset root.

Usage:
    python -m app.training.dataset_validator [DATASET_ROOT] [--json report.json]
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_DIRS = {"__MACOSX", ".git", ".ipynb_checkpoints"}


def is_macos_metadata(path: Path) -> bool:
    return path.name.startswith("._") or "__MACOSX" in path.parts or path.name == ".DS_Store"


def find_label_file(image: Path) -> Path | None:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt, or a sibling x.txt."""
    candidates = [image.with_suffix(".txt")]
    parts = list(image.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        candidates.insert(0, Path(*parts).with_suffix(".txt"))
    return next((c for c in candidates if c.is_file()), None)


def parse_yolo_label(path: Path, num_classes: int | None = None) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse a YOLO detection label file. Returns (boxes, errors)."""
    boxes, errors = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{lineno}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{lineno}: non-numeric value")
            continue
        if cls < 0 or (num_classes is not None and cls >= num_classes):
            errors.append(f"{path.name}:{lineno}: class id {cls} out of range")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            errors.append(f"{path.name}:{lineno}: coordinates must be normalised to (0, 1]")
        boxes.append((cls, x, y, w, h))
    return boxes, errors


def _file_hash(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_data_yaml(root: Path) -> tuple[Path | None, dict | None]:
    for name in ("data.yaml", "dataset.yaml", "data.yml"):
        p = root / name
        if p.is_file():
            try:
                return p, yaml.safe_load(p.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                return p, None
    return None, None


def validate_dataset(root: str | Path, check_duplicates: bool = True) -> dict:
    root = Path(root)
    report: dict = {"root": str(root), "exists": root.is_dir(), "scanned_at": datetime.now().isoformat(timespec="seconds")}
    if not root.is_dir():
        report["issues"] = [f"Dataset root does not exist: {root}"]
        report["ready_for_detection_training"] = False
        return report

    yaml_path, data_yaml = _load_data_yaml(root)
    names = (data_yaml or {}).get("names")
    num_classes = len(names) if isinstance(names, (list, dict)) else None

    folders: dict[str, dict] = defaultdict(lambda: {"images": 0, "labelled": 0, "extensions": Counter(), "bytes": 0,
                                                     "min_wh": None, "max_wh": None})
    metadata_files = 0
    unreadable: list[str] = []
    label_errors: list[str] = []
    class_instances: Counter[int] = Counter()
    other_annotations = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_macos_metadata(path):
            metadata_files += 1
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        ext = path.suffix.lower()
        if ext in (".json", ".xml"):
            other_annotations[ext] += 1
            continue
        if ext not in IMAGE_EXTS:
            continue

        rel = path.relative_to(root)
        folder = folders[str(rel.parent).replace("\\", "/")]
        folder["images"] += 1
        folder["extensions"][ext] += 1
        folder["bytes"] += path.stat().st_size
        try:
            with Image.open(path) as im:
                w, h = im.size
            folder["min_wh"] = [min(folder["min_wh"][0], w), min(folder["min_wh"][1], h)] if folder["min_wh"] else [w, h]
            folder["max_wh"] = [max(folder["max_wh"][0], w), max(folder["max_wh"][1], h)] if folder["max_wh"] else [w, h]
        except Exception:  # noqa: BLE001 - any decode failure means the image is unusable
            unreadable.append(str(rel))
            continue

        label = find_label_file(path)
        if label is not None:
            folder["labelled"] += 1
            boxes, errors = parse_yolo_label(label, num_classes)
            label_errors.extend(errors)
            class_instances.update(b[0] for b in boxes)
        if check_duplicates:
            hashes[_file_hash(path)].append(str(rel))

    total_images = sum(f["images"] for f in folders.values())
    labelled = sum(f["labelled"] for f in folders.values())
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]

    if labelled:
        annotation_format = "yolo"
    elif other_annotations[".json"]:
        annotation_format = "json (possibly COCO; not parsed)"
    elif other_annotations[".xml"]:
        annotation_format = "xml (possibly Pascal VOC; not parsed)"
    else:
        annotation_format = "none"

    issues = []
    if total_images == 0:
        issues.append("No images found.")
    if labelled == 0:
        issues.append(
            "No bounding-box annotations found. Images appear to be organised as one folder per class; "
            "object detection training requires box labels (YOLO .txt) for each image."
        )
    elif labelled < total_images:
        issues.append(f"{total_images - labelled} of {total_images} images have no label file.")
    if yaml_path is None:
        issues.append("No data.yaml found (class names and train/val/test split are undefined).")
    if metadata_files:
        issues.append(f"{metadata_files} macOS metadata files (._*, __MACOSX) present; they are ignored.")
    if unreadable:
        issues.append(f"{len(unreadable)} images could not be read.")
    if duplicates:
        issues.append(f"{len(duplicates)} groups of byte-identical duplicate images found.")
    if label_errors:
        issues.append(f"{len(label_errors)} label format errors.")

    report.update(
        {
            "data_yaml": str(yaml_path) if yaml_path else None,
            "class_names": names,
            "annotation_format": annotation_format,
            "total_images": total_images,
            "labelled_images": labelled,
            "folders": [
                {
                    "path": k,
                    "images": v["images"],
                    "labelled": v["labelled"],
                    "extensions": dict(v["extensions"]),
                    "size_mb": round(v["bytes"] / 1e6, 1),
                    "min_resolution": v["min_wh"],
                    "max_resolution": v["max_wh"],
                }
                for k, v in sorted(folders.items())
            ],
            "class_instances": {str(k): v for k, v in sorted(class_instances.items())},
            "macos_metadata_files": metadata_files,
            "unreadable_images": unreadable,
            "duplicate_groups": duplicates,
            "label_errors": label_errors[:200],
            "issues": issues,
            "ready_for_detection_training": bool(labelled and yaml_path and not label_errors and not unreadable),
        }
    )
    return report


def main() -> None:
    from app.core.config import get_settings

    parser = argparse.ArgumentParser(description="Validate a dataset for YOLO detection training (read-only).")
    parser.add_argument("root", nargs="?", default=None, help="Dataset root (default: DATASET_ROOT from config)")
    parser.add_argument("--json", type=Path, help="Write the full report to this JSON file")
    parser.add_argument("--no-duplicates", action="store_true", help="Skip duplicate hashing")
    args = parser.parse_args()

    root = Path(args.root) if args.root else get_settings().resolve(get_settings().dataset_root)
    report = validate_dataset(root, check_duplicates=not args.no_duplicates)
    print(f"Dataset: {report['root']}")
    print(f"Images: {report.get('total_images', 0)}  labelled: {report.get('labelled_images', 0)}  "
          f"annotation format: {report.get('annotation_format')}")
    for f in report.get("folders", []):
        print(f"  {f['path']:<45} {f['images']:>5} images  {f['labelled']:>5} labelled  {f['size_mb']:>7} MB")
    print("Issues:")
    for issue in report.get("issues", []):
        print(f"  - {issue}")
    print(f"Ready for detection training: {report['ready_for_detection_training']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.json}")


if __name__ == "__main__":
    main()
