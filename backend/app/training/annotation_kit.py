"""Annotation kit: everything needed to box-annotate the dataset in CVAT or Label Studio.

Generates (never touching the original images):
  classes.txt               class names in YOLO id order (import/export class list)
  cvat_labels.json          CVAT project "Raw" label constructor
  label_studio_config.xml   Label Studio labeling interface
  manifest.csv              image key -> original path, folder class, near-duplicate group, size
  exports/                  drop YOLO exports from the annotation tool here

Usage:
    python -m app.training.annotation_kit [--out ../annotation]
"""

import argparse
import csv
import json
import re
from pathlib import Path

from app.training.audit_dataset import audit
from app.training.dataset_validator import IMAGE_EXTS, is_macos_metadata
from app.training.taxonomy import CLASS_NAMES, FOLDER_TO_CLASS, NUM_CLASSES

# One distinct colour per class (shared by CVAT / Label Studio so annotators see the same scheme)
CLASS_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
assert len(CLASS_COLORS) >= NUM_CLASSES

_LS_PREFIX = re.compile(r"^[0-9a-f]{8}-")  # Label Studio prefixes uploaded files with "<8 hex>-"
MANIFEST_FIELDS = ["key", "file_name", "original_path", "folder", "folder_class_id", "folder_class_name",
                   "near_duplicate_group", "width", "height"]


def image_key(file_name: str) -> str:
    """Normalised key used to pair exported label files with original images.

    Tolerates the renaming annotation tools do: Label Studio's "<uuid8>-" prefix, spaces turned
    into underscores, case changes, and different extensions.
    """
    stem = Path(file_name).stem
    stem = _LS_PREFIX.sub("", stem.lower())
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def cvat_labels() -> list[dict]:
    return [{"name": n, "color": CLASS_COLORS[i], "type": "rectangle", "attributes": []} for i, n in CLASS_NAMES.items()]


def label_studio_config() -> str:
    labels = "\n".join(f'    <Label value="{n}" background="{CLASS_COLORS[i]}"/>' for i, n in CLASS_NAMES.items())
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image" strokeWidth="2" canRotate="false">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def build_manifest(records) -> list[dict]:
    rows, seen = [], {}
    for r in records:
        key = image_key(r.path.rsplit("/", 1)[-1])
        if key in seen:
            raise ValueError(f"Image key collision: {r.path} and {seen[key]} (rename is required before annotation)")
        seen[key] = r.path
        rows.append({
            "key": key,
            "file_name": r.path.rsplit("/", 1)[-1],
            "original_path": r.path,
            "folder": r.folder,
            "folder_class_id": r.class_id,
            "folder_class_name": CLASS_NAMES.get(r.class_id, ""),
            "near_duplicate_group": r.group,
            "width": r.width,
            "height": r.height,
        })
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["folder_class_id"] = int(r["folder_class_id"]) if r["folder_class_id"] != "" else None
        r["near_duplicate_group"] = int(r["near_duplicate_group"])
        r["width"], r["height"] = int(r["width"]), int(r["height"])
    return {r["key"]: r for r in rows}


def check_kit(kit_dir: Path, dataset: Path) -> list[str]:
    """Verify a generated kit against the taxonomy and the dataset. Returns problems (empty = OK)."""
    problems = []
    expected = list(CLASS_NAMES.values())
    try:
        classes = [ln.strip() for ln in (kit_dir / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        if classes != expected:
            problems.append(f"classes.txt order {classes} != taxonomy {expected}")
        cvat = json.loads((kit_dir / "cvat_labels.json").read_text(encoding="utf-8"))
        if [l["name"] for l in cvat] != expected:
            problems.append("cvat_labels.json label order differs from the taxonomy")
        if any(l.get("type") not in (None, "rectangle", "any") for l in cvat):
            problems.append("cvat_labels.json has non-rectangle label types")
        xml = (kit_dir / "label_studio_config.xml").read_text(encoding="utf-8")
        if re.findall(r'<Label value="([^"]+)"', xml) != expected:
            problems.append("label_studio_config.xml label order differs from the taxonomy")
        manifest = read_manifest(kit_dir / "manifest.csv")
    except (OSError, KeyError, ValueError) as exc:
        return problems + [f"Kit incomplete or unreadable: {exc}"]

    for row in manifest.values():
        cid = FOLDER_TO_CLASS.get(row["folder"])
        if cid is None:
            problems.append(f"{row['original_path']}: folder {row['folder']!r} is not in the taxonomy mapping")
        elif row["folder_class_id"] != cid or row["folder_class_name"] != CLASS_NAMES[cid]:
            problems.append(f"{row['original_path']}: manifest class {row['folder_class_name']} != {CLASS_NAMES[cid]}")
        if not (dataset / row["original_path"]).is_file():
            problems.append(f"{row['original_path']}: missing from the dataset")
    on_disk = {p.relative_to(dataset).as_posix() for p in dataset.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_macos_metadata(p)}
    unlisted = on_disk - {r["original_path"] for r in manifest.values()}
    if unlisted:
        problems.append(f"{len(unlisted)} dataset images are not in the manifest (regenerate the kit)")
    missing_folders = set(FOLDER_TO_CLASS) - {r["folder"] for r in manifest.values()}
    if missing_folders:
        problems.append(f"No images for mapped folders: {sorted(missing_folders)}")
    return problems


def main() -> None:
    from app.core.config import PROJECT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Generate the annotation kit (read-only on the dataset).")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "annotation")
    parser.add_argument("--check", action="store_true", help="Verify an existing kit instead of generating it")
    args = parser.parse_args()

    dataset = args.dataset or get_settings().resolve(get_settings().dataset_root)
    if args.check:
        problems = check_kit(args.out, dataset)
        manifest = read_manifest(args.out / "manifest.csv") if (args.out / "manifest.csv").is_file() else {}
        print(f"{'ID':<4}{'class':<20}{'folder':<20}{'images':>7}")
        for cid, name in CLASS_NAMES.items():
            folder = next(f for f, c in FOLDER_TO_CLASS.items() if c == cid)
            n = sum(r["folder_class_id"] == cid for r in manifest.values())
            print(f"{cid:<4}{name:<20}{folder:<20}{n:>7}")
        for p in problems:
            print(f"  PROBLEM {p}")
        print("KIT OK" if not problems else f"KIT INVALID ({len(problems)} problems)")
        raise SystemExit(1 if problems else 0)

    _, records = audit(dataset)
    rows = build_manifest(records)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exports").mkdir(exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(CLASS_NAMES.values()) + "\n", encoding="utf-8")
    (args.out / "cvat_labels.json").write_text(json.dumps(cvat_labels(), indent=2), encoding="utf-8")
    (args.out / "label_studio_config.xml").write_text(label_studio_config(), encoding="utf-8")
    write_manifest(rows, args.out / "manifest.csv")

    print(f"Annotation kit written to {args.out}")
    print(f"  {len(rows)} images in manifest; upload each dataset folder as its own task:")
    for folder in sorted({r['folder'] for r in rows}):
        n = sum(r["folder"] == folder for r in rows)
        print(f"    {dataset / folder}  ({n} images)")


if __name__ == "__main__":
    main()
