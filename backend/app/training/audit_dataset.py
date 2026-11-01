"""Read-only audit of the collected Digital Twin dataset against the seven-class taxonomy.

Checks per-class counts, formats and dimensions, fully decodes every image (corruption),
finds exact duplicates (MD5) and near-duplicates (difference hash on the content area),
and searches for any bounding-box annotations.

Usage:
    python -m app.training.audit_dataset [DATASET_ROOT] [--out ../experiments/dataset_audit]
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.training.dataset_validator import IMAGE_EXTS, find_label_file, is_macos_metadata, parse_yolo_label
from app.training.taxonomy import CLASS_NAMES, FOLDER_TO_CLASS, NUM_CLASSES

ANNOTATION_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
HASH_SIZE = 16  # 16x16 = 256-bit difference hash
# Calibrated on this dataset: groups are stable for thresholds 10-40 (a clear gap between
# re-captures of the same photo and different photos); 30 also catches re-cropped screenshots.
NEAR_DUPLICATE_MAX_DISTANCE = 30


@dataclass
class ImageRecord:
    path: str  # relative to the dataset root, forward slashes
    folder: str
    class_id: int | None
    ext: str
    bytes: int
    width: int | None = None
    height: int | None = None
    readable: bool = True
    md5: str | None = None
    dhash: str | None = None  # hex, HASH_SIZE**2 bits
    label_file: str | None = None
    group: int | None = None  # near-duplicate group id (images that must share a split)
    extra: dict = field(default_factory=dict)


def read_image(path: Path) -> np.ndarray | None:
    """Fully decode an image (handles non-ASCII / spaced Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def content_crop(gray: np.ndarray, dark: int = 16, light: int = 235) -> np.ndarray:
    """Crop away uniform near-black (screenshot letterboxing) and near-white (product-shot
    background) borders, so the hash describes the object rather than the empty frame."""
    mask = (gray > dark) & (gray < light)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return gray
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def dhash(image_bgr: np.ndarray, size: int = HASH_SIZE) -> str:
    """size*size-bit difference hash of the content area (hex); robust to rescaling, cropping
    margins and recompression."""
    gray = content_crop(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def _bits(hex_hash: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_hash), dtype=np.uint8)).astype(bool)


def hamming(a: str, b: str) -> int:
    return int((_bits(a) != _bits(b)).sum())


def group_near_duplicates(hashes: list[str | None], max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE) -> list[int]:
    """Single-linkage grouping (union-find) of all pairs within max_distance.
    Returns a group id per input index; unhashable entries get their own group."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    valid = [i for i, h in enumerate(hashes) if h is not None]
    if len(valid) > 1:
        bits = np.stack([_bits(hashes[i]) for i in valid])
        for row in range(len(valid) - 1):
            dists = (bits[row + 1:] != bits[row]).sum(axis=1)
            for col in np.flatnonzero(dists <= max_distance):
                ri, rj = find(valid[row]), find(valid[row + 1 + col])
                if ri != rj:
                    parent[rj] = ri
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(hashes))]


def scan_images(root: Path) -> tuple[list[ImageRecord], dict]:
    records: list[ImageRecord] = []
    other = {"macos_metadata_files": 0, "annotation_like_files": [], "unmapped_files": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_macos_metadata(path):
            other["macos_metadata_files"] += 1
            continue
        rel = path.relative_to(root)
        ext = path.suffix.lower()
        if ext in ANNOTATION_EXTS:
            other["annotation_like_files"].append(rel.as_posix())
            continue
        if ext not in IMAGE_EXTS:
            other["unmapped_files"].append(rel.as_posix())
            continue
        folder = rel.parts[0]
        rec = ImageRecord(path=rel.as_posix(), folder=folder, class_id=FOLDER_TO_CLASS.get(folder), ext=ext,
                          bytes=path.stat().st_size)
        raw = path.read_bytes()
        rec.md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rec.readable = False
        else:
            rec.height, rec.width = img.shape[:2]
            rec.dhash = dhash(img)
        label = find_label_file(path)
        rec.label_file = label.relative_to(root).as_posix() if label else None
        records.append(rec)
    return records, other


def audit(root: str | Path, max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE) -> tuple[dict, list[ImageRecord]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    records, other = scan_images(root)

    groups = group_near_duplicates([r.dhash for r in records], max_distance)
    for rec, g in zip(records, groups):
        rec.group = g

    by_md5: dict[str, list[str]] = defaultdict(list)
    by_group: dict[int, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_md5[r.md5].append(r.path)
        by_group[r.group].append(r)
    exact_dups = [p for p in by_md5.values() if len(p) > 1]
    near_groups = [g for g in by_group.values() if len(g) > 1]

    # annotation check
    label_errors: list[str] = []
    labelled = 0
    for r in records:
        if r.label_file:
            labelled += 1
            _, errs = parse_yolo_label(root / r.label_file, NUM_CLASSES)
            label_errors.extend(errs)

    per_class = []
    for cid, name in CLASS_NAMES.items():
        recs = [r for r in records if r.class_id == cid]
        readable = [r for r in recs if r.readable]
        widths = [r.width for r in readable]
        heights = [r.height for r in readable]
        dims = Counter(f"{r.width}x{r.height}" for r in readable)
        per_class.append({
            "class_id": cid,
            "class_name": name,
            "folder": next(f for f, c in FOLDER_TO_CLASS.items() if c == cid),
            "images": len(recs),
            "readable": len(readable),
            "labelled": sum(1 for r in recs if r.label_file),
            "formats": dict(Counter(r.ext for r in recs)),
            "width_range": [min(widths), max(widths)] if widths else None,
            "height_range": [min(heights), max(heights)] if heights else None,
            "most_common_size": dims.most_common(1)[0] if dims else None,
            "near_duplicate_groups": len({r.group for r in recs if len(by_group[r.group]) > 1}),
            "independent_groups": len({r.group for r in recs}),
            "size_mb": round(sum(r.bytes for r in recs) / 1e6, 1),
        })

    unmapped_folders = sorted({r.folder for r in records if r.class_id is None})
    cross_class_groups = [g for g in near_groups if len({r.class_id for r in g}) > 1]
    unreadable = [r.path for r in records if not r.readable]

    annotations_exist = labelled > 0 or bool(other["annotation_like_files"])
    blockers = []
    if not labelled:
        blockers.append("No bounding-box annotations exist. Folder names give an image-level class only; "
                        "YOLO detection needs a box + class id for every object instance.")
    if label_errors:
        blockers.append(f"{len(label_errors)} label errors.")
    if unreadable:
        blockers.append(f"{len(unreadable)} unreadable images.")

    report = {
        "dataset_root": str(root),
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "taxonomy": {str(k): v for k, v in CLASS_NAMES.items()},
        "total_images": len(records),
        "readable_images": len(records) - len(unreadable),
        "per_class": per_class,
        "formats": dict(Counter(r.ext for r in records)),
        "unreadable_images": unreadable,
        "unmapped_folders": unmapped_folders,
        "macos_metadata_files": other["macos_metadata_files"],
        "annotation_like_files": other["annotation_like_files"],
        "other_files": other["unmapped_files"],
        "labelled_images": labelled,
        "label_errors": label_errors[:200],
        "annotations_exist": annotations_exist,
        "annotations_match_taxonomy": (not label_errors) if labelled else None,
        "exact_duplicate_groups": exact_dups,
        "near_duplicate": {
            "max_hamming_distance": max_distance,
            "groups": len(near_groups),
            "images_in_groups": sum(len(g) for g in near_groups),
            "cross_class_groups": [[r.path for r in g] for g in cross_class_groups],
            "largest_groups": [[r.path for r in g] for g in sorted(near_groups, key=len, reverse=True)[:10]],
        },
        "ready_for_detection_training": not blockers,
        "blockers": blockers,
    }
    return report, records


def to_markdown(report: dict) -> str:
    lines = [
        "# Dataset audit report",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Audited: {report['audited_at']}",
        f"- Total images: **{report['total_images']}** (readable: {report['readable_images']})",
        f"- Formats: {', '.join(f'{k} ×{v}' for k, v in report['formats'].items())}",
        f"- macOS metadata files ignored: {report['macos_metadata_files']}",
        f"- Annotation files found: {len(report['annotation_like_files'])}; labelled images: {report['labelled_images']}",
        f"- **Ready for detection training: {'YES' if report['ready_for_detection_training'] else 'NO'}**",
        "",
        "## Images per class",
        "",
        "| ID | Class | Folder | Images | Labelled | Formats | Width range | Height range | Most common size | Independent groups |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["per_class"]:
        mc = f"{c['most_common_size'][0]} (×{c['most_common_size'][1]})" if c["most_common_size"] else "—"
        wr = "–".join(map(str, c["width_range"])) if c["width_range"] else "—"
        hr = "–".join(map(str, c["height_range"])) if c["height_range"] else "—"
        fm = ", ".join(f"{k} ×{v}" for k, v in c["formats"].items())
        lines.append(f"| {c['class_id']} | {c['class_name']} | {c['folder']} | {c['images']} | {c['labelled']} | {fm} | "
                     f"{wr} | {hr} | {mc} | {c['independent_groups']} |")
    nd = report["near_duplicate"]
    lines += [
        "",
        "## Integrity",
        "",
        f"- Unreadable / corrupt images: {len(report['unreadable_images'])}",
        f"- Exact duplicate groups (identical bytes): {len(report['exact_duplicate_groups'])}",
        f"- Near-duplicate groups (dHash distance ≤ {nd['max_hamming_distance']}): {nd['groups']} "
        f"covering {nd['images_in_groups']} images",
        f"- Near-duplicate groups spanning different classes: {len(nd['cross_class_groups'])}",
        f"- Unmapped folders: {', '.join(report['unmapped_folders']) or 'none'}",
        "",
        "Near-duplicate groups are kept inside a single split by `prepare_dataset` to avoid leakage.",
        "Cross-class groups are the same photo saved in two class folders; such scenes usually contain",
        "several component types, so every instance must be boxed during annotation.",
        "",
        "Cross-class groups:",
        "",
        *[f"- {', '.join(f'`{p}`' for p in g)}" for g in nd["cross_class_groups"]],
        "",
        "## Blockers",
        "",
    ]
    lines += [f"- {b}" for b in report["blockers"]] or ["- none"]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, records: list[ImageRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "audit_report.md").write_text(to_markdown(report), encoding="utf-8")
    (out_dir / "image_manifest.json").write_text(json.dumps([asdict(r) for r in records], indent=1), encoding="utf-8")


def main() -> None:
    from app.core.config import PROJECT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Audit the Digital Twin dataset (read-only).")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "experiments" / "dataset_audit")
    parser.add_argument("--max-distance", type=int, default=NEAR_DUPLICATE_MAX_DISTANCE)
    args = parser.parse_args()

    root = args.root or get_settings().resolve(get_settings().dataset_root)
    report, records = audit(root, args.max_distance)
    write_outputs(report, records, args.out)
    print(to_markdown(report))
    print(f"Reports written to {args.out}")


if __name__ == "__main__":
    main()
