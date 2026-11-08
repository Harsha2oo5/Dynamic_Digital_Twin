"""Build the processed YOLO detection dataset (train/val/test) from validated annotations.

- Original images are only READ (copied from the dataset root); labels come from the tool export.
- Annotations are validated first; any validation error aborts. Only human-reviewed exports are accepted:
  the export must carry the review tool's review_manifest.json and every label must be a REVIEWED image.
- The split is not forced: too few images, a class with too few boxes, or a class missing from train/val
  aborts with the limitation reported (strict mode, the CLI default).
- Split 70/20/10 by *near-duplicate group* (from the annotation manifest), so re-captures of the
  same photo never end up in different splits, stratified by folder class, with a fixed seed.
- Writes data.yaml with the seven taxonomy classes, a split manifest and a per-class report,
  then re-validates the output. An existing output directory is never overwritten.

Pseudo-label mode (`--pseudo <run dir>`): builds the same split from an automatic pseudo-label run
(app.training.pseudo_label). The run's pseudo_manifest.json must be intact (every label file matches its sha256)
and only images the pipeline marked trainable are used. The dataset is stamped `label_source: pseudo`,
`ground_truth: false` in split_report.json and in a marker file, so training, evaluation and the model card never
present it as human-verified. The folder-name check of reviewed exports is not applied: folder names are not
definitive labels.

Usage:
    python -m app.training.prepare_dataset --labels ../annotation/exports/<export> \
        [--manifest ../annotation/manifest.csv] [--out dataset/digital_twin_yolo] [--seed 42]
    python -m app.training.prepare_dataset --pseudo ../annotation/pseudo_labels/full_v1 \
        --out dataset/digital_twin_pseudo_v1
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from app.training.annotation_kit import read_manifest
from app.training.taxonomy import CLASS_NAMES, names_match_taxonomy
from app.training.validate_annotations import VERIFIED_NEGATIVES, read_label_file, validate_export

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.7, 0.2, 0.1)
MIN_IMAGES = 30  # below this a 70/20/10 split leaves val/test with a handful of images
MIN_INSTANCES_PER_CLASS = 10  # boxes per class across the whole reviewed set


def group_stratum(rows: list[dict]) -> int:
    """Stratify a group by its most common folder class (ties -> lowest class id)."""
    counts = Counter(r["folder_class_id"] for r in rows)
    return min(counts, key=lambda c: (-counts[c], c))


def split_groups(groups: dict[int, list[dict]], ratios=DEFAULT_RATIOS, seed: int = 42) -> dict[str, list[int]]:
    """Assign whole groups to splits, per stratum, greedily filling the split with the largest
    deficit against its target image count. Deterministic for a given seed."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    by_stratum: dict[int, list[int]] = defaultdict(list)
    for gid in sorted(groups):
        by_stratum[group_stratum(groups[gid])].append(gid)

    rng = random.Random(seed)
    assignment: dict[str, list[int]] = {s: [] for s in SPLITS}
    for stratum in sorted(by_stratum):
        gids = by_stratum[stratum]
        rng.shuffle(gids)
        # place large groups first so they don't overshoot the small val/test targets late
        gids.sort(key=lambda g: -len(groups[g]))
        total = sum(len(groups[g]) for g in gids)
        targets = [r * total for r in ratios]
        filled = [0, 0, 0]
        for gid in gids:
            deficits = [targets[i] - filled[i] for i in range(3)]
            i = max(range(3), key=lambda k: (deficits[k], -k))
            assignment[SPLITS[i]].append(gid)
            filled[i] += len(groups[gid])
    return assignment


def data_yaml(dataset_dir: Path) -> dict:
    return {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": dict(CLASS_NAMES),
    }


def write_data_yaml(dataset_dir: Path) -> Path:
    path = dataset_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data_yaml(dataset_dir), sort_keys=False), encoding="utf-8")
    return path


def verify_prepared(dataset_dir: Path) -> list[str]:
    """Post-build checks: names, image/label pairing, label validity, no group across splits."""
    problems = []
    cfg = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    if not names_match_taxonomy(cfg["names"]):
        problems.append("data.yaml names do not match the taxonomy")
    negatives_file = dataset_dir / VERIFIED_NEGATIVES
    negatives = set(negatives_file.read_text(encoding="utf-8").split()) if negatives_file.is_file() else set()
    for split in SPLITS:
        images = {p.stem: p for p in (dataset_dir / "images" / split).iterdir()}
        labels = {p.stem: p for p in (dataset_dir / "labels" / split).glob("*.txt")}
        if images.keys() != labels.keys():
            problems.append(f"{split}: {len(images.keys() ^ labels.keys())} images/labels are unpaired")
        for lbl in labels.values():
            boxes, errors, _ = read_label_file(lbl)
            problems += [f"{split}/{e}" for e in errors]
            if not boxes and lbl.stem not in negatives:
                problems.append(f"{split}/{lbl.name}: empty and not a verified negative")
    with (dataset_dir / "split_manifest.csv").open(encoding="utf-8") as f:
        split_of_group: dict[str, set[str]] = defaultdict(set)
        for row in csv.DictReader(f):
            split_of_group[row["near_duplicate_group"]].add(row["split"])
    leaks = [g for g, s in split_of_group.items() if len(s) > 1]
    if leaks:
        problems.append(f"{len(leaks)} near-duplicate groups span multiple splits")
    return problems


def split_limitations(assignment: dict[str, list[int]], groups: dict[int, list[dict]],
                      boxes_of: dict[str, list[int]]) -> tuple[list[str], list[str]]:
    """(blocking problems, warnings) of a proposed split. Blocking: too few images, a class with too few boxes,
    an empty split, a class absent from train or val. Warning: a class absent from test."""
    blocking, warnings = [], []
    total = sum(len(g) for g in groups.values())
    if total < MIN_IMAGES:
        blocking.append(f"only {total} reviewed images; at least {MIN_IMAGES} are needed for a 70/20/10 split")
    per_class = Counter(c for bs in boxes_of.values() for c in bs)
    for cid, name in CLASS_NAMES.items():
        if per_class[cid] < MIN_INSTANCES_PER_CLASS:
            blocking.append(f"class {name}: {per_class[cid]} boxes, at least {MIN_INSTANCES_PER_CLASS} needed")
    for split in SPLITS:
        keys = [r["key"] for g in assignment[split] for r in groups[g]]
        if not keys:
            blocking.append(f"split {split} would be empty")
            continue
        present = {c for k in keys for c in boxes_of[k]}
        for cid, name in CLASS_NAMES.items():
            if cid not in present and per_class[cid]:
                (warnings if split == "test" else blocking).append(f"class {name} has no box in {split}")
    return blocking, warnings


PSEUDO_DATASET_MARKER = "PSEUDO_LABELS_NOT_GROUND_TRUTH.txt"


def check_pseudo_run(run_dir: Path, manifest: dict[str, dict]) -> tuple[dict[str, Path], dict, list[str]]:
    """Verify a pseudo-label run. Returns (key -> label file, pseudo manifest, errors)."""
    from app.training.pseudo_label.outputs import LABEL_SOURCE, MANIFEST, MARKER
    from app.training.review_store import label_sha256

    errors: list[str] = []
    mpath = run_dir / MANIFEST
    if not (run_dir / MARKER).is_file() or not mpath.is_file():
        return {}, {}, [f"{run_dir} is not a pseudo-label run ({MARKER} / {MANIFEST} missing)"]
    pm = json.loads(mpath.read_text(encoding="utf-8"))
    if pm.get("label_source") != LABEL_SOURCE or pm.get("ground_truth") is not False:
        errors.append(f"{MANIFEST}: label_source must be '{LABEL_SOURCE}' and ground_truth false")
    classes = run_dir / "classes.txt"
    if classes.is_file():
        names = [ln.strip() for ln in classes.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not names_match_taxonomy(names):
            errors.append(f"classes.txt {names} does not match the taxonomy")
    trainable = {k for k, e in pm.get("images", {}).items() if e.get("trainable")}
    paired: dict[str, Path] = {}
    for key in sorted(trainable):
        e = pm["images"][key]
        label = run_dir / "labels" / f"{key}.txt"
        if key not in manifest:
            errors.append(f"{key}: not in the annotation manifest")
        elif not label.is_file():
            errors.append(f"{label.name}: listed as trainable but missing")
        elif label_sha256(label.read_text(encoding="utf-8")) != e.get("label_sha256"):
            errors.append(f"{label.name}: content differs from {MANIFEST} (modified after generation)")
        else:
            boxes, box_errors, _ = read_label_file(label)
            if box_errors:
                errors += [f"{label.name}: {x}" for x in box_errors]
            elif not boxes:
                errors.append(f"{label.name}: trainable but empty")
            else:
                paired[key] = label
    labels_dir = run_dir / "labels"
    extra = ({p.stem for p in labels_dir.glob("*.txt")} - trainable) if labels_dir.is_dir() else set()
    if extra:
        errors.append(f"{len(extra)} label files are not listed as trainable in {MANIFEST}")
    if not paired and not errors:
        errors.append("no trainable pseudo-labelled images")
    return paired, pm, errors


def prepare_pseudo(run_dir: Path, manifest_path: Path, dataset_root: Path, out: Path, seed: int = 42,
                   ratios=DEFAULT_RATIOS, strict: bool = True) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    manifest = read_manifest(manifest_path)
    paired, pm, errors = check_pseudo_run(run_dir, manifest)
    if errors:
        raise ValueError(f"Pseudo-label run failed verification with {len(errors)} errors. First: {errors[0]}")
    provenance = {"label_source": "pseudo", "ground_truth": False, "pseudo_run": pm.get("run"),
                  "pseudo_run_dir": str(run_dir), "pseudo_config_hash": pm.get("config_hash"),
                  "pseudo_models": pm.get("models"), "disclaimer": pm.get("disclaimer"), "review_manifest": None}
    missing = [r["original_path"] for k, r in manifest.items() if k not in paired]
    summary = _materialize(paired, manifest, dataset_root, out, seed, ratios, strict, missing, [], provenance)
    (out / PSEUDO_DATASET_MARKER).write_text((pm.get("disclaimer") or "Pseudo-labels: not ground truth.") + "\n",
                                             encoding="utf-8")
    return summary


def prepare(labels_dir: Path, manifest_path: Path, dataset_root: Path, out: Path, seed: int = 42,
            ratios=DEFAULT_RATIOS, require_review: bool = True, strict: bool = True) -> dict:
    """`require_review=False` / `strict=False` exist for unit tests of the split mechanics on tiny synthetic
    sets; the CLI always uses both safeguards."""
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    manifest = read_manifest(manifest_path)
    report = validate_export(labels_dir, manifest, require_review=require_review)
    if not report.ok:
        raise ValueError(f"Annotation validation failed with {len(report.errors)} errors; "
                         f"run validate_annotations for details. First: {report.errors[0]}")
    provenance = {"label_source": "reviewed" if report.review_manifest else "unverified",
                  "ground_truth": bool(report.review_manifest), "labels_dir": str(labels_dir),
                  "review_manifest": report.review_manifest, "validation_warnings": len(report.warnings)}
    return _materialize(report.paired, manifest, dataset_root, out, seed, ratios, strict, report.missing_labels,
                        report.verified_negatives, provenance)


def _materialize(paired: dict[str, Path], manifest: dict[str, dict], dataset_root: Path, out: Path, seed: int,
                 ratios, strict: bool, missing: list[str], negatives: list[str], provenance: dict) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    for key in paired:
        groups[manifest[key]["near_duplicate_group"]].append(manifest[key])
    assignment = split_groups(groups, ratios, seed)
    boxes_of = {key: [b.cls for b in read_label_file(label)[0]] for key, label in paired.items()}
    blocking, limitation_warnings = split_limitations(assignment, groups, boxes_of)
    if strict and blocking:
        raise ValueError("Reviewed dataset is not ready for a train/val/test split: " + "; ".join(blocking))
    limitation_warnings = blocking + limitation_warnings  # non-strict: report everything

    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    rows = []
    images_per = {s: Counter() for s in SPLITS}  # folder class -> images
    instances_per = {s: Counter() for s in SPLITS}  # class -> boxes
    for split, gids in assignment.items():
        for gid in gids:
            for row in groups[gid]:
                src = dataset_root / row["original_path"]
                dst_name = row["key"] + Path(row["file_name"]).suffix.lower()
                shutil.copy2(src, out / "images" / split / dst_name)
                label = paired[row["key"]]
                shutil.copyfile(label, out / "labels" / split / f"{row['key']}.txt")
                boxes, _, _ = read_label_file(label)
                images_per[split][row["folder_class_id"]] += 1
                instances_per[split].update(b.cls for b in boxes)
                rows.append({"split": split, "file": dst_name, "original_path": row["original_path"],
                             "folder_class": row["folder_class_name"], "near_duplicate_group": gid,
                             "label_source": str(label)})

    if negatives:  # keys == output file stems
        (out / VERIFIED_NEGATIVES).write_text("\n".join(negatives) + "\n", encoding="utf-8")
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_data_yaml(out)

    summary = {
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        **provenance,
        "images": {s: sum(images_per[s].values()) for s in SPLITS},
        "groups": {s: len(assignment[s]) for s in SPLITS},
        "images_per_folder_class": {CLASS_NAMES[c]: {s: images_per[s][c] for s in SPLITS} for c in CLASS_NAMES},
        "instances_per_class": {CLASS_NAMES[c]: {s: instances_per[s][c] for s in SPLITS} for c in CLASS_NAMES},
        "excluded_unlabelled_images": len(missing),
        "verified_negative_images": len(negatives),
        "limitations": limitation_warnings,
    }
    summary["post_build_problems"] = verify_prepared(out)
    (out / "split_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(s: dict) -> None:
    print(f"Images: {s['images']}  (groups: {s['groups']}, unlabelled excluded: {s['excluded_unlabelled_images']})")
    print(f"{'class':<20}{'images train/val/test':>24}{'boxes train/val/test':>24}")
    for name in CLASS_NAMES.values():
        im, bx = s["images_per_folder_class"][name], s["instances_per_class"][name]
        print(f"{name:<20}{'/'.join(str(im[k]) for k in SPLITS):>24}{'/'.join(str(bx[k]) for k in SPLITS):>24}")
    for w in s.get("limitations", []):
        print(f"  LIMITATION {w}")
    for p in s["post_build_problems"]:
        print(f"  PROBLEM {p}")


def main() -> None:
    from app.core.config import BACKEND_DIR, PROJECT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Create the YOLO train/val/test dataset from validated annotations.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", type=Path, help="Human-reviewed annotation export directory (YOLO .txt labels)")
    src.add_argument("--pseudo", type=Path, help="Pseudo-label run directory (annotation/pseudo_labels/<run>)")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "annotation" / "manifest.csv")
    parser.add_argument("--dataset", type=Path, default=None, help="Original dataset root (read-only)")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "dataset" / "digital_twin_yolo")
    parser.add_argument("--split", nargs=3, type=float, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset or get_settings().resolve(get_settings().dataset_root)
    if args.out.resolve().is_relative_to(dataset.resolve()):
        sys.exit("Output directory must not be inside the original dataset")
    try:
        if args.pseudo:
            summary = prepare_pseudo(args.pseudo, args.manifest, dataset, args.out, args.seed, tuple(args.split))
        else:
            summary = prepare(args.labels, args.manifest, dataset, args.out, args.seed, tuple(args.split))
    except (FileExistsError, ValueError) as exc:
        sys.exit(str(exc))
    print_summary(summary)
    if summary.get("label_source") == "pseudo":
        print("NOTE: labels are automatic pseudo-labels (not ground truth); metrics on this dataset measure "
              "agreement with pseudo-labels, not accuracy.")
    print(f"Wrote {args.out / 'data.yaml'}")
    sys.exit(1 if summary["post_build_problems"] else 0)


if __name__ == "__main__":
    main()
