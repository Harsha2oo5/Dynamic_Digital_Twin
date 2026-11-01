"""Select a diverse annotation batch per class and stage copies for upload to CVAT.

Selection (deterministic for a given seed):
  - at most ONE image per near-duplicate group across the whole batch (groups from the audit),
    so re-captures of the same photo are never annotated together; a group shared by two class
    folders is used only once;
  - classes with the fewest independent groups choose first, so scarce classes get priority;
  - within a class, farthest-point sampling on the content hash spreads the batch over visually
    different images; each group is represented by its highest-resolution member;
  - images (and, preferably, groups) already in earlier batches are skipped.

Staged copies keep the original file names (labels map back via the manifest). The original
dataset is only read.

Usage:
    python -m app.training.annotation_batch --name batch_01 --per-class 25
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from app.training.audit_dataset import ImageRecord, _bits, audit
from app.training.taxonomy import CLASS_NAMES

FIELDS = ["class_id", "class_name", "file_name", "original_path", "near_duplicate_group", "width", "height",
          "min_distance_to_batch"]


def read_previous_batches(batches_dir: Path) -> tuple[set[str], set[int]]:
    paths, groups = set(), set()
    for manifest in batches_dir.glob("*/batch_manifest.csv"):
        with manifest.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                paths.add(row["original_path"])
                groups.add(int(row["near_duplicate_group"]))
    return paths, groups


def farthest_point_sample(items: list[ImageRecord], n: int, rng: random.Random) -> list[tuple[ImageRecord, int]]:
    """Greedy max-min selection by Hamming distance. Returns (record, distance to the already-selected set)."""
    if not items:
        return []
    bits = [_bits(r.dhash) for r in items]
    first = rng.randrange(len(items))
    chosen = [(items[first], -1)]
    min_dist = [int((bits[i] != bits[first]).sum()) for i in range(len(items))]
    min_dist[first] = -1
    while len(chosen) < min(n, len(items)):
        i = max(range(len(items)), key=lambda k: (min_dist[k], -k))
        chosen.append((items[i], min_dist[i]))
        min_dist[i] = -1
        for k in range(len(items)):
            if min_dist[k] >= 0:
                min_dist[k] = min(min_dist[k], int((bits[k] != bits[i]).sum()))
    return chosen


def select_batch(records: list[ImageRecord], per_class: int, seed: int = 42,
                 exclude_paths: set[str] = frozenset(), covered_groups: set[int] = frozenset()) -> list[dict]:
    rng = random.Random(seed)
    usable = [r for r in records if r.readable and r.dhash and r.class_id is not None and r.path not in exclude_paths]

    # one representative per (class, group): the highest-resolution member
    reps: dict[int, dict[int, ImageRecord]] = defaultdict(dict)
    for r in sorted(usable, key=lambda r: (-(r.width or 0) * (r.height or 0), r.path)):
        reps[r.class_id].setdefault(r.group, r)

    used_groups: set[int] = set()
    selected: list[dict] = []
    for cid in sorted(reps, key=lambda c: (len(reps[c]), c)):  # scarce classes first
        pool = [r for g, r in sorted(reps[cid].items()) if g not in used_groups]
        fresh = [r for r in pool if r.group not in covered_groups]
        picked = farthest_point_sample(fresh, per_class, rng)
        if len(picked) < per_class:  # top up from groups already annotated in earlier batches
            rest = [r for r in pool if r.group in covered_groups]
            picked += farthest_point_sample(rest, per_class - len(picked), rng)
        for rec, dist in picked:
            used_groups.add(rec.group)
            selected.append({
                "class_id": cid, "class_name": CLASS_NAMES[cid], "file_name": rec.path.rsplit("/", 1)[-1],
                "original_path": rec.path, "near_duplicate_group": rec.group, "width": rec.width,
                "height": rec.height, "min_distance_to_batch": dist,
            })
    return sorted(selected, key=lambda s: (s["class_id"], s["file_name"]))


def stage(selected: list[dict], dataset: Path, out: Path) -> None:
    for row in selected:
        folder = out / f"{row['class_id']}_{row['class_name']}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / row["original_path"], folder / row["file_name"])
    with (out / "batch_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(selected)


def restage(batch_dir: Path, dataset: Path) -> int:
    """Re-create a batch's staged copies from its tracked batch_manifest.csv (staged images are not in git).
    Copies only missing files; returns how many were copied."""
    with (batch_dir / "batch_manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    copied = 0
    for row in rows:
        dst = batch_dir / f"{row['class_id']}_{row['class_name']}" / row["file_name"]
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dataset / row["original_path"], dst)
            copied += 1
    return copied


def main() -> None:
    from app.core.config import PROJECT_DIR, get_settings

    parser = argparse.ArgumentParser(description="Select and stage a diverse annotation batch (read-only on the dataset).")
    parser.add_argument("--name", default="batch_01")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_DIR / "annotation" / "batches")
    parser.add_argument("--restage", action="store_true",
                        help="re-copy the images of an existing batch from its batch_manifest.csv (after a fresh clone)")
    args = parser.parse_args()

    out = args.batches_dir / args.name
    if args.restage:
        dataset = args.dataset or get_settings().resolve(get_settings().dataset_root)
        n = restage(out, dataset)
        print(f"Batch '{args.name}': {n} images copied into {out} (existing files kept)")
        return
    if out.exists():
        sys.exit(f"Refusing to overwrite existing batch: {out}")
    dataset = args.dataset or get_settings().resolve(get_settings().dataset_root)
    prev_paths, prev_groups = read_previous_batches(args.batches_dir)
    _, records = audit(dataset)
    selected = select_batch(records, args.per_class, args.seed, prev_paths, prev_groups)
    stage(selected, dataset, out)

    groups_per_class = defaultdict(set)
    for r in records:
        if r.class_id is not None:
            groups_per_class[r.class_id].add(r.group)
    print(f"Batch '{args.name}' staged in {out} ({len(selected)} images)")
    print(f"{'ID':<4}{'class':<20}{'selected':>9}{'groups avail.':>15}{'min dist':>10}")
    for cid, name in CLASS_NAMES.items():
        rows = [s for s in selected if s["class_id"] == cid]
        dists = [s["min_distance_to_batch"] for s in rows if s["min_distance_to_batch"] >= 0]
        print(f"{cid:<4}{name:<20}{len(rows):>9}{len(groups_per_class[cid]):>15}{min(dists, default='-'):>10}")
    assert len({s["near_duplicate_group"] for s in selected}) == len(selected), "group used twice"
    print("Each near-duplicate group appears at most once in this batch.")


if __name__ == "__main__":
    main()
