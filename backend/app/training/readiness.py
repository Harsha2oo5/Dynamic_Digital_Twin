"""Training-readiness report, computed from the live filesystem (never from older reports).

Answers: how many original images exist, how many have candidate boxes, how many are human-reviewed,
which reviewed exports exist and whether they validate, whether a prepared dataset and a custom checkpoint
exist, and -- the gate -- whether a pilot training run may start, with the exact remaining work if not.

Usage (from backend/):
    python -m app.training.readiness            # prints the report
    python -m app.training.readiness --write    # also saves experiments/readiness/<timestamp>/readiness.{json,md}
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.core.config import BACKEND_DIR, PROJECT_DIR, get_settings
from app.training.dataset_validator import IMAGE_EXTS, is_macos_metadata
from app.training.prepare_dataset import MIN_IMAGES, MIN_INSTANCES_PER_CLASS
from app.training.review_store import NEEDS_SECOND_REVIEW, REVIEWED, STATE_FILE, ReviewStore
from app.training.taxonomy import CLASS_NAMES, FOLDER_TO_CLASS, names_match_taxonomy

MINUTES_PER_IMAGE = (1.5, 3.0)  # measured guess for correcting candidate boxes; used only for the effort estimate


def _images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not is_macos_metadata(p)]


def original_dataset(root: Path) -> dict:
    per_class = {CLASS_NAMES[cid]: len(_images(root / folder)) if (root / folder).is_dir() else 0
                 for folder, cid in FOLDER_TO_CLASS.items()}
    label_files = [p for p in root.rglob("*.txt") if not is_macos_metadata(p)] if root.is_dir() else []
    return {"root": str(root), "images": sum(per_class.values()), "per_class": per_class,
            "label_files_in_dataset": len(label_files)}


def candidate_runs(cand_root: Path) -> list[dict]:
    runs = []
    for d in sorted(p for p in cand_root.iterdir() if p.is_dir()) if cand_root.is_dir() else []:
        rep = d / "report.json"
        r = json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}
        runs.append({"run": d.name, "model": r.get("model"), "images": r.get("images"),
                     "candidate_boxes": r.get("candidate_boxes"), "human_reviewed": 0})
    return runs


def review_states(rev_root: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in rev_root.iterdir() if (p / STATE_FILE).is_file()) if rev_root.is_dir() else []:
        store = ReviewStore(d)
        s = store.summary()
        reviewed = [i for i in store.images if i["status"] == REVIEWED]
        boxes = Counter(b["class_id"] for i in reviewed for b in i["boxes"])
        images_with = Counter(c for i in reviewed for c in {b["class_id"] for b in i["boxes"]})
        out.append({
            "batch": d.name, "images": s["images"], "status": s["status"],
            "candidate_boxes_unverified": s["remaining_candidate_boxes"],
            "reviewed_images": len(reviewed),
            "reviewed_boxes_per_class": {CLASS_NAMES[c]: boxes[c] for c in CLASS_NAMES},
            "reviewed_images_per_class": {CLASS_NAMES[c]: images_with[c] for c in CLASS_NAMES},
            "verified_negatives": sum(not i["boxes"] for i in reviewed),
            "human_actions_logged": max(0, sum(1 for _ in (d / "review_log.jsonl").open(encoding="utf-8")) - 1)
            if (d / "review_log.jsonl").is_file() else 0,
        })
    return out


def exports(exp_root: Path, manifest_path: Path) -> list[dict]:
    from app.training.annotation_kit import read_manifest
    from app.training.validate_annotations import validate_export

    if not exp_root.is_dir() or not manifest_path.is_file():
        return []
    manifest = read_manifest(manifest_path)
    out = []
    for d in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        rep = validate_export(d, manifest, require_review=True)
        s = rep.summary()
        out.append({"export": d.name, "valid": s["ok"], "errors": len(s["errors"]), "first_errors": s["errors"][:3],
                    "paired_images": s["paired_images"], "annotations": s["annotations"],
                    "instances_per_class": s["instances_per_class"], "review_manifest": bool(s["review_manifest"])})
    return out


def prepared_datasets(ds_root: Path) -> list[dict]:
    from app.training.train import check_training_ready

    out = []
    for y in sorted(ds_root.glob("*/data.yaml")) if ds_root.is_dir() else []:
        report = y.parent / "split_report.json"
        rep = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
        out.append({"dataset": y.parent.name, "images": rep.get("images"), "limitations": rep.get("limitations", []),
                    "training_gate_problems": check_training_ready(y)})
    return out


def checkpoints(custom_dir: Path, load: bool) -> list[dict]:
    out = []
    for p in sorted(custom_dir.glob("*.pt")) if custom_dir.is_dir() else []:
        entry = {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), "loads": None, "taxonomy_match": None}
        if load:
            try:
                from ultralytics import YOLO

                names = YOLO(str(p)).names
                entry["loads"], entry["classes"] = True, [names[k] for k in sorted(names)]
                entry["taxonomy_match"] = names_match_taxonomy(names)
            except Exception as exc:  # noqa: BLE001 - report, never raise
                entry["loads"], entry["error"] = False, str(exc)
        out.append(entry)
    return out


def training_runs(runs_root: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()) if runs_root.is_dir() else []:
        summ = d / "run_summary.json"
        out.append({"run": d.name, "has_best": (d / "weights" / "best.pt").is_file(),
                    "summary": json.loads(summ.read_text(encoding="utf-8")) if summ.is_file() else None})
    return out


def pilot_gate(reviews: list[dict], exps: list[dict], prepared: list[dict]) -> dict:
    """Blocking reasons for a pilot run, in the order they must be resolved."""
    reviewed_images = sum(r["reviewed_images"] for r in reviews)
    boxes = Counter()
    for r in reviews:
        boxes.update(r["reviewed_boxes_per_class"])
    blocking = []
    if reviewed_images < MIN_IMAGES:
        blocking.append(f"{reviewed_images} human-reviewed images; at least {MIN_IMAGES} needed")
    short = {n: boxes[n] for n in CLASS_NAMES.values() if boxes[n] < MIN_INSTANCES_PER_CLASS}
    if short:
        blocking.append("classes below {} reviewed boxes: {}".format(
            MIN_INSTANCES_PER_CLASS, ", ".join(f"{n} {c}" for n, c in short.items())))
    if not any(e["valid"] for e in exps):
        blocking.append("no valid reviewed export in annotation/exports/ (review_tool export)")
    if not any(not p["training_gate_problems"] for p in prepared):
        blocking.append("no prepared dataset passing the training gate (prepare_dataset)")
    unreviewed = sum(r["images"] - r["reviewed_images"] for r in reviews)
    lo, hi = (round(unreviewed * m / 60, 1) for m in MINUTES_PER_IMAGE)
    return {"ready": not blocking, "blocking": blocking, "reviewed_images": reviewed_images,
            "reviewed_boxes_per_class": dict(boxes), "unreviewed_staged_images": unreviewed,
            "estimated_review_hours": [lo, hi]}


def build_report(load_checkpoints: bool = True) -> dict:
    settings = get_settings()
    ann = PROJECT_DIR / "annotation"
    reviews = review_states(ann / "reviewed")
    exps = exports(ann / "exports", ann / "manifest.csv")
    prepared = prepared_datasets(BACKEND_DIR / "dataset")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "original_dataset": original_dataset(settings.resolve(settings.dataset_root)),
        "staged_batches": {d.name: len(_images(d)) for d in sorted((ann / "batches").iterdir()) if d.is_dir()}
        if (ann / "batches").is_dir() else {},
        "candidate_runs": candidate_runs(ann / "candidates"),
        "review_states": reviews,
        "reviewed_exports": exps,
        "prepared_datasets": prepared,
        "custom_checkpoints": checkpoints(BACKEND_DIR / "models" / "custom", load_checkpoints),
        "training_runs": training_runs(PROJECT_DIR / "experiments" / "training"),
        "pilot_gate": pilot_gate(reviews, exps, prepared),
        "thresholds": {"min_images": MIN_IMAGES, "min_boxes_per_class": MIN_INSTANCES_PER_CLASS},
    }


def to_markdown(r: dict) -> str:
    o, g = r["original_dataset"], r["pilot_gate"]
    lines = [f"# Training readiness ({r['generated_at']})", "",
             f"**Pilot training: {'READY' if g['ready'] else 'BLOCKED'}**", ""]
    lines += [f"- {b}" for b in g["blocking"]] or ["- all gates pass"]
    lines += ["", "## Original dataset (read-only)", "",
              f"{o['images']} images; YOLO label files inside it: {o['label_files_in_dataset']}", "",
              "| Class | Images |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in o["per_class"].items()]
    lines += ["", "## Staged batches", ""] + [f"- {k}: {v} images" for k, v in r["staged_batches"].items()]
    lines += ["", "## Candidate runs (not ground truth)", ""]
    lines += [f"- {c['run']}: {c['images']} images, {c['candidate_boxes']} boxes ({c['model']})" for c in r["candidate_runs"]] or ["- none"]
    lines += ["", "## Human review", "", "| Batch | Images | Reviewed | Second review | Unverified candidate boxes | Human actions |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {v['batch']} | {v['images']} | {v['reviewed_images']} | {v['status'][NEEDS_SECOND_REVIEW]} | "
              f"{v['candidate_boxes_unverified']} | {v['human_actions_logged']} |" for v in r["review_states"]]
    lines += ["", "Reviewed boxes per class (REVIEWED images only): "
              + ", ".join(f"{n} {g['reviewed_boxes_per_class'].get(n, 0)}" for n in CLASS_NAMES.values())]
    lines += ["", "## Reviewed exports", ""]
    lines += [f"- {e['export']}: {'valid' if e['valid'] else 'INVALID'}, {e['paired_images']} images, {e['annotations']} boxes"
              for e in r["reviewed_exports"]] or ["- none"]
    lines += ["", "## Prepared datasets", ""]
    lines += [f"- {p['dataset']}: {p['images']}; gate problems: {p['training_gate_problems'] or 'none'}"
              for p in r["prepared_datasets"]] or ["- none"]
    lines += ["", "## Custom checkpoints", ""]
    lines += [f"- {c['file']} ({c['size_mb']} MB): loads={c['loads']}, taxonomy_match={c['taxonomy_match']}"
              for c in r["custom_checkpoints"]] or ["- none (the app runs the pretrained YOLO11n)"]
    lines += ["", "## Training runs", ""]
    lines += [f"- {t['run']}: best.pt={'yes' if t['has_best'] else 'no'}" for t in r["training_runs"]] or ["- none"]
    lo, hi = g["estimated_review_hours"]
    lines += ["", "## Remaining human work", "",
              f"{g['unreviewed_staged_images']} staged images are not reviewed yet (rough estimate {lo}-{hi} h at "
              f"{MINUTES_PER_IMAGE[0]}-{MINUTES_PER_IMAGE[1]} min per image)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report whether custom-model training may start.")
    parser.add_argument("--write", action="store_true", help="save JSON + Markdown under experiments/readiness/")
    parser.add_argument("--no-load", action="store_true", help="do not load custom checkpoints")
    args = parser.parse_args()
    r = build_report(load_checkpoints=not args.no_load)
    md = to_markdown(r)
    print(md)
    if args.write:
        out = PROJECT_DIR / "experiments" / "readiness" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True)
        (out / "readiness.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        (out / "readiness.md").write_text(md, encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
