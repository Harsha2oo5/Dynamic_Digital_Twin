"""Human-in-the-loop review of candidate boxes (local, single machine, no extra dependencies).

Commands (run from backend/):
  init         create annotation/reviewed/<batch>/ from a staged batch (+ optional candidate run)
  serve        browser review UI at http://127.0.0.1:8765 (state saved to disk on every change)
  status       review progress summary
  export       write YOLO labels of REVIEWED images only (+ review_manifest.json) to annotation/exports/
  import-yolo  load a human-edited YOLO export (e.g. from CVAT) into the review state for sign-off
  check-cvat   verify a candidate run's CVAT zips against candidates.json and write candidate_metadata.json

Examples:
  python -m app.training.review_tool init --batch ..\\annotation\\batches\\batch_01 --candidates ..\\annotation\\candidates\\gdino_tiny_batch01
  python -m app.training.review_tool serve --review ..\\annotation\\reviewed\\batch_01
  python -m app.training.review_tool export --review ..\\annotation\\reviewed\\batch_01 --out ..\\annotation\\exports\\2026-11-10_batch01_reviewed
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

from app.training.review_store import STATUSES, ReviewError, ReviewStore
from app.training.taxonomy import CLASS_NAMES

UI_FILE = Path(__file__).with_name("review_ui.html")


# --- CVAT candidate archives ---------------------------------------------------------------------
def check_cvat_archives(candidates_dir: Path, batch_dir: Path) -> dict:
    """Check every cvat_import/*.zip: CVAT YOLO 1.1 layout, taxonomy order, one label file per batch image, and
    each line equal to the candidate box in candidates.json (same order). Writes cvat_import/candidate_metadata.json
    with class name, confidence and flags per line, because YOLO 1.1 cannot carry them."""
    from app.training.auto_annotate import Candidate, to_yolo

    records = {r["file"]: r for r in json.loads((candidates_dir / "candidates.json").read_text(encoding="utf-8"))}
    problems, metadata, lines_checked = [], {}, 0
    zips = sorted((candidates_dir / "cvat_import").glob("*.zip"))
    if not zips:
        problems.append("no CVAT archives found")
    for zp in zips:
        task = zp.stem.removeprefix(f"{batch_dir.name}_")
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for required in ("obj.names", "obj.data", "train.txt"):
                if required not in names:
                    problems.append(f"{zp.name}: missing {required}")
            if "obj.names" in names:
                order = [ln.strip() for ln in z.read("obj.names").decode("utf-8").splitlines() if ln.strip()]
                if order != list(CLASS_NAMES.values()):
                    problems.append(f"{zp.name}: obj.names order {order} differs from the taxonomy")
            listed = [ln.strip().removeprefix("data/obj_train_data/")
                      for ln in z.read("train.txt").decode("utf-8").splitlines() if ln.strip()] if "train.txt" in names else []
            task_dir = batch_dir / task
            expected = sorted(p.name for p in task_dir.iterdir()) if task_dir.is_dir() else []
            if sorted(listed) != expected:
                problems.append(f"{zp.name}: train.txt lists {len(listed)} images, batch folder {task} has {len(expected)}")
            for image in listed:
                member = f"obj_train_data/{Path(image).stem}.txt"
                rec = records.get(image)
                if member not in names or rec is None:
                    problems.append(f"{zp.name}: {image} has no label file or no candidate record")
                    continue
                got = [ln for ln in z.read(member).decode("utf-8").splitlines() if ln.strip()]
                want = [to_yolo(Candidate(c["class_id"], c["class_name"], c["confidence"], tuple(c["xyxy"]),
                                          c["source_label"], c["mapping"]), rec["width"], rec["height"])
                        for c in rec["candidates"]]
                lines_checked += len(got)
                if len(got) != len(want):
                    problems.append(f"{zp.name}: {image}: {len(got)} lines, {len(want)} candidates")
                else:
                    for n, (g, w) in enumerate(zip(got, want)):
                        gv, wv = g.split(), w.split()
                        # candidates.json rounds pixels to 0.1, the zip was written from unrounded boxes
                        tol = max(0.15 / rec["width"], 0.15 / rec["height"]) + 1e-6
                        if gv[0] != wv[0] or any(abs(float(a) - float(b)) > tol for a, b in zip(gv[1:], wv[1:])):
                            problems.append(f"{zp.name}: {image} line {n + 1}: {g!r} != candidate {w!r}")
                metadata.setdefault(zp.name, {})[image] = [
                    {"line": n + 1, "class_id": c["class_id"], "class_name": c["class_name"],
                     "confidence": c["confidence"], "tier": c["tier"], "flags": c["flags"]}
                    for n, c in enumerate(rec["candidates"])]
    out = candidates_dir / "cvat_import" / "candidate_metadata.json"
    out.write_text(json.dumps({"note": "Machine candidates, NOT ground truth. Line numbers refer to "
                                       "obj_train_data/<image>.txt inside each zip.", "archives": metadata}, indent=1),
                   encoding="utf-8")
    return {"archives": len(zips), "images": sum(len(v) for v in metadata.values()), "lines_checked": lines_checked,
            "problems": problems, "metadata": str(out)}


# --- web UI ---------------------------------------------------------------------------------------
def create_app(store: ReviewStore):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="Digital Twin annotation review")

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return UI_FILE.read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return {"batch": store.state["batch"], "class_names": list(CLASS_NAMES.values()), "statuses": list(STATUSES),
                "images": store.images, "summary": store.summary()}

    @app.get("/api/images/{idx}/file")
    def image_file(idx: int):
        guard(store.get, idx)
        return FileResponse(store.image_path(idx))

    @app.put("/api/images/{idx}/boxes")
    def put_boxes(idx: int, body: dict = Body(...)):
        img = guard(store.set_boxes, idx, body.get("boxes", []), body.get("reviewer"))
        return {"image": img, "summary": store.summary()}

    @app.post("/api/images/{idx}/status")
    def post_status(idx: int, body: dict = Body(...)):
        img = guard(store.set_status, idx, body.get("status"), body.get("reviewer"), bool(body.get("confirm")),
                    body.get("note"))
        return {"image": img, "summary": store.summary()}

    return app


# --- CLI ------------------------------------------------------------------------------------------
def print_summary(s: dict) -> None:
    print(f"Images: {s['images']}  reviewed: {s['reviewed']}  unreviewed: {s['unreviewed']}  "
          f"needs second review: {s['needs_second_review']}")
    print(f"Status: {s['status']}")
    print(f"Boxes: {s['boxes']}  remaining unverified candidate boxes: {s['remaining_candidate_boxes']}  "
          f"rejected candidates: {s['rejected_candidates']}")
    print(f"Boxes per class: {s['boxes_per_class']}")


def main() -> None:
    from app.core.config import PROJECT_DIR

    parser = argparse.ArgumentParser(description="Human review of candidate annotations.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--candidates", type=Path, help="candidate run directory (omit to start with no boxes)")
    p.add_argument("--review", type=Path, help="default: annotation/reviewed/<batch name>")
    p = sub.add_parser("serve")
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("status")
    p.add_argument("--review", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("import-yolo")
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True, help="human-edited YOLO export (e.g. CVAT YOLO 1.1, unzipped)")
    p = sub.add_parser("check-cvat")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.cmd == "init":
            review = args.review or PROJECT_DIR / "annotation" / "reviewed" / args.batch.resolve().name
            store = ReviewStore.init_from_candidates(review, args.batch, args.candidates)
            print(f"Review state created: {store.path}")
            print_summary(store.summary())
        elif args.cmd == "serve":
            import uvicorn

            store = ReviewStore(args.review)
            print(f"Review UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop; every change is saved to {store.path})")
            uvicorn.run(create_app(store), host="127.0.0.1", port=args.port, log_level="warning")
        elif args.cmd == "status":
            print_summary(ReviewStore(args.review).summary())
        elif args.cmd == "export":
            r = ReviewStore(args.review).export_reviewed(args.out)
            print(f"Exported {r['exported']} REVIEWED images ({r['verified_negatives']} verified negatives) to {r['out']}")
            print(f"Not exported: {r['not_exported']}")
            print(f"Next: python -m app.training.validate_annotations {args.out} --require-review")
        elif args.cmd == "import-yolo":
            r = ReviewStore(args.review).import_labels(args.export, source=str(args.export))
            print(f"Imported {r['imported']} images as IN_PROGRESS (sign them off in the review UI).")
            if r["skipped_reviewed"]:
                print(f"Skipped {len(r['skipped_reviewed'])} images already REVIEWED (re-open them first to replace)")
            if r["unmatched"]:
                print(f"{len(r['unmatched'])} label files match no image in this batch: {r['unmatched'][:10]}")
        elif args.cmd == "check-cvat":
            r = check_cvat_archives(args.candidates, args.batch)
            print(f"{r['archives']} archives, {r['images']} images, {r['lines_checked']} box lines checked")
            for prob in r["problems"]:
                print(f"  PROBLEM {prob}")
            print(f"Candidate metadata (class, confidence, flags per line): {r['metadata']}")
            print("CVAT ARCHIVES OK" if not r["problems"] else f"{len(r['problems'])} problems")
            sys.exit(1 if r["problems"] else 0)
    except (ReviewError, FileExistsError, FileNotFoundError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
