"""Human review state for one annotation batch: the only place where boxes become "reviewed".

Directory layout (one per batch, e.g. annotation/reviewed/batch_01/):
  review_state.json   every image: status, current boxes (with provenance), rejected candidates
  review_log.jsonl    append-only audit trail of every change

Statuses (never set automatically to REVIEWED):
  NOT_REVIEWED        initial; boxes are machine candidates
  IN_PROGRESS         a person has started editing (any edit also re-opens a REVIEWED image)
  NEEDS_SECOND_REVIEW the reviewer is unsure; never exported for training
  REVIEWED            a person confirmed the image is complete; requires every box verified,
                      no box marked uncertain, a reviewer name and an explicit confirmation

Box provenance: origin = candidate | human | imported; `verified` turns true only through a human action
(accepting, re-classing, moving/resizing, drawing or importing a human-edited export).
Coordinates are original-image pixels (x1, y1, x2, y2).
"""

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from app.training.annotation_kit import image_key
from app.training.audit_dataset import read_image
from app.training.dataset_validator import IMAGE_EXTS
from app.training.taxonomy import CLASS_NAMES, NUM_CLASSES

STATE_FILE = "review_state.json"
LOG_FILE = "review_log.jsonl"
REVIEW_MANIFEST = "review_manifest.json"  # written into every export by export_reviewed

NOT_REVIEWED, IN_PROGRESS, REVIEWED, NEEDS_SECOND_REVIEW = "NOT_REVIEWED", "IN_PROGRESS", "REVIEWED", "NEEDS_SECOND_REVIEW"
STATUSES = (NOT_REVIEWED, IN_PROGRESS, REVIEWED, NEEDS_SECOND_REVIEW)
TRANSITIONS = {
    NOT_REVIEWED: {IN_PROGRESS, NEEDS_SECOND_REVIEW},
    IN_PROGRESS: {NOT_REVIEWED, REVIEWED, NEEDS_SECOND_REVIEW},
    NEEDS_SECOND_REVIEW: {IN_PROGRESS, REVIEWED},
    REVIEWED: {IN_PROGRESS, NEEDS_SECOND_REVIEW},
}
MIN_BOX_PX = 2.0
COORD_TOL = 0.5  # px


class ReviewError(ValueError):
    """A request that would break the review rules (bad box, illegal transition, ...)."""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clip_box(xyxy, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [round(min(max(x1, 0.0), w), 1), round(min(max(y1, 0.0), h), 1),
            round(min(max(x2, 0.0), w), 1), round(min(max(y2, 0.0), h), 1)]


def check_box(box: dict, w: int, h: int) -> None:
    cid = box.get("class_id")
    if not isinstance(cid, int) or isinstance(cid, bool) or not 0 <= cid < NUM_CLASSES:
        raise ReviewError(f"class_id {cid!r} outside 0..{NUM_CLASSES - 1}")
    xyxy = box.get("xyxy")
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        raise ReviewError("xyxy must be 4 numbers")
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if x1 < -COORD_TOL or y1 < -COORD_TOL or x2 > w + COORD_TOL or y2 > h + COORD_TOL:
        raise ReviewError(f"box {xyxy} outside the {w}x{h} image")
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        raise ReviewError(f"box {xyxy} has (near) zero width or height")


def to_yolo_line(box: dict, w: int, h: int) -> str:
    x1, y1, x2, y2 = clip_box(box["xyxy"], w, h)
    return f"{box['class_id']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def label_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def review_blockers(img: dict) -> list[str]:
    """Why this image cannot be marked REVIEWED yet (empty = it can)."""
    out = []
    unverified = [b for b in img["boxes"] if not b.get("verified")]
    if unverified:
        out.append(f"{len(unverified)} candidate box(es) not yet accepted, corrected or deleted")
    uncertain = [b for b in img["boxes"] if b.get("uncertain")]
    if uncertain:
        out.append(f"{len(uncertain)} box(es) marked uncertain: use NEEDS_SECOND_REVIEW")
    return out


class ReviewStore:
    def __init__(self, review_dir: Path):
        self.dir = Path(review_dir)
        self.path = self.dir / STATE_FILE
        if not self.path.is_file():
            raise FileNotFoundError(f"No review state at {self.path}; run `review_tool init` first")
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text(encoding="utf-8"))
        self.by_key = {img["key"]: i for i, img in enumerate(self.state["images"])}

    # --- creation --------------------------------------------------------------------------
    @staticmethod
    def init_from_candidates(review_dir: Path, batch_dir: Path, candidates_dir: Path | None) -> "ReviewStore":
        """Create the state for every image of the batch. Candidate boxes (if any) are loaded as unverified.
        Refuses to overwrite an existing state: human work is never discarded."""
        review_dir = Path(review_dir)
        if (review_dir / STATE_FILE).exists():
            raise FileExistsError(f"Refusing to overwrite existing review state {review_dir / STATE_FILE}")
        cands = {}
        if candidates_dir is not None:
            for rec in json.loads((Path(candidates_dir) / "candidates.json").read_text(encoding="utf-8")):
                cands[rec["file"]] = rec
        folder_class = {}
        manifest = Path(batch_dir) / "batch_manifest.csv"
        if manifest.is_file():
            import csv

            with manifest.open(encoding="utf-8") as f:
                folder_class = {r["file_name"]: int(r["class_id"]) for r in csv.DictReader(f)}
        images = []
        for p in sorted(p for p in Path(batch_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            img = read_image(p)
            if img is None:
                raise ReviewError(f"unreadable image {p}")
            h, w = img.shape[:2]
            rec = cands.get(p.name)
            if rec is not None and (rec["width"], rec["height"]) != (w, h):
                raise ReviewError(f"{p.name}: candidate size {rec['width']}x{rec['height']} != image {w}x{h}")
            boxes = []
            for n, c in enumerate(rec["candidates"] if rec else []):
                boxes.append({"id": f"c{n}", "class_id": int(c["class_id"]), "xyxy": clip_box(c["xyxy"], w, h),
                              "origin": "candidate", "verified": False, "uncertain": False,
                              "candidate": {"class_id": int(c["class_id"]), "confidence": c["confidence"],
                                            "flags": c.get("flags", []), "source_label": c.get("source_label")}})
            images.append({"file": p.name, "task": p.parent.name, "rel_path": p.relative_to(batch_dir).as_posix(),
                           "key": image_key(p.name), "folder_class_id": folder_class.get(p.name),
                           "width": w, "height": h, "status": NOT_REVIEWED, "reviewer": None, "reviewed_at": None,
                           "updated_at": None, "note": "", "boxes": boxes, "rejected_candidates": [],
                           "candidate_count": len(boxes)})
        if not images:
            raise FileNotFoundError(f"No images in {batch_dir}")
        review_dir.mkdir(parents=True, exist_ok=True)
        state = {"schema": 1, "batch": Path(batch_dir).name, "created_at": now(),
                 "batch_dir": os.path.relpath(Path(batch_dir).resolve(), review_dir.resolve()),
                 "candidates_dir": (os.path.relpath(Path(candidates_dir).resolve(), review_dir.resolve())
                                    if candidates_dir else None),
                 "class_names": list(CLASS_NAMES.values()), "images": images}
        _atomic_write(review_dir / STATE_FILE, state)
        store = ReviewStore(review_dir)
        store._log("init", None, {"images": len(images), "candidate_boxes": sum(len(i["boxes"]) for i in images)})
        return store

    # --- access -----------------------------------------------------------------------------
    @property
    def images(self) -> list[dict]:
        return self.state["images"]

    @property
    def batch_dir(self) -> Path:
        return (self.dir / self.state["batch_dir"]).resolve()

    def image_path(self, idx: int) -> Path:
        return self.batch_dir / self.images[idx]["rel_path"]

    def get(self, idx: int) -> dict:
        if not 0 <= idx < len(self.images):
            raise ReviewError(f"image index {idx} out of range")
        return self.images[idx]

    def summary(self) -> dict:
        imgs = self.images
        status = {s: sum(i["status"] == s for i in imgs) for s in STATUSES}
        boxes = [b for i in imgs for b in i["boxes"]]
        return {
            "images": len(imgs),
            "status": status,
            "reviewed": status[REVIEWED],
            "unreviewed": len(imgs) - status[REVIEWED],
            "needs_second_review": status[NEEDS_SECOND_REVIEW],
            "boxes": len(boxes),
            "remaining_candidate_boxes": sum(not b.get("verified") for b in boxes),
            "rejected_candidates": sum(len(i["rejected_candidates"]) for i in imgs),
            "boxes_per_class": {n: sum(b["class_id"] == c for b in boxes) for c, n in CLASS_NAMES.items()},
            "unverified_candidates_per_class": {n: sum(b["class_id"] == c and not b.get("verified") for b in boxes)
                                                for c, n in CLASS_NAMES.items()},
        }

    # --- mutations --------------------------------------------------------------------------
    def set_boxes(self, idx: int, boxes: list[dict], reviewer: str | None = None) -> dict:
        """Replace an image's boxes with the reviewer's edited list. Candidate boxes missing from the new list
        are kept in `rejected_candidates` (never silently discarded). Any edit re-opens a REVIEWED image."""
        with self.lock:
            img = self.get(idx)
            w, h = img["width"], img["height"]
            old = {b["id"]: b for b in img["boxes"]}
            new, seen = [], set()
            for b in boxes:
                check_box(b, w, h)
                bid = str(b.get("id") or "")
                if not bid or bid in seen:
                    raise ReviewError(f"box ids must be unique and non-empty (got {bid!r})")
                seen.add(bid)
                prev = old.get(bid)
                origin = prev["origin"] if prev else "human"
                if prev is None and not bid.startswith("h"):
                    raise ReviewError(f"unknown box id {bid!r}; new boxes must use ids starting with 'h'")
                changed = prev is None or prev["class_id"] != b["class_id"] or \
                    any(abs(float(a) - float(c)) > COORD_TOL for a, c in zip(prev["xyxy"], b["xyxy"]))
                item = {"id": bid, "class_id": int(b["class_id"]), "xyxy": clip_box(b["xyxy"], w, h),
                        "origin": origin, "uncertain": bool(b.get("uncertain", False)),
                        # verified only by a human action: explicit accept, or any change made by the reviewer
                        "verified": bool(b.get("verified", False)) or changed or origin != "candidate"}
                if prev and "candidate" in prev:
                    item["candidate"] = prev["candidate"]
                new.append(item)
            for bid, prev in old.items():
                if bid not in seen and prev["origin"] == "candidate":
                    img["rejected_candidates"].append({**prev, "rejected_at": now(), "rejected_by": reviewer})
            img["boxes"] = new
            before = img["status"]
            if before in (NOT_REVIEWED, REVIEWED):
                img["status"] = IN_PROGRESS
                img["reviewed_at"] = None
            img["updated_at"] = now()
            self._save()
            self._log("set_boxes", img, {"boxes": len(new), "status_before": before, "reviewer": reviewer})
            return img

    def set_status(self, idx: int, status: str, reviewer: str | None = None, confirm: bool = False,
                   note: str | None = None) -> dict:
        with self.lock:
            img = self.get(idx)
            if status not in STATUSES:
                raise ReviewError(f"unknown status {status!r}")
            if status == img["status"]:
                return img
            if status not in TRANSITIONS[img["status"]]:
                raise ReviewError(f"{img['status']} -> {status} is not allowed")
            if status == REVIEWED:
                if not confirm:
                    raise ReviewError("REVIEWED needs an explicit confirmation that the image is complete")
                if not (reviewer or "").strip():
                    raise ReviewError("REVIEWED needs a reviewer name")
                blockers = review_blockers(img)
                if blockers:
                    raise ReviewError("; ".join(blockers))
                img["reviewer"] = reviewer.strip()
                img["reviewed_at"] = now()
            else:
                img["reviewed_at"] = None
            if note is not None:
                img["note"] = note
            before, img["status"] = img["status"], status
            img["updated_at"] = now()
            self._save()
            self._log("set_status", img, {"from": before, "to": status, "reviewer": reviewer})
            return img

    def import_labels(self, export_dir: Path, source: str) -> dict:
        """Load human-edited YOLO labels (e.g. a CVAT export) as verified boxes. Imported images go to
        IN_PROGRESS: they still need the explicit REVIEWED sign-off here."""
        from app.training.validate_annotations import candidate_marker, is_image_list_file, NON_LABEL_TXT, parse_line

        if candidate_marker(export_dir) is not None:
            raise ReviewError(f"{export_dir} contains machine-generated candidates, not human-edited labels")
        counts = {"imported": 0, "unmatched": [], "skipped_reviewed": []}
        with self.lock:
            for p in sorted(Path(export_dir).rglob("*.txt")):
                if p.name.lower() in NON_LABEL_TXT or is_image_list_file(p):
                    continue
                idx = self.by_key.get(image_key(p.name))
                if idx is None:
                    counts["unmatched"].append(p.name)
                    continue
                img = self.images[idx]
                if img["status"] == REVIEWED:  # never overwrite a signed-off image silently
                    counts["skipped_reviewed"].append(img["file"])
                    continue
                w, h = img["width"], img["height"]
                boxes = []
                for n, line in enumerate(ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()):
                    yb = parse_line(line)
                    b = {"id": f"i{n}", "class_id": yb.cls, "origin": "imported", "verified": True, "uncertain": False,
                         "xyxy": clip_box(((yb.x - yb.w / 2) * w, (yb.y - yb.h / 2) * h,
                                           (yb.x + yb.w / 2) * w, (yb.y + yb.h / 2) * h), w, h), "source": source}
                    check_box(b, w, h)
                    boxes.append(b)
                for prev in img["boxes"]:
                    if prev["origin"] == "candidate":
                        img["rejected_candidates"].append({**prev, "rejected_at": now(),
                                                           "rejected_by": f"superseded by import {source}"})
                img["boxes"] = boxes
                img["status"] = IN_PROGRESS
                img["reviewed_at"] = None
                img["updated_at"] = now()
                counts["imported"] += 1
            self._save()
            self._log("import_labels", None, {"source": source, "imported": counts["imported"],
                                              "unmatched": len(counts["unmatched"])})
        return counts

    # --- export -----------------------------------------------------------------------------
    def export_reviewed(self, out: Path) -> dict:
        """Write YOLO labels for REVIEWED images only, plus the review manifest the validator checks."""
        out = Path(out)
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing export {out}")
        with self.lock:
            reviewed = [i for i in self.images if i["status"] == REVIEWED]
            for img in reviewed:
                blockers = review_blockers(img)
                if blockers:  # cannot happen through set_status; guards against hand-edited state files
                    raise ReviewError(f"{img['file']} is REVIEWED but {'; '.join(blockers)}")
            (out / "labels").mkdir(parents=True)
            (out / "classes.txt").write_text("\n".join(CLASS_NAMES.values()) + "\n", encoding="utf-8")
            entries, negatives = {}, []
            for img in reviewed:
                text = "".join(to_yolo_line(b, img["width"], img["height"]) + "\n" for b in img["boxes"])
                (out / "labels" / f"{Path(img['file']).stem}.txt").write_text(text, encoding="utf-8", newline="\n")
                if not img["boxes"]:
                    negatives.append(img["file"])
                entries[img["key"]] = {"file": img["file"], "status": REVIEWED, "reviewer": img["reviewer"],
                                       "reviewed_at": img["reviewed_at"], "boxes": len(img["boxes"]),
                                       "label_sha256": label_sha256(text)}
            if negatives:
                (out / "verified_negatives.txt").write_text("\n".join(negatives) + "\n", encoding="utf-8")
            not_exported = {s: sum(i["status"] == s for i in self.images) for s in STATUSES if s != REVIEWED}
            manifest = {"schema": 1, "exported_at": now(), "source_review": str(self.dir.resolve()),
                        "batch": self.state["batch"], "images": entries, "not_exported": not_exported}
            (out / REVIEW_MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            self._log("export", None, {"out": str(out), "exported": len(entries), "not_exported": not_exported})
        return {"exported": len(entries), "verified_negatives": len(negatives), "not_exported": not_exported,
                "out": str(out)}

    # --- persistence ------------------------------------------------------------------------
    def _save(self) -> None:
        _atomic_write(self.path, self.state)

    def _log(self, action: str, img: dict | None, detail: dict) -> None:
        entry = {"at": now(), "action": action, "file": img["file"] if img else None, **detail}
        with (self.dir / LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)
