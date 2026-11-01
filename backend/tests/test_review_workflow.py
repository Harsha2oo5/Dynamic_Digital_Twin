"""Human-in-the-loop review: candidate import, status rules, reviewed-only export, validation and splitting."""

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.training import taxonomy
from app.training.annotation_kit import build_manifest, read_manifest, write_manifest
from app.training.audit_dataset import audit
from app.training.auto_annotate import Candidate, to_yolo, write_cvat_zip
from app.training.prepare_dataset import SPLITS, prepare
from app.training.review_store import (
    IN_PROGRESS,
    NEEDS_SECOND_REVIEW,
    NOT_REVIEWED,
    REVIEW_MANIFEST,
    REVIEWED,
    ReviewError,
    ReviewStore,
)
from app.training.review_tool import check_cvat_archives, create_app
from app.training.validate_annotations import validate_export

PER_CLASS = 6
W, H = 96, 72


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)


@pytest.fixture
def env(tmp_path):
    """Original dataset (real folder names), staged batch copies, candidate run with confidences, manifest."""
    root, batch, cands_dir = tmp_path / "dataset", tmp_path / "batch_01", tmp_path / "candidates" / "run"
    rows, records = [], []
    for folder, cid in taxonomy.FOLDER_TO_CLASS.items():
        task = f"{cid}_{taxonomy.CLASS_NAMES[cid]}"
        (root / folder).mkdir(parents=True)
        (batch / task).mkdir(parents=True)
        for i in range(PER_CLASS):
            name = f"c{cid}_img{i}.png"
            img = _noise(cid * 100 + i)
            cv2.imwrite(str(root / folder / name), img)
            shutil.copy2(root / folder / name, batch / task / name)
            rows.append({"class_id": cid, "class_name": taxonomy.CLASS_NAMES[cid], "file_name": name})
            # a wrong-class candidate partly outside the image, and a right-class one
            records.append({"file": name, "task": task, "width": W, "height": H, "candidates": [
                {"class_id": 6, "class_name": "radio_unit", "confidence": 0.42, "xyxy": [-3.0, 5.0, 40.0, 30.0],
                 "source_label": "radio_unit", "mapping": "direct", "tier": "low", "flags": ["alt_class:battery(0.3)"]},
                {"class_id": cid, "class_name": taxonomy.CLASS_NAMES[cid], "confidence": 0.31,
                 "xyxy": [50.0, 20.0, 90.5, 70.0], "source_label": "x", "mapping": "direct", "tier": "low", "flags": []}]})
    with (batch / "batch_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cands_dir.mkdir(parents=True)
    (cands_dir / "candidates.json").write_text(json.dumps(records))
    _, audit_records = audit(root)
    write_manifest(build_manifest(audit_records), tmp_path / "manifest.csv")
    return {"root": root, "batch": batch, "cands": cands_dir, "manifest": tmp_path / "manifest.csv",
            "review": tmp_path / "reviewed" / "batch_01", "tmp": tmp_path}


def _digest(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _review_all(store: ReviewStore, extra_box=True) -> None:
    """Simulate a reviewer: delete the wrong radio_unit candidate, accept the right one, add a second box."""
    for i, img in enumerate(store.images):
        cid = img["folder_class_id"]
        boxes = [{**b, "verified": True} for b in img["boxes"] if b["class_id"] == cid and b["origin"] == "candidate"]
        if extra_box:
            boxes.append({"id": "h1", "class_id": cid, "xyxy": [2, 40, 30, 70]})
        store.set_boxes(i, boxes, "tester")
        store.set_status(i, REVIEWED, "tester", confirm=True)


# --- candidate import ---------------------------------------------------------------------------
def test_init_preserves_classes_coordinates_and_confidence(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    assert len(store.images) == 7 * PER_CLASS
    img = next(i for i in store.images if i["file"] == "c3_img0.png")
    assert img["status"] == NOT_REVIEWED and img["folder_class_id"] == 3 and (img["width"], img["height"]) == (W, H)
    wrong, right = img["boxes"]
    assert wrong["class_id"] == 6 and wrong["xyxy"] == [0.0, 5.0, 40.0, 30.0]  # clipped like the CVAT zip
    assert right["class_id"] == 3 and right["xyxy"] == [50.0, 20.0, 90.5, 70.0]
    assert right["candidate"]["confidence"] == 0.31 and wrong["candidate"]["flags"] == ["alt_class:battery(0.3)"]
    assert not wrong["verified"] and not right["verified"] and right["origin"] == "candidate"
    assert store.summary()["remaining_candidate_boxes"] == 2 * 7 * PER_CLASS
    with pytest.raises(FileExistsError):  # human work is never overwritten
        ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])


def test_cvat_archives_match_candidates_and_carry_metadata(env):
    recs = json.loads((env["cands"] / "candidates.json").read_text())
    (env["cands"] / "cvat_import").mkdir()
    by_task = {}
    for r in recs:
        lines = [to_yolo(Candidate(c["class_id"], c["class_name"], c["confidence"], tuple(c["xyxy"]), c["source_label"],
                                   c["mapping"]), W, H) for c in r["candidates"]]
        by_task.setdefault(r["task"], []).append((r["file"], lines))
    for task, items in by_task.items():
        write_cvat_zip(env["cands"] / "cvat_import" / f"batch_01_{task}.zip", items)
    rep = check_cvat_archives(env["cands"], env["batch"])
    assert rep["problems"] == [] and rep["images"] == 7 * PER_CLASS and rep["lines_checked"] == 2 * 7 * PER_CLASS
    meta = json.loads((env["cands"] / "cvat_import" / "candidate_metadata.json").read_text())
    assert meta["archives"]["batch_01_3_battery.zip"]["c3_img0.png"][1] == {
        "line": 2, "class_id": 3, "class_name": "battery", "confidence": 0.31, "tier": "low", "flags": []}

    # a tampered class id is detected
    zp = env["cands"] / "cvat_import" / "batch_01_3_battery.zip"
    with zipfile.ZipFile(zp) as z:
        files = {n: z.read(n) for n in z.namelist()}
    first, second = files["obj_train_data/c3_img0.txt"].decode().splitlines()
    assert second.startswith("3 ")
    files["obj_train_data/c3_img0.txt"] = f"{first}\n4{second[1:]}\n".encode()
    with zipfile.ZipFile(zp, "w") as z:
        for n, data in files.items():
            z.writestr(n, data)
    assert any("c3_img0.png line 2" in p for p in check_cvat_archives(env["cands"], env["batch"])["problems"])


# --- review status rules ------------------------------------------------------------------------
def test_status_transitions_and_review_requirements(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    with pytest.raises(ReviewError, match="not allowed"):
        store.set_status(0, REVIEWED, "tester", confirm=True)  # never straight from NOT_REVIEWED
    store.set_status(0, IN_PROGRESS)
    with pytest.raises(ReviewError, match="not yet accepted"):
        store.set_status(0, REVIEWED, "tester", confirm=True)  # unverified candidates block sign-off
    boxes = [{**b, "verified": True} for b in store.get(0)["boxes"]]
    boxes[0]["uncertain"] = True
    store.set_boxes(0, boxes)
    with pytest.raises(ReviewError, match="uncertain"):
        store.set_status(0, REVIEWED, "tester", confirm=True)
    store.set_status(0, NEEDS_SECOND_REVIEW, note="is this a battery?")
    boxes[0]["uncertain"] = False
    store.set_boxes(0, boxes)
    assert store.get(0)["status"] == NEEDS_SECOND_REVIEW  # editing keeps it in second review
    with pytest.raises(ReviewError, match="confirmation"):
        store.set_status(0, REVIEWED, "second", confirm=False)
    with pytest.raises(ReviewError, match="reviewer name"):
        store.set_status(0, REVIEWED, "  ", confirm=True)
    img = store.set_status(0, REVIEWED, "second", confirm=True)
    assert img["status"] == REVIEWED and img["reviewer"] == "second" and img["reviewed_at"]
    store.set_boxes(0, boxes[:1])  # any later edit re-opens the image
    assert store.get(0)["status"] == IN_PROGRESS and store.get(0)["reviewed_at"] is None
    # state is persisted and every change is logged
    assert ReviewStore(env["review"]).get(0)["status"] == IN_PROGRESS
    actions = [json.loads(ln)["action"] for ln in (env["review"] / "review_log.jsonl").read_text().splitlines()]
    assert actions == ["init", "set_status", "set_boxes", "set_status", "set_boxes", "set_status", "set_boxes"]


def test_edits_validate_boxes_and_keep_rejected_candidates(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    wrong, right = store.get(0)["boxes"]
    for bad, msg in (({**right, "class_id": 7}, "class_id"), ({**right, "xyxy": [0, 0, W + 5, 10]}, "outside"),
                     ({**right, "xyxy": [10, 10, 10.5, 30]}, "zero"), ({**right, "id": "zz"}, "unknown box id")):
        with pytest.raises(ReviewError, match=msg):
            store.set_boxes(0, [bad])
    img = store.set_boxes(0, [{**wrong, "class_id": 2}])  # re-class the first, delete the second
    assert img["boxes"][0]["verified"] and img["boxes"][0]["candidate"]["class_id"] == 6
    assert [r["id"] for r in img["rejected_candidates"]] == ["c1"]
    moved = store.set_boxes(0, [{**img["boxes"][0], "verified": False, "xyxy": [1, 5, 40, 30]}])["boxes"][0]
    assert moved["verified"]  # a geometry change is a human action


# --- export, validation, training gate -------------------------------------------------------------
def test_only_reviewed_images_are_exported_and_validated(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    _review_all(store)
    store.set_status(1, NEEDS_SECOND_REVIEW)
    store.set_status(2, IN_PROGRESS)
    out = env["tmp"] / "exports" / "reviewed"
    r = store.export_reviewed(out)
    assert r["exported"] == 7 * PER_CLASS - 2 and r["not_exported"][NEEDS_SECOND_REVIEW] == 1
    assert not (out / "labels" / f"{Path(store.get(1)['file']).stem}.txt").exists()
    label = (out / "labels" / "c3_img0.txt").read_text().splitlines()
    assert label == ["3 0.731771 0.625000 0.421875 0.694444", "3 0.166667 0.763889 0.291667 0.416667"]

    manifest = read_manifest(env["manifest"])
    rep = validate_export(out, manifest, require_review=True)
    assert rep.ok, rep.errors[:3]
    s = rep.summary()
    assert s["images_ready_for_training"] == 7 * PER_CLASS - 2 and s["images_requiring_review"] == 2
    assert s["annotations"] == sum(len(i["boxes"]) for i in store.images if i["status"] == REVIEWED)
    assert s["invalid_annotations"] == 0

    # an unreviewed candidate label slipped into the export is caught
    shutil.copy2(out / "labels" / "c3_img0.txt", out / "labels" / f"{Path(store.get(1)['file']).stem}.txt")
    assert any("not listed in review_manifest" in e for e in validate_export(out, manifest).errors)
    (out / "labels" / f"{Path(store.get(1)['file']).stem}.txt").unlink()
    # so is a label edited after sign-off
    (out / "labels" / "c3_img0.txt").write_text("6 0.5 0.5 0.2 0.2\n")
    assert any("hash mismatch" in e for e in validate_export(out, manifest).errors)


def test_training_gate_refuses_unreviewed_exports(env):
    plain = env["tmp"] / "plain_export"
    plain.mkdir()
    (plain / "classes.txt").write_text("\n".join(taxonomy.CLASS_NAMES.values()) + "\n")
    for name in ("c0_img0", "c3_img1"):
        (plain / f"{name}.txt").write_text(f"{name[1]} 0.5 0.5 0.2 0.2\n")
    manifest = read_manifest(env["manifest"])
    assert validate_export(plain, manifest).ok  # structurally fine ...
    assert any(REVIEW_MANIFEST in e for e in validate_export(plain, manifest, require_review=True).errors)
    with pytest.raises(ValueError, match="validation failed"):  # ... but never training data by default
        prepare(plain, env["manifest"], env["root"], env["tmp"] / "prepared")
    assert not (env["tmp"] / "prepared").exists()
    with pytest.raises(ValueError, match="validation failed"):  # candidate runs are refused too
        (env["cands"] / "CANDIDATES_NOT_GROUND_TRUTH.txt").write_text("x")
        prepare(env["cands"], env["manifest"], env["root"], env["tmp"] / "prepared")


def test_prepare_from_reviewed_export_keeps_groups_and_classes(env):
    before = _digest(env["root"])
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    _review_all(store)
    out = env["tmp"] / "exports" / "reviewed"
    store.export_reviewed(out)
    summary = prepare(out, env["manifest"], env["root"], env["tmp"] / "prepared", seed=42)  # strict + review required
    assert summary["post_build_problems"] == [] and summary["review_manifest"]
    assert sum(summary["images"].values()) == 7 * PER_CLASS
    for name in taxonomy.CLASS_NAMES.values():
        assert summary["instances_per_class"][name]["train"] > 0 and summary["instances_per_class"][name]["val"] > 0
    rows = list(csv.DictReader((env["tmp"] / "prepared" / "split_manifest.csv").open()))
    splits_of_group = {}
    for r in rows:
        splits_of_group.setdefault(r["near_duplicate_group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits_of_group.values())
    assert {r["split"] for r in rows} == set(SPLITS)
    # deterministic for the recorded seed
    again = prepare(out, env["manifest"], env["root"], env["tmp"] / "prepared2", seed=42)
    assert again["images"] == summary["images"]
    assert _digest(env["root"]) == before  # original dataset untouched


def test_small_reviewed_sets_are_not_force_split(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    for i in range(5):
        cid = store.get(i)["folder_class_id"]
        store.set_boxes(i, [{**b, "verified": True} for b in store.get(i)["boxes"] if b["class_id"] == cid])
        store.set_status(i, REVIEWED, "tester", confirm=True)
    out = env["tmp"] / "exports" / "small"
    store.export_reviewed(out)
    with pytest.raises(ValueError, match="not ready for a train/val/test split"):
        prepare(out, env["manifest"], env["root"], env["tmp"] / "prepared")
    assert not (env["tmp"] / "prepared").exists()


def test_import_of_human_edited_yolo_export_round_trips(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    _review_all(store)
    out = env["tmp"] / "exports" / "reviewed"
    store.export_reviewed(out)
    reviewed_boxes = {i["file"]: [(b["class_id"], b["xyxy"]) for b in i["boxes"]] for i in store.images}

    fresh = ReviewStore.init_from_candidates(env["tmp"] / "reviewed2", env["batch"], env["cands"])
    shutil.copytree(out / "labels", env["tmp"] / "cvat_like")
    r = fresh.import_labels(env["tmp"] / "cvat_like", source="cvat test")
    assert r["imported"] == 7 * PER_CLASS and not r["unmatched"]
    for img in fresh.images:
        assert img["status"] == IN_PROGRESS  # still needs the explicit sign-off
        got = [(b["class_id"], b["xyxy"]) for b in img["boxes"]]
        for (ca, a), (cb, b) in zip(got, reviewed_boxes[img["file"]], strict=True):
            assert ca == cb and all(abs(x - y) <= 0.15 for x, y in zip(a, b))
        assert all(b["verified"] and b["origin"] == "imported" for b in img["boxes"])
        assert len(img["rejected_candidates"]) == 2  # superseded candidates are kept for the record
    (env["cands"] / "CANDIDATES_NOT_GROUND_TRUTH.txt").write_text("x")
    with pytest.raises(ReviewError, match="machine-generated"):
        fresh.import_labels(env["cands"], source="x")


# --- web API ------------------------------------------------------------------------------------
def test_review_api(env):
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    client = TestClient(create_app(store))
    assert "Annotation Review" in client.get("/").text
    state = client.get("/api/state").json()
    assert len(state["images"]) == 7 * PER_CLASS and state["class_names"] == list(taxonomy.CLASS_NAMES.values())
    assert client.get("/api/images/0/file").content == (env["batch"] / store.get(0)["rel_path"]).read_bytes()
    assert client.post("/api/images/0/status", json={"status": REVIEWED, "confirm": True, "reviewer": "x"}).status_code == 422
    boxes = [{**b, "verified": True} for b in state["images"][0]["boxes"]]
    r = client.put("/api/images/0/boxes", json={"boxes": boxes, "reviewer": "x"})
    assert r.status_code == 200 and r.json()["image"]["status"] == IN_PROGRESS
    assert client.put("/api/images/0/boxes", json={"boxes": [{**boxes[0], "class_id": 9}]}).status_code == 422
    r = client.post("/api/images/0/status", json={"status": REVIEWED, "confirm": True, "reviewer": "x"})
    assert r.status_code == 200 and r.json()["summary"]["reviewed"] == 1
    assert client.get("/api/images/999/file").status_code == 422


def test_workflow_never_modifies_batch_or_dataset(env):
    before = (_digest(env["root"]), _digest(env["batch"]), _digest(env["cands"]))
    store = ReviewStore.init_from_candidates(env["review"], env["batch"], env["cands"])
    _review_all(store)
    store.export_reviewed(env["tmp"] / "exports" / "e")
    assert (_digest(env["root"]), _digest(env["batch"]), _digest(env["cands"])) == before
