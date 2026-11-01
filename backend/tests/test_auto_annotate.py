"""Assisted-annotation rules and ground-truth protection (stub proposer; no model download)."""

import csv
import json
import zipfile

import cv2
import numpy as np
import pytest

from app.training.annotation_kit import build_manifest, read_manifest, write_manifest
from app.training.audit_dataset import audit
from app.training.auto_annotate import (
    MARKER,
    Candidate,
    RawCandidate,
    LetterboxCropProposer,
    Thresholds,
    content_box,
    image_status,
    merge_cable_bundles,
    merge_cross_class,
    overlap_ratio,
    postprocess,
    run,
    to_yolo,
    write_cvat_zip,
)
from app.training.prepare_dataset import prepare
from app.training.taxonomy import CLASS_NAMES
from app.training.validate_annotations import parse_line, read_label_file, validate_export

MAP = {"antenna": (0, "direct"), "cable": (2, "direct"), "wire": (2, "direct"), "cabinet": (5, "direct"),
       "server": (1, "ambiguous")}
T = Thresholds()


def raw(label, conf, box):
    return RawCandidate(label, conf, box)


def test_mapping_drops_unmapped_and_low_confidence():
    cands, unmapped = postprocess([raw("antenna", 0.9, (10, 10, 50, 80)), raw("bottle", 0.9, (0, 0, 5, 5)),
                                   raw("cabinet", 0.05, (0, 0, 90, 90))], MAP, 100, 100, T)
    assert [(c.class_name, c.tier) for c in cands] == [("antenna", "high")]
    assert unmapped == {"bottle": 1}


def test_same_class_duplicates_across_vocabulary_terms_are_suppressed():
    cands, _ = postprocess([raw("antenna", 0.9, (10, 10, 50, 80)), raw("antenna", 0.6, (11, 11, 50, 81))], MAP, 100, 100, T)
    assert len(cands) == 1 and cands[0].confidence == 0.9


def test_overlapping_cables_become_one_bundle_box():
    cands, _ = postprocess([raw("cable", 0.8, (10, 10, 40, 90)), raw("wire", 0.7, (20, 10, 50, 90)),
                            raw("cable", 0.6, (80, 10, 95, 90))], MAP, 100, 100, T)
    cables = sorted((c for c in cands if c.class_name == "cable"), key=lambda c: c.xyxy[0])
    assert len(cables) == 2  # one merged bundle + one separate, distinguishable cable
    assert cables[0].xyxy == (10, 10, 50, 90) and cables[0].tier == "low"
    assert any(f.startswith("cable_bundle_merged") for f in cables[0].flags)
    assert cables[1].flags == []
    for i, a in enumerate(cables):  # never overlapping cable boxes in the output
        for b in cables[i + 1:]:
            assert overlap_ratio(a.xyxy, b.xyxy) <= T.cable_overlap


def test_chained_cable_merges_are_stable():
    cs = [Candidate(2, "cable", 0.5, (i * 10.0, 0, i * 10.0 + 15, 50), "cable", "direct") for i in range(5)]
    merged = merge_cable_bundles(cs, 0.3)
    assert len(merged) == 1 and merged[0].xyxy == (0, 0, 55, 50)


def test_flags_force_review_tier():
    cands, _ = postprocess([raw("server", 0.95, (0, 0, 40, 40)), raw("antenna", 0.9, (0, 0, 99, 99)),
                            raw("cabinet", 0.9, (1, 1, 99, 99)), raw("antenna", 0.9, (0, 0, 2, 2))], MAP, 100, 100, T)
    by = {(c.class_name, round(c.xyxy[2])): c for c in cands}
    assert by[("router", 40)].tier == "low" and "ambiguous_label:server" in by[("router", 40)].flags
    assert "near_whole_image_box" in by[("antenna", 99)].flags
    assert any(f.startswith("class_conflict") for f in by[("equipment_cabinet", 99)].flags)
    assert "tiny_box" in by[("antenna", 2)].flags


def test_image_status_uses_folder_only_as_a_check():
    high = Candidate(0, "antenna", 0.9, (0, 0, 10, 10), "antenna", "direct", tier="high")
    assert image_status([], 0)[0] == "no_detection"
    assert image_status([high], 0) == ("high_confidence", [])
    status, reasons = image_status([high], 5)
    assert status == "needs_review" and "equipment_cabinet" in reasons[0]
    assert high.class_name == "antenna"  # the folder never relabels a candidate


def test_to_yolo_clips_and_normalises():
    line = to_yolo(Candidate(3, "battery", 0.9, (-5, 10, 50, 120), "battery", "direct"), 100, 100)
    box = parse_line(line)
    assert box.cls == 3 and abs(box.x - 0.25) < 1e-6 and abs(box.h - 0.9) < 1e-6


def test_cvat_zip_layout(tmp_path):
    write_cvat_zip(tmp_path / "t.zip", [("a b.png", ["0 0.5 0.5 0.2 0.2"]), ("c.png", [])])
    z = zipfile.ZipFile(tmp_path / "t.zip")
    assert set(z.namelist()) == {"obj.names", "obj.data", "train.txt", "obj_train_data/a b.txt", "obj_train_data/c.txt"}
    assert z.read("obj.names").decode().split() == list(CLASS_NAMES.values())


class StubProposer:
    """Deterministic proposer for tests; boxes depend only on the image size."""
    name = "stub"

    def propose(self, image):
        h, w = image.shape[:2]
        return [RawCandidate("antenna", 0.9, (w * 0.1, h * 0.1, w * 0.5, h * 0.6)),
                RawCandidate("cable", 0.3, (w * 0.6, h * 0.2, w * 0.9, h * 0.8))]


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "dataset"
    batch_dir = tmp_path / "batch"
    rows = []
    for cid, folder, sub in ((0, "antenna", "0_antenna"), (2, "Archive (6)", "2_cable"), (5, "Equipment Cabinet", "5_equipment_cabinet")):
        (root / folder).mkdir(parents=True)
        (batch_dir / sub).mkdir(parents=True)
        img = np.random.default_rng(cid).integers(0, 255, (60, 80, 3), dtype=np.uint8)
        name = f"img{cid}.png"
        cv2.imwrite(str(root / folder / name), img)
        cv2.imwrite(str(batch_dir / sub / name), img)
        rows.append({"class_id": cid, "class_name": CLASS_NAMES[cid], "file_name": name})
    with (batch_dir / "batch_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root, batch_dir


def test_run_outputs_and_ground_truth_protection(tmp_path, batch):
    root, batch_dir = batch
    out = tmp_path / "candidates" / "run1"
    report = run(batch_dir, StubProposer(), MAP, out, T)

    assert (out / MARKER).is_file()
    assert report["images"] == 3 and report["candidate_boxes"] == 6 and report["human_reviewed_annotations"] == 0
    # every image also gets the low-confidence (0.3) cable candidate -> nothing is "high confidence"
    assert report["status_counts"] == {"needs_review": 3}
    assert report["multi_class_images"] == 3
    assert sorted(p.name for p in (out / "cvat_import").glob("*.zip")) == [
        "batch_0_antenna.zip", "batch_2_cable.zip", "batch_5_equipment_cabinet.zip"]
    recs = {r["file"]: r for r in json.loads((out / "candidates.json").read_text())}
    assert any("equipment_cabinet' not proposed" in s for s in recs["img5.png"]["reasons"])
    assert not any("not proposed" in s for s in recs["img0.png"]["reasons"])
    assert (out / "review.html").read_text().count("img0.png") >= 1
    with (out / "manual_review.csv").open() as f:
        assert {r["file"] for r in csv.DictReader(f)} == {"img0.png", "img2.png", "img5.png"}
    assert len(list((out / "previews" / "needs_review").glob("*.jpg"))) == 3
    assert len(list((out / "contact_sheets").glob("*.jpg"))) == 3
    assert sum(report["confidence_histogram"].values()) == report["candidate_boxes"]
    with pytest.raises(FileExistsError):
        run(batch_dir, StubProposer(), MAP, out, T)

    # candidates can never pass as ground truth, nor be prepared into a training set
    _, records = audit(root)
    write_manifest(build_manifest(records), tmp_path / "manifest.csv")
    manifest = read_manifest(tmp_path / "manifest.csv")
    rep = validate_export(out, manifest)
    assert not rep.ok and "machine-generated candidates" in rep.errors[0]
    with pytest.raises(ValueError, match="validation failed"):
        prepare(out / "labels", tmp_path / "manifest.csv", root, tmp_path / "prepared")


# --- quality control on reviewed exports -----------------------------------------------------
def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_duplicate_boxes_are_errors_and_cable_overlap_warns(tmp_path):
    _, errors, _ = read_label_file(_write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2"]))
    assert any("duplicate" in e for e in errors)
    _, errors, warnings = read_label_file(_write(tmp_path, "b.txt", ["2 0.40 0.5 0.2 0.6", "2 0.48 0.5 0.2 0.6"]))
    assert not errors and any("ONE box" in w for w in warnings)
    _, _, warnings = read_label_file(_write(tmp_path, "c.txt", ["5 0.5 0.5 0.99 0.99"]))
    assert any("near-whole-image" in w for w in warnings)


def test_verified_negatives(tmp_path, batch):
    root, _ = batch
    _, records = audit(root)
    write_manifest(build_manifest(records), tmp_path / "manifest.csv")
    manifest = read_manifest(tmp_path / "manifest.csv")
    export = tmp_path / "export"
    export.mkdir()
    (export / "classes.txt").write_text("\n".join(CLASS_NAMES.values()) + "\n")
    (export / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (export / "img2.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    (export / "img5.txt").write_text("")
    assert any("label file is empty" in e for e in validate_export(export, manifest).errors)
    (export / "verified_negatives.txt").write_text("img5.png\n")
    rep = validate_export(export, manifest)
    assert rep.ok, rep.errors
    assert rep.summary()["verified_negative_images"] == 1
    summary = prepare(export, tmp_path / "manifest.csv", root, tmp_path / "prepared", seed=1,
                      require_review=False, strict=False)
    assert summary["post_build_problems"] == [] and summary["verified_negative_images"] == 1
    (export / "img5.txt").write_text("5 0.5 0.5 0.2 0.2\n")
    assert any("verified negative but has" in e for e in validate_export(export, manifest).errors)


def test_content_box_trims_letterbox_bars():
    img = np.zeros((300, 100, 3), np.uint8)
    img[100:200] = 255
    img[120:180, 20:80] = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    x0, y0, x1, y1 = content_box(img, pad=0)
    assert (y0, y1) == (120, 180) and (x0, x1) == (20, 80)
    assert content_box(np.zeros((50, 50, 3), np.uint8)) == (0, 0, 50, 50)  # nothing to keep -> full image


def test_letterbox_crop_maps_boxes_back_to_full_image():
    class Stub:
        name = "stub"

        def __init__(self):
            self.seen = None

        def propose(self, image):
            self.seen = image.shape[:2]
            return [raw("antenna", 0.9, (1, 2, 11, 12))]

    img = np.zeros((300, 100, 3), np.uint8)
    img[120:180, 20:80] = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    inner = Stub()
    out = LetterboxCropProposer(inner).propose(img)
    x0, y0, _, _ = content_box(img)
    assert inner.seen[0] < 300
    assert out[0].xyxy == (1 + x0, 2 + y0, 11 + x0, 12 + y0)


def test_cross_class_merge_keeps_one_box_and_records_alternatives():
    t = Thresholds(merge_cross_class=True)
    cands, _ = postprocess([raw("antenna", 0.6, (10, 10, 100, 100)), raw("cabinet", 0.4, (12, 10, 100, 102)),
                            raw("cabinet", 0.5, (300, 300, 400, 400))], MAP, 1000, 1000, t)
    assert sorted(c.class_name for c in cands) == ["antenna", "equipment_cabinet"]
    a = next(c for c in cands if c.class_name == "antenna")
    assert any(f.startswith("alt_class:equipment_cabinet") for f in a.flags) and a.tier == "low"
    assert len(merge_cross_class([], 0.7)) == 0
