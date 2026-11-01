"""Tests for the custom-model training pipeline on a synthetic dataset (tmp_path only).

Synthetic images/labels here are isolated test fixtures; they are never written to the real
dataset, the annotation workspace or backend/dataset.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from app.training import taxonomy
from app.training.annotation_kit import build_manifest, cvat_labels, image_key, label_studio_config, read_manifest, write_manifest
from app.training.audit_dataset import audit, dhash, group_near_duplicates, hamming
from app.training.evaluate import class_flags, f1_score
from app.training.prepare_dataset import SPLITS, data_yaml, prepare, split_groups, verify_prepared
from app.training.train import (
    PRETRAINED,
    build_train_args,
    check_training_ready,
    project_workload,
    promote_checkpoint,
    read_results_csv,
)
from app.training.validate_annotations import Box, check_box, parse_line, validate_export

PER_CLASS = 10


def _noise(seed: int, w=96, h=72) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)  # blocky texture, distinct per seed


@pytest.fixture
def dataset(tmp_path) -> Path:
    """Folder-per-class dataset using the real folder names, with planted near-duplicates."""
    root = tmp_path / "dataset"
    for folder, cid in taxonomy.FOLDER_TO_CLASS.items():
        d = root / folder
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            cv2.imwrite(str(d / f"c{cid}_img{i}.png"), _noise(cid * 100 + i))
    # same photo re-captured at another size within a class, and across two classes
    cv2.imwrite(str(root / "antenna" / "c0_recapture.png"), cv2.resize(_noise(0), (192, 144)))
    cv2.imwrite(str(root / "Archive (5)" / "c3_cabinet_shot.png"), cv2.resize(_noise(400), (120, 90)))
    (root / "antenna" / "._c0_img0.png").write_bytes(b"mac")  # macOS metadata, must be ignored
    return root


def _export(tmp_path: Path, manifest: dict, *, names=None, skip=(), mutate=None) -> Path:
    """Write a YOLO export: one box of the folder class per image (+ a second-class box on cross-class shots)."""
    out = tmp_path / "export"
    (out / "labels").mkdir(parents=True)
    (out / "classes.txt").write_text("\n".join(names or taxonomy.CLASS_NAMES.values()) + "\n")
    for key, row in manifest.items():
        if key in skip:
            continue
        lines = [f"{row['folder_class_id']} 0.5 0.5 0.4 0.4"]
        if row["file_name"] == "c3_cabinet_shot.png":
            lines.append("4 0.3 0.3 0.2 0.2")
        if mutate:
            lines = mutate(key, lines)
        # Label Studio style name: uuid prefix + spaces replaced
        (out / "labels" / f"1a2b3c4d-{Path(row['file_name']).stem}.txt").write_text("\n".join(lines) + "\n")
    return out


@pytest.fixture
def kit(tmp_path, dataset):
    _, records = audit(dataset)
    rows = build_manifest(records)
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    return read_manifest(path)


# --- taxonomy ------------------------------------------------------------------------------
def test_class_mapping_is_the_confirmed_one():
    assert taxonomy.CLASS_NAMES == {0: "antenna", 1: "router", 2: "cable", 3: "battery",
                                    4: "power_supply_unit", 5: "equipment_cabinet", 6: "radio_unit"}
    assert taxonomy.FOLDER_TO_CLASS == {"antenna": 0, "Archive (7)": 1, "Archive (6)": 2, "Archive (5)": 3,
                                        "Archive (4)": 4, "Equipment Cabinet": 5, "radio unit": 6}
    assert taxonomy.names_match_taxonomy(list(taxonomy.CLASS_NAMES.values()))
    assert not taxonomy.names_match_taxonomy(["router", "antenna", "cable", "battery", "power_supply_unit",
                                              "equipment_cabinet", "radio_unit"])


# --- audit -----------------------------------------------------------------------------------
def test_audit_counts_and_readiness(dataset):
    report, records = audit(dataset)
    per = {c["class_name"]: c["images"] for c in report["per_class"]}
    assert per["antenna"] == PER_CLASS + 1 and per["battery"] == PER_CLASS + 1 and per["router"] == PER_CLASS
    assert report["total_images"] == 7 * PER_CLASS + 2
    assert report["macos_metadata_files"] == 1
    assert report["unreadable_images"] == []
    assert report["annotations_exist"] is False
    assert report["ready_for_detection_training"] is False
    assert any("bounding-box" in b for b in report["blockers"])


def test_audit_finds_near_duplicates(dataset):
    report, records = audit(dataset)
    group = {r.path: r.group for r in records}
    assert group["antenna/c0_img0.png"] == group["antenna/c0_recapture.png"]
    assert group["Archive (4)/c4_img0.png"] == group["Archive (5)/c3_cabinet_shot.png"]
    assert group["antenna/c0_img0.png"] != group["antenna/c0_img1.png"]
    assert len(report["near_duplicate"]["cross_class_groups"]) == 1


def test_audit_flags_corrupt_images(dataset):
    (dataset / "radio unit" / "broken.png").write_bytes(b"not a png")
    report, _ = audit(dataset)
    assert report["unreadable_images"] == ["radio unit/broken.png"]


def test_dhash_properties():
    a, b = _noise(1), _noise(2)
    assert hamming(dhash(a), dhash(cv2.resize(a, (300, 200)))) <= 10
    assert hamming(dhash(a), dhash(b)) > 60
    letterboxed = cv2.copyMakeBorder(a, 200, 200, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    assert hamming(dhash(a), dhash(letterboxed)) <= 10  # black bars are cropped before hashing
    assert group_near_duplicates([dhash(a), None, dhash(a)]) == [0, 1, 0]


# --- annotation kit --------------------------------------------------------------------------
def test_image_key_tolerates_tool_renaming():
    assert image_key("Screenshot 2026-10-07 112112.png") == "screenshot_2026_10_07_112112"
    assert image_key("1a2b3c4d-Screenshot_2026-10-07_112112.txt") == "screenshot_2026_10_07_112112"
    assert image_key("IMG_6429.PNG") == image_key("img_6429.txt")


def test_tool_configs_use_taxonomy_order():
    assert [l["name"] for l in cvat_labels()] == list(taxonomy.CLASS_NAMES.values())
    xml = label_studio_config()
    positions = [xml.index(f'"{n}"') for n in taxonomy.CLASS_NAMES.values()]
    assert positions == sorted(positions)


def test_manifest_round_trip(kit):
    assert len(kit) == 7 * PER_CLASS + 2
    row = kit[image_key("c3_cabinet_shot.png")]
    assert row["folder"] == "Archive (5)" and row["folder_class_name"] == "battery"
    assert isinstance(row["near_duplicate_group"], int)


def test_check_kit_detects_misconfiguration(tmp_path, dataset):
    from app.training import annotation_kit as ak
    from app.training.annotation_kit import check_kit

    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    _, records = audit(dataset)
    write_manifest(build_manifest(records), kit_dir / "manifest.csv")
    (kit_dir / "classes.txt").write_text("\n".join(taxonomy.CLASS_NAMES.values()) + "\n")
    (kit_dir / "cvat_labels.json").write_text(json.dumps(ak.cvat_labels()))
    (kit_dir / "label_studio_config.xml").write_text(ak.label_studio_config())
    assert check_kit(kit_dir, dataset) == []

    swapped = ak.cvat_labels()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    (kit_dir / "cvat_labels.json").write_text(json.dumps(swapped))
    cv2.imwrite(str(dataset / "radio unit" / "new_image.png"), _noise(999))
    problems = check_kit(kit_dir, dataset)
    assert any("cvat_labels.json" in p for p in problems)
    assert any("not in the manifest" in p for p in problems)


def test_annotation_batch_selection(dataset):
    from app.training.annotation_batch import select_batch

    _, records = audit(dataset)
    batch = select_batch(records, per_class=4, seed=1)
    groups = [b["near_duplicate_group"] for b in batch]
    assert len(groups) == len(set(groups))  # never two members of one near-duplicate group
    per_class = {cid: sum(b["class_id"] == cid for b in batch) for cid in taxonomy.CLASS_NAMES}
    assert set(per_class.values()) == {4}
    assert batch == select_batch(records, per_class=4, seed=1)  # deterministic
    # the re-captured antenna photo is represented by its higher-resolution copy if chosen
    assert "c0_img0.png" not in {b["file_name"] for b in batch} or "c0_recapture.png" not in {b["file_name"] for b in batch}

    # a second batch skips images and prefers groups not yet covered
    batch2 = select_batch(records, per_class=4, seed=1, exclude_paths={b["original_path"] for b in batch},
                          covered_groups=set(groups))
    assert not {b["original_path"] for b in batch} & {b["original_path"] for b in batch2}
    assert not set(groups) & {b["near_duplicate_group"] for b in batch2}


def test_cvat_subset_list_files_are_not_labels(tmp_path, kit):
    export = _export(tmp_path, kit)
    (export / "default.txt").write_text("data/obj_train_data/c0_img0.png\ndata/obj_train_data/c0_img1.png\n")
    (export / "obj.names").write_text("\n".join(taxonomy.CLASS_NAMES.values()) + "\n")
    (export / "classes.txt").unlink()
    rep = validate_export(export, kit)
    assert rep.ok, rep.errors
    assert rep.names_source.endswith("obj.names")


# --- annotation validation -------------------------------------------------------------------
@pytest.mark.parametrize(
    "line,error",
    [("7 0.5 0.5 0.1 0.1", "outside 0..6"), ("0 1.2 0.5 0.1 0.1", "normalised"), ("0 0.95 0.5 0.2 0.1", "outside the image"),
     ("0 0.5 0.5 0 0.1", "non-positive")],
)
def test_check_box_errors(line, error):
    errors, _ = check_box(parse_line(line))
    assert any(error in e for e in errors)


def test_parse_line_rejects_malformed():
    for bad in ("0 0.5 0.5 0.1", "a 0.5 0.5 0.1 0.1", "1.5 0.5 0.5 0.1 0.1"):
        with pytest.raises(ValueError):
            parse_line(bad)
    assert parse_line("3 0.5 0.5 0.25 0.25") == Box(3, 0.5, 0.5, 0.25, 0.25)


def test_valid_export(tmp_path, kit):
    rep = validate_export(_export(tmp_path, kit), kit)
    assert rep.ok, rep.errors
    assert len(rep.paired) == len(kit)
    assert rep.instances[3] == PER_CLASS + 1 and rep.instances[4] == PER_CLASS + 1


def test_export_with_wrong_class_order_is_rejected(tmp_path, kit):
    names = list(taxonomy.CLASS_NAMES.values())
    names[0], names[1] = names[1], names[0]
    rep = validate_export(_export(tmp_path, kit, names=names), kit)
    assert any("does not match the taxonomy" in e for e in rep.errors)


def test_export_missing_folder_class_box_is_rejected(tmp_path, kit):
    target = image_key("c1_img0.png")
    rep = validate_export(_export(tmp_path, kit, mutate=lambda k, lines: ["0 0.5 0.5 0.4 0.4"] if k == target else lines), kit)
    assert any("no 'router' box" in e for e in rep.errors)


def test_export_empty_and_missing_labels(tmp_path, kit):
    empty, missing = image_key("c2_img0.png"), image_key("c2_img1.png")
    rep = validate_export(_export(tmp_path, kit, skip={missing}, mutate=lambda k, lines: [] if k == empty else lines), kit)
    assert any("empty" in e for e in rep.errors)
    assert rep.missing_labels == ["Archive (6)/c2_img1.png"]
    rep_strict = validate_export(_export(tmp_path / "b", kit, skip={missing}), kit, require_all=True)
    assert any("no label file" in e for e in rep_strict.errors)


def test_export_unmatched_label_file(tmp_path, kit):
    export = _export(tmp_path, kit)
    (export / "labels" / "unknown_image.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    rep = validate_export(export, kit)
    assert rep.unmatched_label_files == [str(Path("labels") / "unknown_image.txt")]
    assert not rep.ok


# --- splitting & preparation -----------------------------------------------------------------
def _groups(n_groups: int, sizes=None):
    sizes = sizes or [1] * n_groups
    return {g: [{"folder_class_id": g % 3}] * sizes[g] for g in range(n_groups)}


def test_split_is_deterministic_complete_and_proportional():
    groups = _groups(90)
    a, b = split_groups(groups, seed=7), split_groups(groups, seed=7)
    assert a == b
    assert sorted(g for s in SPLITS for g in a[s]) == list(range(90))
    assert [len(a[s]) for s in SPLITS] == [63, 18, 9]
    assert split_groups(groups, seed=8) != a


def test_split_never_breaks_groups():
    groups = _groups(30, sizes=[1 + (g % 4) for g in range(30)])
    a = split_groups(groups, seed=1)
    owners = {}
    for s in SPLITS:
        for g in a[s]:
            assert g not in owners
            owners[g] = s
    for s in ("val", "test"):  # every stratum reaches val and test
        assert {groups[g][0]["folder_class_id"] for g in a[s]} == {0, 1, 2}


def test_data_yaml_generation(tmp_path):
    cfg = data_yaml(tmp_path)
    assert cfg["nc"] == 7 and cfg["names"] == taxonomy.CLASS_NAMES
    assert (cfg["train"], cfg["val"], cfg["test"]) == ("images/train", "images/val", "images/test")
    assert Path(cfg["path"]) == tmp_path.resolve()


def test_prepare_end_to_end(tmp_path, dataset, kit):
    export = _export(tmp_path, kit)
    manifest_path = tmp_path / "manifest.csv"
    out = tmp_path / "prepared"
    summary = prepare(export, manifest_path, dataset, out, seed=42, require_review=False, strict=False)

    assert summary["post_build_problems"] == []
    assert sum(summary["images"].values()) == len(kit)
    for name in taxonomy.CLASS_NAMES.values():
        assert all(summary["images_per_folder_class"][name][s] >= 1 for s in SPLITS), name
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert taxonomy.names_match_taxonomy(cfg["names"])
    rows = list(csv.DictReader((out / "split_manifest.csv").open()))
    split_of = {r["original_path"]: r["split"] for r in rows}
    assert split_of["antenna/c0_img0.png"] == split_of["antenna/c0_recapture.png"]
    assert split_of["Archive (4)/c4_img0.png"] == split_of["Archive (5)/c3_cabinet_shot.png"]
    assert " " not in "".join(p.name for p in (out / "images").rglob("*"))
    assert check_training_ready(out / "data.yaml", require_review=False) == []
    # original dataset untouched, output never overwritten
    assert (dataset / "antenna" / "c0_img0.png").is_file()
    with pytest.raises(FileExistsError):
        prepare(export, manifest_path, dataset, out, require_review=False, strict=False)


def test_prepare_refuses_invalid_annotations(tmp_path, dataset, kit):
    export = _export(tmp_path, kit, mutate=lambda k, lines: ["9 0.5 0.5 0.1 0.1"])
    with pytest.raises(ValueError, match="validation failed"):
        prepare(export, tmp_path / "manifest.csv", dataset, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_verify_prepared_detects_leakage(tmp_path, dataset, kit):
    out = tmp_path / "prepared"
    prepare(_export(tmp_path, kit), tmp_path / "manifest.csv", dataset, out, require_review=False, strict=False)
    rows = list(csv.DictReader((out / "split_manifest.csv").open()))
    rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
    group = rows[0]["near_duplicate_group"]
    rows.append({**rows[0], "split": "val" if rows[0]["split"] != "val" else "train", "near_duplicate_group": group})
    with (out / "split_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert any("span multiple splits" in p for p in verify_prepared(out))


# --- training configuration ------------------------------------------------------------------
def test_train_args_are_cpu_reproducible_and_non_destructive(tmp_path):
    args = build_train_args(tmp_path / "data.yaml", epochs=5, imgsz=640, batch=4, patience=8, workers=2, seed=42,
                            name="pilot_x", project=tmp_path)
    assert args["device"] == "cpu" and args["amp"] is False
    assert args["seed"] == 42 and args["deterministic"] is True
    assert args["exist_ok"] is False and args["save"] is True and args["plots"] is True
    assert (args["epochs"], args["imgsz"], args["batch"], args["patience"]) == (5, 640, 4, 8)


def test_training_gate_blocks_unprepared_data(tmp_path):
    assert check_training_ready(tmp_path / "missing.yaml")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": str(tmp_path), "names": ["a", "b"]}))
    problems = check_training_ready(tmp_path / "data.yaml")
    assert any("taxonomy" in p for p in problems) and any("No labels" in p for p in problems)


def test_project_workload():
    p = project_workload({"batch": 4, "train_step_s": 2.0, "val_image_s": 0.5}, train_images=10, val_images=4, epochs=3)
    assert p["steps_per_epoch"] == 3 and p["epoch_s_lower_bound"] == 8.0


def test_promote_never_overwrites_pretrained(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    with pytest.raises(ValueError):
        promote_checkpoint(best, PRETRAINED)
    target = tmp_path / "custom" / "digital_twin_best.pt"
    promote_checkpoint(best, target)
    promote_checkpoint(best, target)  # second promotion backs up the first
    assert len(list(target.parent.glob("digital_twin_best_*.pt"))) == 1


def test_read_results_csv(tmp_path):
    (tmp_path / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.2,0.1\n2,0.5,0.3\n3,0.4,0.25\n")
    r = read_results_csv(tmp_path)
    assert r["epochs_completed"] == 3 and r["best"]["epoch"] == 2 and r["final"]["epoch"] == 3


def test_evaluation_helpers():
    assert f1_score(0.5, 0.5) == 0.5 and f1_score(0, 0) == 0
    flags = class_flags({"antenna": {"instances": 3, "mAP50": 0.9}, "router": {"instances": 40, "mAP50": 0.2}})
    assert "unreliable" in flags["antenna"][0] and "weak" in flags["router"][0]
    assert flags["cable"] == ["no instances in this split"]
    json.dumps(flags)


def test_confusion_analysis_reads_ultralytics_orientation():
    from app.training.evaluate import confusion_analysis

    names = dict(taxonomy.CLASS_NAMES)
    m = [[0.0] * 8 for _ in range(8)]  # matrix[predicted][true], index 7 = background
    m[0][0] = 5  # antenna correct
    m[6][0] = 2  # antenna predicted as radio_unit
    m[7][0] = 1  # antenna missed
    m[4][3] = 3  # battery predicted as power_supply_unit
    m[5][7] = 4  # cabinet predicted on background
    r = confusion_analysis(m, names)
    a = r["per_class"]["antenna"]
    assert (a["ground_truth"], a["correct"], a["missed"], a["confused_as"]) == (8, 5, 1, {"radio_unit": 2})
    assert r["per_class"]["equipment_cabinet"]["background_false_positives"] == 4
    pairs = {p["pair"]: p for p in r["pairs"]}
    assert pairs["antenna <-> radio_unit"]["antenna_predicted_as_radio_unit"] == 2
    assert pairs["battery <-> power_supply_unit"]["battery_predicted_as_power_supply_unit"] == 3
    assert pairs["battery <-> power_supply_unit"]["power_supply_unit_predicted_as_battery"] == 0


def test_training_gate_requires_reviewed_provenance(tmp_path, dataset, kit):
    out = tmp_path / "prepared"
    prepare(_export(tmp_path, kit), tmp_path / "manifest.csv", dataset, out, require_review=False, strict=False)
    problems = check_training_ready(out / "data.yaml")  # default: reviewed provenance required
    assert any("human-reviewed export" in p for p in problems)
    assert check_training_ready(out / "data.yaml", require_review=False) == []


def test_restage_recreates_batch_from_manifest(tmp_path, dataset):
    from app.training.annotation_batch import restage, select_batch, stage

    _, records = audit(dataset)
    out = tmp_path / "batch_x"
    stage(select_batch(records, per_class=2, seed=3), dataset, out)
    before = sorted(p.relative_to(out) for p in out.rglob("*.png"))
    for p in out.rglob("*.png"):
        p.unlink()  # a fresh clone has only batch_manifest.csv
    assert restage(out, dataset) == len(before)
    assert sorted(p.relative_to(out) for p in out.rglob("*.png")) == before
    assert restage(out, dataset) == 0  # idempotent
