"""Automated pseudo-labelling: fusion, thresholds, suppression, YOLO output, manifest and end-to-end run.

Models are replaced by fakes (nothing is downloaded); the dataset is a synthetic one in tmp_path.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.training import taxonomy
from app.training.annotation_kit import build_manifest, read_manifest, write_manifest
from app.training.audit_dataset import audit
from app.training.pseudo_label import outputs
from app.training.pseudo_label.config import CLASSES, PseudoLabelConfig, load_config
from app.training.pseudo_label.filters import image_decision, suppress
from app.training.pseudo_label.fusion import (
    ACCEPTED,
    REJECTED,
    UNCERTAIN,
    classify_box,
    folder_prior,
    fuse,
    gdino_votes,
    softmax,
)
from app.training.pseudo_label.geometry import crop, to_yolo_line
from app.training.pseudo_label.pipeline import StoredDetections, label_image
from app.training.pseudo_label.run import run, select_rows
from app.training.review_store import label_sha256


def sims(best: str, top: float = 0.32, rest: float = 0.25, second: str | None = None, second_v: float = 0.0):
    s = {c: rest for c in CLASSES}
    s[best] = top
    if second:
        s[second] = second_v
    return s


def det(cls="router", conf=0.4, alternatives=()):
    return {"gdino_class": cls, "gdino_confidence": conf, "alternatives": list(alternatives)}


@pytest.fixture
def cfg() -> PseudoLabelConfig:
    return PseudoLabelConfig()


# --- fusion -----------------------------------------------------------------------------------------
def test_softmax_and_votes_are_distributions():
    p = softmax(sims("battery"), 100.0)
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == "battery"
    v = gdino_votes("router", 0.4, [{"gdino_class": "cable", "gdino_confidence": 0.3}], 0.05)
    assert abs(sum(v.values()) - 1) < 1e-9
    assert v["router"] > v["cable"] > v["antenna"]


def test_folder_prior_scales_with_box_size_and_can_be_off(cfg):
    assert folder_prior("battery", 0.5, cfg)["battery"] == pytest.approx(cfg.folder_prior_boost)
    assert folder_prior("battery", 0.05, cfg)["battery"] == pytest.approx(1 + (cfg.folder_prior_boost - 1) * 0.25)
    assert set(folder_prior(None, 0.5, cfg).values()) == {1.0}
    cfg.folder_prior_boost = 1.0
    assert set(folder_prior("battery", 0.9, cfg).values()) == {1.0}


def test_clear_evidence_is_accepted_with_full_provenance(cfg):
    r = classify_box(det("battery", 0.45), sims("battery", 0.33), 0.15, 0.4, None, cfg,
                     flip_sims=sims("battery", 0.33), alt_sims=sims("battery", 0.30))
    assert r["tier"] == ACCEPTED and r["final_class"] == "battery" and r["reason"] is None
    for key in ("p_final", "p_clip", "p_gdino", "clip_similarities", "clip_negative_similarity", "gdino_top1",
                "folder_prior_decisive", "flip_stable", "alt_prompt_stable"):
        assert key in r
    assert abs(sum(r["p_final"].values()) - 1) < 1e-3


@pytest.mark.parametrize("kwargs,tier,needle", [
    ({"area": 0.001}, REJECTED, "too small"),
    ({"neg": 0.40}, REJECTED, "background"),
    ({"gconf": 0.20}, REJECTED, "detector confidence"),
    ({"clip": sims("battery", 0.30, second="power_supply_unit", second_v=0.299), "gcls": "antenna"}, UNCERTAIN,
     "margin"),
    ({"flip": sims("radio_unit", 0.40)}, UNCERTAIN, "flipped"),
])
def test_weak_evidence_is_not_silently_labelled(cfg, kwargs, tier, needle):
    r = classify_box(det(kwargs.get("gcls", "battery"), kwargs.get("gconf", 0.45)), kwargs.get("clip", sims("battery", 0.33)),
                     kwargs.get("neg", 0.15), kwargs.get("area", 0.4), None, cfg,
                     flip_sims=kwargs.get("flip", sims("battery", 0.33)))
    assert r["tier"] == tier and needle in r["reason"]


def test_class_without_clip_support_stays_uncertain_even_with_prior(cfg):
    """The folder prior can never be the only evidence: CLIP must rank the class in its top-k."""
    cfg.folder_prior_boost = 1e5
    clip = {"antenna": 0.34, "router": 0.33, "cable": 0.32, "battery": 0.31, "power_supply_unit": 0.20,
            "equipment_cabinet": 0.20, "radio_unit": 0.20}
    r = classify_box(det("antenna", 0.45), clip, 0.1, 0.6, "battery", cfg, flip_sims=clip)
    assert r["final_class"] == "battery" and r["tier"] == UNCERTAIN and "no CLIP support" in r["reason"]


def test_folder_prior_decision_is_recorded(cfg):
    clip = sims("power_supply_unit", 0.300, second="radio_unit", second_v=0.297)
    r = classify_box(det("equipment_cabinet", 0.45), clip, 0.1, 0.6, "radio_unit", cfg, flip_sims=clip)
    assert r["final_class"] == "radio_unit" and r["folder_prior_decisive"] is True
    cfg.folder_prior_boost = 1.0
    r2 = classify_box(det("equipment_cabinet", 0.45), clip, 0.1, 0.6, "radio_unit", cfg, flip_sims=clip)
    assert r2["final_class"] == "power_supply_unit" and r2["folder_prior_decisive"] is False


def test_fuse_weights(cfg):
    p_clip = softmax(sims("cable", 0.30), 100)
    p_gd = gdino_votes("antenna", 0.9, [], 0.05)
    neutral = {c: 1.0 for c in CLASSES}
    cfg.w_gdino = 0.0
    assert max(fuse(p_clip, p_gd, neutral, cfg).items(), key=lambda kv: kv[1])[0] == "cable"


# --- suppression and image decision ---------------------------------------------------------------
def box(i, xyxy, cls, tier=ACCEPTED, conf=0.8, share=0.3):
    return {"id": i, "xyxy": xyxy, "final_class": cls, "tier": tier, "final_confidence": conf, "area_share": share,
            "reason": None if tier == ACCEPTED else "x"}


def test_duplicates_and_conflicts_keep_the_stronger_box(cfg):
    bs = [box(0, (0, 0, 100, 100), "router", conf=0.9), box(1, (2, 2, 100, 100), "router", conf=0.7),
          box(2, (1, 1, 99, 99), "battery", conf=0.6), box(3, (200, 200, 300, 300), "battery")]
    suppress(bs, cfg)
    assert [b["tier"] for b in bs] == [ACCEPTED, REJECTED, REJECTED, ACCEPTED]
    assert "duplicate" in bs[1]["reason"] and "conflicts" in bs[2]["reason"]
    assert bs[1]["tier_before_suppression"] == ACCEPTED


def test_parts_are_rejected_except_inside_cabinets_and_nestable_modules(cfg):
    bs = [box(0, (0, 0, 200, 200), "radio_unit"), box(1, (10, 10, 40, 40), "antenna"),
          box(2, (50, 50, 90, 90), "cable"), box(3, (300, 0, 600, 300), "equipment_cabinet"),
          box(4, (320, 20, 400, 100), "battery"), box(5, (150, 150, 260, 260), "cable"),
          box(6, (700, 0, 900, 200), "battery"), box(7, (710, 10, 790, 90), "battery")]
    suppress(bs, cfg)
    assert bs[1]["tier"] == REJECTED and "part of box 0" in bs[1]["reason"]
    assert bs[2]["tier"] == REJECTED  # a 'cable' entirely inside a radio unit is a handle or rack ear
    assert bs[5]["tier"] == ACCEPTED  # a cable leaving the unit is a cable
    assert bs[4]["tier"] == ACCEPTED  # battery inside a cabinet
    assert bs[7]["tier"] == ACCEPTED  # battery module inside a battery rack
    cfg.allowed_inside_any = ["cable"]
    bs[2].update(tier=ACCEPTED, reason=None)
    suppress(bs, cfg)
    assert bs[2]["tier"] == ACCEPTED


def test_uncertain_objects_still_contain_parts(cfg):
    bs = [box(0, (0, 0, 200, 200), "battery", tier=UNCERTAIN), box(1, (150, 150, 190, 190), "router")]
    suppress(bs, cfg)
    assert bs[1]["tier"] == REJECTED and "part of box 0" in bs[1]["reason"]


def test_image_decision(cfg):
    assert image_decision([box(0, (0, 0, 1, 1), "router", tier=UNCERTAIN)], cfg) == (False, "no accepted box")
    covered = [box(0, (0, 0, 100, 100), "router"), box(1, (0, 0, 98, 98), "battery", tier=UNCERTAIN, share=0.4)]
    assert image_decision(covered, cfg)[0] is True
    uncovered = [box(0, (0, 0, 100, 100), "router"), box(1, (300, 300, 500, 500), "battery", tier=UNCERTAIN, share=0.2)]
    ok, why = image_decision(uncovered, cfg)
    assert not ok and "uncovered uncertain box 1" in why
    small = [box(0, (0, 0, 100, 100), "router"), box(1, (300, 300, 310, 310), "battery", tier=UNCERTAIN, share=0.01)]
    assert image_decision(small, cfg)[0] is True


# --- geometry / config ------------------------------------------------------------------------------
def test_yolo_line_is_normalised_and_clipped():
    assert to_yolo_line(3, (-10, 0, 50, 100), 100, 200) == "3 0.250000 0.250000 0.500000 0.500000"


def test_crop_padding_and_degenerate_box():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop(img, (40, 40, 60, 60), 0.5).shape[:2] == (40, 40)
    with pytest.raises(ValueError):
        crop(img, (150, 150, 160, 160))


def test_config_overrides_validation_and_hash(tmp_path):
    base = load_config()
    y = tmp_path / "c.yaml"
    y.write_text("min_final_confidence:\n  battery: 0.6\nfolder_prior_boost: 1.0\n")
    c = load_config(y)
    assert c.min_final_confidence["battery"] == 0.6 and c.min_final_confidence["router"] == 0.45
    assert c.hash() != base.hash() and load_config().hash() == base.hash()
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides={"not_a_key": 1})
    with pytest.raises(ValueError, match="seven classes"):
        load_config(overrides={"min_gdino_confidence": {"antenna": 0.3, "toaster": 0.1}})


# --- end to end with fakes ----------------------------------------------------------------------------
class FakeScorer:
    """Large crops look like the box's detector class to 'CLIP'; small crops look like cables."""

    def __init__(self):
        self.calls = 0

    def score(self, crops):
        self.calls += 1
        out = []
        for c in crops:
            cls = "cable" if c.shape[1] < 40 else self.big_class
            s = sims(cls, 0.34)
            out.append({"sims": s, "flip_sims": s, "alt_sims": s, "negative_sim": 0.1})
        return out


class FakeSource:
    name = "fake"

    def __init__(self, folder_of):
        self.folder_of = folder_of

    def get(self, original_path, image):
        h, w = image.shape[:2]
        cls = self.folder_of[original_path]
        return ([{"xyxy": (2, 2, w - 2, h - 2), "gdino_confidence": 0.5, "prompt": "p", "gdino_class": cls,
                  "alternatives": []},
                 {"xyxy": (w * 0.8, 0, w, h * 0.3), "gdino_confidence": 0.4, "prompt": "c", "gdino_class": "cable",
                  "alternatives": []}], [0, 0, w, h])


def _noise(seed, w=120, h=90):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)


@pytest.fixture
def synthetic(tmp_path):
    root = tmp_path / "dataset"
    for folder, cid in taxonomy.FOLDER_TO_CLASS.items():
        (root / folder).mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(root / folder / f"c{cid}_{i}.png"), _noise(cid * 10 + i))
    _, records = audit(root)
    mpath = tmp_path / "manifest.csv"
    write_manifest(build_manifest(records), mpath)
    return root, mpath


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_end_to_end_run_writes_labels_manifest_report_and_leaves_dataset_untouched(tmp_path, synthetic):
    root, mpath = synthetic
    before = _tree_hash(root)
    manifest = read_manifest(mpath)
    folder_of = {r["original_path"]: r["folder_class_name"] for r in manifest.values()}
    cfg = PseudoLabelConfig(dataset_root=str(root), manifest=str(mpath))
    scorer = FakeScorer()
    scorer.big_class = "router"  # CLIP thinks every big object is a router
    s = run("t1", cfg, tmp_path / "pl", scorer=scorer, source=FakeSource(folder_of), log=lambda *_: None)
    out = tmp_path / "pl" / "t1"

    assert _tree_hash(root) == before
    assert (out / outputs.MARKER).is_file() and (out / "classes.txt").read_text().split() == CLASSES
    m = json.loads((out / outputs.MANIFEST).read_text())
    assert m["label_source"] == "pseudo" and m["ground_truth"] is False and len(m["images"]) == 21
    assert s["images"] == 21 and s["boxes"] == 42
    for key, e in m["images"].items():
        label = out / "labels" / f"{key}.txt"
        assert label.is_file() == e["trainable"]
        if e["trainable"]:
            assert label_sha256(label.read_text()) == e["label_sha256"]
    # router folder: detector, CLIP and prior agree -> labelled router; cable box accepted (allowed over anything)
    router_keys = [k for k, r in manifest.items() if r["folder_class_name"] == "router"]
    lines = (out / "labels" / f"{router_keys[0]}.txt").read_text().split("\n")
    assert sorted(int(ln.split()[0]) for ln in lines if ln) == [taxonomy.CLASS_IDS["router"], taxonomy.CLASS_IDS["cable"]]
    recs = [json.loads(ln) for ln in (out / "records.jsonl").read_text().splitlines()]
    assert all("p_final" in b and "tier" in b for r in recs for b in r["boxes"])
    assert (out / "report.md").read_text().count("NOT ground truth") >= 1
    assert (out / "gallery.html").is_file() and len(list((out / "previews").glob("*.jpg"))) == 21
    with pytest.raises(FileExistsError):
        run("t1", cfg, tmp_path / "pl", scorer=scorer, source=FakeSource(folder_of))


def test_limit_round_robins_over_folders(synthetic):
    _, mpath = synthetic
    rows = select_rows(read_manifest(mpath), 7)
    assert len({r["folder"] for r in rows}) == 7


def test_stored_detections_map_by_dataset_relative_path(tmp_path):
    root = tmp_path / "ds"
    run_dir = tmp_path / "run"
    (run_dir / "predictions").mkdir(parents=True)
    rec = {"source_path": str(root / "Archive (4)" / "IMG_1.PNG"), "photo_box": [0, 0, 10, 10],
           "detections": [{"xyxy": [1, 1, 5, 5], "gdino_confidence": 0.3, "prompt": "rru", "gdino_class": "radio_unit",
                           "alternatives": [{"gdino_class": "router", "gdino_confidence": 0.26}]}]}
    (run_dir / "predictions" / "predictions.json").write_text(json.dumps({"run": {}, "images": [rec]}))
    src = StoredDetections(run_dir, root)
    dets, photo = src.get("Archive (4)/IMG_1.PNG", np.zeros((10, 10, 3), np.uint8))
    assert dets[0]["gdino_class"] == "radio_unit" and dets[0]["alternatives"] and photo == [0, 0, 10, 10]
    with pytest.raises(KeyError):
        src.get("Archive (4)/missing.png", None)


def test_label_image_skips_degenerate_boxes(cfg):
    img = _noise(1)
    row = {"key": "k", "file_name": "k.png", "original_path": "x/k.png", "folder": "x", "folder_class_name": None}
    scorer = FakeScorer()
    scorer.big_class = "battery"
    rec = label_image(row, img, [{"xyxy": (200, 200, 210, 210), "gdino_confidence": 0.5, "gdino_class": "battery"}],
                      [0, 0, 120, 90], scorer, cfg)
    assert rec["boxes"] == [] and rec["trainable"] is False


# --- visual prototypes --------------------------------------------------------------------------------
def _entry(folder_class, group, vec, share=0.6):
    v = np.asarray(vec, np.float32)
    v = v / np.linalg.norm(v)
    return {"row": {"folder_class_name": folder_class, "near_duplicate_group": group},
            "boxes": [{"area_share": share}], "emb": v[None, :]}


def test_prototypes_are_leave_group_out_and_need_enough_anchors(cfg):
    from app.training.pseudo_label.prototypes import Prototypes

    cfg.proto_min_anchors = 2
    rng = np.random.default_rng(0)
    base = {c: rng.normal(size=16) for c in CLASSES}
    entries = [_entry(c, g * 10 + CLASSES.index(c), base[c] + 0.05 * rng.normal(size=16)) for c in CLASSES
               for g in range(3)]
    entries.append(_entry("radio_unit", 999, base["radio_unit"], share=0.05))  # too small to be an anchor
    p = Prototypes(entries, cfg)
    assert p.anchors()["radio_unit"] == 3
    # an image's own group is excluded: with 3 anchors and min 2, removing one still leaves a prototype
    m, names = p.matrix_for(group=0 * 10 + CLASSES.index("antenna"))
    assert "antenna" in names
    own = entries[0]["emb"][0]
    full = p.total["antenna"] / np.linalg.norm(p.total["antenna"])
    loo = m[names.index("antenna")]
    assert not np.allclose(full, loo)
    assert abs(float(own @ loo)) < abs(float(own @ full)) + 1e-6
    sims = p.similarities(entries[4]["emb"], group=-1)[0]
    assert max(sims, key=sims.get) == entries[4]["row"]["folder_class_name"]
    cfg.proto_min_anchors = 3  # leaving a group out drops antenna below the minimum
    assert "antenna" not in p.matrix_for(group=CLASSES.index("antenna"))[1]


def test_prototype_evidence_supplies_visual_support(cfg):
    """CLIP text ranks radio_unit 5th; the visual prototype ranks it 1st -> accepted, with provenance."""
    clip = {"antenna": 0.25, "router": 0.30, "cable": 0.24, "battery": 0.27, "power_supply_unit": 0.31,
            "equipment_cabinet": 0.28, "radio_unit": 0.26}
    proto = {c: 0.55 for c in CLASSES}
    proto["radio_unit"] = 0.80
    r = classify_box(det("equipment_cabinet", 0.5), clip, 0.1, 0.9, "radio_unit", cfg, flip_sims=clip,
                     proto_sims=proto, flip_proto_sims=proto)
    assert r["final_class"] == "radio_unit" and r["tier"] == ACCEPTED
    assert r["proto_top1"] == "radio_unit" and r["clip_rank_of_final"] == 5
    cfg.use_prototypes = False
    r2 = classify_box(det("equipment_cabinet", 0.5), clip, 0.1, 0.9, "radio_unit", cfg, flip_sims=clip,
                      proto_sims=proto)
    assert r2["tier"] != ACCEPTED or r2["final_class"] != "radio_unit"
