"""Zero-shot pipeline logic with fake models (no downloads). Run from the project root:
    backend\\.venv\\Scripts\\python.exe -m pytest experiments/zero_shot_bts/tests -q
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.zero_shot_bts.boxes import clip_xyxy, crop, cxcywh_norm_to_xyxy, iou, mask_to_box, nms, offset
from experiments.zero_shot_bts.config import CLIP_DESCRIPTIONS, GDINO_PROMPTS, UNCERTAIN, UNKNOWN, Config
from experiments.zero_shot_bts.models import build_prompt, decide, phrase_token_index
from experiments.zero_shot_bts.outputs import pick_examples, report_markdown, summarize, write_gallery
from experiments.zero_shot_bts.run import combine, process, select

from app.training.taxonomy import CLASS_NAMES


# --- geometry -------------------------------------------------------------------------------------
def test_coordinate_conversion_and_clipping():
    assert cxcywh_norm_to_xyxy((0.5, 0.5, 0.2, 0.4), 200, 100) == pytest.approx((80, 30, 120, 70))
    assert clip_xyxy((-5, -1, 250, 120), 200, 100) == (0, 0, 200, 100)
    assert offset((1, 2, 3, 4), 10, 20) == (11, 22, 13, 24)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


def test_crop_extraction_with_padding_and_bounds():
    img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    c = crop(img, (50, 20, 150, 60))
    assert c.shape == (40, 100, 3) and np.array_equal(c, img[20:60, 50:150])
    assert crop(img, (50, 20, 150, 60), padding=0.1).shape == (48, 120, 3)  # +10 % on each side
    # padded to (155, 65, 295, 165), then clipped to the 200x100 image -> (155, 65, 200, 100)
    assert crop(img, (190, 90, 260, 140), padding=0.5).shape[:2] == (35, 45)
    with pytest.raises(ValueError):
        crop(img, (300, 300, 400, 400))
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5) and mask_to_box(np.zeros((4, 4), bool)) is None


def test_duplicate_suppression_keeps_best_and_records_alternatives():
    dets = [{"xyxy": (0, 0, 100, 100), "gdino_confidence": 0.4, "prompt": "battery module", "gdino_class": "battery"},
            {"xyxy": (2, 1, 101, 99), "gdino_confidence": 0.6, "prompt": "equipment cabinet", "gdino_class": "equipment_cabinet"},
            {"xyxy": (300, 300, 350, 350), "gdino_confidence": 0.3, "prompt": "rru", "gdino_class": "radio_unit"}]
    kept = nms(dets, 0.5)
    assert [k["prompt"] for k in kept] == ["equipment cabinet", "rru"]
    assert kept[0]["alternatives"] == [{"prompt": "battery module", "gdino_class": "battery", "gdino_confidence": 0.4}]


# --- prompts and class mapping ----------------------------------------------------------------------
def test_prompts_cover_exactly_the_seven_classes():
    assert set(GDINO_PROMPTS) == set(CLIP_DESCRIPTIONS) == set(CLASS_NAMES)
    assert CLASS_NAMES == {0: "antenna", 1: "router", 2: "cable", 3: "battery", 4: "power_supply_unit",
                           5: "equipment_cabinet", 6: "radio_unit"}
    text, spans = build_prompt({0: ["Telecom Antenna"], 6: ["rru"]})
    assert text == "telecom antenna. rru."
    assert [(c, p, text[a:b]) for c, p, a, b in spans] == [(0, "telecom antenna", "telecom antenna"), (6, "rru", "rru")]
    # offsets as a tokenizer would return them: [CLS], telecom, antenna, ., rru, ., [SEP]
    offsets = [(0, 0), (0, 7), (8, 15), (15, 16), (17, 20), (20, 21), (0, 0)]
    assert phrase_token_index(offsets, spans) == [(0, "telecom antenna", [1, 2]), (6, "rru", [4])]


# --- CLIP decision ----------------------------------------------------------------------------------
def sims(**kw):
    base = {c: 0.20 for c in CLASS_NAMES}
    for name, v in kw.items():
        base[next(c for c, n in CLASS_NAMES.items() if n == name)] = v
    return base


def test_similarity_ranking_and_cautious_decisions():
    d = decide(sims(battery=0.31, power_supply_unit=0.28, router=0.25), 0.21, 0.20, 0.005)
    assert d["clip_label"] == "battery" and d["clip_reason"] is None
    assert [t["class"] for t in d["clip_top3"]] == ["battery", "power_supply_unit", "router"]
    assert d["clip_margin"] == pytest.approx(0.03)
    assert decide(sims(battery=0.31, power_supply_unit=0.308), 0.2, 0.20, 0.005)["clip_label"] == UNCERTAIN  # margin
    assert decide(sims(battery=0.19, router=0.15), 0.1, 0.20, 0.005)["clip_label"] == UNCERTAIN  # low similarity
    unk = decide(sims(battery=0.25), 0.27, 0.20, 0.005)  # a background description wins
    assert unk["clip_label"] == UNKNOWN and "background" in unk["clip_reason"]


def det(gclass="battery", conf=0.5, clip_label=None, box=(10, 10, 50, 50)):
    d = {"xyxy": box, "gdino_confidence": conf, "gdino_class": gclass, "prompt": "x", "prompt_score": 0.3}
    if clip_label:
        d["clip_label"] = clip_label
    return d


def test_combination_never_forces_a_class():
    assert combine(det(clip_label="battery"), True, 1e4)["final_label"] == "battery"
    dis = combine(det(clip_label="power_supply_unit"), True, 1e4)
    assert dis["final_label"] == UNCERTAIN and dis["agreement"] == "disagree"
    assert any(f.startswith("disagreement:gdino=battery,clip=power_supply_unit") for f in dis["flags"])
    assert combine(det(clip_label=UNCERTAIN), True, 1e4)["final_label"] == UNCERTAIN
    assert combine(det(clip_label=UNKNOWN), True, 1e4)["final_label"] == UNKNOWN
    g = combine(det(conf=0.3, box=(0, 0, 100, 100)), False, 100 * 100)
    assert g["final_label"] == "battery" and g["agreement"] == "no_clip"
    assert {"low_gdino_confidence", "near_whole_photo", "unverified_by_clip"} <= set(g["flags"])


def test_limited_runs_cover_every_folder():
    images = [(Path(f"{f}{i}.png"), f) for f in "abc" for i in range(5)]
    assert [f for _, f in select(images, 4)] == ["a", "b", "c", "a"]
    assert select(images, None) == images


# --- end-to-end outputs with fake models ---------------------------------------------------------------
class FakeDetector:
    load_seconds = 0.0

    def detect(self, image):  # boxes relative to the (letterbox-cropped) photo
        h, w = image.shape[:2]
        return [{"xyxy": (0.1 * w, 0.1 * h, 0.5 * w, 0.6 * h), "gdino_confidence": 0.45, "prompt": "battery module",
                 "gdino_class_id": 3, "gdino_class": "battery", "prompt_score": 0.3},
                {"xyxy": (0.12 * w, 0.1 * h, 0.5 * w, 0.62 * h), "gdino_confidence": 0.30, "prompt": "rru",
                 "gdino_class_id": 6, "gdino_class": "radio_unit", "prompt_score": 0.2},
                {"xyxy": (0.6 * w, 0.2 * h, 0.9 * w, 0.8 * h), "gdino_confidence": 0.33, "prompt": "coaxial cable",
                 "gdino_class_id": 2, "gdino_class": "cable", "prompt_score": 0.25}]


class FakeClassifier:
    load_seconds = 0.0

    def __init__(self):
        self.crops = []

    def classify(self, crops):
        self.crops += crops
        return [decide(sims(battery=0.3), 0.2, 0.2, 0.005), decide(sims(cable=0.22, antenna=0.219), 0.2, 0.2, 0.005)]


def test_process_writes_crops_visualization_and_reports(tmp_path):
    img = np.zeros((400, 200, 3), np.uint8)  # phone-style letterbox: photo in the middle
    img[150:250] = np.random.default_rng(0).integers(0, 255, (100, 200, 3), dtype=np.uint8)
    src = tmp_path / "data" / "Archive (5)" / "a b.png"
    src.parent.mkdir(parents=True)
    cv2.imwrite(str(src), img)
    before = src.read_bytes()
    out = tmp_path / "run"
    clf = FakeClassifier()
    rec = process(src, "Archive (5)", Config(), FakeDetector(), clf, None, out)

    assert src.read_bytes() == before  # dataset read-only
    assert rec["folder_class"] == "battery" and rec["photo_box"][1] >= 140
    assert len(rec["detections"]) == 2  # the rru duplicate was suppressed
    battery, cable = rec["detections"]
    assert battery["alternatives"][0]["prompt"] == "rru"
    assert battery["xyxy"][1] >= 150  # mapped back from the cropped photo to full-image pixels
    assert battery["final_label"] == "battery" and battery["agreement"] == "agree"
    assert cable["final_label"] == UNCERTAIN and cable["agreement"] == "clip_uncertain"
    assert all((out / d["crop"]).is_file() for d in rec["detections"]) and len(clf.crops) == 2
    assert (out / rec["visualization"]).is_file()

    empty = {**rec, "file": "empty.png", "detections": [], "timing": {"total_s": 1.0, "gdino_s": 0.9}}
    s = summarize([rec, empty], Config().to_dict(), {"wall_minutes": 0.1})
    assert s["images_processed"] == 2 and s["images_without_detections"] == 1 and s["total_detections"] == 2
    assert s["final_label_counts"] == {"battery": 1, UNCERTAIN: 1}
    assert s["gdino_vs_clip"] == {"battery -> battery": 1, f"cable -> {UNCERTAIN}": 1}
    assert "precision" not in json.dumps(s).lower().replace("no precision", "")
    md = report_markdown(s, pick_examples([rec, empty]))
    assert "No human-verified ground truth" in md and "no_detections" in md
    write_gallery(out / "gallery.html", [rec, empty], "test")
    html = (out / "gallery.html").read_text(encoding="utf-8")
    assert "Zero-shot BTS Gallery" in html and "a b.png" in html and "</script><" not in html.split("const D=")[1][:50]
