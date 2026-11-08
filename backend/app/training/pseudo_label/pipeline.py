"""Per-image pseudo-labelling in two passes.

Pass 1  `score_image`   detections -> crops -> CLIP (text similarities, mirrored crop, alt prompts, image embeddings)
Pass 2  `decide_image`  visual prototypes (built from all pass-1 results) -> fusion -> suppression -> YOLO lines

Detection sources:
  StoredDetections  Grounding DINO boxes recorded by an earlier zero-shot run (no detector re-run)
  LiveDetections    run Grounding DINO now, on the photo inside letterboxed screenshots
Scores can come from the live CLIP model or from an earlier run's cache (CachedScores). Everything is injectable
so tests can use fakes.
"""

import json
import time
from pathlib import Path

import numpy as np

from app.training.auto_annotate import content_box
from app.training.pseudo_label.config import CLASSES, PseudoLabelConfig
from app.training.pseudo_label.filters import image_decision, suppress
from app.training.pseudo_label.fusion import ACCEPTED, classify_box
from app.training.pseudo_label.geometry import area, crop, offset, to_yolo_line
from app.training.taxonomy import CLASS_IDS

CACHE_FILE = "clip_cache.npz"
SCORE_KEYS = ("clip_model", "clip_prompts", "clip_alt_prompts", "clip_negatives", "crop_padding")


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").lower()


class StoredDetections:
    """Detections from experiments/zero_shot_bts/runs/<run>/predictions/predictions.json, keyed by the image's
    path relative to the dataset root."""

    def __init__(self, run_dir: Path, dataset_root: Path):
        path = Path(run_dir) / "predictions" / "predictions.json"
        if not path.is_file():
            raise FileNotFoundError(f"No stored predictions at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        root = _norm_path(str(Path(dataset_root).resolve())).rstrip("/") + "/"
        self.by_path: dict[str, dict] = {}
        for rec in data["images"]:
            sp = _norm_path(rec["source_path"])
            self.by_path[sp[len(root):] if sp.startswith(root) else sp] = rec
        self.name = f"stored:{Path(run_dir).name}"

    def get(self, original_path: str, image: np.ndarray) -> tuple[list[dict], list[int]]:
        rec = self.by_path.get(_norm_path(original_path))
        if rec is None:
            raise KeyError(f"{original_path} is not in the stored run")
        dets = [{"xyxy": tuple(d["xyxy"]), "gdino_confidence": d["gdino_confidence"], "prompt": d["prompt"],
                 "gdino_class": d["gdino_class"], "alternatives": d.get("alternatives", [])} for d in rec["detections"]]
        return dets, list(rec["photo_box"])


class LiveDetections:
    def __init__(self, detector, letterbox_crop: bool = True):
        self.detector, self.letterbox_crop = detector, letterbox_crop
        self.name = "live:grounding_dino"

    def get(self, original_path: str, image: np.ndarray) -> tuple[list[dict], list[int]]:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = content_box(image) if self.letterbox_crop else (0, 0, w, h)
        dets = self.detector.detect(image[y0:y1, x0:x1])
        for d in dets:
            d["xyxy"] = offset(d["xyxy"], x0, y0)
        return dets, [x0, y0, x1, y1]


class CachedScores:
    """CLIP scores and embeddings recorded by an earlier pseudo-label run, so thresholds, fusion weights and
    prototypes can be re-tuned in seconds. Refuses a run whose CLIP settings differ."""

    def __init__(self, run_dir: Path, cfg: PseudoLabelConfig):
        run_dir = Path(run_dir)
        prev = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cur = cfg.to_dict()
        diff = [k for k in SCORE_KEYS if prev.get(k) != cur[k]]
        if diff:
            raise ValueError(f"Cannot reuse scores from {run_dir}: CLIP settings differ ({', '.join(diff)})")
        cache = np.load(run_dir / CACHE_FILE) if (run_dir / CACHE_FILE).is_file() else None
        self.scores: dict[str, dict[tuple, dict]] = {}
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            per = {}
            for i, b in enumerate(r["boxes"]):
                s = dict(b["clip_raw"])
                if cache is not None and f"{r['key']}__emb" in cache:
                    s["emb"] = cache[f"{r['key']}__emb"][i]
                    s["flip_emb"] = cache[f"{r['key']}__flip"][i]
                per[tuple(b["xyxy"])] = s
            self.scores[r["key"]] = per
        self.name = f"reused:{run_dir.name}"

    def score_boxes(self, key: str, boxes: list[dict], crops) -> list[dict]:
        cached = self.scores.get(key)
        if cached is None:
            raise KeyError(f"{key} has no cached scores")
        try:
            return [cached[tuple(b["xyxy"])] for b in boxes]
        except KeyError as exc:
            raise KeyError(f"{key}: box {exc} has no cached score (different detections?)") from None


def score_image(row: dict, image: np.ndarray, dets: list[dict], photo_box: list[int], scorer,
                cfg: PseudoLabelConfig) -> dict:
    """Pass 1: crops and CLIP scores for every usable detection of one image."""
    t0 = time.perf_counter()
    h, w = image.shape[:2]
    photo_area = float(max(1, (photo_box[2] - photo_box[0]) * (photo_box[3] - photo_box[1])))
    boxes, crops = [], []
    for d in dets:
        b = {"id": len(boxes), "xyxy": [round(float(v), 1) for v in d["xyxy"]], "gdino_class": d["gdino_class"],
             "gdino_confidence": round(float(d["gdino_confidence"]), 4), "prompt": d.get("prompt"),
             "alternatives": d.get("alternatives", [])}
        try:
            crops.append(crop(image, b["xyxy"], cfg.crop_padding))
        except ValueError:
            continue  # degenerate box: nothing to classify
        b["area_share"] = round(area(b["xyxy"]) / photo_area, 4)
        boxes.append(b)
    scores = scorer.score_boxes(row["key"], boxes, crops) if hasattr(scorer, "score_boxes") else scorer.score(crops)
    emb = np.stack([s["emb"] for s in scores]).astype(np.float32) if scores and "emb" in scores[0] else None
    flip = np.stack([s["flip_emb"] for s in scores]).astype(np.float32) if scores and "flip_emb" in scores[0] else None
    return {"row": row, "width": w, "height": h, "photo_box": photo_box, "boxes": boxes, "scores": scores,
            "emb": emb if emb is not None else np.zeros((0, 512), np.float32), "flip_emb": flip,
            "score_s": time.perf_counter() - t0}


def _rounded(d: dict | None) -> dict | None:
    return {c: round(float(v), 5) for c, v in d.items()} if d else None


def decide_image(entry: dict, cfg: PseudoLabelConfig, protos=None, refiner=None, image: np.ndarray | None = None) -> dict:
    """Pass 2: fuse the evidence, apply suppression and the image-level rule, produce the YOLO lines."""
    t0 = time.perf_counter()
    row, boxes, scores = entry["row"], entry["boxes"], entry["scores"]
    w, h = entry["width"], entry["height"]
    folder_class = row.get("folder_class_name") or None
    proto, flip_proto = [None] * len(boxes), [None] * len(boxes)
    if protos is not None and cfg.use_prototypes and len(entry["emb"]):
        group = int(row.get("near_duplicate_group", -1))
        proto = protos.similarities(entry["emb"], group)
        if entry.get("flip_emb") is not None:
            flip_proto = protos.similarities(entry["flip_emb"], group)
    for b, s, ps, fps in zip(boxes, scores, proto, flip_proto):
        b["clip_raw"] = {"sims": _rounded(s["sims"]), "flip_sims": _rounded(s.get("flip_sims")),
                         "alt_sims": _rounded(s.get("alt_sims")), "negative_sim": round(float(s["negative_sim"]), 5)}
        b.update(classify_box(b, s["sims"], s["negative_sim"], b["area_share"], folder_class, cfg,
                              flip_sims=s.get("flip_sims"), alt_sims=s.get("alt_sims"), proto_sims=ps,
                              flip_proto_sims=fps))
    suppress(boxes, cfg)

    accepted = [b for b in boxes if b["tier"] == ACCEPTED]
    if refiner is not None and accepted and image is not None:
        for b, r in zip(accepted, refiner.refine(image, [b["xyxy"] for b in accepted])):
            b.update(r)
    for b in boxes:
        b["label_xyxy"] = b["sam_xyxy"] if b.get("sam_used") else b["xyxy"]
        b["box_source"] = "mobile_sam" if b.get("sam_used") else "grounding_dino"

    trainable, exclusion = image_decision(boxes, cfg)
    lines = [to_yolo_line(CLASS_IDS[b["final_class"]], b["label_xyxy"], w, h) for b in boxes if b["tier"] == ACCEPTED]
    return {
        "key": row["key"], "file": row["file_name"], "original_path": row["original_path"], "folder": row["folder"],
        "folder_class": folder_class, "near_duplicate_group": row.get("near_duplicate_group"), "width": w, "height": h,
        "photo_box": entry["photo_box"], "boxes": boxes, "trainable": trainable, "exclusion_reason": exclusion,
        "yolo_lines": lines, "timing_s": round(entry.get("score_s", 0.0) + time.perf_counter() - t0, 3),
    }


def label_image(row: dict, image: np.ndarray, dets: list[dict], photo_box: list[int], scorer, cfg: PseudoLabelConfig,
                refiner=None, protos=None) -> dict:
    """Both passes for a single image (no prototypes unless given)."""
    return decide_image(score_image(row, image, dets, photo_box, scorer, cfg), cfg, protos, refiner, image)


__all__ = ["CACHE_FILE", "CLASSES", "CachedScores", "LiveDetections", "StoredDetections", "decide_image",
           "label_image", "score_image"]
