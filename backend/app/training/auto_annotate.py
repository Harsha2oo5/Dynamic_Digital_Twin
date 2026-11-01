"""Assisted annotation: generate CANDIDATE boxes for human review (never ground truth).

A proposer (any detector) suggests boxes; detector labels are mapped to the seven-class taxonomy
through an explicit table; post-processing applies the annotation rules; every candidate and
image gets a review tier. Outputs go to annotation/candidates/<run>/ together with a marker file
that makes validate_annotations / prepare_dataset refuse the directory as training labels.

Folder classes are used ONLY as a quality-control signal ("expected component not proposed"),
never to assign or relabel a box. No whole-image or synthetic boxes are produced.

Proposers:
  vocabulary  any Ultralytics detector whose class names are mapped by a table, e.g. the
              prompt-free YOLOE (yoloe-11l-seg-pf.pt, 4585-term vocabulary) or, later, our own
              custom checkpoint (identity mapping)
  text        YOLOE / YOLO-World with text prompts (requires the `clip` tokenizer package)

Usage:
    python -m app.training.auto_annotate --batch ../annotation/batches/batch_01 \
        --model models/annotation_assist/yoloe-11l-seg-pf.pt --name yoloe_pf_batch01
"""

import argparse
import csv
import html
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.core.device import resolve_device
from app.training.audit_dataset import read_image
from app.training.dataset_validator import IMAGE_EXTS
from app.training.taxonomy import CLASS_IDS, CLASS_NAMES

MARKER = "CANDIDATES_NOT_GROUND_TRUTH.txt"

# Detector vocabulary term -> (class id, mapping quality). "direct": the term names the component;
# "ambiguous": the term often but not always denotes the component -> always needs review.
# Measured on batch_01 (2026-11-01): the prompt-free YOLOE proposed the folder's own component in only
# 10/175 images (1/175 at conf >= 0.5) while mapped terms fired 89 times on other folders, so this map
# is kept for evaluation only; it is NOT recommended for CVAT pre-annotation (see annotation-guide.md).
YOLOE_PF_VOCAB_MAP: dict[str, tuple[int, str]] = {
    "antenna": (0, "direct"),
    "dish antenna": (0, "direct"),
    "router": (1, "direct"),
    "switch": (1, "ambiguous"),
    "server": (1, "ambiguous"),
    "cable": (2, "direct"),
    "wire": (2, "direct"),
    "extension cord": (2, "ambiguous"),
    "connector": (2, "ambiguous"),
    "battery": (3, "direct"),
    "car battery": (3, "direct"),
    "portable battery": (3, "ambiguous"),
    "solar battery": (3, "ambiguous"),
    "cabinet": (5, "direct"),
    "enclosure": (5, "ambiguous"),
    "locker": (5, "ambiguous"),
    "file cabinet": (5, "ambiguous"),
    "side cabinet": (5, "ambiguous"),
}

# Text prompts per class for open-vocabulary detectors (several phrasings per class are allowed).
DEFAULT_PROMPTS: dict[int, list[str]] = {
    0: ["antenna"],
    1: ["telecommunications router"],
    2: ["electrical or telecom cable"],
    3: ["battery module"],
    4: ["power supply unit or rectifier module"],
    5: ["equipment cabinet"],
    6: ["radio unit or remote radio unit"],
}

CLASS_COLORS_BGR = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245), (180, 30, 145),
                    (244, 212, 66)]


@dataclass
class RawCandidate:
    source_label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class Candidate:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    source_label: str
    mapping: str  # direct | ambiguous
    tier: str = "low"  # high | low
    flags: list[str] = field(default_factory=list)


@dataclass
class Thresholds:
    min_conf: float = 0.10  # proposals below this are discarded
    high_conf: float = 0.50  # high tier needs this AND a direct mapping AND no flags
    nms_iou: float = 0.60  # same-class duplicate suppression
    conflict_iou: float = 0.70  # same region, different classes
    cable_overlap: float = 0.30  # IoU (or containment of the smaller box) above which cables form one bundle
    tiny_area: float = 0.002  # fraction of the image
    huge_area: float = 0.90
    merge_cross_class: bool = False  # collapse different-class boxes on one object (open-vocabulary proposers)


class Proposer(Protocol):
    name: str

    def propose(self, image: np.ndarray) -> list[RawCandidate]: ...


class UltralyticsProposer:
    """Wraps any Ultralytics detector; `label_map` turns its class names into taxonomy classes."""

    def __init__(self, model_path: Path, label_map: dict[str, tuple[int, str]], imgsz: int = 640, conf: float = 0.1,
                 prompts: dict[int, list[str]] | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.name = Path(model_path).name
        self.imgsz, self.conf = imgsz, conf
        if prompts:  # text-prompt mode: the prompt index defines the class
            texts, label_map = [], {}
            for cid, phrases in prompts.items():
                for p in phrases:
                    texts.append(p)
                    label_map[p] = (cid, "direct")
            try:
                self.model.set_classes(texts)
            except Exception as exc:  # noqa: BLE001 - missing text-encoder dependency, download blocked, ...
                raise RuntimeError(f"Text prompts unavailable for {self.name}: {exc}") from exc
            self.name += " (text prompts)"
        self.label_map = label_map

    def propose(self, image: np.ndarray) -> list[RawCandidate]:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=resolve_device("cpu"), verbose=False)[0]
        return [RawCandidate(r.names[int(c)], float(p), tuple(float(v) for v in b))
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def content_box(image: np.ndarray, std_thr: float = 8.0, pad: int = 4) -> tuple[int, int, int, int]:
    """Bounding box of the photo inside a letterboxed screenshot: rows/columns whose grey-level spread is
    below `std_thr` (uniform black or white bars) are trimmed. Falls back to the full image."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = grey.shape
    rows = np.where(grey.std(axis=1) > std_thr)[0]
    if len(rows) == 0:
        return 0, 0, w, h
    # columns are measured inside the kept rows only, otherwise black-over-white bars look like content
    cols = np.where(grey[rows[0]:rows[-1] + 1].std(axis=0) > std_thr)[0]
    if len(cols) == 0:
        return 0, 0, w, h
    x0, y0 = max(0, int(cols[0]) - pad), max(0, int(rows[0]) - pad)
    x1, y1 = min(w, int(cols[-1]) + 1 + pad), min(h, int(rows[-1]) + 1 + pad)
    if (x1 - x0) * (y1 - y0) < 0.05 * w * h:  # implausibly small: do not trust the trim
        return 0, 0, w, h
    return x0, y0, x1, y1


class LetterboxCropProposer:
    """Runs the inner proposer on the photo content only (screenshots are up to ~80% padding) and maps
    the boxes back to full-image pixels, so labels stay valid for the staged (uncropped) images."""

    def __init__(self, inner: Proposer):
        self.inner = inner
        self.name = f"{inner.name} + letterbox crop"

    def propose(self, image: np.ndarray) -> list[RawCandidate]:
        x0, y0, x1, y1 = content_box(image)
        raw = self.inner.propose(image[y0:y1, x0:x1])
        return [RawCandidate(r.source_label, r.confidence,
                             (r.xyxy[0] + x0, r.xyxy[1] + y0, r.xyxy[2] + x0, r.xyxy[3] + y0)) for r in raw]


# Grounding DINO phrases; several phrases may map to one class. Short nouns ground better than descriptions.
GDINO_PROMPTS: dict[int, list[str]] = {
    0: ["antenna"],
    1: ["router"],
    2: ["cable"],
    3: ["battery"],
    4: ["power supply"],
    5: ["cabinet"],
    6: ["radio unit"],
}


class GroundingDinoProposer:
    """Open-vocabulary candidates from Grounding DINO (Hugging Face transformers, CPU).

    The stock label decoding concatenates every phrase whose token score passes a threshold, giving
    labels like "power supply unit equipment cabinet". Here each query box is scored per class as the mean
    token probability inside that class's phrase span (the max let generic tokens such as "unit" win), and
    gets exactly one class; if the runner-up class is close (`ambiguity_ratio`), the mapping is marked
    ambiguous so the box is always flagged for review."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", prompts: dict[int, list[str]] | None = None,
                 short_side: int = 512, long_side: int = 896, conf: float = 0.1, ambiguity_ratio: float = 0.8):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval()
        self.name = model_id.split("/")[-1]
        self.size = {"shortest_edge": short_side, "longest_edge": long_side}
        self.conf, self.ambiguity_ratio = conf, ambiguity_ratio
        prompts = prompts or GDINO_PROMPTS
        phrases = [(cid, p.lower()) for cid, ps in prompts.items() for p in ps]
        self.text, spans, pos = "", [], 0
        for cid, p in phrases:
            spans.append((cid, pos, pos + len(p)))
            self.text += p + ". "
            pos = len(self.text)
        self.text = self.text.strip()
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        self.class_tokens: dict[int, list[int]] = defaultdict(list)
        for ti, (s, e) in enumerate(enc["offset_mapping"]):
            for cid, a, b in spans:
                if e > s and s >= a and e <= b:
                    self.class_tokens[cid].append(ti)
        self.label_map = {CLASS_NAMES[cid]: (cid, "direct") for cid in self.class_tokens}
        self.label_map.update({f"{CLASS_NAMES[cid]}?": (cid, "ambiguous") for cid in self.class_tokens})

    def propose(self, image: np.ndarray) -> list[RawCandidate]:
        from PIL import Image

        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.text, return_tensors="pt", size=self.size)
        with self.torch.no_grad():
            out = self.model(**inputs)
        probs = out.logits[0].sigmoid()  # (queries, text tokens)
        boxes = out.pred_boxes[0]  # normalised cx, cy, w, h
        cids = sorted(self.class_tokens)
        per_class = self.torch.stack([probs[:, self.class_tokens[c]].mean(dim=1) for c in cids], dim=1)
        top2 = per_class.topk(min(2, len(cids)), dim=1)
        raw = []
        for q in range(per_class.shape[0]):
            score = float(top2.values[q, 0])
            if score < self.conf:
                continue
            cid = cids[int(top2.indices[q, 0])]
            ambiguous = len(cids) > 1 and float(top2.values[q, 1]) >= self.ambiguity_ratio * score
            cx, cy, bw, bh = (float(v) for v in boxes[q])
            xyxy = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            raw.append(RawCandidate(CLASS_NAMES[cid] + ("?" if ambiguous else ""), score, xyxy))
        return raw


# --- geometry ----------------------------------------------------------------------------------
def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a, b) -> float:
    """max(IoU, intersection / smaller area): high when one box sits inside the other."""
    inter = intersection(a, b)
    smaller = min(box_area(a), box_area(b))
    return max(iou(a, b), inter / smaller if smaller > 0 else 0.0)


# --- rules -------------------------------------------------------------------------------------
def map_candidates(raw: list[RawCandidate], label_map: dict[str, tuple[int, str]], min_conf: float
                   ) -> tuple[list[Candidate], Counter]:
    mapped, unmapped = [], Counter()
    for r in raw:
        if r.confidence < min_conf:
            continue
        hit = label_map.get(r.source_label)
        if hit is None:
            unmapped[r.source_label] += 1
            continue
        cid, quality = hit
        mapped.append(Candidate(cid, CLASS_NAMES[cid], round(r.confidence, 4), r.xyxy, r.source_label, quality))
    return mapped, unmapped


def suppress_duplicates(cands: list[Candidate], iou_thr: float) -> list[Candidate]:
    """Same-class NMS across vocabulary terms (e.g. 'cable' and 'wire' on one object)."""
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.confidence):
        if all(k.class_id != c.class_id or iou(k.xyxy, c.xyxy) < iou_thr for k in kept):
            kept.append(c)
    return kept


def merge_cross_class(cands: list[Candidate], iou_thr: float) -> list[Candidate]:
    """One box per object: a box overlapping a higher-scoring box of another class is dropped, and its class
    is kept on the survivor as an `alt_class` flag (survivor becomes ambiguous, so it is always reviewed)."""
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.confidence):
        owner = next((k for k in kept if k.class_id != c.class_id and iou(k.xyxy, c.xyxy) > iou_thr), None)
        if owner is None:
            kept.append(c)
            continue
        alt = f"alt_class:{c.class_name}({c.confidence:.2f})"
        if not any(f.startswith(f"alt_class:{c.class_name}(") for f in owner.flags):
            owner.flags.append(alt)
        owner.mapping = "ambiguous"
    return kept


def merge_cable_bundles(cands: list[Candidate], overlap_thr: float) -> list[Candidate]:
    """Cable rule: overlapping cable boxes are treated as ONE inseparable bundle (union box, flagged).
    Non-overlapping cables stay separate. Never emits overlapping cable boxes."""
    cables = [c for c in cands if c.class_id == CLASS_IDS["cable"]]
    others = [c for c in cands if c.class_id != CLASS_IDS["cable"]]
    clusters: list[list[Candidate]] = []
    for c in sorted(cables, key=lambda c: -c.confidence):
        joined = [cl for cl in clusters if any(overlap_ratio(c.xyxy, m.xyxy) > overlap_thr for m in cl)]
        merged = [c] + [m for cl in joined for m in cl]
        clusters = [cl for cl in clusters if cl not in joined] + [merged]
    # union boxes can grow into each other; repeat until stable
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if overlap_ratio(_union(clusters[i]), _union(clusters[j])) > overlap_thr:
                    clusters[i] += clusters.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = []
    for cl in clusters:
        best = max(cl, key=lambda c: c.confidence)
        if len(cl) == 1:
            out.append(best)
        else:
            out.append(Candidate(best.class_id, best.class_name, best.confidence, _union(cl), best.source_label,
                                 best.mapping, flags=[f"cable_bundle_merged({len(cl)})"]))
    return others + out


def _union(cl: list[Candidate]) -> tuple[float, float, float, float]:
    return (min(c.xyxy[0] for c in cl), min(c.xyxy[1] for c in cl), max(c.xyxy[2] for c in cl), max(c.xyxy[3] for c in cl))


def flag_and_tier(cands: list[Candidate], width: int, height: int, t: Thresholds) -> list[Candidate]:
    area = float(width * height)
    for c in cands:
        frac = box_area(c.xyxy) / area
        if frac < t.tiny_area:
            c.flags.append("tiny_box")
        if frac > t.huge_area:
            c.flags.append("near_whole_image_box")
        if c.mapping != "direct":
            c.flags.append(f"ambiguous_label:{c.source_label}")
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if a.class_id != b.class_id and iou(a.xyxy, b.xyxy) > t.conflict_iou:
                a.flags.append(f"class_conflict:{b.class_name}")
                b.flags.append(f"class_conflict:{a.class_name}")
    for c in cands:
        c.tier = "high" if c.confidence >= t.high_conf and c.mapping == "direct" and not c.flags else "low"
    return cands


def postprocess(raw: list[RawCandidate], label_map, width: int, height: int, t: Thresholds
                ) -> tuple[list[Candidate], Counter]:
    cands, unmapped = map_candidates(raw, label_map, t.min_conf)
    cands = suppress_duplicates(cands, t.nms_iou)
    if t.merge_cross_class:
        cands = merge_cross_class(cands, t.conflict_iou)
    cands = merge_cable_bundles(cands, t.cable_overlap)
    return flag_and_tier(cands, width, height, t), unmapped


def image_status(cands: list[Candidate], expected_class: int | None) -> tuple[str, list[str]]:
    """Review priority for one image. Every status still needs human confirmation."""
    reasons = []
    if not cands:
        return "no_detection", ["no candidate for any of the seven classes"]
    classes = {c.class_id for c in cands}
    if expected_class is not None and expected_class not in classes:
        reasons.append(f"expected component '{CLASS_NAMES[expected_class]}' not proposed")
    low = [c for c in cands if c.tier == "low"]
    if low:
        reasons.append(f"{len(low)} low-confidence/flagged candidate(s)")
    return ("needs_review" if reasons else "high_confidence"), reasons


# --- outputs -----------------------------------------------------------------------------------
def to_yolo(c: Candidate, w: int, h: int) -> str:
    x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(c.xyxy, (w, h, w, h)))
    return f"{c.class_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def draw_preview(img: np.ndarray, cands: list[Candidate], title: str, max_side: int = 1280) -> np.ndarray:
    scale = min(1.0, max_side / max(img.shape[:2]))
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    for c in cands:
        color = CLASS_COLORS_BGR[c.class_id]
        x1, y1, x2, y2 = (int(v * scale) for v in c.xyxy)
        if c.tier == "high":
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        else:  # dashed outline = low confidence / flagged
            for k in range(x1, x2, 16):
                cv2.line(out, (k, y1), (min(k + 8, x2), y1), color, 2)
                cv2.line(out, (k, y2), (min(k + 8, x2), y2), color, 2)
            for k in range(y1, y2, 16):
                cv2.line(out, (x1, k), (x1, min(k + 8, y2)), color, 2)
                cv2.line(out, (x2, k), (x2, min(k + 8, y2)), color, 2)
        label = f"{c.class_name} {c.confidence:.2f}{'' if c.tier == 'high' else ' ?'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def write_contact_sheets(out: Path, records: list[dict], cols: int = 5, tile: int = 320) -> list[Path]:
    """One JPEG grid of candidate previews per batch task (folder class), for a quick visual pass."""
    sheets = []
    (out / "contact_sheets").mkdir(exist_ok=True)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        rows = (len(rs) + cols - 1) // cols
        sheet = np.full((rows * tile, cols * tile, 3), 30, np.uint8)
        for i, r in enumerate(rs):
            img = read_image(out / r["preview"])
            if img is None:
                continue
            s = min(tile / img.shape[1], tile / img.shape[0])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
            y, x = (i // cols) * tile, (i % cols) * tile
            sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
        path = out / "contact_sheets" / f"{task}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(path))
        sheets.append(path)
    return sheets


def write_cvat_zip(path: Path, items: list[tuple[str, list[str]]]) -> None:
    """CVAT 'YOLO 1.1' annotation archive for one task: upload via Task → Actions → Upload annotations."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASS_NAMES.values()) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASS_NAMES)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{name}\n" for name, _ in items))
        for name, lines in items:
            z.writestr(f"obj_train_data/{Path(name).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))


REVIEW_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candidate review</title><style>
:root{--bg:#0b0f14;--card:#111821;--line:#263241;--ink:#e6edf3;--ink2:#a4b1c0;--good:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#2fb5d2}
body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;z-index:1}
h1{font-size:16px;margin:0 0 8px}.note{color:var(--warn);font-size:12px;margin:4px 0 8px}button{background:#17202b;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 10px;margin:2px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}.card img{width:100%;max-height:420px;object-fit:contain;background:#000;display:block}
.meta{padding:8px 10px}.fn{font-family:monospace;font-size:12px;word-break:break-all}.badge{display:inline-block;border-radius:4px;padding:1px 6px;font-size:11px;margin:2px 4px 2px 0;border:1px solid}
.high_confidence{color:var(--good);border-color:var(--good)}.needs_review{color:var(--warn);border-color:var(--warn)}.no_detection{color:var(--bad);border-color:var(--bad)}.multi{color:var(--acc);border-color:var(--acc)}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}td{border-top:1px solid var(--line);padding:3px 4px;color:var(--ink2)}.reason{font-size:12px;color:var(--warn)}
</style></head><body><header><h1>Candidate review: __RUN__</h1>
<div class="note">Machine-generated candidates; none of them is ground truth. Correct every image in CVAT before export.</div>
<div id="filters"></div></header><main id="grid"></main>
<script>
const DATA=__DATA__;const F=[["all","All"],["high_confidence","High confidence"],["needs_review","Needs review"],["no_detection","No detection"],["multi","Multiple classes"]];
let cur="all";function show(){const g=document.getElementById("grid");g.innerHTML="";
for(const d of DATA){if(cur==="multi"?!d.multi_class:cur!=="all"&&d.status!==cur)continue;
const rows=d.candidates.map(c=>`<tr><td>${c.class_name}</td><td>${c.confidence.toFixed(2)}</td><td>${c.tier}</td><td>${c.source_label}</td><td>${c.flags.join(", ")}</td></tr>`).join("");
g.insertAdjacentHTML("beforeend",`<div class="card"><img loading="lazy" src="${d.preview}"><div class="meta"><div class="fn">${d.file}</div>
<span class="badge ${d.status}">${d.status.replace("_"," ")}</span>${d.multi_class?'<span class="badge multi">multiple classes</span>':""}
<span class="badge" style="color:#a4b1c0;border-color:#263241">folder: ${d.folder_class}</span>
<div class="reason">${d.reasons.join("; ")}</div><table>${rows}</table></div></div>`)}}
document.getElementById("filters").innerHTML=F.map(([k,l])=>`<button data-k="${k}">${l} (${k==="all"?DATA.length:k==="multi"?DATA.filter(d=>d.multi_class).length:DATA.filter(d=>d.status===k).length})</button>`).join("");
document.querySelectorAll("#filters button").forEach(b=>b.onclick=()=>{cur=b.dataset.k;document.querySelectorAll("#filters button").forEach(x=>x.classList.toggle("on",x===b));show()});
document.querySelector("#filters button").classList.add("on");show();
</script></body></html>"""


def run(batch_dir: Path, proposer: Proposer, label_map: dict, out: Path, t: Thresholds) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite candidate run {out}")
    expected = {}
    manifest = batch_dir / "batch_manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as f:
            expected = {r["file_name"]: int(r["class_id"]) for r in csv.DictReader(f)}
    images = sorted(p for p in batch_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images in {batch_dir}")

    (out / "labels").mkdir(parents=True)
    (out / "cvat_import").mkdir()
    for status in ("high_confidence", "needs_review", "no_detection"):
        (out / "previews" / status).mkdir(parents=True)
    (out / MARKER).write_text(
        "Files in this directory are MACHINE-GENERATED CANDIDATES, not verified annotations.\n"
        "Import them into CVAT as pre-annotations, correct every image, and export the reviewed result to\n"
        "annotation/exports/. validate_annotations and prepare_dataset refuse this directory.\n", encoding="utf-8")

    records, per_task, unmapped_total = [], defaultdict(list), Counter()
    started = time.time()
    for path in images:
        img = read_image(path)
        if img is None:
            print(f"  unreadable: {path}")
            continue
        h, w = img.shape[:2]
        cands, unmapped = postprocess(proposer.propose(img), label_map, w, h, t)
        unmapped_total.update(unmapped)
        exp = expected.get(path.name)
        status, reasons = image_status(cands, exp)
        lines = [to_yolo(c, w, h) for c in cands]
        (out / "labels" / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        preview = out / "previews" / status / f"{path.stem}.jpg"
        cv2.imencode(".jpg", draw_preview(img, cands, f"{path.name} | {status}"))[1].tofile(str(preview))
        per_task[path.parent.name].append((path.name, lines))
        records.append({
            "file": path.name, "task": path.parent.name, "folder_class": CLASS_NAMES.get(exp, "?"),
            "status": status, "reasons": reasons, "multi_class": len({c.class_id for c in cands}) > 1,
            "preview": preview.relative_to(out).as_posix(), "width": w, "height": h,
            "candidates": [{**asdict(c), "xyxy": [round(v, 1) for v in c.xyxy]} for c in cands],
        })
    elapsed = time.time() - started

    for task, items in per_task.items():
        write_cvat_zip(out / "cvat_import" / f"{batch_dir.name}_{task}.zip", items)
    with (out / "manual_review.csv").open("w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["file", "task", "status", "multi_class", "reasons"])
        for r in records:
            if r["status"] != "high_confidence":
                w_.writerow([r["file"], r["task"], r["status"], r["multi_class"], "; ".join(r["reasons"])])
    (out / "candidates.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    write_contact_sheets(out, records)
    (out / "review.html").write_text(
        REVIEW_HTML.replace("__RUN__", html.escape(out.name)).replace("__DATA__", json.dumps(records)), encoding="utf-8")

    report = summarize(records, proposer.name, t, elapsed, unmapped_total)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def summarize(records: list[dict], model: str, t: Thresholds, elapsed: float, unmapped: Counter) -> dict:
    per_folder = {}
    for cid, name in CLASS_NAMES.items():
        rs = [r for r in records if r["folder_class"] == name]
        per_folder[name] = {
            "images": len(rs),
            "high_confidence": sum(r["status"] == "high_confidence" for r in rs),
            "needs_review": sum(r["status"] == "needs_review" for r in rs),
            "no_detection": sum(r["status"] == "no_detection" for r in rs),
            "multi_class": sum(r["multi_class"] for r in rs),
            "expected_class_proposed": sum(any(c["class_id"] == cid for c in r["candidates"]) for r in rs),
        }
    all_c = [c for r in records for c in r["candidates"]]
    per_class_candidates = {
        name: {"boxes": sum(c["class_id"] == cid for c in all_c),
               "high_tier": sum(c["class_id"] == cid and c["tier"] == "high" for c in all_c),
               "low_tier": sum(c["class_id"] == cid and c["tier"] == "low" for c in all_c),
               "images_with_candidate": sum(any(c["class_id"] == cid for c in r["candidates"]) for r in records)}
        for cid, name in CLASS_NAMES.items()}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "thresholds": asdict(t),
        "images": len(records),
        "seconds": round(elapsed, 1),
        "candidate_boxes": len(all_c),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "multi_class_images": sum(r["multi_class"] for r in records),
        "multi_box_images": sum(len(r["candidates"]) > 1 for r in records),
        "confidence_histogram": {f"{lo / 10:.1f}-{(lo + 1) / 10:.1f}": sum(lo <= c["confidence"] * 10 < lo + 1 for c in all_c)
                                 for lo in range(10)},
        "images_requiring_manual_review": sum(r["status"] != "high_confidence" for r in records),
        "per_folder": per_folder,
        "per_class_candidates": per_class_candidates,
        "top_unmapped_detector_labels": unmapped.most_common(25),
        "human_reviewed_annotations": 0,
    }


def report_markdown(r: dict) -> str:
    lines = [
        f"# Candidate annotation report ({r['model']})", "",
        "> Machine-generated candidates. **0 boxes are human-reviewed.** Nothing here is training data.", "",
        f"- Images: {r['images']} in {r['seconds']} s on CPU; candidate boxes: {r['candidate_boxes']}",
        f"- Status: {r['status_counts']}; multiple-class images: {r['multi_class_images']}; "
        f"images with more than one box: {r.get('multi_box_images', '?')}",
        "- Detector confidence histogram (score, NOT accuracy): "
        + ", ".join(f"{k}: {v}" for k, v in r.get("confidence_histogram", {}).items() if v),
        f"- Images requiring manual review (all non-high-confidence): {r['images_requiring_manual_review']}", "",
        "## Per source folder", "",
        "| Folder class | Images | High confidence | Needs review | No detection | Multi-class | Expected class proposed |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, p in r["per_folder"].items():
        lines.append(f"| {name} | {p['images']} | {p['high_confidence']} | {p['needs_review']} | {p['no_detection']} | "
                     f"{p['multi_class']} | {p['expected_class_proposed']} |")
    lines += ["", "## Candidate boxes per class", "", "| Class | Boxes | High tier | Low tier | Images with a candidate |",
              "|---|---|---|---|---|"]
    for name, p in r["per_class_candidates"].items():
        lines.append(f"| {name} | {p['boxes']} | {p['high_tier']} | {p['low_tier']} | {p['images_with_candidate']} |")
    lines += ["", "Most frequent detector labels with no taxonomy mapping (ignored): "
              + ", ".join(f"{k} ×{v}" for k, v in r["top_unmapped_detector_labels"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    from app.core.config import BACKEND_DIR, PROJECT_DIR

    parser = argparse.ArgumentParser(description="Generate candidate boxes for human review (not ground truth).")
    parser.add_argument("--batch", type=Path, default=PROJECT_DIR / "annotation" / "batches" / "batch_01")
    parser.add_argument("--model", type=Path, default=BACKEND_DIR / "models" / "annotation_assist" / "yoloe-11l-seg-pf.pt")
    parser.add_argument("--mode", choices=["vocabulary", "text", "custom", "gdino"], default="vocabulary",
                        help="vocabulary: prompt-free YOLOE map; text: open-vocabulary prompts; custom: our 7-class model; "
                             "gdino: Grounding DINO (transformers)")
    parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--no-crop", action="store_true", help="do not trim letterbox bars before proposing")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-conf", type=float, default=Thresholds.min_conf)
    parser.add_argument("--high-conf", type=float, default=Thresholds.high_conf)
    args = parser.parse_args()

    t = Thresholds(min_conf=args.min_conf, high_conf=args.high_conf)
    if args.mode == "custom":
        label_map = {n: (cid, "direct") for cid, n in CLASS_NAMES.items()}
        proposer = UltralyticsProposer(args.model, label_map, args.imgsz, t.min_conf)
    elif args.mode == "text":
        try:
            proposer = UltralyticsProposer(args.model, {}, args.imgsz, t.min_conf, prompts=DEFAULT_PROMPTS)
        except RuntimeError as exc:
            sys.exit(str(exc))
        label_map = proposer.label_map
    elif args.mode == "gdino":
        t.merge_cross_class = True
        proposer = GroundingDinoProposer(args.gdino_model, conf=t.min_conf)
        label_map = proposer.label_map
    else:
        label_map = YOLOE_PF_VOCAB_MAP
        proposer = UltralyticsProposer(args.model, label_map, args.imgsz, t.min_conf)
    if not args.no_crop:
        proposer = LetterboxCropProposer(proposer)

    out = PROJECT_DIR / "annotation" / "candidates" / args.name
    report = run(args.batch, proposer, label_map, out, t)
    print(report_markdown(report))
    print(f"Review: {out / 'review.html'}\nCVAT pre-annotations: {out / 'cvat_import'}")


if __name__ == "__main__":
    main()
