"""Run outputs: YOLO label files, pseudo_manifest.json, records, report and a browsable gallery.

Layout of annotation/pseudo_labels/<run>/:
  PSEUDO_LABELS_NOT_GROUND_TRUTH.txt  marker: the validator refuses this directory as reviewed ground truth
  classes.txt                         the seven taxonomy classes in id order
  labels/<key>.txt                    YOLO labels, ONLY for images the pipeline marked trainable
  pseudo_manifest.json                provenance + sha256 of every label file (checked by prepare_dataset --pseudo)
  records.jsonl                       every image and every box with all evidence and the decision reason
  report.json / report.md             counts, distributions, rejection/uncertainty reasons, consistency measures
  gallery.html + previews/*.jpg       visual check: solid = accepted, dashed = uncertain, thin grey = rejected
"""

import html
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.training.pseudo_label.config import CLASSES
from app.training.pseudo_label.fusion import ACCEPTED, REJECTED, UNCERTAIN
from app.training.review_store import label_sha256

MARKER = "PSEUDO_LABELS_NOT_GROUND_TRUTH.txt"
MANIFEST = "pseudo_manifest.json"
LABEL_SOURCE = "pseudo"
DISCLAIMER = ("Pseudo-labels are generated automatically by Grounding DINO + CLIP (+ optional folder prior and "
              "MobileSAM). They are NOT ground truth. Agreement between models and consistency with folder names are "
              "not accuracy, precision, recall or mAP. No human has verified these labels.")
COLORS_BGR = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245), (180, 30, 145),
              (244, 212, 66)]

REASON_CODES = [("too small", "too_small"), ("background", "background"), ("detector confidence", "low_detector_conf"),
                ("no CLIP support", "no_clip_support"), ("final confidence", "low_final_conf"),
                ("margin", "low_margin"), ("class changes", "flip_unstable"), ("duplicate", "duplicate"),
                ("conflicts", "conflict"), ("part of", "part_of_larger_object")]


def reason_code(reason: str | None) -> str | None:
    if reason is None:
        return None
    return next((code for prefix, code in REASON_CODES if reason.startswith(prefix)), "other")


def write_marker(out: Path) -> None:
    (out / MARKER).write_text(DISCLAIMER + "\n", encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def write_labels(out: Path, record: dict) -> str | None:
    """Write the label file of a trainable image; returns its sha256 (None for excluded images)."""
    if not record["trainable"]:
        return None
    text = "\n".join(record["yolo_lines"]) + "\n"
    (out / "labels").mkdir(exist_ok=True)
    (out / "labels" / f"{record['key']}.txt").write_text(text, encoding="utf-8")
    return label_sha256(text)


def build_manifest(run: str, cfg, cfg_hash: str, source: str, records: list[dict], hashes: dict[str, str]) -> dict:
    return {
        "label_source": LABEL_SOURCE,
        "ground_truth": False,
        "disclaimer": DISCLAIMER,
        "run": run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg_hash,
        "detection_source": source,
        "models": {"detector": cfg.gdino_model, "classifier": cfg.clip_model,
                   "box_refiner": "mobile_sam" if cfg.use_sam else None},
        "folder_prior_boost": cfg.folder_prior_boost,
        "images": {r["key"]: {"file": r["file"], "original_path": r["original_path"], "trainable": r["trainable"],
                              "exclusion_reason": r["exclusion_reason"],
                              "accepted_boxes": sum(b["tier"] == ACCEPTED for b in r["boxes"]),
                              "label_sha256": hashes.get(r["key"])} for r in records},
    }


def _hist(values: list[float], edges: list[float]) -> dict[str, int]:
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        out[f"{lo:.1f}-{hi:.1f}"] = sum(lo <= v < hi or (hi == edges[-1] and v == hi) for v in values)
    return out


def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def summarize(records: list[dict], meta: dict) -> dict:
    boxes = [b for r in records for b in r["boxes"]]
    by_tier = Counter(b["tier"] for b in boxes)
    per_class = {}
    for c in CLASSES:
        cb = [b for b in boxes if b["final_class"] == c]
        acc = [b for b in cb if b["tier"] == ACCEPTED]
        per_class[c] = {
            "accepted": len(acc), "uncertain": sum(b["tier"] == UNCERTAIN for b in cb),
            "rejected": sum(b["tier"] == REJECTED for b in cb),
            "accepted_in_training_images": sum(b["tier"] == ACCEPTED for r in records if r["trainable"]
                                               for b in r["boxes"] if b["final_class"] == c),
            "training_images_with_class": sum(any(b["tier"] == ACCEPTED and b["final_class"] == c for b in r["boxes"])
                                              for r in records if r["trainable"]),
            "final_confidence_accepted": _hist([b["final_confidence"] for b in acc], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "median_final_confidence_accepted": round(statistics.median([b["final_confidence"] for b in acc]), 4)
            if acc else None,
            "median_detector_confidence_accepted": round(statistics.median([b["gdino_confidence"] for b in acc]), 4)
            if acc else None,
            "uncertain_reasons": dict(Counter(reason_code(b["reason"]) for b in cb if b["tier"] == UNCERTAIN)),
            "rejected_reasons": dict(Counter(reason_code(b["reason"]) for b in cb if b["tier"] == REJECTED)),
        }
    acc = [b for b in boxes if b["tier"] == ACCEPTED]
    decided = [b for b in boxes if b["tier"] != REJECTED]
    folder_rows = defaultdict(Counter)
    folder_images = Counter()
    folder_consistent = Counter()
    for r in records:
        f = r["folder_class"] or "(none)"
        folder_images[f] += 1
        classes = {b["final_class"] for b in r["boxes"] if b["tier"] == ACCEPTED}
        folder_rows[f].update(classes)
        if r["trainable"] and r["folder_class"] in classes:
            folder_consistent[f] += 1
    consistency = {
        "note": "Consistency between automatic signals; NOT accuracy.",
        "accepted_where_clip_top1_is_final": _share(sum(b["clip_top1"] == b["final_class"] for b in acc), len(acc)),
        "accepted_where_detector_class_is_final": _share(sum(b["gdino_top1"] == b["final_class"] for b in acc), len(acc)),
        "accepted_where_folder_prior_was_decisive": _share(sum(b["folder_prior_decisive"] for b in acc), len(acc)),
        "accepted_where_prototype_top1_is_final": _share(sum(b.get("proto_top1") == b["final_class"] for b in acc),
                                                         len(acc)),
        "flip_stable_share_of_non_rejected": _share(sum(bool(b.get("flip_stable")) for b in decided), len(decided)),
        "alt_prompt_stable_share_of_non_rejected": _share(sum(bool(b.get("alt_prompt_stable")) for b in decided),
                                                          len(decided)),
        "alt_prompt_stable_share_of_accepted": _share(sum(bool(b.get("alt_prompt_stable")) for b in acc), len(acc)),
        "folder_class_among_labels_of_training_images": {
            f: {"training_images_with_folder_class": folder_consistent[f], "images": folder_images[f]}
            for f in folder_images},
        "folder_to_accepted_class_images": {f: dict(folder_rows[f]) for f in folder_rows},
    }
    trainable = [r for r in records if r["trainable"]]
    return {
        "disclaimer": DISCLAIMER,
        **meta,
        "images": len(records),
        "training_images": len(trainable),
        "excluded_images": len(records) - len(trainable),
        "exclusion_reasons": dict(Counter((r["exclusion_reason"] or "").split(" (")[0].split(" box ")[0]
                                          for r in records if not r["trainable"])),
        "boxes": len(boxes),
        "boxes_by_tier": {t: by_tier[t] for t in (ACCEPTED, UNCERTAIN, REJECTED)},
        "training_instances": sum(len(r["yolo_lines"]) for r in trainable),
        "sam_refined_boxes": sum(bool(b.get("sam_used")) for b in acc),
        "per_class": per_class,
        "consistency": consistency,
        "mean_seconds_per_image": round(statistics.mean(r["timing_s"] for r in records), 3) if records else None,
    }


def report_markdown(s: dict) -> str:
    L = [f"# Pseudo-label report: `{s['run']}`", "", f"> {s['disclaimer']}", "",
         f"Created {s['created_at']} · config hash `{s['config_hash']}` · detections: {s['detection_source']} · "
         f"CLIP scores: {s.get('scores', 'clip:live')} · "
         f"folder prior boost {s['folder_prior_boost']} · visual prototypes "
         f"{'on, anchors ' + str(s.get('prototype_anchors')) if s.get('use_prototypes') else 'off'} · "
         f"MobileSAM {'on' if s['use_sam'] else 'off'} · "
         f"{s['elapsed_s']} s ({s['mean_seconds_per_image']} s/image)", "",
         "## Totals", "",
         "| | Count |", "|---|---|",
         f"| Images | {s['images']} |",
         f"| Images usable for training | {s['training_images']} |",
         f"| Excluded images | {s['excluded_images']} |",
         f"| Candidate boxes | {s['boxes']} |"]
    L += [f"| {t} boxes | {n} |" for t, n in s["boxes_by_tier"].items()]
    L += [f"| Label instances in training images | {s['training_instances']} |",
          f"| Accepted boxes refined by MobileSAM | {s['sam_refined_boxes']} |", "",
          "Exclusion reasons: " + (", ".join(f"{k or 'n/a'} {v}" for k, v in s["exclusion_reasons"].items()) or "none"),
          "", "## Per class (by final class)", "",
          "| Class | Accepted | Uncertain | Rejected | Instances in training images | Training images with class | "
          "Median final conf. (accepted) | Median detector conf. (accepted) |",
          "|---|---|---|---|---|---|---|---|"]
    for c, p in s["per_class"].items():
        L.append(f"| {c} | {p['accepted']} | {p['uncertain']} | {p['rejected']} | {p['accepted_in_training_images']} | "
                 f"{p['training_images_with_class']} | {p['median_final_confidence_accepted'] or '—'} | "
                 f"{p['median_detector_confidence_accepted'] or '—'} |")
    L += ["", "### Final-confidence distribution of accepted boxes", "",
          "| Class | " + " | ".join(next(iter(s["per_class"].values()))["final_confidence_accepted"]) + " |",
          "|---|" + "---|" * 6]
    for c, p in s["per_class"].items():
        L.append(f"| {c} | " + " | ".join(str(v) for v in p["final_confidence_accepted"].values()) + " |")
    L += ["", "### Why boxes were uncertain or rejected", "", "| Class | Uncertain | Rejected |", "|---|---|---|"]
    for c, p in s["per_class"].items():
        fmt = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "—"  # noqa: E731
        L.append(f"| {c} | {fmt(p['uncertain_reasons'])} | {fmt(p['rejected_reasons'])} |")
    cs = s["consistency"]
    L += ["", "## Consistency (NOT accuracy)", "", f"_{cs['note']}_", "",
          "| Measure | Share |", "|---|---|",
          f"| Accepted boxes where CLIP's top-1 is the final class | {cs['accepted_where_clip_top1_is_final']} |",
          f"| Accepted boxes where Grounding DINO's class is the final class | {cs['accepted_where_detector_class_is_final']} |",
          f"| Accepted boxes where the visual prototype's top-1 is the final class | {cs['accepted_where_prototype_top1_is_final']} |",
          f"| Accepted boxes where the folder prior changed the class | {cs['accepted_where_folder_prior_was_decisive']} |",
          f"| Non-rejected boxes whose class survives a mirrored crop | {cs['flip_stable_share_of_non_rejected']} |",
          f"| Non-rejected boxes whose class survives the alternative prompt set | {cs['alt_prompt_stable_share_of_non_rejected']} |",
          f"| Accepted boxes whose class survives the alternative prompt set | {cs['alt_prompt_stable_share_of_accepted']} |",
          "", "### Folder → accepted classes (images containing each class)", "",
          "| Folder | Images | Training images with the folder's class | " + " | ".join(CLASSES) + " |",
          "|---|---|---|" + "---|" * len(CLASSES)]
    for f, row in cs["folder_to_accepted_class_images"].items():
        fc = cs["folder_class_among_labels_of_training_images"][f]
        L.append(f"| {f} | {fc['images']} | {fc['training_images_with_folder_class']} | "
                 + " | ".join(str(row.get(c, 0)) for c in CLASSES) + " |")
    L += ["", "Folders are image-level collection labels, not object-level ground truth; an image may legitimately "
          "contain several classes.", ""]
    return "\n".join(L)


def _dashed(img, p1, p2, color, t, dash=10):
    (x1, y1), (x2, y2) = p1, p2
    for a, b, fixed, horiz in ((x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)):
        for s in range(a, b, dash * 2):
            e = min(s + dash, b)
            cv2.line(img, (s, fixed) if horiz else (fixed, s), (e, fixed) if horiz else (fixed, e), color, t)


def draw_preview(image: np.ndarray, record: dict, max_side: int) -> np.ndarray:
    x0, y0, x1, y1 = record["photo_box"]
    img = image[y0:y1, x0:x1].copy()
    scale = min(1.0, max_side / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    order = {REJECTED: 0, UNCERTAIN: 1, ACCEPTED: 2}
    for b in sorted(record["boxes"], key=lambda b: order[b["tier"]]):
        bx = b["label_xyxy"]
        p1 = (int((bx[0] - x0) * scale), int((bx[1] - y0) * scale))
        p2 = (int((bx[2] - x0) * scale), int((bx[3] - y0) * scale))
        color = COLORS_BGR[CLASSES.index(b["final_class"])]
        if b["tier"] == REJECTED:
            cv2.rectangle(img, p1, p2, (150, 150, 150), 1)
            continue
        if b["tier"] == ACCEPTED:
            cv2.rectangle(img, p1, p2, color, 3)
        else:
            _dashed(img, p1, p2, color, 2)
        text = f"{b['id']} {b['final_class']} {b['final_confidence']:.2f}" + ("" if b["tier"] == ACCEPTED else " ?")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(img, text, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def write_preview(out: Path, image: np.ndarray, record: dict, max_side: int) -> str:
    rel = f"previews/{record['key']}.jpg"
    (out / "previews").mkdir(exist_ok=True)
    ok, buf = cv2.imencode(".jpg", draw_preview(image, record, max_side), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        (out / rel).write_bytes(buf.tobytes())
    return rel


GALLERY = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pseudo-labels: {run}</title><style>
:root{{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#d9dde3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111318;--card:#1a1d24;--ink:#e8eaee;--mute:#9aa1ad;--line:#2c313b}}}}
body{{margin:0;padding:16px;font:14px system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
.warn{{border:1px solid #c98a00;background:#c98a0018;padding:8px 12px;border-radius:6px;max-width:1100px}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}} select{{padding:4px 6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block;background:#000}} .meta{{padding:6px 8px;color:var(--mute);font-size:12px}}
.meta b{{color:var(--ink)}} .ex{{color:#c0392b}}
</style></head><body><h1>Pseudo-labels: {run}</h1><p class="warn">{disclaimer}<br>Solid box = accepted (written as a label) ·
dashed = uncertain (not written) · thin grey = rejected. Number = final fused confidence.</p>
<div class="bar"><label>Folder <select id="f"><option value="">all</option>{folders}</select></label>
<label>Image <select id="t"><option value="">all</option><option value="1">training</option><option value="0">excluded</option></select></label>
<label>Contains accepted <select id="c"><option value="">any</option>{classes}</select></label> <span id="n"></span></div>
<div class="grid" id="g">{cards}</div><script>
const q=s=>document.querySelector(s),cards=[...document.querySelectorAll('.card')];
function apply(){{const f=q('#f').value,t=q('#t').value,c=q('#c').value;let n=0;for(const e of cards){{
const ok=(!f||e.dataset.folder===f)&&(!t||e.dataset.train===t)&&(!c||e.dataset.classes.split(' ').includes(c));
e.style.display=ok?'':'none';n+=ok}}q('#n').textContent=n+' images'}}
for(const s of ['#f','#t','#c'])q(s).onchange=apply;apply();</script></body></html>"""


def write_gallery(out: Path, run: str, records: list[dict]) -> None:
    folders = sorted({r["folder_class"] or "(none)" for r in records})
    cards = []
    for r in records:
        acc = sorted({b["final_class"] for b in r["boxes"] if b["tier"] == ACCEPTED})
        tiers = Counter(b["tier"] for b in r["boxes"])
        status = "training" if r["trainable"] else f"<span class=ex>excluded: {html.escape(r['exclusion_reason'] or '')}</span>"
        cards.append(
            f'<div class="card" data-folder="{html.escape(r["folder_class"] or "(none)")}" '
            f'data-train="{int(r["trainable"])}" data-classes="{" ".join(acc)}">'
            f'<img loading="lazy" src="{r.get("preview", "")}" alt="{html.escape(r["file"])}">'
            f'<div class="meta"><b>{html.escape(r["file"])}</b> · folder {html.escape(r["folder_class"] or "—")}<br>'
            f'{status}<br>accepted {tiers[ACCEPTED]} · uncertain {tiers[UNCERTAIN]} · rejected {tiers[REJECTED]}'
            f'<br>labels: {html.escape(", ".join(acc) or "none")}</div></div>')
    (out / "gallery.html").write_text(GALLERY.format(
        run=html.escape(run), disclaimer=html.escape(DISCLAIMER),
        folders="".join(f"<option>{html.escape(f)}</option>" for f in folders),
        classes="".join(f"<option>{c}</option>" for c in CLASSES), cards="\n".join(cards)), encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
