"""Weak-label diagnostic metrics (pure functions; no models).

Unit definitions used throughout:
  primary box   the detection with the highest Grounding DINO confidence in an image. In these mostly
                single-product photos it is usually the whole product, so it is the fairest single box to
                compare with the image's folder.
  any box       the folder class appears for at least one detection of the image (more boxes = more chances,
                so this is an optimistic upper view).
  box level     share of all boxes in a folder's images whose top-1 is the folder class (pessimistic: parts and
                other components count against it).
A "ranking" maps each detection key (image index, detection id) to its classes ordered by similarity."""

import statistics
from collections import Counter

from app.training.taxonomy import CLASS_NAMES

CLASSES = list(CLASS_NAMES.values())
SIZE_BUCKETS = [(0.0, 0.05, "tiny <5%"), (0.05, 0.20, "small 5-20%"), (0.20, 0.60, "medium 20-60%"),
                (0.60, 1.01, "large >60%")]


def ranking_from_sims(sims: dict[str, float]) -> list[str]:
    return [c for c, _ in sorted(sims.items(), key=lambda kv: -kv[1])]


def stored_rankings(records: list[dict]) -> dict[tuple[int, int], list[str]]:
    """Rankings from the run's stored CLIP similarities (all seven classes per detection)."""
    return {(i, d["id"]): ranking_from_sims(d["clip_all"]) for i, r in enumerate(records) for d in r["detections"]
            if d.get("clip_all")}


def primary(rec: dict) -> dict | None:
    return max(rec["detections"], key=lambda d: d["gdino_confidence"]) if rec["detections"] else None


def photo_area(rec: dict) -> float:
    x0, y0, x1, y1 = rec["photo_box"]
    return float(max(1, (x1 - x0) * (y1 - y0)))


def box_fraction(rec: dict, det: dict) -> float:
    x1, y1, x2, y2 = det["xyxy"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / photo_area(rec)


def size_bucket(frac: float) -> str:
    return next(name for lo, hi, name in SIZE_BUCKETS if lo <= frac < hi)


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def weak_metrics(records: list[dict], ranks: dict[tuple[int, int], list[str]]) -> dict:
    """Per folder class and overall (micro over images, macro over classes)."""
    per = {c: Counter() for c in CLASSES}
    for i, r in enumerate(records):
        cls = r["folder_class"]
        if cls not in per:
            continue
        c = per[cls]
        c["images"] += 1
        p = primary(r)
        if p is None:
            c["no_detection"] += 1
            continue
        pr = ranks.get((i, p["id"]))
        if pr:
            c["primary_top1"] += pr[0] == cls
            c["primary_top2"] += cls in pr[:2]
            c["primary_top3"] += cls in pr[:3]
        c["gdino_primary"] += p["gdino_class"] == cls
        boxes = [ranks.get((i, d["id"])) for d in r["detections"]]
        c["any_top1"] += any(b and b[0] == cls for b in boxes)
        c["any_top3"] += any(b and cls in b[:3] for b in boxes)
        c["gdino_any"] += any(d["gdino_class"] == cls for d in r["detections"])
        c["boxes"] += len(r["detections"])
        c["boxes_top1"] += sum(bool(b) and b[0] == cls for b in boxes)
    keys = ["primary_top1", "primary_top2", "primary_top3", "any_top1", "any_top3", "gdino_primary", "gdino_any"]
    out = {"per_class": {}, "overall": {}}
    for cls, c in per.items():
        out["per_class"][cls] = {"images": c["images"], **{k: _rate(c[k], c["images"]) for k in keys},
                                 "box_level_top1": _rate(c["boxes_top1"], c["boxes"]), "boxes": c["boxes"]}
    total = sum(c["images"] for c in per.values())
    for k in keys:
        out["overall"][f"{k}_micro"] = _rate(sum(c[k] for c in per.values()), total)
        vals = [v[k] for v in out["per_class"].values() if v[k] is not None]
        out["overall"][f"{k}_macro"] = round(sum(vals) / len(vals), 4) if vals else None
    out["overall"]["images"] = total
    return out


def confusion(records: list[dict], ranks: dict, use: str = "clip") -> dict[str, Counter]:
    """Folder class -> top-1 of the primary box (CLIP ranking, or Grounding DINO prompt class)."""
    table = {c: Counter() for c in CLASSES}
    for i, r in enumerate(records):
        p = primary(r)
        if r["folder_class"] not in table:
            continue
        if p is None:
            table[r["folder_class"]]["(no detection)"] += 1
        elif use == "clip":
            table[r["folder_class"]][ranks[(i, p["id"])][0]] += 1
        else:
            table[r["folder_class"]][p["gdino_class"]] += 1
    return table


def prediction_profile(table: dict[str, Counter]) -> dict:
    """Column view of a confusion table: how often each class is *predicted* for primary boxes, and which share of
    those predictions come from that class's own folder ("folder purity"). A class predicted for everything can
    reach a high folder-match rate while its purity stays low, which reveals a bias rather than recognition."""
    out = {}
    for c in CLASSES:
        predicted = sum(row.get(c, 0) for row in table.values())
        out[c] = {"predicted": predicted, "folder_purity": _rate(table[c].get(c, 0), predicted)}
    return out


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    q = lambda p: round(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))], 4)  # noqa: E731
    return {"n": len(v), "min": round(v[0], 4), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": round(v[-1], 4),
            "mean": round(statistics.mean(v), 4)}


def distributions(records: list[dict]) -> dict:
    """Stored CLIP top-1 similarity, top-1/top-2 margin and Grounding DINO confidence, per folder class."""
    out = {}
    for cls in CLASSES:
        dets = [d for r in records if r["folder_class"] == cls for d in r["detections"] if d.get("clip_all")]
        sims = [sorted(d["clip_all"].values(), reverse=True) for d in dets]
        out[cls] = {"clip_top1_similarity": quantiles([s[0] for s in sims]),
                    "clip_margin_top1_top2": quantiles([s[0] - s[1] for s in sims]),
                    "gdino_confidence": quantiles([d["gdino_confidence"] for d in dets]),
                    "near_tie_share_margin_lt_0.005": _rate(sum(s[0] - s[1] < 0.005 for s in sims), len(sims))}
    return out


def size_analysis(records: list[dict], ranks: dict) -> dict:
    """Box size relative to the photo vs. how often the box's CLIP top-1 equals the folder class."""
    buckets = {name: Counter() for *_, name in SIZE_BUCKETS}
    prim = Counter()
    for i, r in enumerate(records):
        p = primary(r)
        for d in r["detections"]:
            b = buckets[size_bucket(box_fraction(r, d))]
            b["boxes"] += 1
            b["top1_is_folder"] += ranks[(i, d["id"])][0] == r["folder_class"]
            b["unknown"] += d.get("clip_label") == "UNKNOWN"
        if p is not None:
            prim[size_bucket(box_fraction(r, p))] += 1
    return {"by_size": {k: {"boxes": v["boxes"], "top1_is_folder": _rate(v["top1_is_folder"], v["boxes"]),
                            "clip_unknown": _rate(v["unknown"], v["boxes"])} for k, v in buckets.items()},
            "primary_box_size": dict(prim)}


def rank_change_rate(a: dict, b: dict) -> float | None:
    """Share of detections whose top-1 differs between two rankings (prompt sensitivity)."""
    keys = a.keys() & b.keys()
    return _rate(sum(a[k][0] != b[k][0] for k in keys), len(keys))
