"""Image-level rules applied after the per-box decisions (pure functions).

1. duplicates   two ACCEPTED boxes of the same class on one object -> the weaker is REJECTED
2. conflicts    two ACCEPTED boxes of different classes on one object -> the weaker is REJECTED
3. parts        a box mostly inside a larger ACCEPTED or UNCERTAIN non-container box (a handle, rack ear, label or fan
                module of a radio unit, ...) is REJECTED: an uncertain box is still an object, only its class is
                unknown. Cabinets may contain other components, and
                battery modules / rectifiers may sit inside a battery rack / PSU shelf (annotation rules: one box
                per battery module, per PSU shelf, nested components boxed separately)
4. training     an image is used for training only if it has at least one ACCEPTED box and no large UNCERTAIN box
                that no ACCEPTED box covers: YOLO would otherwise learn that real equipment is background
"""

from app.training.pseudo_label.config import PseudoLabelConfig
from app.training.pseudo_label.fusion import ACCEPTED, REJECTED, UNCERTAIN
from app.training.pseudo_label.geometry import area, containment, iou


def _reject(box: dict, reason: str) -> None:
    box["tier_before_suppression"] = box["tier"]
    box["tier"], box["reason"] = REJECTED, reason


def suppress(boxes: list[dict], cfg: PseudoLabelConfig) -> list[dict]:
    """Mutates and returns `boxes` (dicts with id, xyxy, tier, final_class, final_confidence)."""
    accepted = sorted((b for b in boxes if b["tier"] == ACCEPTED), key=lambda b: -b["final_confidence"])
    kept: list[dict] = []
    for b in accepted:
        clash = None
        for k in kept:
            o = iou(b["xyxy"], k["xyxy"])
            if k["final_class"] == b["final_class"] and o > cfg.duplicate_iou:
                clash = f"duplicate of box {k['id']} (IoU {o:.2f})"
            elif k["final_class"] != b["final_class"] and o > cfg.conflict_iou:
                clash = f"conflicts with box {k['id']} ({k['final_class']}, IoU {o:.2f})"
            if clash:
                break
        if clash:
            _reject(b, clash)
        else:
            kept.append(b)

    containers = [k for k in boxes if k["tier"] in (ACCEPTED, UNCERTAIN) and k["final_class"] not in cfg.container_classes]
    for b in boxes:
        if b["tier"] == REJECTED or b["final_class"] in cfg.allowed_inside_any:
            continue
        for k in containers:
            if k is b or area(b["xyxy"]) > cfg.part_max_area_ratio * area(k["xyxy"]):
                continue
            if b["final_class"] == k["final_class"] and b["final_class"] in cfg.nestable_classes:
                continue  # a battery module in a battery rack, a rectifier in a PSU shelf
            c = containment(b["xyxy"], k["xyxy"])
            if c >= cfg.part_containment:
                _reject(b, f"part of box {k['id']} ({k['final_class']}, {c:.0%} inside)")
                break
    return boxes


def image_decision(boxes: list[dict], cfg: PseudoLabelConfig) -> tuple[bool, str | None]:
    accepted = [b for b in boxes if b["tier"] == ACCEPTED]
    if not accepted:
        return False, "no accepted box"
    for b in boxes:
        if b["tier"] != UNCERTAIN or b["area_share"] < cfg.blocking_uncertain_area:
            continue
        if not any(iou(b["xyxy"], a["xyxy"]) >= cfg.covered_iou for a in accepted):
            return False, f"uncovered uncertain box {b['id']} ({b['area_share']:.0%} of the photo: {b['reason']})"
    return True, None
