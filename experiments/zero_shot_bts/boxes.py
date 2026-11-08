"""Box geometry: coordinate conversion, clipping, IoU, duplicate suppression, crop extraction."""

import numpy as np


def cxcywh_norm_to_xyxy(box, width: int, height: int) -> tuple[float, float, float, float]:
    """Grounding DINO outputs (cx, cy, w, h) normalised to the input image; convert to pixel x1, y1, x2, y2."""
    cx, cy, w, h = (float(v) for v in box)
    return ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def clip_xyxy(box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (min(max(x1, 0.0), width), min(max(y1, 0.0), height), min(max(x2, 0.0), width), min(max(y2, 0.0), height))


def offset(box, dx: float, dy: float) -> tuple[float, float, float, float]:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(items: list[dict], iou_thr: float, key: str = "gdino_confidence") -> list[dict]:
    """Class-agnostic suppression: the same object proposed under several prompts keeps only the
    highest-scoring box; suppressed prompts are recorded on the survivor as `alternatives`."""
    kept: list[dict] = []
    for it in sorted(items, key=lambda d: -d[key]):
        owner = next((k for k in kept if iou(k["xyxy"], it["xyxy"]) > iou_thr), None)
        if owner is None:
            kept.append({**it, "alternatives": list(it.get("alternatives", []))})
        else:
            owner["alternatives"].append({"prompt": it["prompt"], "gdino_class": it["gdino_class"],
                                          "gdino_confidence": round(it[key], 4)})
    return kept


def crop(image: np.ndarray, box, padding: float = 0.0) -> np.ndarray:
    """Pixel crop of `box` with `padding` (fraction of the box size) of context, clipped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1, x2, y2 = clip_xyxy((x1 - px, y1 - py, x2 + px, y2 + py), w, h)
    x1, y1, x2, y2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop for box {box}")
    return image[y1:y2, x1:x2]


def mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
