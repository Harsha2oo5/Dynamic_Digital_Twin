"""Box geometry in pixel xyxy coordinates (pure functions)."""

import numpy as np

Box = tuple[float, float, float, float]


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def clip_box(b, width: int, height: int) -> Box:
    x1, y1, x2, y2 = (float(v) for v in b)
    return (min(max(x1, 0.0), width), min(max(y1, 0.0), height), min(max(x2, 0.0), width), min(max(y2, 0.0), height))


def cxcywh_norm_to_xyxy(b, width: int, height: int) -> Box:
    cx, cy, w, h = (float(v) for v in b)
    return ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def offset(b, dx: float, dy: float) -> Box:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def crop(image: np.ndarray, b, padding: float = 0.0) -> np.ndarray:
    """Crop of `b` with `padding` (fraction of the box size) of context, clipped to the image."""
    h, w = image.shape[:2]
    px, py = (b[2] - b[0]) * padding, (b[3] - b[1]) * padding
    x1, y1, x2, y2 = clip_box((b[0] - px, b[1] - py, b[2] + px, b[3] + py), w, h)
    x1, y1, x2, y2 = int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop for box {b}")
    return image[y1:y2, x1:x2]


def mask_to_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def to_yolo_line(class_id: int, b, width: int, height: int) -> str:
    """YOLO 'cls cx cy w h' normalised to the full image."""
    x1, y1, x2, y2 = clip_box(b, width, height)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
