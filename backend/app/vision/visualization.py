"""Draw detections onto frames with stable per-track / per-class colours."""

import base64
import colorsys

import cv2
import numpy as np

from app.schemas.detection import Detection


def color_for(key: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour (golden-ratio hue stepping)."""
    hue = (key * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def label_for(det: Detection) -> str:
    prefix = f"#{det.track_id} " if det.track_id is not None else ""
    return f"{prefix}{det.class_name} {det.confidence * 100:.0f}%"


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return an annotated copy of the frame; the original is left untouched."""
    out = frame.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(h, w) / 400))
    font_scale = max(0.4, min(h, w) / 1200)
    for det in detections:
        color = color_for(det.track_id if det.track_id is not None else det.class_id)
        x1, y1, x2, y2 = (int(round(v)) for v in (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness + 1, cv2.LINE_AA)

        label = label_for(det)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y1 - th - baseline - 4 if y1 - th - baseline - 4 >= 0 else y1  # keep label inside the frame
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + baseline + 4), color, -1)
        cv2.putText(out, label, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return out


def encode_jpeg_base64(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
