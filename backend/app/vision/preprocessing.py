"""Frame validation and resizing.

YOLO letterboxes internally (aspect-ratio preserving) to the inference size, so the only
resize done here is a cheap downscale of very large frames; the returned scale is used to map
boxes back to original-frame pixel coordinates.
"""

import cv2
import numpy as np


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Validate a frame and normalise it to 3-channel uint8 BGR (no copy when already BGR)."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Frame must be a non-empty numpy array")
    if frame.dtype != np.uint8:
        raise ValueError(f"Frame must be uint8, got {frame.dtype}")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported frame shape {frame.shape}")


def resize_max_dim(frame: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_dim, preserving aspect ratio.

    Returns (frame, scale) where scale = resized / original. Frames already within
    bounds are returned unchanged with scale 1.0.
    """
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame, 1.0
    scale = max_dim / longest
    resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale
