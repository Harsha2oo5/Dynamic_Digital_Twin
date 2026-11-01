"""YOLO object detector (CPU).

One Detector instance (and therefore one model) is created at startup and shared. Ultralytics
predictors are not thread-safe, so every predict() call is serialised by a lock; callers run
it in a worker thread to keep the event loop free.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.device import resolve_device
from app.schemas.detection import BBox
from app.vision.preprocessing import ensure_bgr, resize_max_dim

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass
class RawDetections:
    """Detector output in original-frame pixels. Arrays are aligned by index."""

    xyxy: np.ndarray  # (N, 4) float32
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int
    frame_width: int
    frame_height: int
    inference_ms: float

    def __len__(self) -> int:
        return len(self.conf)


def scale_boxes_to_original(xyxy: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    """Map boxes from a resized frame back to original pixels and clip them to the frame."""
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    if scale != 1.0:
        boxes = boxes / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def to_bbox(row: np.ndarray) -> BBox:
    return BBox(x1=round(float(row[0]), 1), y1=round(float(row[1]), 1), x2=round(float(row[2]), 1), y2=round(float(row[3]), 1))


def infer_model_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "custom" in parts:
        return "custom"
    if "pretrained" in parts:
        return "pretrained"
    return "unknown"


class Detector:
    def __init__(self, model_path: Path, device: str = "cpu", max_frame_dim: int = 1920, max_det: int = 100):
        self.model_path = Path(model_path)
        self.device = resolve_device(device)
        self.max_frame_dim = max_frame_dim
        self.max_det = max_det
        self.model = None
        self.error: str | None = None
        self.load_time_s: float | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def names(self) -> dict[int, str]:
        return dict(self.model.names) if self.model is not None else {}

    @property
    def task(self) -> str | None:
        return getattr(self.model, "task", None) if self.model is not None else None

    def load(self, warmup: bool = True, imgsz: int = 416) -> bool:
        """Load the checkpoint. Returns False (and records the error) instead of raising."""
        if not self.model_path.is_file():
            self.error = f"Model checkpoint not found: {self.model_path}"
            logger.error(self.error)
            return False
        try:
            from ultralytics import YOLO

            start = time.perf_counter()
            model = YOLO(str(self.model_path))
            if getattr(model, "task", "detect") != "detect":
                raise ValueError(f"Checkpoint task is {model.task!r}; a detection model is required")
            model.to(self.device)
            self.model = model
            if warmup:
                self.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), conf=0.5, iou=0.45, imgsz=imgsz)
            self.load_time_s = round(time.perf_counter() - start, 2)
            self.error = None
            logger.info(
                "Loaded %s on %s in %.2fs (%d classes)", self.model_path.name, self.device, self.load_time_s, len(self.names)
            )
            return True
        except Exception as exc:  # noqa: BLE001 - surface any load failure through /api/model/info
            self.model = None
            self.error = f"Failed to load model: {exc}"
            logger.exception("Model load failed")
            return False

    def predict(self, frame: np.ndarray, conf: float, iou: float, imgsz: int) -> RawDetections:
        if self.model is None:
            raise ModelNotLoadedError(self.error or "Model is not loaded")
        frame = ensure_bgr(frame)
        orig_h, orig_w = frame.shape[:2]
        small, scale = resize_max_dim(frame, self.max_frame_dim)

        with self._lock:
            start = time.perf_counter()
            result = self.model.predict(
                source=small, device=self.device, conf=conf, iou=iou, imgsz=imgsz, max_det=self.max_det, verbose=False
            )[0]
            inference_ms = (time.perf_counter() - start) * 1000

        boxes = result.boxes.cpu().numpy()
        return RawDetections(
            xyxy=scale_boxes_to_original(boxes.xyxy, scale, orig_w, orig_h),
            conf=boxes.conf.astype(np.float32),
            cls=boxes.cls.astype(int),
            frame_width=orig_w,
            frame_height=orig_h,
            inference_ms=inference_ms,
        )
