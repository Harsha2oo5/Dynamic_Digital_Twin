"""Model management: versioned checkpoints with model cards, runtime model switching, safe fallback.

- Lists every *.pt under the models directory with its model card (``<stem>.model_card.json`` written by
  ``train --promote-versioned``). The pretrained COCO checkpoint gets a built-in card (label source ``coco``).
- ``select()`` loads the new checkpoint in the calling (worker) thread and swaps it in only when it loaded and is a
  detection model; on failure the running model stays active and the error is returned. No restart is needed.
  The selection is runtime state: after a restart DETECTION_MODEL / MODEL_PATH decide again.
- Backends: anything with the Detector interface (load / predict / names / loaded) can be plugged in; only
  Ultralytics .pt checkpoints are implemented. ONNX / OpenVINO export is not part of this milestone.
"""

import json
import logging
import threading
from pathlib import Path

from app.identification.component_classifier import ComponentClassifier
from app.schemas.detection import ModelEntry
from app.vision.detector import Detector, infer_model_type

logger = logging.getLogger(__name__)

PRETRAINED_CARD = {
    "label_source": "coco",
    "ground_truth": None,
    "architecture": "yolo11n",
    "metrics_meaning": "Ultralytics COCO checkpoint: generic objects, not BTS equipment classes",
    "limitations": ["Detects the 80 COCO classes; it does not know the seven BTS component classes."],
}


class ModelSelectionError(RuntimeError):
    pass


def model_card_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".model_card.json")


def read_model_card(checkpoint: Path) -> dict | None:
    p = model_card_path(checkpoint)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Unreadable model card %s", p)
    if infer_model_type(checkpoint) == "pretrained":
        return dict(PRETRAINED_CARD)
    return None


class ModelManager:
    def __init__(self, service):
        self.service = service
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def models_dir(self) -> Path:
        return self.service.settings.resolve(self.service.settings.models_dir)

    def entries(self) -> list[ModelEntry]:
        root = self.models_dir
        if not root.is_dir():
            return []
        active = self.service.model_path.resolve()
        out = []
        for p in sorted(root.rglob("*.pt")):
            card = read_model_card(p)
            out.append(ModelEntry(
                model_id=p.relative_to(root).as_posix(),
                name=p.name,
                model_type=infer_model_type(p),
                version=(card or {}).get("version"),
                label_source=(card or {}).get("label_source") or "unknown",
                ground_truth=(card or {}).get("ground_truth"),
                size_mb=round(p.stat().st_size / 1e6, 2),
                active=p.resolve() == active,
                has_model_card=model_card_path(p).is_file(),
                model_card=card,
            ))
        return out

    def resolve_id(self, model_id: str) -> Path:
        root = self.models_dir.resolve()
        path = (root / model_id).resolve()
        if not path.is_relative_to(root) or path.suffix != ".pt":
            raise ModelSelectionError(f"Invalid model id {model_id!r}")
        if not path.is_file():
            raise ModelSelectionError(f"Model not found: {model_id}")
        return path

    def select(self, model_id: str) -> ModelEntry:
        """Blocking (loads and warms up the model): call from a worker thread."""
        path = self.resolve_id(model_id)
        s = self.service.settings
        with self._lock:
            detector = Detector(path, self.service.device, s.max_frame_dim, s.max_detections)
            if not detector.load(warmup=s.warmup, imgsz=s.image_size):
                self.last_error = detector.error
                raise ModelSelectionError(f"{detector.error}; the current model stays active")
            model_type = infer_model_type(path)
            svc = self.service
            svc.classifier = ComponentClassifier(model_type, detector.names)
            svc.detector, svc.model_path, svc.model_type = detector, path, model_type
            svc.fallback_reason = None
            self.last_error = None
            logger.info("Switched detection model to %s (%s)", path.name, model_type)
        return next(e for e in self.entries() if e.model_id == model_id)

    def active_card(self) -> dict | None:
        return read_model_card(self.service.model_path)

    def performance(self) -> dict:
        """Measured runtime performance of the active model plus the offline metrics from its model card."""
        stats = self.service.detection_stats()
        card = self.active_card() or {}
        return {
            "model": self.service.model_path.name,
            "model_type": self.service.model_type,
            "label_source": card.get("label_source", "unknown"),
            "runtime": {"frames_processed": stats.frames_processed, "avg_inference_ms": stats.avg_inference_ms,
                        "avg_inference_fps": stats.avg_inference_fps, "min_inference_ms": stats.min_inference_ms,
                        "max_inference_ms": stats.max_inference_ms, "avg_confidence": stats.avg_confidence,
                        "class_counts": [c.model_dump() for c in stats.class_counts]},
            "offline_metrics": card.get("validation_metrics"),
            "metrics_meaning": card.get("metrics_meaning"),
            "limitations": card.get("limitations", []),
        }
