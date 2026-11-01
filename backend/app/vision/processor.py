"""Per-frame pipeline: preprocess -> detect -> (track) -> classify -> FrameResult."""

from datetime import datetime

import numpy as np

from app.identification.component_classifier import ComponentClassifier
from app.schemas.detection import Detection, FrameResult
from app.vision.detector import Detector, RawDetections, to_bbox
from app.vision.tracker import ObjectTracker


def build_detections(
    raw: RawDetections,
    names: dict[int, str],
    track_ids: np.ndarray | None = None,
    classifier: ComponentClassifier | None = None,
) -> list[Detection]:
    detections = []
    for i in range(len(raw)):
        class_id = int(raw.cls[i])
        track_id = int(track_ids[i]) if track_ids is not None and track_ids[i] >= 0 else None
        detections.append(
            Detection(
                track_id=track_id,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=round(float(raw.conf[i]), 4),
                bbox=to_bbox(raw.xyxy[i]),
                component_type=classifier.component_type(class_id) if classifier else None,
            )
        )
    return detections


class FrameProcessor:
    """Processes frames for one session. Pass a tracker to enable tracking for that session."""

    def __init__(self, detector: Detector, classifier: ComponentClassifier | None, tracker: ObjectTracker | None = None):
        self.detector = detector
        self.classifier = classifier
        self.tracker = tracker

    def process(self, frame: np.ndarray, frame_id: int, conf: float, iou: float, imgsz: int) -> FrameResult:
        raw = self.detector.predict(frame, conf=conf, iou=iou, imgsz=imgsz)
        track_ids = self.tracker.update(raw) if self.tracker is not None else None
        detections = build_detections(raw, self.detector.names, track_ids, self.classifier)
        return FrameResult(
            frame_id=frame_id,
            timestamp=datetime.now(),
            device=self.detector.device,
            fps=round(1000.0 / raw.inference_ms, 2) if raw.inference_ms > 0 else 0.0,
            inference_ms=round(raw.inference_ms, 2),
            frame_width=raw.frame_width,
            frame_height=raw.frame_height,
            detections=detections,
        )
