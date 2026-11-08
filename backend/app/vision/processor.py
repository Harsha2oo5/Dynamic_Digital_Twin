"""Per-frame pipeline: preprocess -> detect -> (track) -> classify -> FrameResult."""

import time
from datetime import datetime

import numpy as np

from app.identification.component_classifier import ComponentClassifier
from app.schemas.detection import Detection, FrameResult
from app.vision.detector import Detector, RawDetections, to_bbox
from app.vision.track_manager import TrackingSession
from app.vision.tracker import ObjectTracker


def build_detections(
    raw: RawDetections,
    names: dict[int, str],
    track_ids: np.ndarray | None = None,
    classifier: ComponentClassifier | None = None,
    infos: list | None = None,
) -> list[Detection]:
    detections = []
    for i in range(len(raw)):
        class_id = int(raw.cls[i])
        track_id = int(track_ids[i]) if track_ids is not None and track_ids[i] >= 0 else None
        info = infos[i] if infos is not None else None
        extra = {}
        if info is not None:
            extra = dict(track_state=info.track_state, stable_class=names.get(info.stable_class_id,
                                                                              str(info.stable_class_id)),
                         class_consistency=info.class_consistency, smoothed_confidence=info.smoothed_confidence,
                         track_age_s=info.track_age_s, equipment_id=info.equipment_id)
        detections.append(
            Detection(
                track_id=track_id,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=round(float(raw.conf[i]), 4),
                bbox=to_bbox(raw.xyxy[i]),
                component_type=classifier.component_type(class_id) if classifier else None,
                **extra,
            )
        )
    return detections


class FrameProcessor:
    """Processes frames for one session. Pass a tracker to enable tracking for that session: an ObjectTracker
    (track IDs only) or a TrackingSession (track IDs + lifecycle + session equipment identity)."""

    def __init__(self, detector: Detector, classifier: ComponentClassifier | None,
                 tracker: "ObjectTracker | TrackingSession | None" = None):
        self.detector = detector
        self.classifier = classifier
        self.tracker = tracker
        self.last_tracking_ms = 0.0

    def process(self, frame: np.ndarray, frame_id: int, conf: float, iou: float, imgsz: int,
                session_time: float | None = None) -> FrameResult:
        """`session_time`: clock for track lifecycle (video time for files); defaults to the monotonic clock."""
        raw = self.detector.predict(frame, conf=conf, iou=iou, imgsz=imgsz)
        track_ids, infos = None, None
        start = time.perf_counter()
        if isinstance(self.tracker, TrackingSession):
            now = session_time if session_time is not None else time.monotonic()
            track_ids, infos = self.tracker.step(raw, frame, now)
        elif self.tracker is not None:
            track_ids = self.tracker.update(raw)
        self.last_tracking_ms = (time.perf_counter() - start) * 1000
        detections = build_detections(raw, self.detector.names, track_ids, self.classifier, infos)
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
