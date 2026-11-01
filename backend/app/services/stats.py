"""Aggregated detection statistics. Every number is derived from measured inference results."""

import threading
from collections import Counter, deque

from app.schemas.detection import ClassCount, FrameResult, PerfSample


class StatsCollector:
    def __init__(self, window: int = 120):
        self._lock = threading.Lock()
        self._recent: deque[PerfSample] = deque(maxlen=window)
        self.frames_processed = 0
        self.total_detections = 0
        self._conf_sum = 0.0
        self._class_counts: Counter[str] = Counter()
        self._sources: dict[str, str] = {}  # source_id -> human-readable label

    def record(self, result: FrameResult) -> None:
        with self._lock:
            self.frames_processed += 1
            self.total_detections += len(result.detections)
            for det in result.detections:
                self._conf_sum += det.confidence
                self._class_counts[det.class_name] += 1
            self._recent.append(
                PerfSample(timestamp=result.timestamp, inference_ms=result.inference_ms, detections=len(result.detections))
            )

    def source_started(self, source_id: str, label: str) -> None:
        with self._lock:
            self._sources[source_id] = label

    def source_stopped(self, source_id: str) -> None:
        with self._lock:
            self._sources.pop(source_id, None)

    def snapshot(self) -> dict:
        with self._lock:
            recent = list(self._recent)
            avg_ms = sum(s.inference_ms for s in recent) / len(recent) if recent else None
            return {
                "frames_processed": self.frames_processed,
                "total_detections": self.total_detections,
                "avg_inference_ms": round(avg_ms, 2) if avg_ms else None,
                "avg_inference_fps": round(1000.0 / avg_ms, 2) if avg_ms else None,
                "avg_confidence": round(self._conf_sum / self.total_detections, 4) if self.total_detections else None,
                "active_sources": list(self._sources.values()),
                "class_counts": [ClassCount(class_name=k, count=v) for k, v in self._class_counts.most_common()],
                "recent": recent,
            }

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self.frames_processed = 0
            self.total_detections = 0
            self._conf_sum = 0.0
            self._class_counts.clear()
