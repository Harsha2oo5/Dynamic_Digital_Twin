"""Aggregated detection statistics. Every number is derived from measured inference results.

Besides the counters, the collector keeps the *activity* facts the dashboard needs to tell apart
"backend up but idle", "source connected but no frames yet", "frames being processed" and "failing":
per-source frame counts and last-frame times, the global last-frame time, and the last processing error.
"""

import threading
import time
from collections import Counter, deque
from datetime import datetime

from app.schemas.detection import ClassCount, EngineError, FrameResult, PerfSample, SourceState

ACTIVE_WINDOW_S = 5.0  # a source/engine that produced a frame within this window counts as processing
ERROR_WINDOW_S = 60.0  # an error newer than the last good frame keeps the engine in "error" this long


class StatsCollector:
    def __init__(self, window: int = 120):
        self._lock = threading.Lock()
        self.window = window
        self._recent: deque[PerfSample] = deque(maxlen=window)
        self._reset_counters()
        self._sources: dict[str, dict] = {}  # source_id -> label, type, timings

    def _reset_counters(self) -> None:
        self.session_started_at = datetime.now()
        self.frames_processed = 0
        self.total_detections = 0
        self._conf_sum = 0.0
        self._class_counts: Counter[tuple[int, str]] = Counter()
        self._last_frame_mono: float | None = None
        self._last_frame_at: datetime | None = None
        self._errors = 0
        self._last_error: dict | None = None  # message, source, at (datetime), mono, seq
        # event order: monotonic() ticks only every ~16 ms on Windows, too coarse to order a frame and an error
        self._seq = 0
        self._last_frame_seq = 0

    def record(self, result: FrameResult, source_id: str | None = None) -> None:
        now_mono = time.monotonic()
        with self._lock:
            self.frames_processed += 1
            self.total_detections += len(result.detections)
            for det in result.detections:
                self._conf_sum += det.confidence
                self._class_counts[(det.class_id, det.class_name)] += 1
            self._recent.append(PerfSample(timestamp=result.timestamp, inference_ms=result.inference_ms,
                                           detections=len(result.detections), seq=self.frames_processed))
            self._last_frame_mono, self._last_frame_at = now_mono, datetime.now()
            self._seq += 1
            self._last_frame_seq = self._seq
            src = self._sources.get(source_id) if source_id else None
            if src is not None:  # one-shot image requests are not registered sources
                src["frames"] += 1
                src["last_frame_mono"], src["last_frame_at"] = now_mono, self._last_frame_at

    def record_error(self, message: str, source_id: str | None = None) -> None:
        with self._lock:
            self._errors += 1
            self._seq += 1
            self._last_error = {"message": message, "source": source_id, "at": datetime.now(), "mono": time.monotonic(),
                                "seq": self._seq}

    def source_started(self, source_id: str, label: str) -> None:
        with self._lock:
            self._sources[source_id] = {"label": label, "type": source_id.split(":", 1)[0], "started_at": datetime.now(),
                                        "frames": 0, "last_frame_mono": None, "last_frame_at": None}

    def source_stopped(self, source_id: str) -> None:
        with self._lock:
            self._sources.pop(source_id, None)

    def engine_state(self, model_loaded: bool) -> str:
        """error | running | idle, from measured activity (never assumed)."""
        now = time.monotonic()
        with self._lock:
            if not model_loaded:
                return "error"
            err, last = self._last_error, self._last_frame_mono
            if err and err["seq"] > self._last_frame_seq and now - err["mono"] < ERROR_WINDOW_S:
                return "error"
            if last is not None and now - last < ACTIVE_WINDOW_S:
                return "running"
            return "idle"

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            recent = list(self._recent)
            times = [s.inference_ms for s in recent]
            avg_ms = sum(times) / len(times) if times else None
            sources = [SourceState(
                source_id=sid, label=s["label"], source_type=s["type"], frames=s["frames"],
                started_at=s["started_at"], last_frame_at=s["last_frame_at"],
                state="processing" if s["last_frame_mono"] is not None and now - s["last_frame_mono"] < ACTIVE_WINDOW_S
                else "connected") for sid, s in self._sources.items()]
            err = self._last_error
            return {
                "frames_processed": self.frames_processed,
                "total_detections": self.total_detections,
                "avg_inference_ms": round(avg_ms, 2) if avg_ms else None,
                "avg_inference_fps": round(1000.0 / avg_ms, 2) if avg_ms else None,
                "min_inference_ms": round(min(times), 2) if times else None,
                "max_inference_ms": round(max(times), 2) if times else None,
                "latency_window": self.window,
                "avg_confidence": round(self._conf_sum / self.total_detections, 4) if self.total_detections else None,
                "active_sources": [s.label for s in sources],
                "sources": sources,
                "class_counts": [ClassCount(class_id=cid, class_name=name, count=v)
                                 for (cid, name), v in self._class_counts.most_common()],
                "recent": recent,
                "session_started_at": self.session_started_at,
                "last_frame_at": self._last_frame_at,
                "seconds_since_last_frame": round(now - self._last_frame_mono, 1) if self._last_frame_mono else None,
                "error_count": self._errors,
                "last_error": EngineError(message=err["message"], source=err["source"], at=err["at"]) if err else None,
            }

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._reset_counters()
            for s in self._sources.values():
                s["frames"], s["last_frame_mono"], s["last_frame_at"] = 0, None, None
