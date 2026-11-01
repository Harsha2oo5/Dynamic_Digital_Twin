"""Registry of observed (tracked) objects for the Object Inventory.

Entries are keyed by source + temporary track ID. They describe *observations*, not physical
components: the same physical item seen in two videos produces two entries. A permanent
component registry belongs to a later milestone.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

from app.schemas.detection import BBox, FrameResult, TrackedObject


@dataclass
class _Entry:
    source_id: str
    source_type: str
    track_id: int | None
    class_id: int
    class_name: str
    component_type: str | None
    confidence: float
    max_confidence: float
    conf_sum: float
    first_seen: datetime
    last_seen: datetime
    frames_seen: int
    last_bbox: BBox
    frame_width: int
    frame_height: int


class ObjectRegistry:
    def __init__(self, lost_after_s: float = 3.0, max_entries: int = 5000):
        self.lost_after_s = lost_after_s
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._live_sources: set[str] = set()
        self._lock = threading.Lock()

    def source_started(self, source_id: str) -> None:
        with self._lock:
            self._live_sources.add(source_id)

    def source_stopped(self, source_id: str) -> None:
        with self._lock:
            self._live_sources.discard(source_id)

    def update(self, source_id: str, source_type: str, result: FrameResult) -> None:
        now = result.timestamp
        with self._lock:
            for i, det in enumerate(result.detections):
                if source_type == "image":
                    key = f"{source_id}:{i}"
                elif det.track_id is not None:
                    key = f"{source_id}:{det.track_id}"
                else:
                    continue  # unconfirmed detection in a tracked stream
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _Entry(
                        source_id=source_id,
                        source_type=source_type,
                        track_id=det.track_id,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        component_type=det.component_type,
                        confidence=det.confidence,
                        max_confidence=det.confidence,
                        conf_sum=det.confidence,
                        first_seen=now,
                        last_seen=now,
                        frames_seen=1,
                        last_bbox=det.bbox,
                        frame_width=result.frame_width,
                        frame_height=result.frame_height,
                    )
                else:
                    entry.class_id, entry.class_name = det.class_id, det.class_name
                    entry.component_type = det.component_type
                    entry.confidence = det.confidence
                    entry.max_confidence = max(entry.max_confidence, det.confidence)
                    entry.conf_sum += det.confidence
                    entry.last_seen = now
                    entry.frames_seen += 1
                    entry.last_bbox = det.bbox
                    self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def _status(self, entry: _Entry, now: datetime) -> str:
        if entry.source_type == "image":
            return "snapshot"
        recent = (now - entry.last_seen).total_seconds() < self.lost_after_s
        return "active" if recent and entry.source_id in self._live_sources else "lost"

    def _to_model(self, key: str, entry: _Entry, now: datetime) -> TrackedObject:
        return TrackedObject(
            object_id=key,
            source_id=entry.source_id,
            source_type=entry.source_type,
            track_id=entry.track_id,
            class_id=entry.class_id,
            class_name=entry.class_name,
            component_type=entry.component_type,
            confidence=entry.confidence,
            max_confidence=entry.max_confidence,
            avg_confidence=round(entry.conf_sum / entry.frames_seen, 4),
            first_seen=entry.first_seen,
            last_seen=entry.last_seen,
            frames_seen=entry.frames_seen,
            status=self._status(entry, now),
            last_bbox=entry.last_bbox,
            frame_width=entry.frame_width,
            frame_height=entry.frame_height,
        )

    def list_objects(self) -> list[TrackedObject]:
        now = datetime.now()
        with self._lock:
            return [self._to_model(k, e, now) for k, e in reversed(self._entries.items())]

    def get(self, object_id: str) -> TrackedObject | None:
        now = datetime.now()
        with self._lock:
            entry = self._entries.get(object_id)
            return self._to_model(object_id, entry, now) if entry else None

    def counts(self) -> tuple[int, int]:
        """(currently active tracks, total tracked objects ever seen) excluding image snapshots."""
        now = datetime.now()
        with self._lock:
            tracked = [e for e in self._entries.values() if e.source_type != "image"]
            return sum(self._status(e, now) == "active" for e in tracked), len(tracked)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
