"""Registry of observed equipment for the Object Inventory.

With lifecycle tracking (TrackingSession), entries are keyed by source + *session equipment ID*, so one physical
item seen in one session is one entry even when the tracker's ID changes (re-linked items accumulate their
tracker IDs). Tentative tracks (`new`) are not registered. Without an equipment ID the entry falls back to
source + temporary track ID (older clients / plain ObjectTracker), and image uploads produce snapshots.

Entries describe *observations within a session*, not permanent components: the same physical item seen in two
videos produces two entries. A permanent component registry belongs to a later milestone.
"""

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

from app.schemas.detection import BBox, FrameResult, InventorySummary, SessionInfo, TrackedObject

INVENTORY_NOTE = ("Counts of equipment items identified within each session (stream, video or image). IDs are "
                  "session-level, not permanent: the same physical item seen in two sessions is counted twice.")


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
    equipment_id: str | None = None
    track_ids: list[int] = field(default_factory=list)
    track_state: str | None = None
    class_consistency: float | None = None


@dataclass
class _Source:
    source_type: str
    label: str | None
    started_at: datetime
    live: bool = True
    tracking: object | None = None  # TrackingSession of a live source (for lifecycle counters)


class ObjectRegistry:
    def __init__(self, lost_after_s: float = 3.0, max_entries: int = 5000, removed_after_s: float | None = None):
        self.lost_after_s = lost_after_s
        self.removed_after_s = removed_after_s if removed_after_s is not None else max(10.0, 3 * lost_after_s)
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._live_sources: set[str] = set()
        self._sources: dict[str, _Source] = {}
        self._lock = threading.Lock()

    def source_started(self, source_id: str, source_type: str | None = None, label: str | None = None,
                       tracking=None) -> None:
        with self._lock:
            self._live_sources.add(source_id)
            stype = source_type or source_id.split(":", 1)[0]
            self._sources[source_id] = _Source(stype, label, datetime.now(), True, tracking)

    def source_stopped(self, source_id: str) -> None:
        with self._lock:
            self._live_sources.discard(source_id)
            src = self._sources.get(source_id)
            if src is not None:
                src.live, src.tracking = False, None

    def update(self, source_id: str, source_type: str, result: FrameResult) -> None:
        now = result.timestamp
        with self._lock:
            if source_id not in self._sources:
                self._sources[source_id] = _Source(source_type, None, now, source_id in self._live_sources)
            for i, det in enumerate(result.detections):
                if source_type == "image":
                    key = f"{source_id}:{i}"
                elif det.equipment_id is not None:
                    key = f"{source_id}:{det.equipment_id}"
                elif det.track_id is not None and det.track_state is None:
                    key = f"{source_id}:{det.track_id}"
                else:
                    continue  # unconfirmed / tentative detection in a tracked stream
                class_name = det.stable_class or det.class_name
                class_id = det.class_id
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _Entry(
                        source_id=source_id,
                        source_type=source_type,
                        track_id=det.track_id,
                        class_id=class_id,
                        class_name=class_name,
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
                        equipment_id=det.equipment_id,
                        track_ids=[det.track_id] if det.track_id is not None else [],
                        track_state=det.track_state,
                        class_consistency=det.class_consistency,
                    )
                else:
                    entry.class_id, entry.class_name = class_id, class_name
                    entry.component_type = det.component_type
                    entry.confidence = det.confidence
                    entry.max_confidence = max(entry.max_confidence, det.confidence)
                    entry.conf_sum += det.confidence
                    entry.last_seen = now
                    entry.frames_seen += 1
                    entry.last_bbox = det.bbox
                    entry.track_id = det.track_id
                    if det.track_id is not None and det.track_id not in entry.track_ids:
                        entry.track_ids.append(det.track_id)
                    entry.track_state = det.track_state
                    entry.class_consistency = det.class_consistency
                    self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def _status(self, entry: _Entry, now: datetime) -> str:
        if entry.source_type == "image":
            return "snapshot"
        recent = (now - entry.last_seen).total_seconds() < self.lost_after_s
        return "active" if recent and entry.source_id in self._live_sources else "lost"

    def _lifecycle(self, entry: _Entry, now: datetime) -> str | None:
        if entry.track_state is None:
            return None
        age = (now - entry.last_seen).total_seconds()
        if entry.source_id not in self._live_sources or age >= self.removed_after_s:
            return "removed"
        return "lost" if age >= self.lost_after_s else entry.track_state

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
            equipment_id=entry.equipment_id,
            track_ids=list(entry.track_ids),
            track_state=self._lifecycle(entry, now),
            class_consistency=entry.class_consistency,
            duration_s=round((entry.last_seen - entry.first_seen).total_seconds(), 2),
            relinked=max(0, len(entry.track_ids) - 1) if entry.equipment_id else 0,
        )

    def list_objects(self, status: str | None = None, class_name: str | None = None,
                     session: str | None = None) -> list[TrackedObject]:
        """Newest first. Filters: status (active|lost|snapshot), class name, session (source_id or its id part)."""
        now = datetime.now()
        with self._lock:
            out = []
            for k, e in reversed(self._entries.items()):
                if class_name and e.class_name != class_name:
                    continue
                if session and session not in (e.source_id, e.source_id.split(":", 1)[-1]):
                    continue
                m = self._to_model(k, e, now)
                if status and m.status != status:
                    continue
                out.append(m)
            return out

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

    def sessions(self) -> list[SessionInfo]:
        now = datetime.now()
        with self._lock:
            by_source: dict[str, list[_Entry]] = {}
            for e in self._entries.values():
                by_source.setdefault(e.source_id, []).append(e)
            out = []
            for sid in set(self._sources) | set(by_source):
                src = self._sources.get(sid)
                entries = by_source.get(sid, [])
                tracking = getattr(src, "tracking", None) if src else None
                out.append(SessionInfo(
                    source_id=sid,
                    source_type=src.source_type if src else entries[0].source_type,
                    label=src.label if src else None,
                    live=sid in self._live_sources,
                    started_at=src.started_at if src else (min(e.first_seen for e in entries) if entries else None),
                    last_seen=max((e.last_seen for e in entries), default=None),
                    equipment_items=len(entries),
                    active_items=sum(self._status(e, now) == "active" for e in entries),
                    observations=sum(e.frames_seen for e in entries),
                    class_counts=dict(Counter(e.class_name for e in entries)),
                    tracking=tracking.counts() if tracking is not None else {},
                ))
            return sorted(out, key=lambda s: s.started_at or datetime.min, reverse=True)

    def summary(self) -> InventorySummary:
        now = datetime.now()
        with self._lock:
            entries = list(self._entries.values())
            total, active = Counter(), Counter()
            for e in entries:
                total[e.class_name] += 1
                if self._status(e, now) == "active":
                    active[e.class_name] += 1
            by_class = [{"class_name": c, "items": n, "active": active[c]} for c, n in total.most_common()]
            return InventorySummary(note=INVENTORY_NOTE, total_items=len(entries), active_items=sum(active.values()),
                                    by_class=by_class, sessions=len({e.source_id for e in entries}))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sources = {k: v for k, v in self._sources.items() if k in self._live_sources}
