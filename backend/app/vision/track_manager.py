"""Track lifecycle, class voting and session-level identity on top of the tracker.

Lifecycle of a tracker ID (all times in seconds of the session clock: video time for uploaded videos,
wall time for live streams):
  new      seen in fewer than `min_hits` frames (tentative; not counted in the inventory)
  active   confirmed and seen in the current frame
  lost     confirmed, not seen in the current frame, still inside the tracker buffer
  removed  not seen for longer than `lost_timeout_s`; its equipment item can still be re-linked for a while

A confirmed track gets an equipment ID from EquipmentIdentity (re-linked to a recently lost item when class,
position and appearance agree). Class: confidence-weighted vote over the track's detections, which smooths the
per-frame class flicker of look-alike classes; `class_consistency` is the winning class's share of the votes.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from app.identification.equipment_identity import EquipmentIdentity, IdentityConfig, appearance_histogram
from app.vision.detector import RawDetections
from app.vision.tracker import ObjectTracker

NEW, ACTIVE, LOST, REMOVED = "new", "active", "lost", "removed"


@dataclass
class TrackInfo:
    track_id: int
    track_state: str
    stable_class_id: int
    class_consistency: float
    smoothed_confidence: float
    track_age_s: float
    equipment_id: str | None
    relinked: bool = False


@dataclass
class _Track:
    track_id: int
    first_seen: float
    last_seen: float
    box: np.ndarray
    hits: int = 0
    state: str = NEW
    votes: dict[int, float] = field(default_factory=dict)
    conf: float = 0.0
    equipment_id: str | None = None
    appearance: np.ndarray | None = None
    relinked: bool = False

    @property
    def stable_class(self) -> int:
        return max(self.votes, key=lambda c: (self.votes[c], -c))

    @property
    def consistency(self) -> float:
        total = sum(self.votes.values())
        return self.votes[self.stable_class] / total if total > 0 else 0.0


class TrackManager:
    def __init__(self, identity: EquipmentIdentity, names: dict[int, str], min_hits: int = 3,
                 lost_timeout_s: float = 3.0, conf_alpha: float = 0.3, appearance_every: int = 5):
        self.identity = identity
        self.names = names
        self.min_hits = min_hits
        self.lost_timeout_s = lost_timeout_s
        self.conf_alpha = conf_alpha
        self.appearance_every = appearance_every
        self.tracks: dict[int, _Track] = {}
        self.removed = 0
        self.confirmed = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.identity.reset()
        self.removed = self.confirmed = 0

    def _class_name(self, cid: int) -> str:
        return self.names.get(cid, str(cid))

    def update(self, raw: RawDetections, track_ids: np.ndarray, frame: np.ndarray | None, now: float
               ) -> list[TrackInfo | None]:
        infos: list[TrackInfo | None] = [None] * len(raw)
        seen = set()
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid < 0:
                continue
            seen.add(tid)
            box = np.asarray(raw.xyxy[i], float)
            t = self.tracks.get(tid)
            if t is None or t.state == REMOVED:
                t = self.tracks[tid] = _Track(tid, now, now, box)
            t.hits += 1
            t.last_seen, t.box = now, box
            conf, cls = float(raw.conf[i]), int(raw.cls[i])
            t.votes[cls] = t.votes.get(cls, 0.0) + conf
            t.conf = conf if t.hits == 1 else (1 - self.conf_alpha) * t.conf + self.conf_alpha * conf
            if t.state == NEW and t.hits >= self.min_hits:
                t.appearance = appearance_histogram(frame, box)
                t.equipment_id, t.relinked = self.identity.assign(
                    self._class_name(t.stable_class), box, t.appearance, raw.frame_width, raw.frame_height, now)
                self.confirmed += 1
            elif t.equipment_id and t.hits % self.appearance_every == 0:
                h = appearance_histogram(frame, box)
                if h is not None:
                    t.appearance = h if t.appearance is None else 0.7 * t.appearance + 0.3 * h
            if t.equipment_id:
                t.state = ACTIVE
            infos[i] = TrackInfo(tid, t.state, t.stable_class, round(t.consistency, 3), round(t.conf, 4),
                                 round(now - t.first_seen, 2), t.equipment_id, t.relinked)
        for tid, t in list(self.tracks.items()):
            if tid in seen or t.state == REMOVED:
                continue
            if now - t.last_seen > self.lost_timeout_s:
                if t.equipment_id:
                    self.identity.lost(t.equipment_id, self._class_name(t.stable_class), t.box, t.appearance,
                                       t.last_seen)
                t.state = REMOVED
                self.removed += 1
                del self.tracks[tid]
            elif t.state == ACTIVE:
                t.state = LOST
        return infos

    def counts(self) -> dict[str, int]:
        states = [t.state for t in self.tracks.values()]
        return {NEW: states.count(NEW), ACTIVE: states.count(ACTIVE), LOST: states.count(LOST),
                REMOVED: self.removed, "confirmed": self.confirmed, "equipment_items": self.identity.created,
                "relinks": self.identity.relinks}


class TrackingSession:
    """Tracker + lifecycle + identity for ONE session (stream or video). Never shared between sessions."""

    def __init__(self, tracker_args: SimpleNamespace, session_id: str, names: dict[int, str],
                 frame_rate: float = 10.0, min_hits: int = 3, lost_timeout_s: float | None = None,
                 identity_config: IdentityConfig | None = None):
        self.session_id = session_id
        self.tracker = ObjectTracker(tracker_args, frame_rate=frame_rate)
        timeout = lost_timeout_s if lost_timeout_s is not None else self.tracker.buffer_s
        self.manager = TrackManager(EquipmentIdentity(session_id, identity_config or IdentityConfig()), names,
                                    min_hits=min_hits, lost_timeout_s=timeout)

    def reset(self) -> None:
        self.tracker.reset()
        self.manager.reset()

    def step(self, raw: RawDetections, frame: np.ndarray | None, now: float) -> tuple[np.ndarray, list]:
        track_ids = self.tracker.update(raw, frame)
        return track_ids, self.manager.update(raw, track_ids, frame, now)

    def counts(self) -> dict[str, int]:
        return self.manager.counts()
