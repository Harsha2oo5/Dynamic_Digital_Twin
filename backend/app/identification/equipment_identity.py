"""Session-level equipment identity.

The tracker's IDs change whenever a track is lost for longer than its buffer or association fails (equipment
leaving and re-entering the frame while the phone pans). This layer gives each *physical item seen in one
session* a stable equipment ID such as ``RU-003@a1b2c3d4`` (class abbreviation, number, session). When a new track is confirmed, it is re-linked to a
recently lost item of the same class if position and a cheap colour-histogram appearance agree, so one item does
not produce several inventory entries.

These IDs are NOT permanent: they are unique within one session (one stream / one video) only. Two identical
looking items that are never visible at the same time can be merged; an item seen from a very different angle
can get a second ID. A permanent, cross-session registry is a later milestone; ``PersistentEquipmentRegistry``
is the interface it will implement (only an in-memory stand-in exists now).
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

CLASS_ABBREVIATIONS = {"antenna": "ANT", "router": "RTR", "cable": "CBL", "battery": "BAT",
                       "power_supply_unit": "PSU", "equipment_cabinet": "CAB", "radio_unit": "RU"}


def class_abbreviation(class_name: str) -> str:
    return CLASS_ABBREVIATIONS.get(class_name) or "".join(ch for ch in class_name.upper() if ch.isalnum())[:3] or "OBJ"


def appearance_histogram(frame: np.ndarray | None, box: np.ndarray) -> np.ndarray | None:
    """Normalised HSV hue/saturation histogram of the box crop (16x8 bins); None if the crop is unusable."""
    if frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    if max(crop.shape[:2]) > 96:  # cost is independent of the box size
        s = 96 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else None


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.minimum(a, b).sum())  # histogram intersection of normalised histograms, 0..1


@dataclass
class _Candidate:
    equipment_id: str
    class_name: str
    box: np.ndarray
    appearance: np.ndarray | None
    lost_at: float


@dataclass
class IdentityConfig:
    relink_window_s: float = 15.0  # how long a lost item can be re-linked
    max_center_distance: float = 0.35  # fraction of the frame diagonal
    min_appearance: float = 0.55  # histogram intersection required together with the position
    strong_appearance: float = 0.8  # appearance alone (any position) is enough above this; handles camera pans


@dataclass
class EquipmentIdentity:
    session_id: str
    config: IdentityConfig = field(default_factory=IdentityConfig)

    def __post_init__(self):
        self._counters: dict[str, int] = {}
        self._lost: dict[str, _Candidate] = {}
        self.relinks = 0
        self.created = 0

    def reset(self) -> None:
        self.__post_init__()

    def _new_id(self, class_name: str) -> str:
        abbr = class_abbreviation(class_name)
        self._counters[abbr] = self._counters.get(abbr, 0) + 1
        self.created += 1
        return f"{abbr}-{self._counters[abbr]:03d}@{self.session_id}"

    def _match_score(self, c: _Candidate, class_name: str, box: np.ndarray, appearance, diag: float, now: float):
        if c.class_name != class_name or now - c.lost_at > self.config.relink_window_s:
            return None
        sim = appearance_similarity(c.appearance, appearance)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ox, oy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
        dist = math.hypot(cx - ox, cy - oy) / diag
        if sim is not None and sim >= self.config.strong_appearance:
            return 1.0 + sim - dist * 0.1
        if dist <= self.config.max_center_distance and (sim is None or sim >= self.config.min_appearance):
            return (1 - dist / self.config.max_center_distance) * 0.5 + (sim if sim is not None else 0.25)
        return None

    def assign(self, class_name: str, box: np.ndarray, appearance: np.ndarray | None, frame_w: int, frame_h: int,
               now: float) -> tuple[str, bool]:
        """Equipment ID for a newly confirmed track: (id, re-linked to a lost item?)."""
        self.expire(now)
        diag = math.hypot(frame_w, frame_h) or 1.0
        best, best_score = None, None
        for c in self._lost.values():
            score = self._match_score(c, class_name, box, appearance, diag, now)
            if score is not None and (best_score is None or score > best_score):
                best, best_score = c, score
        if best is not None:
            del self._lost[best.equipment_id]
            self.relinks += 1
            return best.equipment_id, True
        return self._new_id(class_name), False

    def lost(self, equipment_id: str, class_name: str, box: np.ndarray, appearance: np.ndarray | None,
             now: float) -> None:
        self._lost[equipment_id] = _Candidate(equipment_id, class_name, np.asarray(box, float), appearance, now)

    def expire(self, now: float) -> None:
        for k in [k for k, c in self._lost.items() if now - c.lost_at > self.config.relink_window_s]:
            del self._lost[k]


class PersistentEquipmentRegistry(Protocol):
    """Interface for a future permanent registry (site-wide IDs that survive sessions). Not implemented yet."""

    def register(self, session_equipment_id: str, class_name: str, attributes: dict) -> str | None: ...

    def lookup(self, permanent_id: str) -> dict | None: ...


class InMemoryEquipmentRegistry:
    """Development stand-in: remembers session equipment IDs; never claims a permanent identity."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    def register(self, session_equipment_id: str, class_name: str, attributes: dict) -> str | None:
        self._items[session_equipment_id] = {"class_name": class_name, **attributes}
        return None  # no permanent ID is issued

    def lookup(self, permanent_id: str) -> dict | None:
        return self._items.get(permanent_id)
