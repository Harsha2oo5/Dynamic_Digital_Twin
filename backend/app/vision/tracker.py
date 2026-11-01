"""Per-session ByteTrack wrapper.

Each stream / video job owns its own ObjectTracker, so concurrent sessions never share
tracking state. Track IDs are temporary visual identifiers, unique only within a session.
"""

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from app.vision.detector import RawDetections


class _BoxesView:
    """Minimal numpy-backed stand-in for ultralytics Boxes, as expected by BYTETracker.update()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2
        return xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_BoxesView":
        return _BoxesView(self.xyxy[idx], self.conf[idx], self.cls[idx])


def load_tracker_args(config: str = "bytetrack.yaml") -> SimpleNamespace:
    cfg = YAML.load(check_yaml(config))
    if cfg.get("tracker_type") != "bytetrack":
        raise ValueError(f"Only ByteTrack is supported, got {cfg.get('tracker_type')!r}")
    return SimpleNamespace(**cfg)


class ObjectTracker:
    def __init__(self, tracker_args: SimpleNamespace):
        self._args = tracker_args
        self._tracker = BYTETracker(tracker_args)

    def reset(self) -> None:
        self._tracker = BYTETracker(self._args)

    def update(self, dets: RawDetections) -> np.ndarray:
        """Return a track ID per detection (-1 where the detection is not part of a confirmed track)."""
        track_ids = np.full(len(dets), -1, dtype=int)
        boxes = _BoxesView(dets.xyxy, dets.conf, dets.cls.astype(np.float32))
        # Always call update, even with no detections, so lost tracks age out correctly.
        tracks = self._tracker.update(boxes)
        for row in tracks:
            det_idx = int(row[-1])
            if 0 <= det_idx < len(track_ids):
                track_ids[det_idx] = int(row[4])
        return track_ids
