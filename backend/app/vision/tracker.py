"""Per-session multi-object tracker (ByteTrack, or BoT-SORT with camera-motion compensation).

Each stream / video job owns its own ObjectTracker, so concurrent sessions never share
tracking state. Track IDs are temporary visual identifiers, unique only within a session.

Tracker buffers in the YAML are frames at a 30 fps reference. Passing the real processed frame rate
(`frame_rate`) scales the buffer, so a lost track survives the same *time* whatever the frame skip or CPU speed.
The default frame_rate of 30 keeps the configured buffer unchanged.
"""

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from app.vision.detector import RawDetections

SUPPORTED_TRACKERS = ("bytetrack", "botsort")


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
    cfg = YAML.load(check_yaml(str(config)))
    if cfg.get("tracker_type") not in SUPPORTED_TRACKERS:
        raise ValueError(f"Unsupported tracker {cfg.get('tracker_type')!r}; use one of {SUPPORTED_TRACKERS}")
    if cfg.get("tracker_type") == "botsort" and cfg.get("with_reid"):
        raise ValueError("BoT-SORT ReID needs an extra model; set with_reid: false (identity re-linking is separate)")
    return SimpleNamespace(**cfg)


class ObjectTracker:
    def __init__(self, tracker_args: SimpleNamespace, frame_rate: float = 30.0):
        self._args = tracker_args
        self.frame_rate = max(1, int(round(frame_rate)))
        self.kind = tracker_args.tracker_type
        self._tracker = self._make()

    def _scaled_args(self) -> SimpleNamespace:
        """track_buffer converted from frames at 30 fps to frames at the session's real processed rate (done here
        rather than via the trackers' frame_rate argument, which newer Ultralytics versions removed)."""
        args = SimpleNamespace(**vars(self._args))
        args.track_buffer = max(1, int(round(self._args.track_buffer * self.frame_rate / 30.0)))
        return args

    def _make(self):
        if self.kind == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT

            return BOTSORT(self._scaled_args())
        return BYTETracker(self._scaled_args())

    @property
    def buffer_s(self) -> float:
        """How long a lost track is kept, in seconds (track_buffer frames at the 30 fps reference)."""
        return float(self._args.track_buffer) / 30.0

    def reset(self) -> None:
        self._tracker = self._make()

    def update(self, dets: RawDetections, frame: np.ndarray | None = None) -> np.ndarray:
        """Return a track ID per detection (-1 where the detection is not part of a confirmed track).
        `frame` is needed only by BoT-SORT's camera-motion compensation."""
        track_ids = np.full(len(dets), -1, dtype=int)
        boxes = _BoxesView(dets.xyxy, dets.conf, dets.cls.astype(np.float32))
        # Always call update, even with no detections, so lost tracks age out correctly.
        if self.kind == "botsort":
            tracks = self._tracker.update(boxes, frame)
        else:
            tracks = self._tracker.update(boxes)
        for row in tracks:
            det_idx = int(row[-1])
            if 0 <= det_idx < len(track_ids):
                track_ids[det_idx] = int(row[4])
        return track_ids
