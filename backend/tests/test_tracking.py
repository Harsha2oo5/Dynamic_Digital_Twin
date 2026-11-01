"""Unit tests for result parsing, tracking and the object registry (synthetic detections, isolated)."""

from datetime import datetime, timedelta

import numpy as np

from app.identification.component_classifier import ComponentClassifier
from app.schemas.detection import FrameResult
from app.services.object_registry import ObjectRegistry
from app.vision.detector import RawDetections
from app.vision.processor import build_detections
from app.vision.tracker import ObjectTracker, load_tracker_args

NAMES = {0: "person", 1: "car"}


def raw(boxes, confs, classes) -> RawDetections:
    return RawDetections(
        xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        conf=np.array(confs, np.float32),
        cls=np.array(classes, int),
        frame_width=640,
        frame_height=480,
        inference_ms=10.0,
    )


def test_build_detections_parsing():
    dets = build_detections(raw([[1, 2, 3, 4], [10, 20, 30, 40]], [0.9, 0.5], [0, 1]), NAMES, np.array([7, -1]))
    assert [d.class_name for d in dets] == ["person", "car"]
    assert dets[0].track_id == 7 and dets[1].track_id is None
    assert dets[0].bbox.x2 == 3 and dets[1].confidence == 0.5
    assert dets[0].component_type is None


def test_component_classifier_only_for_custom_models():
    assert ComponentClassifier("pretrained", NAMES).component_type(0) is None
    assert ComponentClassifier("custom", {0: "rectifier"}).component_type(0) == "rectifier"


def test_tracker_keeps_ids_across_frames():
    tracker = ObjectTracker(load_tracker_args())
    ids_per_frame = []
    for step in range(6):
        dx = step * 5
        ids = tracker.update(raw([[100 + dx, 100, 200 + dx, 300], [400 - dx, 50, 480 - dx, 200]], [0.9, 0.8], [0, 1]))
        ids_per_frame.append(tuple(ids))
    assert len(set(ids_per_frame[1:])) == 1  # stable after activation
    assert -1 not in ids_per_frame[-1]
    assert ids_per_frame[-1][0] != ids_per_frame[-1][1]


def test_tracker_handles_empty_frames_and_reset():
    tracker = ObjectTracker(load_tracker_args())
    assert len(tracker.update(raw([], [], []))) == 0
    for _ in range(3):
        first = tracker.update(raw([[10, 10, 50, 50]], [0.9], [0]))[0]
    tracker.reset()
    for _ in range(3):
        again = tracker.update(raw([[10, 10, 50, 50]], [0.9], [0]))[0]
    assert first == again == 1  # IDs restart per tracker instance


def test_separate_trackers_are_independent():
    a, b = ObjectTracker(load_tracker_args()), ObjectTracker(load_tracker_args())
    for _ in range(3):
        ida = a.update(raw([[10, 10, 50, 50], [100, 100, 150, 150]], [0.9, 0.9], [0, 0]))
        idb = b.update(raw([[300, 300, 350, 350]], [0.9], [1]))
    assert sorted(ida) == [1, 2] and list(idb) == [1]


def _frame(dets, ts) -> FrameResult:
    return FrameResult(frame_id=0, timestamp=ts, device="cpu", fps=10, inference_ms=100, frame_width=640,
                       frame_height=480, detections=dets)


def test_registry_tracks_lifecycle():
    reg = ObjectRegistry(lost_after_s=2.0)
    reg.source_started("webcam:a")
    now = datetime.now()
    det = build_detections(raw([[1, 2, 3, 4], [5, 6, 7, 8]], [0.6, 0.9], [0, 0]), NAMES, np.array([1, -1]))
    reg.update("webcam:a", "webcam", _frame(det, now - timedelta(seconds=1)))
    det2 = build_detections(raw([[2, 3, 4, 5]], [0.8], [0]), NAMES, np.array([1]))
    reg.update("webcam:a", "webcam", _frame(det2, now))

    objs = reg.list_objects()
    assert len(objs) == 1  # unconfirmed detection (track -1) is not registered
    obj = reg.get("webcam:a:1")
    assert obj.frames_seen == 2 and obj.max_confidence == 0.8 and obj.avg_confidence == 0.7
    assert obj.status == "active" and reg.counts() == (1, 1)
    reg.source_stopped("webcam:a")
    assert reg.get("webcam:a:1").status == "lost"


def test_registry_image_snapshots():
    reg = ObjectRegistry()
    det = build_detections(raw([[1, 2, 3, 4], [5, 6, 7, 8]], [0.6, 0.9], [0, 1]), NAMES)
    reg.update("image:x", "image", _frame(det, datetime.now()))
    assert {o.status for o in reg.list_objects()} == {"snapshot"}
    assert reg.counts() == (0, 0)
