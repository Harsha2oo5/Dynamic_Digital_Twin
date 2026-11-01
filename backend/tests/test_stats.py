"""Dashboard statistics: activity states, latency window, class ids, errors (no model needed),
plus the additive API fields against the real YOLO11n model."""

from datetime import datetime

import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.detection import BBox, Detection, FrameResult
from app.services import stats as stats_mod
from app.services.stats import StatsCollector
from tests.conftest import make_settings


def frame(ms: float, dets: list[tuple[int, str, float]] = ()) -> FrameResult:
    return FrameResult(frame_id=0, timestamp=datetime.now(), device="cpu", fps=1000 / ms, inference_ms=ms,
                       frame_width=64, frame_height=48,
                       detections=[Detection(track_id=None, class_id=c, class_name=n, confidence=p,
                                             bbox=BBox(x1=0, y1=0, x2=10, y2=10)) for c, n, p in dets])


def test_counters_latency_window_and_class_ids():
    s = StatsCollector(window=3)
    snap = s.snapshot()
    # nothing measured yet: unavailable, not zero
    assert snap["avg_inference_ms"] is None and snap["min_inference_ms"] is None and snap["avg_confidence"] is None
    assert snap["recent"] == [] and snap["last_frame_at"] is None and snap["seconds_since_last_frame"] is None
    for ms in (40, 10, 30, 20):
        s.record(frame(ms, [(0, "person", 0.5), (2, "car", 0.9)]))
    snap = s.snapshot()
    assert snap["frames_processed"] == 4 and snap["total_detections"] == 8
    assert [r.inference_ms for r in snap["recent"]] == [10, 30, 20]  # rolling window of 3
    assert [r.seq for r in snap["recent"]] == [2, 3, 4]
    assert (snap["min_inference_ms"], snap["max_inference_ms"], snap["avg_inference_ms"]) == (10, 30, 20)
    assert snap["avg_inference_fps"] == 50 and snap["latency_window"] == 3
    assert snap["avg_confidence"] == pytest.approx(0.7)
    assert {(c.class_id, c.class_name, c.count) for c in snap["class_counts"]} == {(0, "person", 4), (2, "car", 4)}


def test_zero_detections_is_a_measured_zero():
    s = StatsCollector()
    s.record(frame(25))
    snap = s.snapshot()
    assert snap["frames_processed"] == 1 and snap["total_detections"] == 0
    assert snap["avg_confidence"] is None  # no detections -> no mean confidence, not 0 %
    assert snap["avg_inference_ms"] == 25


def test_engine_and_source_states(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(stats_mod.time, "monotonic", lambda: clock[0])
    s = StatsCollector()
    assert s.engine_state(model_loaded=False) == "error"
    assert s.engine_state(model_loaded=True) == "idle"  # backend up, nothing processed

    s.source_started("webcam:abc", "Webcam stream (abc)")
    snap = s.snapshot()
    assert [(x.source_type, x.state, x.frames) for x in snap["sources"]] == [("webcam", "connected", 0)]
    assert s.engine_state(True) == "idle"  # connected but no inference yet

    s.record(frame(30), "webcam:abc")
    assert s.snapshot()["sources"][0].state == "processing" and s.engine_state(True) == "running"

    clock[0] += stats_mod.ACTIVE_WINDOW_S + 1
    assert s.snapshot()["sources"][0].state == "connected" and s.engine_state(True) == "idle"

    s.record_error("Frame decode failed: bad bytes", "webcam:abc")
    snap = s.snapshot()
    assert s.engine_state(True) == "error" and snap["error_count"] == 1
    assert snap["last_error"].message.startswith("Frame decode failed") and snap["last_error"].source == "webcam:abc"
    s.record(frame(30), "webcam:abc")  # a later good frame clears the error state
    assert s.engine_state(True) == "running"
    s.record_error("same clock tick")  # ordered by event, not by the (frozen) clock
    assert s.engine_state(True) == "error"
    s.record(frame(30), "webcam:abc")
    assert s.engine_state(True) == "running"
    s.record_error("x")
    clock[0] += stats_mod.ERROR_WINDOW_S + 1
    assert s.engine_state(True) == "idle"  # old errors expire

    s.record(frame(30), "image:1234")  # one-shot images are not registered sources
    assert [x.source_id for x in s.snapshot()["sources"]] == ["webcam:abc"]
    s.source_stopped("webcam:abc")
    assert s.snapshot()["sources"] == [] and s.snapshot()["active_sources"] == []


def test_reset_clears_measurements_but_keeps_connected_sources():
    s = StatsCollector()
    s.source_started("video:j1", "Video: a.mp4")
    s.record(frame(30, [(1, "bicycle", 0.6)]), "video:j1")
    s.record_error("boom")
    s.reset()
    snap = s.snapshot()
    assert snap["frames_processed"] == 0 and snap["class_counts"] == [] and snap["last_error"] is None
    assert [(x.label, x.frames, x.last_frame_at) for x in snap["sources"]] == [("Video: a.mp4", 0, None)]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(make_settings(tmp_path_factory.mktemp("stats")))) as c:
        yield c


def test_api_reports_activity_and_model_metadata(client, bus_image):
    info = client.get("/api/model/info").json()
    assert info["component_taxonomy"] == ["antenna", "router", "cable", "battery", "power_supply_unit",
                                          "equipment_cabinet", "radio_unit"]
    assert info["model_type"] == "pretrained" and info["checkpoint_size_mb"] > 0 and info["checkpoint_modified"]
    assert info["taxonomy_match"] is None  # pretrained COCO model: component analytics unavailable

    stats = client.get("/api/detection/stats").json()
    assert stats["engine_state"] == "idle" and stats["frames_processed"] == 0 and stats["sources"] == []
    assert stats["min_inference_ms"] is None and stats["last_error"] is None

    jpg = cv2.imencode(".jpg", bus_image)[1].tobytes()
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        live = client.get("/api/detection/stats").json()
        assert [s["state"] for s in live["sources"]] == ["connected"]  # connected, no frame yet
        ws.send_bytes(jpg)
        assert ws.receive_json()["type"] == "result"
        live = client.get("/api/detection/stats").json()
        assert live["engine_state"] == "running" and live["sources"][0]["state"] == "processing"
        assert live["sources"][0]["frames"] == 1 and live["min_inference_ms"] > 0
        assert all(c["class_id"] is not None for c in live["class_counts"])
        assert any(c["class_name"] == "bus" and c["class_id"] == 5 for c in live["class_counts"])
        ws.send_bytes(b"not an image")
        assert ws.receive_json()["type"] == "error"
        broken = client.get("/api/detection/stats").json()
        assert broken["engine_state"] == "error" and broken["error_count"] == 1
    assert client.get("/api/detection/stats").json()["sources"] == []
