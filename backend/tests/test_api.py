"""Integration tests against the real YOLO11n model on CPU (no mocks)."""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import BUS_IMAGE, make_settings


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("api")))
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_model_info_reports_cpu_and_classes(client):
    info = client.get("/api/model/info").json()
    assert info["loaded"] and info["device"] == "cpu"
    assert info["model_type"] == "pretrained" and info["num_classes"] == 80
    assert "person" in info["class_names"]
    models = client.get("/api/models").json()
    assert any(m["active"] and m["name"] == "yolo11n.pt" for m in models)


def test_model_runs_on_cpu(client):
    service = client.app.state.service
    assert str(next(service.detector.model.model.parameters()).device) == "cpu"


def test_detect_image_real_inference(client):
    with BUS_IMAGE.open("rb") as f:
        r = client.post("/api/detect/image", files={"file": ("bus.jpg", f, "image/jpeg")}, data={"annotate": "true"})
    assert r.status_code == 200
    body = r.json()
    assert (body["frame_width"], body["frame_height"]) == (810, 1080)
    names = [d["class_name"] for d in body["detections"]]
    assert "bus" in names and names.count("person") >= 3
    for d in body["detections"]:
        b = d["bbox"]
        assert 0 <= b["x1"] < b["x2"] <= 810 and 0 <= b["y1"] < b["y2"] <= 1080
        assert 0.35 <= d["confidence"] <= 1
    assert body["annotated_image"]


def test_detect_image_confidence_threshold(client):
    with BUS_IMAGE.open("rb") as f:
        body = client.post("/api/detect/image", files={"file": ("bus.jpg", f)}, data={"confidence": "0.85"}).json()
    assert all(d["confidence"] >= 0.85 for d in body["detections"])


@pytest.mark.parametrize(
    "name,content,code",
    [("x.txt", b"hello", 415), ("x.jpg", b"not an image", 400), ("x.png", b"", 400)],
)
def test_detect_image_errors(client, name, content, code):
    r = client.post("/api/detect/image", files={"file": (name, content)})
    assert r.status_code == code
    assert "detail" in r.json()


def test_detect_image_rejects_invalid_threshold(client):
    with BUS_IMAGE.open("rb") as f:
        r = client.post("/api/detect/image", files={"file": ("bus.jpg", f)}, data={"confidence": "2"})
    assert r.status_code == 422


def test_video_job_end_to_end(client, bus_video):
    with bus_video.open("rb") as f:
        r = client.post("/api/detect/video", files={"file": ("pan.mp4", f, "video/mp4")}, data={"frame_skip": "1"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(300):
        status = client.get(f"/api/detect/video/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert status["status"] == "completed", status
    results = client.get(f"/api/detect/video/{job_id}/results").json()
    frames = results["frames"]
    assert len(frames) == 12
    assert [f["frame_id"] for f in frames] == list(range(12))
    tracked = [d["track_id"] for f in frames for d in f["detections"] if d["class_name"] == "bus"]
    assert tracked and len({t for t in tracked if t is not None}) == 1  # one bus, one ID
    assert client.get(f"/api/detect/video/{job_id}/annotated").status_code == 200
    frame = client.get(f"/api/detect/video/{job_id}/frames/5")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)
    assert client.get(f"/api/detect/video/{job_id}/frames/99").status_code == 404


def test_video_rejects_invalid(client):
    assert client.post("/api/detect/video", files={"file": ("x.mp4", b"garbage")}).status_code == 400
    assert client.post("/api/detect/video", files={"file": ("x.gif", b"garbage")}).status_code == 415
    assert client.get("/api/detect/video/doesnotexist").status_code == 404


def test_stats_and_inventory(client):
    stats = client.get("/api/detection/stats").json()
    assert stats["device"] == "cpu" and stats["frames_processed"] > 0
    assert stats["avg_inference_fps"] > 0
    objects = client.get("/api/objects").json()
    assert objects
    assert client.get(f"/api/objects/{objects[0]['object_id']}").status_code == 200
    assert client.get("/api/objects/nope").status_code == 404


def test_websocket_stream(client, bus_image):
    jpg = cv2.imencode(".jpg", bus_image)[1].tobytes()
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text('{"type": "config", "confidence": 0.5}')
        ids = []
        for _ in range(3):
            ws.send_bytes(jpg)
            msg = ws.receive_json()
            assert msg["type"] == "result" and msg["device"] == "cpu"
            ids.append(sorted(d["track_id"] for d in msg["detections"] if d["track_id"] is not None))
        assert ids[1] == ids[2] and ids[2]
        ws.send_bytes(b"garbage")
        assert ws.receive_json()["type"] == "error"


def test_degraded_mode_when_model_missing(tmp_path):
    app = create_app(make_settings(tmp_path, model_path=tmp_path / "missing.pt"))
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "degraded"
        info = c.get("/api/model/info").json()
        assert not info["loaded"] and "not found" in info["error"]
        with BUS_IMAGE.open("rb") as f:
            assert c.post("/api/detect/image", files={"file": ("bus.jpg", f)}).status_code == 503
