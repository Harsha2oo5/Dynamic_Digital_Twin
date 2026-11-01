"""Custom-model integration: checkpoint selection, loading, class-name mapping, fallback, and
compatibility of inference output with the existing API / tracking / WebSocket.

The 7-class checkpoint built here has RANDOM, UNTRAINED weights. It exists only in tmp_path to
exercise the code path; its detections are meaningless and nothing about accuracy is asserted.
"""

import cv2
import pytest
import torch
import ultralytics
from fastapi.testclient import TestClient
from ultralytics.nn.tasks import DetectionModel

from app.main import create_app
from app.training.taxonomy import CLASS_NAMES
from tests.conftest import BUS_IMAGE, MODEL_PATH, make_settings


@pytest.fixture(scope="module")
def custom_ckpt(tmp_path_factory):
    model = DetectionModel("yolo11n.yaml", nc=len(CLASS_NAMES), verbose=False)
    model.names = dict(CLASS_NAMES)
    model.args = {"task": "detect"}
    path = tmp_path_factory.mktemp("custom") / "digital_twin_best.pt"
    torch.save({"model": model.half(), "train_args": {"task": "detect"}, "version": ultralytics.__version__}, path)
    return path


def custom_settings(tmp_path, ckpt, **kw):
    return make_settings(tmp_path, model_path=None, detection_model="custom", custom_model_path=ckpt,
                         pretrained_model_path=MODEL_PATH, confidence_threshold=0.01, **kw)


def test_selection_resolves_paths(tmp_path, custom_ckpt):
    assert make_settings(tmp_path, model_path=None).selected_model_path() == MODEL_PATH.resolve()
    assert custom_settings(tmp_path, custom_ckpt).selected_model_path() == custom_ckpt.resolve()
    override = make_settings(tmp_path, model_path=MODEL_PATH, detection_model="custom")
    assert override.selected_model_path() == MODEL_PATH.resolve()  # explicit override wins


def test_custom_model_loads_with_taxonomy(tmp_path, custom_ckpt):
    with TestClient(create_app(custom_settings(tmp_path, custom_ckpt))) as c:
        info = c.get("/api/model/info").json()
        assert info["loaded"] and info["model_type"] == "custom" and info["selected_model"] == "custom"
        assert info["class_names"] == list(CLASS_NAMES.values())
        assert info["taxonomy_match"] is True and info["fallback_active"] is False
        assert info["device"] == "cpu"
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/config").json()["detection_model"] == "custom"


def test_custom_inference_output_is_compatible(tmp_path, custom_ckpt):
    """REST + WebSocket responses keep the existing schema; component_type carries the class name."""
    with TestClient(create_app(custom_settings(tmp_path, custom_ckpt))) as c:
        with BUS_IMAGE.open("rb") as f:
            r = c.post("/api/detect/image", files={"file": ("bus.jpg", f)}, data={"confidence": "0.001"})
        assert r.status_code == 200
        body = r.json()
        assert body["device"] == "cpu" and (body["frame_width"], body["frame_height"]) == (810, 1080)
        assert body["detections"], "random weights still emit low-confidence boxes at conf=0.001"
        for d in body["detections"]:
            assert d["class_name"] in CLASS_NAMES.values()
            assert d["component_type"] == d["class_name"] == CLASS_NAMES[d["class_id"]]
            assert 0 <= d["bbox"]["x1"] <= d["bbox"]["x2"] <= 810

        objects = c.get("/api/objects").json()
        assert objects and all(o["component_type"] in CLASS_NAMES.values() for o in objects)

        jpg = cv2.imencode(".jpg", cv2.imread(str(BUS_IMAGE)))[1].tobytes()
        with c.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_text('{"type": "config", "confidence": 0.001}')
            ws.send_bytes(jpg)
            msg = ws.receive_json()
            assert msg["type"] == "result" and all(d["class_name"] in CLASS_NAMES.values() for d in msg["detections"])


def test_missing_custom_model_is_a_clear_error(tmp_path):
    with TestClient(create_app(custom_settings(tmp_path, tmp_path / "nope.pt"))) as c:
        info = c.get("/api/model/info").json()
        assert not info["loaded"] and info["model_type"] == "custom"
        assert "not found" in info["error"] and info["custom_model_available"] is False
        assert c.get("/api/health").json()["status"] == "degraded"
        with BUS_IMAGE.open("rb") as f:
            r = c.post("/api/detect/image", files={"file": ("bus.jpg", f)})
        assert r.status_code == 503 and "not found" in r.json()["detail"]


def test_documented_fallback_to_pretrained(tmp_path):
    settings = custom_settings(tmp_path, tmp_path / "nope.pt", model_fallback_to_pretrained=True)
    with TestClient(create_app(settings)) as c:
        info = c.get("/api/model/info").json()
        assert info["loaded"] and info["model_type"] == "pretrained" and info["selected_model"] == "custom"
        assert info["fallback_active"] is True and "not found" in info["fallback_reason"]
        assert c.get("/api/health").json()["status"] == "degraded"  # never silently "ok"
