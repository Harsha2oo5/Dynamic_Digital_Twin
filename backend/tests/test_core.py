"""Unit tests: device selection, preprocessing, box conversion, ingestion."""

import cv2
import numpy as np
import pytest

from app.camera.video_stream import InvalidMediaError, VideoSource, decode_image, load_image
from app.core.device import resolve_device
from app.vision.detector import Detector, ModelNotLoadedError, infer_model_type, scale_boxes_to_original, to_bbox
from app.vision.preprocessing import ensure_bgr, resize_max_dim
from pathlib import Path


@pytest.mark.parametrize("requested", ["cpu", "CPU", "cuda", "cuda:0", "mps", "", None])
def test_device_is_always_cpu(requested):
    assert resolve_device(requested) == "cpu"


def test_ensure_bgr_conversions():
    bgr = np.zeros((10, 20, 3), np.uint8)
    assert ensure_bgr(bgr) is bgr  # no copy for valid input
    assert ensure_bgr(np.zeros((10, 20), np.uint8)).shape == (10, 20, 3)
    assert ensure_bgr(np.zeros((10, 20, 4), np.uint8)).shape == (10, 20, 3)
    with pytest.raises(ValueError):
        ensure_bgr(np.zeros((10, 20, 3), np.float32))
    with pytest.raises(ValueError):
        ensure_bgr(np.zeros((0,), np.uint8))


def test_resize_max_dim_preserves_aspect_ratio():
    frame = np.zeros((1000, 2000, 3), np.uint8)
    small, scale = resize_max_dim(frame, 500)
    assert small.shape == (250, 500, 3)
    assert scale == pytest.approx(0.25)
    same, scale = resize_max_dim(frame, 4000)
    assert same is frame and scale == 1.0


def test_scale_boxes_to_original_and_clip():
    boxes = np.array([[10, 20, 110, 220], [-5, -5, 600, 300]], np.float32)
    out = scale_boxes_to_original(boxes, 0.5, width=1000, height=500)
    np.testing.assert_allclose(out[0], [20, 40, 220, 440])
    np.testing.assert_allclose(out[1], [0, 0, 1000, 500])  # clipped to frame


def test_to_bbox_rounds():
    b = to_bbox(np.array([1.234, 2.345, 3.456, 4.567]))
    assert (b.x1, b.y1, b.x2, b.y2) == (1.2, 2.3, 3.5, 4.6)


def test_infer_model_type():
    assert infer_model_type(Path("models/pretrained/yolo11n.pt")) == "pretrained"
    assert infer_model_type(Path("models/custom/digital_twin_best.pt")) == "custom"
    assert infer_model_type(Path("elsewhere/x.pt")) == "unknown"


def test_missing_model_is_handled_gracefully(tmp_path):
    det = Detector(tmp_path / "missing.pt")
    assert det.load() is False
    assert "not found" in det.error
    with pytest.raises(ModelNotLoadedError):
        det.predict(np.zeros((32, 32, 3), np.uint8), 0.5, 0.45, 320)


def test_decode_image_rejects_garbage():
    with pytest.raises(InvalidMediaError):
        decode_image(b"not an image")
    with pytest.raises(InvalidMediaError):
        decode_image(b"")


def test_load_image_non_ascii_path(tmp_path):
    path = tmp_path / "ünïcode image.png"
    cv2.imencode(".png", np.full((8, 8, 3), 255, np.uint8))[1].tofile(str(path))
    assert load_image(path).shape == (8, 8, 3)


def test_video_frame_extraction(bus_video):
    with VideoSource(bus_video) as src:
        info = src.info
        assert (info.width, info.height) == (320, 240)
        assert info.fps == pytest.approx(10, abs=0.1)
        assert info.frame_count == 12
        indices = [i for i, _ in src.frames(skip=3)]
    assert indices == [0, 3, 6, 9]


def test_video_source_invalid(tmp_path):
    with pytest.raises(InvalidMediaError):
        VideoSource(tmp_path / "missing.mp4").open()
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(InvalidMediaError):
        VideoSource(bad).open()
