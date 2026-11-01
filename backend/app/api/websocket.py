"""WebSocket streaming detection.

Protocol (one connection = one session with its own tracker):
  client -> server  binary   encoded frame (JPEG/PNG). The client should wait for the reply
                             before sending the next frame (natural back-pressure).
  client -> server  text     {"type": "config", "confidence": 0.35, "iou": 0.45, "image_size": 416,
                              "track": true, "source": "webcam", "label": "Laptop webcam"}
                             {"type": "reset"}   -> restart tracking (new IDs)
  server -> client  text     {"type": "result", ...FrameResult}
                             {"type": "ready", "session_id": ..., "device": "cpu"}
                             {"type": "error", "message": ...}
"""

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.camera.video_stream import InvalidMediaError, decode_image
from app.vision.tracker import ObjectTracker

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FRAME_BYTES = 8 << 20


@router.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    service = ws.app.state.service
    s = service.settings
    session_id = uuid.uuid4().hex[:8]
    params = {"confidence": s.confidence_threshold, "iou": s.iou_threshold, "image_size": s.image_size}
    source_type = "webcam"
    tracker = ObjectTracker(service.tracker_args)
    track_enabled = True
    source_id = f"{source_type}:{session_id}"
    frame_id = 0

    service.source_started(source_id, f"Webcam stream ({session_id})")
    await ws.send_json({"type": "ready", "session_id": session_id, "device": service.device, **params})
    logger.info("WebSocket session %s connected", session_id)
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    cmd = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue
                if cmd.get("type") == "reset":
                    tracker.reset()
                    frame_id = 0
                elif cmd.get("type") == "config":
                    for key, lo, hi in (("confidence", 0.0, 1.0), ("iou", 0.0, 1.0), ("image_size", 32, 1920)):
                        if key in cmd:
                            params[key] = type(params[key])(min(max(cmd[key], lo), hi))
                    track_enabled = bool(cmd.get("track", track_enabled))
                    if cmd.get("source") in ("webcam", "video") and cmd["source"] != source_type:
                        service.source_stopped(source_id)
                        source_type = cmd["source"]
                        source_id = f"{source_type}:{session_id}"
                        tracker.reset()
                        service.source_started(source_id, cmd.get("label") or f"{source_type.title()} stream ({session_id})")
                continue

            data = message.get("bytes")
            if not data:
                continue
            if not service.detector.loaded:
                service.record_error(source_id, service.detector.error or "Model is not loaded")
                await ws.send_json({"type": "error", "message": service.detector.error or "Model is not loaded"})
                continue
            if len(data) > MAX_FRAME_BYTES:
                await ws.send_json({"type": "error", "message": "Frame too large"})
                continue
            try:
                frame = await run_in_threadpool(decode_image, data)
                processor = service.new_processor(tracker if track_enabled else None)
                result = await run_in_threadpool(
                    processor.process, frame, frame_id, params["confidence"], params["iou"], params["image_size"]
                )
            except InvalidMediaError as exc:
                service.record_error(source_id, f"Frame decode failed: {exc}")
                await ws.send_json({"type": "error", "message": str(exc)})
                continue
            frame_id += 1
            service.record(source_id, source_type, result)
            await ws.send_text(json.dumps({"type": "result", **result.model_dump(mode="json")}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket session %s failed", session_id)
        service.record_error(source_id, f"Stream failed: {exc}")
    finally:
        service.source_stopped(source_id)
        logger.info("WebSocket session %s closed after %d frames", session_id, frame_id)
