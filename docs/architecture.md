# Architecture (Milestone 1)

```
 Browser (React)                                   Backend (FastAPI, CPU)
 ───────────────                                   ──────────────────────
 Webcam  ──JPEG frames──►  WS /ws/stream ─┐
 Image   ──multipart────►  POST /detect/image ─┤──► FrameProcessor (one per session/request)
 Video   ──multipart────►  POST /detect/video ─┘        │
                                    (VideoJobManager,    ├─ preprocessing  (validate, downscale huge frames)
                                     1 worker thread)    ├─ Detector       (shared YOLO, lock-serialised, CPU)
                                                         ├─ ObjectTracker  (own ByteTrack per session)
                                                         └─ ComponentClassifier (custom models only)
                                                                │ FrameResult
                                         StatsCollector ◄───────┤
                                         ObjectRegistry ◄───────┘
 Overview / Inventory  ◄── polling REST ── /detection/stats, /objects
```

Design decisions:

- **One model, many sessions.** YOLO loads once at startup. Ultralytics predictors are not thread-safe,
  so `Detector.predict` takes a lock, and callers run it in a worker thread so the event loop stays free.
- **Tracking state is per session.** `model.track(persist=True)` keeps tracker state on the shared model,
  so two concurrent streams would corrupt each other's tracks. Instead, each WebSocket connection and each
  video job owns a `BYTETracker`. Detection and tracking are separate steps.
- **Coordinates.** The backend always returns original-frame pixels. The frontend maps them to the displayed
  element: frame px → media px → `object-contain` letterboxed rectangle.
- **Video results sync to playback** through `frame_id` (the source frame index): `frame ≈ round(t × fps)`,
  using `requestVideoFrameCallback` where available.
- **Webcam back-pressure.** The browser sends a frame only after the previous result arrives, so throughput
  adapts to CPU speed without queueing latency.
- **Tracking IDs ≠ component IDs.** The inventory keys observations by `source:track_id`. A permanent
  component registry (with physical identity, sensors and telemetry) belongs to a later milestone.
