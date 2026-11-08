# Architecture

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

## Telemetry (Milestone 2)

```
 HTTP POST /api/telemetry/ingest ─┐
 MQTT bts/<site>/<sensor>/telemetry ─┼─► IngestionPipeline ─► TelemetryStore ─► PostgreSQL (SQLite dev fallback)
 Simulator (thread, SIM- devices) ─┘        │   validate · normalise · de-duplicate · order
                                             ├─► latest-value cache ─► GET /api/telemetry/latest
                                             ├─► AlertEngine (thresholds.yaml) ─► alerts
                                             └─► TelemetryHub ─► WS /ws/telemetry ─► dashboard (one shared socket)
 Watchdog thread: sensor online/stale/offline, communication alerts, retention purge
```

- **Equipment identity** lives in the telemetry database (`equipment` table, stable operator IDs). Visual tracks
  stay session-scoped. A track can be linked to equipment only by an explicit operator action
  (`equipment_visual_links`), never automatically.
- Telemetry starts after the vision service and is isolated from it: a database failure disables only the telemetry
  endpoints.
- Details: [sensor-integration.md](sensor-integration.md).

## Analytics (Milestone 3)

```
IngestionPipeline ─(listener: new in-order good readings)─► AnalyticsEngine
   1-min buckets per (equipment, sensor, metric) ─► detectors ─► context ─► anomaly episodes ─► anomalies table
AnalyticsService worker ─► health (30 s) ─► snapshots (5 min) · maintenance trends (15 min) ─► recommendations
events: WS /ws/telemetry {"type": "anomaly" | "health" | "recommendation"}
```

Training (`app.analytics.train`) and validation (`app.analytics.validate`) are offline commands that reuse the same
feature code. Details: [anomaly-detection.md](anomaly-detection.md).
