# Equipment tracking, session identity and inventory

This doc covers how a detection becomes an inventory item: multi-object tracking, the track lifecycle, class
voting, session-level equipment identity with re-linking, the inventory registry, model management and the APIs.

> **IDs are session-level, not permanent.** An equipment ID such as `RU-003@a1b2c3d4` is stable while one item is
> re-identified within **one** stream or video. The same physical item seen in another session gets another ID.
> A permanent, site-wide component registry is a later milestone.

## Pipeline per frame

```
frame ─► Detector (YOLO, CPU) ─► ObjectTracker (ByteTrack / BoT-SORT) ─► TrackManager ─► EquipmentIdentity
            raw boxes               tracker IDs                          lifecycle,         session equipment ID,
                                                                         class vote,        re-link of lost items
                                                                         smoothed conf.
                                         └──────────────► Detection fields ─► ObjectRegistry (inventory)
```

Code:
- `app/vision/tracker.py`: tracker wrapper;
- `app/vision/track_manager.py`: `TrackManager` and `TrackingSession`;
- `app/identification/equipment_identity.py`: identity and re-linking;
- `app/services/object_registry.py`: inventory;
- `app/services/model_manager.py`: model versions and switching.

Every stream (WebSocket connection) and every video job owns one `TrackingSession`; sessions never share state.

## Tracker configuration

`TRACKER_CONFIG` (default `configs/trackers/bts_bytetrack.yaml`):

| Parameter | Ultralytics default | BTS value | Why |
|---|---|---|---|
| `track_high_thresh` | 0.25 | 0.30 | equipment detections are mostly confident; keeps clutter out of the first association stage |
| `new_track_thresh` | 0.25 | 0.35 | fewer spurious tracks from one-frame false positives |
| `track_buffer` | 30 | 90 | at the 30 fps reference this is about 3 s; equipment often leaves the frame briefly while the phone pans |
| `match_thresh` | 0.80 | 0.85 | CPU frame rates (3–15 fps) mean larger box movement between frames |

**Buffers are scaled to time.** `track_buffer` is defined in frames at 30 fps, and `ObjectTracker` converts it to
frames at the session's real processed rate:
- videos: `video_fps / frame_skip`;
- live streams: `STREAM_FRAME_RATE`, default 8.

A lost track therefore survives about 3 s whatever the frame skip or CPU speed.

`configs/trackers/bts_botsort.yaml` is BoT-SORT with **camera-motion compensation** (sparse optical flow), for
handheld footage. It has no ReID model (`with_reid: false`); appearance re-linking is done by the identity layer.
Select it with `TRACKER_CONFIG=configs/trackers/bts_botsort.yaml`.

## Track lifecycle

| State | Meaning |
|---|---|
| `new` | seen in fewer than `TRACK_MIN_HITS` (default 3) frames. Tentative: drawn dashed, **not** put in the inventory, no equipment ID |
| `active` | confirmed and visible in the current frame |
| `lost` | confirmed, not visible, still inside the tracker buffer. Dashed in the overlay |
| `removed` | not seen for longer than the buffer; the item stays re-linkable for 15 s |

The lifecycle clock is **video time** for uploaded videos (frame index / fps), so timeouts mean the same on a fast
or a slow CPU. Live streams use wall time.

## Class voting and smoothed confidence

- **Stable class:** the track's class is a confidence-weighted vote over all its detections. This smooths
  per-frame flicker between look-alike classes, such as radio unit vs antenna.
- **Class consistency:** the winning class's share of the votes, from 0 to 1. The inventory highlights items
  below 0.8.
- **Raw class is kept:** `class_name` on a detection remains the raw per-frame class. `stable_class` is
  additional, and the overlay marks a differing vote with `*`.
- **Smoothed confidence:** an exponential moving average with α = 0.3.

## Session-level equipment identity and re-linking

When a track is confirmed it gets an equipment ID `<CLASS>-<NNN>@<session>`. The class abbreviations are ANT, RTR,
CBL, BAT, PSU, CAB and RU.

**When a re-link happens.** The tracker issues a new ID when an item is lost for longer than its buffer, or when
association fails (a fast pan or a brief occlusion). Without a re-link, that would create a second inventory entry
for the same item. A newly confirmed track is re-linked to a recently lost item when:
- both have the **same stable class**;
- the item was lost **within 15 s**;
- the appearance and position match, in one of two ways:
  - **close match:** the box centre is within 0.35 × the frame diagonal **and** the colour histograms
    (HSV hue/saturation, histogram intersection) are at least 0.55 similar;
  - **strong appearance alone:** a histogram similarity of at least 0.8 is enough at any position, which covers
    camera pans.

The best-scoring candidate wins. Each re-link is counted in the item's `relinked` count and the session's counter.

**Known limitations (by design of a cheap, CPU-only identity):**
- Two identical-looking items that are never visible at the same time can be merged into one.
- An item seen from a very different angle or under different lighting can get a second ID.
- Colour histograms carry no fine detail; a learned ReID embedding would be the next step if this proves
  insufficient on real footage.
- **Only checked on synthetic sequences so far.** The dataset contains still images only, and no BTS site
  footage is available yet.

`PersistentEquipmentRegistry` is the interface a future permanent registry will implement. Only an in-memory
stand-in exists, and it never issues permanent IDs.

## Inventory

The registry key is `source_id:equipment_id`, so one physical item in one session is one row, even across
tracker-ID changes; its tracker IDs accumulate in `track_ids`.
- Tentative tracks are not registered.
- Results without lifecycle tracking keep the old key `source_id:track_id`.
- Image uploads are snapshots, as before.

Each row records:
- first and last seen, duration and observation count;
- last box, last / average / maximum confidence;
- class consistency, lifecycle state and re-link count.

## Model management

- **Versioned checkpoints:** `train --promote-versioned` copies `best.pt` to `models/custom/digital_twin_vN.pt`,
  never overwriting an existing file, and writes `digital_twin_vN.model_card.json` next to it.
- **Model card contents:** version, sha256, label source, pseudo-label run, dataset sizes, training run,
  validation metrics with what they mean, and limitations.
- **Runtime switching:** `POST /api/model/select {"model_id": "custom/digital_twin_v2.pt"}` loads the model in a
  worker thread and swaps it in without a restart.
  - **On failure** (missing, corrupt, or not a detection model) the running model stays active and a 422 explains
    why.
  - **Running video jobs** finish with the model they started with.
  - **The choice is runtime state:** after a restart, `DETECTION_MODEL` / `MODEL_PATH` apply again.
- **Model ids:** paths outside `models/` are refused.

## API (all additive; existing clients keep working)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/objects?status=&class=&session=` | inventory with filters (unfiltered call unchanged) |
| GET | `/api/inventory/summary` | unique equipment items per class, active now |
| GET | `/api/sessions` | sessions (streams / videos / images) with item counts and live lifecycle counters |
| GET | `/api/sessions/{id}/tracks` | every item of one session, including lost / removed ones |
| GET | `/api/model/versions` | checkpoints with version, label source and model card |
| POST | `/api/model/select` | switch the model at runtime |
| GET | `/api/model/performance` | measured runtime latency of the active model plus its model-card metrics |

New `Detection` fields: `track_state`, `stable_class`, `class_consistency`, `smoothed_confidence`, `track_age_s`
and `equipment_id`. New `TrackedObject` fields: `equipment_id`, `track_ids`, `track_state`, `class_consistency`,
`duration_s` and `relinked`.

## Site cameras (preparation only)

`app/camera/frame_source.py` defines the `FrameSource` interface that a server-pulled RTSP camera will implement.
`RtspSource` raises `NotImplementedError` until it is built and tested against a real camera. Camera URLs can
contain credentials, so they must come from the environment and never be logged.

## Tests

`tests/test_vision_lifecycle.py` (17 tests) covers:
- the lifecycle (`new` → `active` → `lost` → `removed`), and that tentative tracks never get an ID;
- one moving item keeping one ID over 30 frames;
- class voting under flicker;
- session isolation;
- re-linking after a forced tracker-ID change, with no re-link for another class, a far and dissimilar item, or
  outside the window;
- time-scaled buffers, and that BoT-SORT works with camera-motion compensation;
- inventory de-duplication and filters;
- model versions, selection, path-traversal and broken-checkpoint refusal;
- the pseudo-label training path;
- a full video job through the API (states, one equipment ID for the bus, session tracks, inventory summary).
