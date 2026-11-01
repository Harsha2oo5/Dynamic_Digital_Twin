# Dynamic Digital Twin — Vision Console (Milestone 1)

A CPU-only computer-vision application that detects objects in images, videos and a laptop webcam with
YOLO11n, tracks them across frames with ByteTrack, and shows the results in a React dashboard.

| Capability | Status |
|---|---|
| Image / video / webcam ingestion (OpenCV, browser camera) | ✅ |
| Object detection (pretrained YOLO11n, CPU) | ✅ |
| Bounding boxes, class names, confidence | ✅ |
| Multi-object tracking (ByteTrack, per-session state) | ✅ |
| Overview dashboard: backend / engine / source / model health, latency window, component distribution | ✅ |
| Annotation review tool (browser, human sign-off, audit log, reviewed-only export) | ✅ |
| Custom 7-class component model (fine-tuned YOLO) | 🧩 pipeline, gates and model switch ready; **blocked on human review of the annotation batches** (0 / 343 images reviewed) |
| Sensors, telemetry, anomaly detection, 3D | ⏳ later milestones |

> **Helping with annotation review?** Skip to [Helping with annotation review](#helping-with-annotation-review).
> It takes about 15 minutes to set up, and no ML experience is needed.

**No GPU, CUDA or NVIDIA drivers are needed.** PyTorch is installed from the CPU wheel index, and the backend
never selects CUDA, even when a GPU is present (`backend/app/core/device.py`).

---

## Helping with annotation review

We are training our own YOLO model to recognise seven kinds of telecom base-station (BTS) equipment. A model
already drew *candidate* boxes, but many have the wrong class (it calls most equipment "radio unit"). A person
has to check every image. That's the job: open an image in the browser, fix the boxes, and mark it reviewed.

### One-time setup (Windows, about 15 minutes)

You need **Git**, **Python 3.12** ([python.org](https://www.python.org/downloads/), tick *Add to PATH*), and about 3 GB of disk.

```powershell
git clone https://github.com/Harsha2oo5/Dynamic_Digital_Twin.git
cd Dynamic_Digital_Twin\backend
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt

# the images to review are copies of the dataset; re-create them after cloning
.venv\Scripts\python.exe -m app.training.annotation_batch --name batch_01 --restage
.venv\Scripts\python.exe -m app.training.annotation_batch --name batch_02 --restage
```

### Who reviews what

**Only one person works on a batch at a time.** The review state of a batch is one file
(`annotation/reviewed/<batch>/review_state.json`), and two people editing it at once will conflict in Git.

| Batch | Images | Starts with | Reviewer |
|---|---|---|---|
| `batch_01` | 175 (25 per class) | machine candidate boxes to correct | *(assign)* |
| `batch_02` | 168 | no boxes: draw them | *(assign)* |

### Each review session

```powershell
cd Dynamic_Digital_Twin
git pull                                           # get everyone's latest reviews first
cd backend
.venv\Scripts\python.exe -m app.training.review_tool serve --review ..\annotation\reviewed\batch_01
# open http://127.0.0.1:8765, type your name in "Reviewer", review; Ctrl+C in the terminal when done
```

Every change is saved immediately. Then share your work:

```powershell
cd ..
git add annotation/reviewed
git commit -m "Review batch_01: <how many> images"
git push
```

### How to review an image

1. **Dashed boxes are the model's guesses.** For each one: if it's right, press **Accept** (`A`). If the class is
   wrong, pick the right one from the dropdown (keys `1`–`7`). If the box is loose, drag its corners. If there's
   no such object, delete it (**✕** / `Del`).
2. **Add what the model missed:** choose the class at the top (*New box class*) and drag on empty space.
3. **Box every one of the seven components you can see,** not only the one the folder is named after. An open
   cabinet usually also contains batteries, PSUs and cables.
4. **Cables:** one box around a bundle you can't tell apart; separate boxes only for clearly separate cables.
5. **Not sure?** Mark the box with **?** (`U`) and press **Needs second review**, with a note. Don't guess.
6. When the image is complete, press **Mark REVIEWED** and tick the checklist. An image with nothing to box can be
   marked reviewed with no boxes.

| Key | Class | What it looks like |
|---|---|---|
| 1 | antenna | panel or mast antennas (usually white or grey, flat and tall) |
| 2 | router | transmission / router chassis with ports and line cards (e.g. Huawei ATN) |
| 3 | cable | feeders, jumpers, power and fibre cables |
| 4 | battery | battery blocks or modules |
| 5 | power_supply_unit | rectifier / PSU shelves and modules |
| 6 | equipment_cabinet | the whole indoor or outdoor cabinet enclosure |
| 7 | radio_unit | RRU / radio units (finned metal boxes with connectors at the bottom) |

The full rules are in [docs/annotation-guide.md](docs/annotation-guide.md#annotation-rules). Expect about 1.5–3
minutes per image. Progress at any time: `.venv\Scripts\python.exe -m app.training.review_tool status --review ..\annotation\reviewed\batch_01`.

---

## Repository layout

```
dynamic_digital_twin/
├── backend/                  FastAPI + YOLO + ByteTrack (Python 3.12)
│   ├── app/
│   │   ├── core/             config (.env), CPU device policy, logging
│   │   ├── camera/           OpenCV ingestion: images, video files, webcams (RTSP-ready)
│   │   ├── vision/           preprocessing, detector, tracker, processor, visualization
│   │   ├── identification/   component classification layer (active for custom models)
│   │   ├── services/         detection service, stats, object registry, video jobs
│   │   ├── api/              REST routes + WebSocket
│   │   ├── schemas/          Pydantic models (the JSON contract)
│   │   └── training/         taxonomy, audit, annotation kit/validation, prepare, train, evaluate, predict
│   ├── models/pretrained/    yolo11n.pt
│   ├── models/custom/        digital_twin_best.pt goes here after fine-tuning
│   ├── dataset/              generated YOLO datasets (train/val/test)
│   ├── tests/                pytest suite
│   └── requirements.txt
├── frontend/                 React 19 + TypeScript + Vite + Tailwind 4 + Recharts
├── digital twin project datasets/   collected images (READ-ONLY, never modified)
├── annotation/               annotation kit (CVAT/Label Studio configs, manifest); exports/ for YOLO exports
├── docs/                     architecture, dataset audit, annotation guide, training guide
└── experiments/              training/evaluation runs (created by the scripts)
```

---

## 1. Setup

Prerequisites: **Python 3.12** (3.13 is not used here, to stay on well-supported wheels), **Node.js 22**.

### Backend

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# CPU-only PyTorch FIRST, so nothing pulls a CUDA build
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt

copy .env.example .env     # optional: adjust settings
```

Verify:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.x.x+cpu False
```

`models/pretrained/yolo11n.pt` is included. If it is missing, download it:
`.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` and move it to that folder.

### Frontend

```powershell
cd frontend
npm.cmd install        # npm.cmd avoids PowerShell's npm.ps1 execution-policy block
```

## 2. Run

Use two terminals:

```powershell
# terminal 1 — API on http://127.0.0.1:8000  (docs at /docs)
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2 — dashboard on http://localhost:5173
cd frontend
npm.cmd run dev
```

The Vite dev server proxies `/api` and `/ws` to the backend, so there's no CORS setup to do. To point the UI at another
backend, set `BACKEND_URL` for the dev proxy, or `VITE_API_BASE` for a production build.

## 3. Using the dashboard

- **Overview**: health strip (backend online/offline, detection engine running/idle/error, camera or video
  source processing/connected/disconnected, model loaded/failed), model information, eight metrics (detections,
  frames, current and unique tracks, FPS, latency, confidence, processing errors), a latency chart over the last
  120 frames with avg/min/max, and the component distribution. The distribution counts the seven components only
  when a matching custom model is loaded; with the pretrained model it shows its COCO classes separately.
  Every number comes from real inference, and a dash (—) is shown until data exists.
- **Live Monitoring**
  - *Image*: upload a JPG/PNG. Boxes are drawn over the original image. You can switch to the
    server-annotated JPEG.
  - *Video file*: upload MP4/AVI/MOV/MKV/WEBM and press **Run detection**. The backend processes it
    frame-by-frame with tracking as a background job, and the UI shows progress. Afterwards the video plays
    with the overlay synced to the playback position (seek and pause work). If your browser can't play a codec
    (e.g. AVI, or OpenCV `mp4v`), use **Annotated frames** to step through server-rendered frames, or
    download the annotated MP4.
  - *Webcam*: **Start** captures the laptop camera in the browser and streams JPEG frames over the WebSocket.
    Only one frame is in flight at a time, so the CPU never builds a backlog. Threshold changes apply live, and
    **Reset IDs** restarts tracking.
  - Controls: confidence, IoU (NMS), inference resolution, frame skip (video), tracking on/off (webcam).
- **Object Inventory**: every tracked object, with ID, class, confidence, first/last seen, frames seen
  and status (active / lost / snapshot). Click a row for details and its last position.
- **Model Management**: the loaded model, its type, checkpoint, device, thresholds and load status, plus
  the available checkpoints and the model's class list.
- **Dataset Management**: a read-only scan of the collected dataset (counts, formats, resolutions,
  annotation status, issues).
- **Settings**: client defaults (stored in the browser) and the backend's runtime configuration.

> **Tracking IDs are temporary.** They identify a visual track within one stream or video only. They are not
> permanent physical component IDs; the object ID shown in the inventory is `source:track_id`.

## 4. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model loaded, device, torch version |
| GET | `/api/model/info` | model, type, classes, thresholds, load status |
| GET | `/api/models` | checkpoints in `backend/models/` |
| GET | `/api/config` | effective runtime configuration |
| POST | `/api/detect/image` | multipart `file` (+ `confidence`, `iou`, `image_size`, `annotate`) → detections |
| POST | `/api/detect/video` | multipart `file` (+ `frame_skip`, …) → 202 + job |
| GET | `/api/detect/video/{id}` | job status/progress |
| GET | `/api/detect/video/{id}/results` | per-frame detections |
| GET | `/api/detect/video/{id}/annotated` | annotated MP4 |
| GET | `/api/detect/video/{id}/frames/{n}` | annotated JPEG of the n-th processed frame |
| POST | `/api/detect/video/{id}/cancel` | cancel a job |
| GET | `/api/detection/stats` | aggregate statistics |
| GET / DELETE | `/api/objects`, `/api/objects/{id}` | object inventory |
| GET | `/api/dataset/summary?refresh=` | dataset validation report |
| WS | `/ws/stream` | binary JPEG frames in → `FrameResult` JSON out |

Frame result schema (shared by REST and WebSocket). Box coordinates are original-frame pixels:

```json
{
  "frame_id": 30, "timestamp": "2026-11-01T00:37:41.2", "device": "cpu",
  "fps": 16.8, "inference_ms": 59.4, "frame_width": 640, "frame_height": 480,
  "detections": [
    { "track_id": 1, "class_id": 5, "class_name": "bus", "confidence": 0.95,
      "bbox": { "x1": 2, "y1": 13, "x2": 639, "y2": 465 }, "component_type": null }
  ]
}
```

WebSocket control messages: `{"type":"config","confidence":0.4,"iou":0.45,"image_size":416,"track":true}`
and `{"type":"reset"}`.

## 5. CPU configuration and performance

All settings live in `backend/.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `DEVICE` | `cpu` | only `cpu` is accepted; other values log a warning and fall back to CPU |
| `DETECTION_MODEL` | `pretrained` | `pretrained` (COCO YOLO11n) or `custom` (fine-tuned 7-class model) |
| `PRETRAINED_MODEL_PATH` / `CUSTOM_MODEL_PATH` | `models/pretrained/yolo11n.pt` / `models/custom/digital_twin_best.pt` | checkpoint for each choice |
| `MODEL_FALLBACK_TO_PRETRAINED` | `false` | if `custom` can't load: `false` = degraded with an error; `true` = run pretrained and flag `fallback_active` |
| `MODEL_PATH` | unset | explicit checkpoint override |
| `CONFIDENCE_THRESHOLD` / `IOU_THRESHOLD` | 0.35 / 0.45 | request-level overrides supported |
| `IMAGE_SIZE` | 416 | inference resolution; 320 is faster, 640 is more accurate |
| `FRAME_SKIP` | 1 | process every Nth video frame |
| `TORCH_THREADS` | 0 | 0 = PyTorch default |
| `MAX_FRAME_DIM` | 1920 | larger frames are downscaled before inference; boxes are mapped back |
| `MAX_VIDEO_MB` / `MAX_VIDEO_FRAMES` | 200 / 5000 | upload and processing caps |

How the backend stays responsive on CPU:
- the model loads **once** at startup, and a warm-up pass absorbs the slow first call;
- all inference runs in worker threads, never on the event loop;
- a lock serialises calls into the shared model (Ultralytics predictors aren't thread-safe);
- video jobs run on a single worker thread;
- each stream or job has its **own ByteTrack instance**, so concurrent sessions can't corrupt each other's tracks;
- skipped video frames are `grab()`bed but not decoded.

**Measured on this development machine** (Intel i7-1255U, 16 GB RAM, YOLO11n, `IMAGE_SIZE=416`):

| Workload | Measured |
|---|---|
| Model load + warm-up | ~2.7–3.4 s |
| First inference without warm-up | ~3.8 s |
| Single image, steady state | ~40–110 ms |
| 640×480 video, every frame, with tracking | ~10–16 FPS inference |
| Browser webcam at 640 px over WebSocket | ~14 FPS end-to-end, ~17 FPS inference |

Your numbers will differ. The dashboard always shows the live measured values. If throughput is too low,
try `IMAGE_SIZE=320`, `FRAME_SKIP=2`, or a 480 px webcam send width (Settings page).

## 6. Tests

```powershell
cd backend
.venv\Scripts\python.exe -m pytest
```

There are 113 tests:
- annotation review (`test_review_workflow.py`, `test_auto_annotate.py`): candidate import with class/coordinate/confidence
  preservation, CVAT archive checks, status transitions and sign-off rules, reviewed-only export, review-manifest
  validation (unreviewed or edited labels are caught), the training gate, split safeguards, YOLO import round trip,
  the review API, and immutability of the dataset and batches;
- dashboard statistics (`test_stats.py`): latency window, class ids, engine and source states, error handling,
  and the additive API fields; readiness gate (`test_readiness.py`);
- unit tests: CPU device policy, preprocessing, box scaling/clipping, video frame extraction, detection
  parsing, ByteTrack ID persistence and session isolation, object registry, dataset validation;
- integration tests against the **real YOLO11n model on CPU**: health, model info, image detection,
  threshold handling, error codes (400/413/415/422/503), a full video job with tracking and annotated output,
  the WebSocket stream, and degraded mode when the model file is missing;
- training pipeline (`test_training_pipeline.py`): class mapping, audit (counts, corruption, near-duplicates),
  annotation-kit configs, annotation validation and YOLO label parsing, group-aware deterministic splitting,
  `data.yaml` generation, end-to-end preparation, training configuration and gate, checkpoint promotion;
- custom model (`test_custom_model.py`): checkpoint selection, loading a 7-class checkpoint, class-name
  mapping, REST/WebSocket/inventory compatibility, the missing-checkpoint error and the fallback.

The training and custom-model tests use synthetic fixtures in temporary folders: generated images, and a
**randomly initialised** 7-class checkpoint used only to exercise code paths. The application itself never
fabricates detections or metrics.

## 7. Custom component model (7 classes)

The classes are antenna, router, cable, battery, power_supply_unit, equipment_cabinet and radio_unit; the
mapping is in `backend/app/training/taxonomy.py`. The collected dataset has **no human-reviewed bounding boxes
yet**, so training is blocked until they exist. Machine candidates for batch 1 are loaded into a local review tool:

```powershell
cd backend
.venv\Scripts\python.exe -m app.training.review_tool serve --review ..\annotation\reviewed\batch_01   # http://127.0.0.1:8765
```

Only images a person marks REVIEWED are exported. `prepare_dataset` and the training gate accept nothing else.
Check whether training may start with `python -m app.training.readiness` (from `backend/`); it reports BLOCKED
with the exact reasons until enough reviewed boxes exist.

- [docs/phase2-progress-report.md](docs/phase2-progress-report.md): current status of the custom-model work.
- [docs/bts-field-testing.md](docs/bts-field-testing.md): how the model will be tested on real BTS sites.

- [docs/dataset-audit.md](docs/dataset-audit.md): audit of the 543 images.
- [docs/annotation-guide.md](docs/annotation-guide.md): how to annotate in CVAT or Label Studio using the kit in `annotation/`.
- [docs/custom-model-training.md](docs/custom-model-training.md): validate → prepare → pilot → train → evaluate → `DETECTION_MODEL=custom`.

## 8. Future GPU migration (out of scope now)

1. Install a CUDA build of PyTorch in a separate environment.
2. Add `"cuda"` to `SUPPORTED_DEVICES` in `backend/app/core/device.py`.
3. Set `DEVICE=cuda`.

No other code changes are required.
