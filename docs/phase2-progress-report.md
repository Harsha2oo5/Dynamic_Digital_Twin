# Phase 2 progress report: custom 7-class detector

Last updated: 2026-11-02. Every statement below was checked against the repository on that date.

Legend: **Implemented** = code exists · **Executed** = it was run on real data · **Validated** = output checked by
tests or by inspection · **Human-reviewed** = checked by a person · **Pending** = not done.

## State recovered at the start of this session

| Question | Answer (evidence) |
|---|---|
| Automated annotation model selected? | Yes, prompt-free YOLOE-11L (`backend/models/annotation_assist/yoloe-11l-seg-pf.pt`) |
| Installed and run? | Yes, on all 175 `batch_01` images |
| Candidates / previews generated? | Yes, then **rejected**: own class proposed in 10/175 images (`annotation/rejected/yoloe_pf_batch01_eval/REJECTED.md`) |
| Text-prompt YOLOE / YOLO-World? | Not run: needs the `clip` tokenizer from GitHub; the install was blocked pending approval |
| Human-reviewed annotations? | **None** (`annotation/exports/` empty) |
| Valid YOLO dataset? | **No** (`backend/dataset/` holds only `.gitkeep`) |
| Training run? | **No** (`experiments/` holds only `dataset_audit/`; `backend/models/custom/` is empty) |
| Live app model | pretrained COCO YOLO11n (`DETECTION_MODEL=pretrained`) |

## Done in this session

| Item | Status |
|---|---|
| Grounding DINO proposer (`--mode gdino`, `transformers` 4.57 from PyPI into `backend/.venv` only; model `IDEA-Research/grounding-dino-tiny`, cached in the user's Hugging Face cache, not in the repo) | Implemented, executed |
| Letterbox crop for proposers (screenshots are 65–82% bars; ~18 s → ~7 s per image) with boxes mapped back to staged-image pixels | Implemented, executed, unit-tested |
| One-class-per-box scoring for Grounding DINO plus a class-agnostic merge with `alt_class:` flags | Implemented, executed, unit-tested |
| Per-class contact sheets, confidence histogram and multi-box count in the candidate report | Implemented, executed, tested |
| Candidates for `batch_01` → `annotation/candidates/gdino_tiny_batch01/` | Executed (see the numbers below); **0 human-reviewed** |
| `batch_02` staged (168 images: 25 per class, cable 18; no near-duplicate group shared with batch_01) | Executed |
| Annotation guide updated (Grounding DINO workflow and findings) | Done |

### Grounding DINO results on batch_01

Measured 2026-11-02 (`annotation/candidates/gdino_tiny_batch01/report.md`): 175 images in 1631 s on CPU
(~9 s/image while sharing the CPU with the test suite).

- Candidate boxes: **553**. Images with 0 boxes: **0**. Images with more than one box: **138**. Multiple-class images: **129**.
- Image status: 171 `needs_review`, 4 `high_confidence` (all in the radio-unit folder, where the model's
  "radio unit" bias happens to agree). **Images requiring human review: all 175** (high confidence is not verification).
- 112 boxes carry `alt_class:` flags (another class scored on the same object).
- Detector score histogram (not accuracy): 0.2–0.3: 176, 0.3–0.4: 193, 0.4–0.5: 98, 0.5–0.6: 32, 0.6–0.7: 37, 0.7–0.8: 16, 0.8–0.9: 1.

| Folder | Images | Folder's class proposed | Boxes per proposed class (whole batch) |
|---|---|---|---|
| antenna | 25 | 9 | antenna 43 |
| router | 25 | **0** | router **1** |
| cable | 25 | 14 | cable 98 |
| battery | 25 | 11 | battery 61 |
| power_supply_unit | 25 | **1** | power_supply_unit 37 |
| equipment_cabinet | 25 | 18 | equipment_cabinet 84 |
| radio_unit | 25 | 25 | radio_unit **229** (bias: most equipment is called a radio unit) |

For comparison, prompt-free YOLOE proposed the folder's class in 10/175 images and gave 116 images no box at all.
Grounding DINO proposed it in 78/175, but that figure is inflated by the radio-unit bias. Router and PSU are effectively never named.

**Interpretation.** On inspection, boxes usually sit on the main objects, but the class assignment is unreliable:
most equipment is labelled with whichever device word dominates the prompt. These candidates are a **box
pre-annotation aid only**; the reviewer must set every label. Detector confidence is not accuracy.

## Human-in-the-loop review workflow (2026-11-02, second session)

| Item | Status |
|---|---|
| `review_store.py`: statuses NOT_REVIEWED / IN_PROGRESS / REVIEWED / NEEDS_SECOND_REVIEW, box provenance, sign-off rules, audit log, reviewed-only export with `review_manifest.json` | Implemented, tested |
| `review_tool.py` + `review_ui.html`: CLI and local browser editor (FastAPI, which the backend already uses; no new dependencies) | Implemented, API tested; UI rendering checked with a headless-browser screenshot. Mouse editing **not yet exercised by a person** |
| CVAT archives of `gdino_tiny_batch01` checked: 7 archives, 175 images, 553 lines match `candidates.json`; `candidate_metadata.json` keeps confidence and flags | Executed |
| `annotation/reviewed/batch_01/` created: 175 images NOT_REVIEWED, 553 unverified candidate boxes | Executed; **0 reviewed** |
| Validator: `--require-review` (manifest, REVIEWED status, label hash), counts of annotations / invalid / ready / requiring review | Implemented, tested |
| `prepare_dataset`: review manifest required; refuses undersized or unrepresentative splits | Implemented, tested |
| Backend tests | **104 passed** |

## Custom-model training task (2026-11-02, third session)

Readiness audit (`python -m app.training.readiness --write`): **BLOCKED**. There are 543 original images and no
labels in the dataset. Batch 1 has 175 images with 553 unverified candidates, batch 2 has 168 images, and
**0 images are human-reviewed** (the batch 1 review log holds only the `init` entry). There are no exports,
no prepared dataset, no custom checkpoint and no training runs. Training was **not** run.

Prepared so that training can start as soon as reviews exist:
- `readiness.py`: a reproducible readiness report and pilot gate (tested).
- Training gate: it now refuses datasets without a human-reviewed origin (tested).
- `evaluate.py`: confusion analysis per class and for the look-alike pairs (tested).
- `annotation/reviewed/batch_02/` created (168 images, no candidate boxes).
- `docs/bts-field-testing.md`: the real-site test protocol.
- Backend tests: 112 passed. Frontend build: passes.

## Not done (and why)

| Item | Status | Reason |
|---|---|---|
| Human review of candidates | Pending | needs a person in CVAT |
| Annotation validation on a real export | Pending | no export exists |
| `prepare_dataset` (70/20/10, group-aware) | Pending | blocked on validated labels |
| 5-epoch pilot / evaluation | **Not run** | the policy forbids training on unreviewed candidates |
| Switching to `DETECTION_MODEL=custom` | Not done | no validated checkpoint; the app stays in pretrained mode |

## Tests

- Backend: `pytest`: **93 passed** (includes 3 new tests for crop/merge and new assertions on contact sheets and the histogram).
- `annotation_kit --check`: **KIT OK**.
- Frontend `npm run build`: **passes** (existing chunk-size warning only).
- Original dataset: `git status` shows **0 changes** under `digital twin project datasets/`.

## Next steps (exact commands, from `backend/`)

1. Look at `annotation/candidates/gdino_tiny_batch01/contact_sheets/*.jpg` and `review.html`.
2. In CVAT (see `docs/annotation-guide.md`): create the project and the 7 `b01_*` tasks, then for each task decide whether to
   upload `cvat_import/batch_01_<task>.zip` (*Actions → Upload annotations → YOLO 1.1*) or draw from scratch.
   **Fix every label**, add missing boxes, delete false ones.
3. Export YOLO 1.1 into `annotation/exports/<date>_batch01_cvat/`, then run:
   ```powershell
   .venv\Scripts\python.exe -m app.training.validate_annotations ..\annotation\exports\<date>_batch01_cvat
   .venv\Scripts\python.exe -m app.training.prepare_dataset --labels ..\annotation\exports\<date>_batch01_cvat --seed 42
   .venv\Scripts\python.exe -m app.training.train --data dataset\digital_twin_yolo\data.yaml --pilot
   .venv\Scripts\python.exe -m app.training.evaluate --model experiments\training\<run>\weights\best.pt --data dataset\digital_twin_yolo\data.yaml --split test
   ```
4. Then use the pilot model for `batch_02` candidates (`--mode custom`), and review again.
