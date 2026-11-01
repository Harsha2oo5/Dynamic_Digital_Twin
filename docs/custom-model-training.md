# Custom YOLO fine-tuning: Digital Twin components (CPU)

Commands run from `backend/` with `.venv\Scripts\python.exe` (written as `python` below).
All scripts are in `backend/app/training/` and are run as modules (`python -m app.training.<script>`).

| Script | Purpose |
|---|---|
| `taxonomy.py` | the seven classes and the folder → class mapping (single source of truth) |
| `audit_dataset.py` | read-only audit: counts, formats, sizes, corruption, duplicates and near-duplicates, annotation search |
| `annotation_kit.py` | CVAT / Label Studio configs, `classes.txt`, image manifest; `--check` verifies the kit |
| `annotation_batch.py` | selects and stages a diverse, near-duplicate-free annotation batch per class |
| `auto_annotate.py` | **candidate** boxes for review (detector → taxonomy mapping → rules → tiers), CVAT pre-annotation zips, previews, `review.html`; output refused as training data |
| `review_store.py` | human review state per batch: statuses, box provenance, sign-off rules, reviewed-only export |
| `review_tool.py` + `review_ui.html` | CLI (`init`, `serve`, `status`, `export`, `import-yolo`, `check-cvat`) and the local browser review UI |
| `validate_annotations.py` | validates a YOLO export against the taxonomy and the manifest (`--require-review`: review manifest and label hashes) |
| `prepare_dataset.py` | duplicate-aware, stratified 70/20/10 split, writes `data.yaml` and reports |
| `train.py` | workload estimate, pilot run, full run, checkpoint promotion |
| `evaluate.py` | test-set metrics, per-class table, plots, prediction images, look-alike confusion pairs |
| `readiness.py` | live readiness report: original images, candidates, review progress, exports, datasets, checkpoints, and the pilot gate |
| `predict.py` | annotated predictions and optional pre-annotation proposals |

## 1. Dataset and class mapping

`digital twin project datasets/` is **read-only**. No script writes to it.

| ID | Class | Folder | Images |
|---|---|---|---|
| 0 | antenna | `antenna/antenna` | 100 |
| 1 | router | `Archive (7)` | 74 |
| 2 | cable | `Archive (6)` | 44 |
| 3 | battery | `Archive (5)` | 62 |
| 4 | power_supply_unit | `Archive (4)` | 93 |
| 5 | equipment_cabinet | `Equipment Cabinet/Equipment Cabinet` | 70 |
| 6 | radio_unit | `radio unit/radio unit` | 100 |
| | **total** | | **543** |

The full audit is in [dataset-audit.md](dataset-audit.md). Re-run it with `python -m app.training.audit_dataset`;
it writes `experiments/dataset_audit/`.

**Annotation status (2026-11-02): 0 human-reviewed images.** `batch_01` (175 images) has 553 Grounding DINO
candidate boxes loaded as unverified in `annotation/reviewed/batch_01/`. Training is blocked until reviewed labels
exist; see [annotation-guide.md](annotation-guide.md).

## 2. Annotate → validate

Only **human-reviewed** labels can reach training. Machine candidates (`annotation/candidates/`) are review aids.
The full workflow is in [annotation-guide.md](annotation-guide.md#review-workflow-from-candidates-to-training-ready-labels).

```powershell
python -m app.training.review_tool init --batch ..\annotation\batches\batch_01 --candidates ..\annotation\candidates\gdino_tiny_batch01   # done for batch_01
python -m app.training.review_tool serve --review ..\annotation\reviewed\batch_01     # review in the browser, mark images REVIEWED
python -m app.training.review_tool export --review ..\annotation\reviewed\batch_01 --out ..\annotation\exports\<export>
python -m app.training.validate_annotations ..\annotation\exports\<export> --require-review
```

`export` writes labels for `REVIEWED` images only, plus `review_manifest.json` (reviewer and label hash per image).
`NOT_REVIEWED`, `IN_PROGRESS` and `NEEDS_SECOND_REVIEW` images are counted but never exported. Corrections made in
CVAT come back through `review_tool import-yolo` and still need the sign-off.

## 3. Prepare the dataset

```powershell
python -m app.training.prepare_dataset --labels ..\annotation\exports\<export> --seed 42
```

What it does:
- Validates the export first, **requiring the review manifest**, and aborts on any error. A raw CVAT/Label Studio
  export or a candidate run is refused.
- Refuses to force a split when the reviewed set is too small: fewer than 30 images, a class with fewer than
  10 boxes, an empty split, or a class with no box in train or val. It reports the limitation and writes
  nothing. A class missing from the small test split is only reported (`limitations` in `split_report.json`).
- Splits **by near-duplicate group**: re-captures of the same photo, found by a 256-bit content hash in the
  audit, always land in the same split. This prevents evaluation leakage. The audit found 39 such groups covering 84 images.
- Stratifies by folder class, 70/20/10, with fixed seed 42, so the result is reproducible.
- **Copies** images (from the original dataset) and labels into `backend/dataset/digital_twin_yolo/`, using
  space-free file names. It never overwrites an existing output directory.
- Writes `data.yaml` (the seven names, `nc: 7`), `split_manifest.csv` (each file → split, source, group) and
  `split_report.json` (images and box instances per class per split). It then re-verifies the output:
  pairing, label validity, taxonomy, and no group spanning splits.

Images that aren't labelled yet are excluded and counted, so you can train on a first batch.

## Readiness check (run first)

```powershell
python -m app.training.readiness            # add --write to save experiments/readiness/<timestamp>/
```

It reads the filesystem, not older reports. It prints **READY** or **BLOCKED** with each blocking reason: at least
30 reviewed images, at least 10 reviewed boxes per class, a valid reviewed export, and a prepared dataset that
passes the training gate. As of 2026-11-02 it reports BLOCKED: 0 of 343 staged images reviewed.

## 4. Train (CPU)

The training gate (`check_training_ready`) also refuses any dataset that was not built by `prepare_dataset` from a
human-reviewed export (`split_report.json` must name the review manifest). A hand-made `data.yaml` over unreviewed
labels cannot be trained on.

### Workload, measured on this laptop (i7-1255U, 16 GB, 10 torch threads)

`python -m app.training.train --estimate --train-images 380 --val-images 108` measured:

| | Measured |
|---|---|
| One training step (batch 4, 640 px, forward + backward) | 2.1 s |
| Validation, per image (640 px) | 0.28 s |
| Projected epoch (~380 train images) | ≥ 3.9 min |
| Projected 5-epoch pilot | ≥ ~20 min |
| Projected 30-epoch run | ≥ ~2 h |

These are lower bounds, because data loading, augmentation and loss assignment are excluded.
The pilot gives the real per-epoch time (`run_summary.json`).

### Pilot (smoke test): run first

```powershell
python -m app.training.train --data dataset\digital_twin_yolo\data.yaml --pilot
```

This runs 5 epochs at 640 px with batch 4. Check that the losses go down and mAP is above zero, and note the real epoch time.

### Full run: start it consciously

```powershell
python -m app.training.train --data dataset\digital_twin_yolo\data.yaml --epochs 30 --patience 8 --promote
```

Configuration:
- Transfer learning from `models/pretrained/yolo11n.pt` (read-only).
- `device=cpu`, `amp=False`, `seed=42`, `deterministic=True`, default optimiser and learning rate.
- `--batch 2` if memory is tight; `--imgsz 512` to roughly halve the time per step.
- The script refuses to start unless `data.yaml` matches the taxonomy, train and val labels exist, and the split verifies.

Outputs go to `experiments/training/<run>/`. Runs are never overwritten. Each run folder contains:
- `weights/best.pt` and `weights/last.pt`;
- `results.csv` (per-epoch train and val losses, P, R, mAP) and `results.png` (training curves);
- `confusion_matrix*.png` and the PR/F1/P/R curves;
- `val_batch*_pred.jpg` (validation predictions);
- `args.yaml` (full configuration);
- `run_summary.json` (duration, environment, final and best epoch metrics).

`--promote` copies `best.pt` to **`backend/models/custom/digital_twin_best.pt`**. Any previous custom
checkpoint is backed up with a timestamp, and the pretrained checkpoint is never touched.

## 5. Evaluate on the held-out test set

```powershell
python -m app.training.evaluate --model models\custom\digital_twin_best.pt --data dataset\digital_twin_yolo\data.yaml --split test
```

This prints precision, recall, F1, mAP50 and mAP50-95, overall and per class. It flags classes with fewer
than 10 test instances (unreliable) or mAP50 below 0.5 (weak). From the confusion matrix it lists, per class:
correct, missed, background false positives, and the classes it was confused with. It also reports the
look-alike pairs: antenna ↔ radio unit, router ↔ PSU, battery ↔ PSU, and cabinet ↔ battery/PSU/router/radio
unit. Cable bundle vs individual-cable behaviour must be checked in the prediction images. Validation and test
splits come from the same catalogue-style collection, so real-site behaviour needs the separate
[field test](bts-field-testing.md). Results are saved to `experiments/evaluation/<run>/`:
- the confusion matrix and curves;
- `metrics.json`;
- annotated predictions for every test image (`test_predictions/`).

**Comparison with the pretrained model:** none of the seven classes exist in COCO, so pretrained YOLO11n
scores zero on them by construction. A numeric comparison isn't meaningful. The useful baseline is the
pilot run versus the full run.

The test split is about 10% of roughly 500 images. Small classes such as cable (44 images) will have only a
handful of test instances, so per-class numbers for them are noisy.

## 6. Use the custom model in the application

In `backend/.env`:

```ini
DETECTION_MODEL=custom          # or pretrained
CUSTOM_MODEL_PATH=models/custom/digital_twin_best.pt
MODEL_FALLBACK_TO_PRETRAINED=false
```

Then restart the backend. The model is loaded once at startup, on CPU.

| Situation | Behaviour |
|---|---|
| `custom`, checkpoint present | custom model loaded; `model_type: custom`; seven class names in the dashboard; inventory `component_type` = class name; ByteTrack, REST and WebSocket unchanged |
| `custom`, checkpoint missing or broken, fallback `false` | API starts **degraded**; `/api/model/info` shows the exact error; detection endpoints return 503 |
| `custom`, checkpoint missing, fallback `true` | pretrained model runs; `fallback_active: true` with the reason; health `degraded`; dashboard warning |
| custom model's classes differ from the taxonomy | it still loads; `taxonomy_match: false` plus a log and dashboard warning |

`MODEL_PATH=<file>` remains available as an explicit override.

## 7. GPU later (no architecture change)

1. Create a separate environment with a CUDA build of PyTorch.
2. Add `"cuda"` to `SUPPORTED_DEVICES` in `backend/app/core/device.py`.
3. Set `DEVICE=cuda`.

The training scripts call `resolve_device()`, so they follow automatically. On a GPU, enable AMP (`amp=True` in
`build_train_args`) and raise `--batch`.
