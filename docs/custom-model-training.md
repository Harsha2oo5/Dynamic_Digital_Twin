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
| `validate_annotations.py` | validates a YOLO export against the taxonomy and the manifest |
| `prepare_dataset.py` | duplicate-aware, stratified 70/20/10 split, writes `data.yaml` and reports |
| `train.py` | workload estimate, pilot run, full run, checkpoint promotion |
| `evaluate.py` | test-set metrics, per-class table, plots, prediction images |
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

**Annotation status: none.** Detection training is blocked until boxes exist; see [annotation-guide.md](annotation-guide.md).

## 2. Annotate → validate

```powershell
python -m app.training.annotation_kit                # writes ../annotation/ (configs + manifest)
# … annotate in CVAT / Label Studio, export YOLO into ../annotation/exports/<export>/ …
python -m app.training.validate_annotations ..\annotation\exports\<export>
```

## 3. Prepare the dataset

```powershell
python -m app.training.prepare_dataset --labels ..\annotation\exports\<export> --seed 42
```

What it does:
- Validates the export first and aborts on any error.
- Splits **by near-duplicate group**: re-captures of the same photo, found by a 256-bit content hash in the
  audit, always land in the same split. This prevents evaluation leakage. The audit found 39 such groups covering 84 images.
- Stratifies by folder class, 70/20/10, with fixed seed 42, so the result is reproducible.
- **Copies** images (from the original dataset) and labels into `backend/dataset/digital_twin_yolo/`, using
  space-free file names. It never overwrites an existing output directory.
- Writes `data.yaml` (the seven names, `nc: 7`), `split_manifest.csv` (each file → split, source, group) and
  `split_report.json` (images and box instances per class per split). It then re-verifies the output:
  pairing, label validity, taxonomy, and no group spanning splits.

Images that aren't labelled yet are excluded and counted, so you can train on a first batch.

## 4. Train (CPU)

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
than 10 test instances (unreliable) or mAP50 below 0.5 (weak), and saves the following to `experiments/evaluation/<run>/`:
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
