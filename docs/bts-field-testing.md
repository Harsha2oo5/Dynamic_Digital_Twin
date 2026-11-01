# BTS field testing protocol (custom component model)

Purpose: measure how the custom seven-class model behaves on **real BTS sites**, separately from the
catalogue-style images it is trained on. Validation numbers from `evaluate.py` say how well the model fits
our dataset. Only this field test says whether it recognises equipment on site.

> Status (2026-11-02): no custom model exists yet (0 human-reviewed training images). This protocol is ready to
> use once a checkpoint passes evaluation. Nothing here has been executed.

## 1. Why a separate field set

The training images are mostly product photos and phone screenshots (white backgrounds, one centred unit).
Real sites have clutter, poor light, distance, occlusion and many units in one frame. A model can score well on
our validation split and still fail on site. Field images must therefore:

- **never** be used for training or for picking thresholds;
- come from sites, cameras and days that are not in the training data;
- be annotated through the same review tool (human-reviewed boxes), so false positives and misses can be counted.

## 2. What to capture

Aim for at least **150 field images** (or frames sampled one per second from walk-around video), covering every
row of the scenario matrix with at least 10 images per row.

| Scenario | What to vary | Watch for |
|---|---|---|
| Lighting | daylight, overcast, dusk, indoor fluorescent, torch / flash, backlit | missed detections in shadow, glare false positives |
| Viewpoint | front, 45°, from below (tower), top-down into an open cabinet | antenna ↔ radio unit confusion from below |
| Distance | close-up (unit fills frame), 2–5 m, tower from the ground | small, distant antennas and RRUs missed |
| Indoor / outdoor | shelter, rooftop, ground-based outdoor cabinet, tower | background false positives on outdoor clutter |
| Occlusion | doors half open, cables across units, people in front | partial units below a third visible are skipped per the annotation rules |
| Vendors | Huawei, Ericsson, Nokia, ZTE, Vertiv, others on site | unseen vendors' designs |
| Density | crowded racks, stacked batteries, many PSU modules | duplicate boxes, merged neighbours |
| Cables | single jumper, distinguishable cables, inseparable bundles | bundle vs individual-cable boxes (annotation rule 5) |

Record per image: site ID (anonymised), date and time, camera or phone, scenario tags. Don't photograph people,
and follow site safety and access rules.

## 3. Procedure

```powershell
cd backend
# 1. stage the field images as their own batch (copies; originals untouched)
#    put them in annotation/batches/field_01/<any subfolder>/ first
.venv\Scripts\python.exe -m app.training.review_tool init --batch ..\annotation\batches\field_01

# 2. (optional) pre-label with the custom model, as candidates only
.venv\Scripts\python.exe -m app.training.auto_annotate --mode custom --model models\custom\digital_twin_best.pt --batch ..\annotation\batches\field_01 --name custom_field01

# 3. review every image in the UI (correct, add misses, delete false boxes) and mark REVIEWED
.venv\Scripts\python.exe -m app.training.review_tool serve --review ..\annotation\reviewed\field_01

# 4. qualitative pass: annotated predictions for every field image
.venv\Scripts\python.exe -m app.training.predict --model models\custom\digital_twin_best.pt --source ..\annotation\batches\field_01 --out ..\experiments\predictions\field_01
```

Limitation: `prepare_dataset` and `validate_annotations` pair labels with the original dataset's manifest, so a
field batch can't yet be turned into a quantitative test split with one command. Until a field-set
preparation step exists, compare the reviewed field boxes with the model's predictions (step 2 candidates vs
step 3 reviewed boxes) per image. Count per class: true positives (IoU ≥ 0.5, same class), misses, false
positives and class confusions.

## 4. What to report

- Per class: field precision, field recall, and the number of field instances (fewer than 10 means unreliable).
- Per scenario row: misses and false positives, to show *where* the model fails.
- Confusions on the look-alike pairs: antenna ↔ radio unit, router ↔ PSU, battery ↔ PSU, cabinet ↔ the
  components inside it (`evaluate.py` prints the same pairs for the validation split).
- Cable behaviour: bundle boxes vs individual-cable boxes, against the annotation rule.
- Inference speed on the target machine (the dashboard shows measured latency).

## 5. Interpreting results

- Field recall far below validation recall means a domain gap: add field images of the failing scenarios to the
  **training** pool (through the review tool), retrain, and re-test on a *new* field set.
- A class with many background false positives needs more negative context in training (wide shots where it
  is absent), not a higher threshold alone.
- A model trained on a few hundred catalogue images is a pilot. Don't describe it as production-ready until it
  holds up across all scenario rows on several sites.
