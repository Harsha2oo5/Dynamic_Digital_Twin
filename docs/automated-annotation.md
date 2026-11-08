# Automated pseudo-labelling (no human annotation)

`backend/app/training/pseudo_label/` turns the 543 raw images into YOLO labels without manual boxes.

> **Pseudo-labels are not ground truth.** They are machine-generated, and no person has checked them. Agreement
> between models, and consistency with folder names, are **not** accuracy, precision, recall or mAP. A model trained
> on them learns their mistakes too. Every run writes `PSEUDO_LABELS_NOT_GROUND_TRUTH.txt` and a manifest with
> `"ground_truth": false`.

## Pipeline

For every image:
1. **Regions.** Use the Grounding DINO-tiny boxes, by default reused from the zero-shot run
   `experiments/zero_shot_bts/runs/full_543_gdino_clip` (`--detections live` re-runs the detector, about 70 min on
   this CPU).
2. **CLIP scores** for each crop (20 % padding):
   - text similarity to the `detailed_ensemble` class descriptions, which did best in the diagnostics;
   - the same for a mirrored crop, and for an independent prompt set (stability checks);
   - similarity to "none of these" negative descriptions;
   - the crop's image embedding.
3. **Visual prototypes.** For each class, the mean CLIP embedding of the largest box in each image of that class's
   folder (when it covers at least 20 % of the photo). Every image is scored against prototypes built **without its
   own near-duplicate group**, so it never votes for itself. Folders act here as image-level weak labels.
4. **Fusion:**
   `log p(c) = log p_clip(c) + 0.5·log p_detector(c) + 2·log p_prototype(c) + log prior(c)`.
   The folder prior is a soft boost (×2 at most, scaled by box size). The chosen class must be in CLIP's text
   top 3 or be the top prototype, so the folder can never decide a class on its own.
5. **Tiers** (thresholds per class in `backend/configs/pseudo_label/default.yaml`):
   - **ACCEPTED**: becomes a YOLO label.
   - **UNCERTAIN**: low fused confidence, small margin, no visual support, or a class that changes on the mirrored
     crop. Kept with its reason, never labelled.
   - **REJECTED**: too small, background, or low detector score.
6. **Suppression:**
   - same-class duplicates and cross-class conflicts are rejected;
   - a box mostly inside a larger object (handle, rack ear, fan module) is rejected as a part, except inside a
     cabinet, and except battery modules in a battery rack or rectifiers in a PSU shelf.
7. **Image decision:** an image is used for training only if it has an accepted box and no large uncertain box that
   no accepted box covers. Otherwise YOLO would learn that real equipment is background.
8. Optional **MobileSAM** tightening of accepted boxes (`--sam`, about +3 s per image).

## Running it

```powershell
cd backend
.venv\Scripts\python.exe -m app.training.pseudo_label.run --name full_v1                       # all 543 images (~13 min)
.venv\Scripts\python.exe -m app.training.pseudo_label.run --name smoke --limit 21              # round-robin over the 7 folders
.venv\Scripts\python.exe -m app.training.pseudo_label.run --name strict --reuse-scores ..\annotation\pseudo_labels\full_v1 --set min_final_margin=0.2
```

`--reuse-scores` re-applies fusion and thresholds to the CLIP scores and embeddings of an earlier run in seconds.
It is refused if the CLIP settings differ. `--set key=value` overrides any config key.

Output in `annotation/pseudo_labels/<run>/`:
- `labels/` (training images only);
- `pseudo_manifest.json` (provenance, config hash, sha256 of every label file);
- `records.jsonl` (every box with every signal and its reason);
- `report.md` / `report.json`;
- `gallery.html` with `previews/`, and `clip_cache.npz`.

## Results of `full_v1` (2026-11-07)

- 543 images in 13 min (1.3 s per image): **441 usable for training, 762 label instances**.
- 102 images excluded: 39 without an accepted box, 63 with an uncovered uncertain object.
- Boxes: 918 accepted, 175 uncertain, 1339 rejected.

| Class | Accepted | In training images | Folder images containing the folder's class |
|---|---|---|---|
| antenna | 140 | 133 | 93 / 100 |
| router | 105 | 97 | 62 / 74 |
| cable | 143 | 110 | 32 / 44 |
| battery | 60 | 38 | **16 / 62** |
| power_supply_unit | 157 | 133 | 44 / 93 |
| equipment_cabinet | 240 | 185 | 48 / 70 |
| radio_unit | 73 | 66 | 60 / 100 |

Consistency (not accuracy):
- CLIP's text top-1 equals the final class for 73 % of accepted boxes, the prototype top-1 for 73 %, and Grounding
  DINO's class for only 35 %.
- The folder prior changed the class of 2 % of accepted boxes.
- 92 % of non-rejected boxes keep their class on a mirrored crop, and 90 % of accepted boxes keep it under the
  alternative prompt set.

Weak spots seen in the report and gallery:
- **Battery images** are often labelled power supply unit (26) or cabinet (24).
- **Power-supply images** are often labelled cabinet (58).
- Some primary boxes are confidently wrong when several signals agree on the wrong class.

Expect these errors in any model trained on this set.

## Tuning (70-image sweep on cached scores)

| Setting | Usable images | Main box matches folder* |
|---|---|---|
| CLIP text + detector only | 52 | 32 / 52 |
| + prototypes (weight 2), no folder prior | 58 | 46 / 58 |
| + prototypes (weight 2) + folder prior 2 (**default**) | 62 | 50 / 62 |

\*A weak consistency signal, not accuracy.

## Recommended next step

An optional audit of about 50 test images with the existing review tool (`app.training.review_tool`) would give
the only real accuracy estimate. It is not required by the pipeline.
