# Pseudo-label report: `smoke_70`

> Pseudo-labels are generated automatically by Grounding DINO + CLIP (+ optional folder prior and MobileSAM). They are NOT ground truth. Agreement between models and consistency with folder names are not accuracy, precision, recall or mAP. No human has verified these labels.

Created 2026-11-07T20:12:33 · config hash `5d1d184475400ab2` · detections: stored:full_543_gdino_clip · CLIP scores: clip:live · folder prior boost 2.0 · MobileSAM off · 156.7 s (2.17 s/image)

## Totals

| | Count |
|---|---|
| Images | 70 |
| Images usable for training | 55 |
| Excluded images | 15 |
| Candidate boxes | 321 |
| ACCEPTED boxes | 136 |
| UNCERTAIN boxes | 44 |
| REJECTED boxes | 141 |
| Label instances in training images | 107 |
| Accepted boxes refined by MobileSAM | 0 |

Exclusion reasons: uncovered uncertain 11, no accepted box 4

## Per class (by final class)

| Class | Accepted | Uncertain | Rejected | Instances in training images | Training images with class | Median final conf. (accepted) | Median detector conf. (accepted) |
|---|---|---|---|---|---|---|---|
| antenna | 15 | 2 | 19 | 11 | 10 | 0.8212 | 0.3561 |
| router | 18 | 3 | 23 | 15 | 15 | 0.8761 | 0.4672 |
| cable | 36 | 8 | 18 | 24 | 19 | 0.9165 | 0.3275 |
| battery | 3 | 7 | 29 | 0 | 0 | 0.6548 | 0.3555 |
| power_supply_unit | 32 | 20 | 25 | 29 | 19 | 0.6529 | 0.3932 |
| equipment_cabinet | 28 | 2 | 10 | 24 | 13 | 0.8807 | 0.3509 |
| radio_unit | 4 | 2 | 17 | 4 | 2 | 0.7561 | 0.3754 |

### Final-confidence distribution of accepted boxes

| Class | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|
| antenna | 0 | 0 | 2 | 5 | 2 | 6 |
| router | 1 | 5 | 1 | 1 | 1 | 9 |
| cable | 0 | 3 | 6 | 3 | 4 | 20 |
| battery | 0 | 1 | 1 | 1 | 0 | 0 |
| power_supply_unit | 2 | 9 | 9 | 5 | 7 | 0 |
| equipment_cabinet | 0 | 4 | 2 | 3 | 9 | 10 |
| radio_unit | 0 | 1 | 0 | 1 | 2 | 0 |

### Why boxes were uncertain or rejected

| Class | Uncertain | Rejected |
|---|---|---|
| antenna | low_final_conf 2 | part_of_larger_object 13, low_detector_conf 5, too_small 1 |
| router | low_final_conf 3 | part_of_larger_object 16, low_detector_conf 6, background 1 |
| cable | low_final_conf 6, low_margin 1, flip_unstable 1 | background 9, low_detector_conf 7, too_small 2 |
| battery | low_final_conf 6, low_margin 1 | part_of_larger_object 14, low_detector_conf 8, background 6, too_small 1 |
| power_supply_unit | low_final_conf 16, flip_unstable 2, no_clip_support 1, low_margin 1 | low_detector_conf 16, part_of_larger_object 7, background 1, too_small 1 |
| equipment_cabinet | low_final_conf 2 | low_detector_conf 7, too_small 1, background 1, part_of_larger_object 1 |
| radio_unit | low_final_conf 2 | part_of_larger_object 7, background 6, low_detector_conf 3, too_small 1 |

## Consistency (NOT accuracy)

_Consistency between automatic signals; NOT accuracy._

| Measure | Share |
|---|---|
| Accepted boxes where CLIP's top-1 is the final class | 0.9265 |
| Accepted boxes where Grounding DINO's class is the final class | 0.4118 |
| Accepted boxes where the folder prior changed the class | 0.0074 |
| Non-rejected boxes whose class survives a mirrored crop | 0.9111 |
| Non-rejected boxes whose class survives the alternative prompt set | 0.7667 |
| Accepted boxes whose class survives the alternative prompt set | 0.8456 |

### Folder → accepted classes (images containing each class)

| Folder | Images | Training images with the folder's class | antenna | router | cable | battery | power_supply_unit | equipment_cabinet | radio_unit |
|---|---|---|---|---|---|---|---|---|---|
| power_supply_unit | 10 | 7 | 0 | 2 | 1 | 0 | 7 | 1 | 0 |
| battery | 10 | 0 | 1 | 4 | 3 | 2 | 5 | 2 | 0 |
| cable | 10 | 8 | 1 | 0 | 8 | 0 | 0 | 1 | 0 |
| router | 10 | 8 | 1 | 8 | 0 | 0 | 1 | 1 | 0 |
| equipment_cabinet | 10 | 8 | 1 | 4 | 1 | 1 | 3 | 10 | 0 |
| antenna | 10 | 6 | 8 | 0 | 10 | 0 | 1 | 0 | 0 |
| radio_unit | 10 | 2 | 1 | 0 | 2 | 0 | 5 | 1 | 2 |

Folders are image-level collection labels, not object-level ground truth; an image may legitimately contain several classes.
