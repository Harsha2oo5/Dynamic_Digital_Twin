# Pseudo-label report: `smoke_70_proto`

> Pseudo-labels are generated automatically by Grounding DINO + CLIP (+ optional folder prior and MobileSAM). They are NOT ground truth. Agreement between models and consistency with folder names are not accuracy, precision, recall or mAP. No human has verified these labels.

Created 2026-11-07T20:19:13 · config hash `966769ac322be818` · detections: stored:full_543_gdino_clip · CLIP scores: clip:live · folder prior boost 2.0 · visual prototypes on, anchors {'antenna': 10, 'router': 9, 'cable': 8, 'battery': 10, 'power_supply_unit': 10, 'equipment_cabinet': 10, 'radio_unit': 10} · MobileSAM off · 144.0 s (1.889 s/image)

## Totals

| | Count |
|---|---|
| Images | 70 |
| Images usable for training | 52 |
| Excluded images | 18 |
| Candidate boxes | 321 |
| ACCEPTED boxes | 157 |
| UNCERTAIN boxes | 34 |
| REJECTED boxes | 130 |
| Label instances in training images | 113 |
| Accepted boxes refined by MobileSAM | 0 |

Exclusion reasons: uncovered uncertain 15, no accepted box 3

## Per class (by final class)

| Class | Accepted | Uncertain | Rejected | Instances in training images | Training images with class | Median final conf. (accepted) | Median detector conf. (accepted) |
|---|---|---|---|---|---|---|---|
| antenna | 17 | 2 | 18 | 13 | 10 | 0.9749 | 0.3629 |
| router | 13 | 3 | 15 | 12 | 12 | 0.9928 | 0.4437 |
| cable | 52 | 8 | 41 | 38 | 28 | 0.9623 | 0.3107 |
| battery | 14 | 6 | 30 | 3 | 2 | 0.783 | 0.3353 |
| power_supply_unit | 21 | 9 | 13 | 15 | 10 | 0.7813 | 0.4062 |
| equipment_cabinet | 33 | 3 | 6 | 27 | 12 | 0.9571 | 0.3433 |
| radio_unit | 7 | 3 | 7 | 5 | 5 | 0.7624 | 0.4504 |

### Final-confidence distribution of accepted boxes

| Class | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|
| antenna | 0 | 1 | 3 | 0 | 3 | 10 |
| router | 0 | 1 | 0 | 0 | 1 | 11 |
| cable | 0 | 0 | 3 | 5 | 7 | 37 |
| battery | 0 | 3 | 2 | 3 | 5 | 1 |
| power_supply_unit | 0 | 4 | 4 | 3 | 3 | 7 |
| equipment_cabinet | 1 | 1 | 4 | 4 | 4 | 19 |
| radio_unit | 0 | 1 | 1 | 2 | 1 | 2 |

### Why boxes were uncertain or rejected

| Class | Uncertain | Rejected |
|---|---|---|
| antenna | flip_unstable 1, low_final_conf 1 | part_of_larger_object 14, low_detector_conf 4 |
| router | low_final_conf 2, flip_unstable 1 | part_of_larger_object 8, low_detector_conf 7 |
| cable | low_final_conf 6, no_clip_support 1, flip_unstable 1 | background 22, low_detector_conf 14, too_small 5 |
| battery | low_final_conf 5, flip_unstable 1 | part_of_larger_object 16, low_detector_conf 10, background 3, too_small 1 |
| power_supply_unit | low_final_conf 7, low_margin 1, flip_unstable 1 | low_detector_conf 9, part_of_larger_object 4 |
| equipment_cabinet | low_final_conf 2, flip_unstable 1 | low_detector_conf 6 |
| radio_unit | low_final_conf 2, low_margin 1 | part_of_larger_object 5, low_detector_conf 2 |

## Consistency (NOT accuracy)

_Consistency between automatic signals; NOT accuracy._

| Measure | Share |
|---|---|
| Accepted boxes where CLIP's top-1 is the final class | 0.7962 |
| Accepted boxes where Grounding DINO's class is the final class | 0.3376 |
| Accepted boxes where the visual prototype's top-1 is the final class | 0.7197 |
| Accepted boxes where the folder prior changed the class | 0.0318 |
| Non-rejected boxes whose class survives a mirrored crop | 0.911 |
| Non-rejected boxes whose class survives the alternative prompt set | 0.8639 |
| Accepted boxes whose class survives the alternative prompt set | 0.9554 |

### Folder → accepted classes (images containing each class)

| Folder | Images | Training images with the folder's class | antenna | router | cable | battery | power_supply_unit | equipment_cabinet | radio_unit |
|---|---|---|---|---|---|---|---|---|---|
| power_supply_unit | 10 | 7 | 0 | 1 | 4 | 0 | 7 | 1 | 0 |
| battery | 10 | 1 | 1 | 0 | 5 | 6 | 4 | 2 | 0 |
| cable | 10 | 8 | 1 | 1 | 9 | 1 | 0 | 1 | 0 |
| router | 10 | 8 | 1 | 8 | 2 | 0 | 1 | 1 | 0 |
| equipment_cabinet | 10 | 7 | 0 | 3 | 6 | 1 | 0 | 10 | 0 |
| antenna | 10 | 8 | 10 | 0 | 9 | 0 | 0 | 0 | 0 |
| radio_unit | 10 | 5 | 0 | 0 | 2 | 0 | 2 | 1 | 6 |

Folders are image-level collection labels, not object-level ground truth; an image may legitimately contain several classes.
