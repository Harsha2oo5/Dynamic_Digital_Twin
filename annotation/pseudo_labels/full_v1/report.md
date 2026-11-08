# Pseudo-label report: `full_v1`

> Pseudo-labels are generated automatically by Grounding DINO + CLIP (+ optional folder prior and MobileSAM). They are NOT ground truth. Agreement between models and consistency with folder names are not accuracy, precision, recall or mAP. No human has verified these labels.

Created 2026-11-07T20:41:34 · config hash `bdf7dbdff06f62c9` · detections: stored:full_543_gdino_clip · CLIP scores: clip:live · folder prior boost 2.0 · visual prototypes on, anchors {'antenna': 99, 'router': 70, 'cable': 39, 'battery': 61, 'power_supply_unit': 93, 'equipment_cabinet': 68, 'radio_unit': 100} · MobileSAM off · 777.4 s (1.34 s/image)

## Totals

| | Count |
|---|---|
| Images | 543 |
| Images usable for training | 441 |
| Excluded images | 102 |
| Candidate boxes | 2432 |
| ACCEPTED boxes | 918 |
| UNCERTAIN boxes | 175 |
| REJECTED boxes | 1339 |
| Label instances in training images | 762 |
| Accepted boxes refined by MobileSAM | 0 |

Exclusion reasons: no accepted box 39, uncovered uncertain 63

## Per class (by final class)

| Class | Accepted | Uncertain | Rejected | Instances in training images | Training images with class | Median final conf. (accepted) | Median detector conf. (accepted) |
|---|---|---|---|---|---|---|---|
| antenna | 140 | 16 | 347 | 133 | 121 | 0.9941 | 0.3704 |
| router | 105 | 13 | 86 | 97 | 95 | 0.9879 | 0.4608 |
| cable | 143 | 22 | 553 | 110 | 87 | 0.9568 | 0.3164 |
| battery | 60 | 23 | 102 | 38 | 23 | 0.8478 | 0.3625 |
| power_supply_unit | 157 | 54 | 116 | 133 | 94 | 0.8099 | 0.3922 |
| equipment_cabinet | 240 | 33 | 58 | 185 | 93 | 0.9438 | 0.3648 |
| radio_unit | 73 | 14 | 77 | 66 | 65 | 0.8453 | 0.4239 |

### Final-confidence distribution of accepted boxes

| Class | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|
| antenna | 0 | 1 | 13 | 9 | 12 | 105 |
| router | 1 | 5 | 4 | 6 | 12 | 77 |
| cable | 0 | 7 | 14 | 10 | 27 | 85 |
| battery | 2 | 4 | 7 | 9 | 20 | 18 |
| power_supply_unit | 5 | 20 | 21 | 31 | 29 | 51 |
| equipment_cabinet | 2 | 18 | 19 | 21 | 25 | 155 |
| radio_unit | 3 | 8 | 11 | 11 | 12 | 28 |

### Why boxes were uncertain or rejected

| Class | Uncertain | Rejected |
|---|---|---|
| antenna | flip_unstable 8, low_final_conf 6, no_clip_support 1, low_margin 1 | part_of_larger_object 244, low_detector_conf 63, background 27, too_small 13 |
| router | low_final_conf 9, flip_unstable 3, low_margin 1 | part_of_larger_object 65, low_detector_conf 21 |
| cable | low_final_conf 18, flip_unstable 4 | part_of_larger_object 216, low_detector_conf 171, background 147, too_small 19 |
| battery | low_final_conf 13, flip_unstable 7, low_margin 3 | part_of_larger_object 69, low_detector_conf 27, background 6 |
| power_supply_unit | low_final_conf 26, flip_unstable 16, low_margin 12 | part_of_larger_object 60, low_detector_conf 55, background 1 |
| equipment_cabinet | low_final_conf 13, low_margin 8, flip_unstable 7, no_clip_support 5 | low_detector_conf 37, part_of_larger_object 16, background 3, too_small 2 |
| radio_unit | low_margin 7, low_final_conf 6, flip_unstable 1 | part_of_larger_object 53, low_detector_conf 20, too_small 2, background 2 |

## Consistency (NOT accuracy)

_Consistency between automatic signals; NOT accuracy._

| Measure | Share |
|---|---|
| Accepted boxes where CLIP's top-1 is the final class | 0.7342 |
| Accepted boxes where Grounding DINO's class is the final class | 0.3508 |
| Accepted boxes where the visual prototype's top-1 is the final class | 0.7277 |
| Accepted boxes where the folder prior changed the class | 0.0207 |
| Non-rejected boxes whose class survives a mirrored crop | 0.9167 |
| Non-rejected boxes whose class survives the alternative prompt set | 0.8463 |
| Accepted boxes whose class survives the alternative prompt set | 0.9009 |

### Folder → accepted classes (images containing each class)

| Folder | Images | Training images with the folder's class | antenna | router | cable | battery | power_supply_unit | equipment_cabinet | radio_unit |
|---|---|---|---|---|---|---|---|---|---|
| power_supply_unit | 93 | 44 | 3 | 15 | 12 | 4 | 58 | 26 | 1 |
| battery | 62 | 16 | 4 | 2 | 16 | 26 | 21 | 24 | 1 |
| cable | 44 | 32 | 4 | 3 | 33 | 2 | 2 | 8 | 3 |
| router | 74 | 62 | 3 | 65 | 4 | 1 | 4 | 4 | 2 |
| equipment_cabinet | 70 | 48 | 17 | 13 | 27 | 4 | 13 | 66 | 0 |
| antenna | 100 | 93 | 94 | 1 | 8 | 0 | 0 | 2 | 0 |
| radio_unit | 100 | 60 | 3 | 3 | 8 | 1 | 17 | 1 | 62 |

Folders are image-level collection labels, not object-level ground truth; an image may legitimately contain several classes.
