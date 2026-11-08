# Pseudo-label report: `smoke_21`

> Pseudo-labels are generated automatically by Grounding DINO + CLIP (+ optional folder prior and MobileSAM). They are NOT ground truth. Agreement between models and consistency with folder names are not accuracy, precision, recall or mAP. No human has verified these labels.

Created 2026-11-07T20:08:00 · config hash `4a7066dfd40c9e48` · detections: stored:full_543_gdino_clip · folder prior boost 2.0 · MobileSAM off · 134.1 s (6.325 s/image)

## Totals

| | Count |
|---|---|
| Images | 21 |
| Images usable for training | 16 |
| Excluded images | 5 |
| Candidate boxes | 93 |
| ACCEPTED boxes | 36 |
| UNCERTAIN boxes | 12 |
| REJECTED boxes | 45 |
| Label instances in training images | 25 |
| Accepted boxes refined by MobileSAM | 0 |

Exclusion reasons: uncovered uncertain 4, no accepted box 1

## Per class (by final class)

| Class | Accepted | Uncertain | Rejected | Instances in training images | Training images with class | Median final conf. (accepted) | Median detector conf. (accepted) |
|---|---|---|---|---|---|---|---|
| antenna | 4 | 0 | 5 | 4 | 3 | 0.8712 | 0.3942 |
| router | 7 | 2 | 7 | 5 | 5 | 0.7779 | 0.4918 |
| cable | 9 | 4 | 5 | 6 | 5 | 0.9238 | 0.2849 |
| battery | 3 | 2 | 12 | 0 | 0 | 0.6548 | 0.3555 |
| power_supply_unit | 5 | 4 | 8 | 4 | 4 | 0.6177 | 0.539 |
| equipment_cabinet | 8 | 0 | 2 | 6 | 3 | 0.7983 | 0.3537 |
| radio_unit | 0 | 0 | 6 | 0 | 0 | — | — |

### Final-confidence distribution of accepted boxes

| Class | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|
| antenna | 0 | 0 | 0 | 1 | 1 | 2 |
| router | 0 | 3 | 0 | 1 | 0 | 3 |
| cable | 0 | 1 | 1 | 1 | 0 | 6 |
| battery | 0 | 1 | 1 | 1 | 0 | 0 |
| power_supply_unit | 0 | 2 | 1 | 1 | 1 | 0 |
| equipment_cabinet | 0 | 2 | 1 | 1 | 3 | 1 |
| radio_unit | 0 | 0 | 0 | 0 | 0 | 0 |

### Why boxes were uncertain or rejected

| Class | Uncertain | Rejected |
|---|---|---|
| antenna | — | part_of_larger_object 3, low_detector_conf 1, too_small 1 |
| router | low_final_conf 2 | part_of_larger_object 4, low_detector_conf 2, background 1 |
| cable | low_final_conf 4 | background 3, low_detector_conf 1, too_small 1 |
| battery | low_final_conf 2 | part_of_larger_object 7, background 3, too_small 1, low_detector_conf 1 |
| power_supply_unit | low_final_conf 3, flip_unstable 1 | part_of_larger_object 5, low_detector_conf 3 |
| equipment_cabinet | — | too_small 1, low_detector_conf 1 |
| radio_unit | — | background 3, part_of_larger_object 2, low_detector_conf 1 |

## Consistency (NOT accuracy)

_Consistency between automatic signals; NOT accuracy._

| Measure | Share |
|---|---|
| Accepted boxes where CLIP's top-1 is the final class | 0.8889 |
| Accepted boxes where Grounding DINO's class is the final class | 0.4444 |
| Accepted boxes where the folder prior changed the class | 0.0 |
| Non-rejected boxes whose class survives a mirrored crop | 0.8958 |
| Non-rejected boxes whose class survives the alternative prompt set | 0.7292 |
| Accepted boxes whose class survives the alternative prompt set | 0.8333 |

### Folder → accepted classes (images containing each class)

| Folder | Images | Training images with the folder's class | antenna | router | cable | battery | power_supply_unit | equipment_cabinet | radio_unit |
|---|---|---|---|---|---|---|---|---|---|
| power_supply_unit | 3 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| battery | 3 | 0 | 0 | 2 | 1 | 2 | 1 | 1 | 0 |
| cable | 3 | 3 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| router | 3 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| equipment_cabinet | 3 | 2 | 0 | 1 | 0 | 1 | 1 | 3 | 0 |
| antenna | 3 | 2 | 2 | 0 | 3 | 0 | 0 | 0 | 0 |
| radio_unit | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |

Folders are image-level collection labels, not object-level ground truth; an image may legitimately contain several classes.
