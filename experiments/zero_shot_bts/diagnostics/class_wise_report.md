# Class-wise diagnostic report (zero-shot Grounding DINO + CLIP)

> Weak-label diagnostics: folder names are image-level labels, not object-level ground truth. An image can show several components and a box can be a part, so these rates are NOT accuracy, precision, recall or mAP, and matching a folder does not prove a prediction is correct.

Run analysed: `full_543_gdino_clip` (543 images, 2432 detections).
Definitions: *primary box* = highest Grounding DINO confidence in the image; *any box* = at least one box; *box level* = share of all boxes in the folder's images.

## 1. Weak-label rates of the stored run (original prompts, 5 % padding)

| Folder class | Images | CLIP top-1 (primary) | CLIP top-3 (primary) | CLIP top-1 (any box) | CLIP top-3 (any box) | GD class (primary) | GD class (any box) | Box-level CLIP top-1 |
|---|---|---|---|---|---|---|---|---|
| antenna | 100 | 57.0% | 86.0% | 90.0% | 99.0% | 17.0% | 74.0% | 42.8% |
| router | 74 | 85.1% | 94.6% | 93.2% | 97.3% | 13.5% | 46.0% | 44.2% |
| cable | 44 | 27.3% | 40.9% | 52.3% | 72.7% | 2.3% | 45.5% | 23.3% |
| battery | 62 | 19.4% | 50.0% | 53.2% | 96.8% | 0.0% | 62.9% | 14.2% |
| power_supply_unit | 93 | 22.6% | 87.1% | 74.2% | 100.0% | 3.2% | 46.2% | 25.8% |
| equipment_cabinet | 70 | 54.3% | 81.4% | 82.9% | 98.6% | 70.0% | 97.1% | 28.9% |
| radio_unit | 100 | 18.0% | 69.0% | 24.0% | 85.0% | 5.0% | 30.0% | 9.5% |
| **all (micro / macro)** | 543 | 40.7% / 40.5% | 75.9% / 72.7% | 67.4% | 93.9% | 15.7% | 56.7% | |

Chance level for a top-1 pick among 7 classes is 14.3 %, and 42.9 % for top-3.

### Confusion: folder class → clip primary top-1

| folder \ predicted | antenna | router | cable | battery | power_sup | equipment | radio_uni |
|---|---|---|---|---|---|---|---|
| antenna | 57 | 9 | 16 | 3 | 6 | 4 | 5 |
| router | 0 | 63 | 0 | 1 | 6 | 3 | 1 |
| cable | 13 | 12 | 12 | 0 | 3 | 0 | 4 |
| battery | 1 | 15 | 0 | 12 | 16 | 11 | 7 |
| power_supply_unit | 2 | 26 | 0 | 4 | 21 | 28 | 12 |
| equipment_cabinet | 1 | 17 | 1 | 3 | 10 | 38 | 0 |
| radio_unit | 7 | 13 | 1 | 5 | 51 | 5 | 18 |

### Confusion: folder class → gdino primary top-1

| folder \ predicted | antenna | router | cable | battery | power_sup | equipment | radio_uni |
|---|---|---|---|---|---|---|---|
| antenna | 17 | 21 | 9 | 2 | 3 | 45 | 3 |
| router | 0 | 10 | 0 | 0 | 3 | 53 | 8 |
| cable | 5 | 6 | 1 | 1 | 0 | 29 | 2 |
| battery | 1 | 10 | 0 | 0 | 1 | 50 | 0 |
| power_supply_unit | 1 | 11 | 0 | 0 | 3 | 74 | 4 |
| equipment_cabinet | 1 | 11 | 0 | 5 | 3 | 49 | 1 |
| radio_unit | 2 | 4 | 1 | 1 | 2 | 85 | 5 |

## 2. Score distributions (stored run)

| Folder class | CLIP top-1 sim (median, p25-p75) | top-1 − top-2 margin (median) | near ties (< 0.005) | GD confidence (median, p25-p75) |
|---|---|---|---|---|
| antenna | 0.2718 (0.2513-0.2899) | 0.0103 | 30.1% | 0.3299 (0.2899-0.3862) |
| router | 0.2918 (0.2697-0.3053) | 0.0099 | 31.5% | 0.33 (0.2761-0.411) |
| cable | 0.2855 (0.2672-0.3015) | 0.0091 | 33.0% | 0.3051 (0.2723-0.3623) |
| battery | 0.2829 (0.2615-0.3023) | 0.0055 | 46.2% | 0.315 (0.2747-0.3803) |
| power_supply_unit | 0.2814 (0.2608-0.3031) | 0.0063 | 41.9% | 0.3134 (0.2778-0.3796) |
| equipment_cabinet | 0.2894 (0.264-0.3079) | 0.0067 | 40.4% | 0.3075 (0.2732-0.3684) |
| radio_unit | 0.2818 (0.259-0.3004) | 0.0055 | 46.7% | 0.3223 (0.2771-0.4177) |

## 3. Box size vs. folder match (stored run)

| Box size (share of photo) | Boxes | CLIP top-1 = folder | CLIP UNKNOWN |
|---|---|---|---|
| tiny <5% | 1078 | 14.8% | 15.5% |
| small 5-20% | 491 | 27.5% | 3.0% |
| medium 20-60% | 395 | 41.0% | 1.3% |
| large >60% | 468 | 45.7% | 0.2% |

Primary-box sizes: {'medium 20-60%': 78, 'large >60%': 358, 'tiny <5%': 52, 'small 5-20%': 55}

## 4. Prompt comparison (same boxes, 5 % padding)

Sanity check: the re-computed `original_ensemble` reproduces the stored top-1 for 99.0% of boxes.

| Configuration | Primary top-1 (macro) | Primary top-3 (macro) | Any-box top-1 | Half A / half B (top-1 macro) | Decided / uncertain / unknown | Median margin | Top-1 changed vs original |
|---|---|---|---|---|---|---|---|
| name_only | 35.3% | 74.7% | 58.3% | 34.1% / 36.5% | 63.8% / 26.6% / 9.5% | 0.009 | 46.6% |
| short_ensemble | 39.1% | 76.3% | 66.8% | 37.8% / 40.5% | 63.7% / 30.0% / 6.2% | 0.0085 | 32.6% |
| original_ensemble | 40.6% | 72.7% | 66.7% | 38.3% / 42.9% | 58.8% / 33.5% / 7.7% | 0.0068 | 0.0% |
| original_max | 42.9% | 74.0% | 70.9% | 41.6% / 44.1% | 63.9% / 28.1% / 8.1% | 0.0084 | 31.3% |
| detailed_ensemble | 46.2% | 82.3% | 71.7% | 46.4% / 46.1% | 63.4% / 28.0% / 8.7% | 0.0088 | 36.5% |
| detailed_max | 44.2% | 76.3% | 73.8% | 45.1% / 43.4% | 59.4% / 29.4% / 11.2% | 0.0077 | 46.1% |
| all_ensemble | 42.6% | 78.0% | 70.7% | 41.9% / 43.4% | 62.3% / 31.2% / 6.4% | 0.008 | 23.2% |

Per class, primary-box top-1 (macro shown above):

| Configuration | antenna | router | cable | battery | power_sup | equipment | radio_uni |
|---|---|---|---|---|---|---|---|
| name_only | 29.0% | 78.4% | 22.7% | 6.5% | 72.0% | 31.4% | 7.0% |
| short_ensemble | 51.0% | 77.0% | 27.3% | 21.0% | 33.3% | 61.4% | 3.0% |
| original_ensemble | 57.0% | 85.1% | 27.3% | 19.4% | 22.6% | 55.7% | 17.0% |
| original_max | 50.0% | 81.1% | 27.3% | 22.6% | 41.9% | 61.4% | 16.0% |
| detailed_ensemble | 59.0% | 77.0% | 38.6% | 30.6% | 48.4% | 60.0% | 10.0% |
| detailed_max | 60.0% | 83.8% | 29.5% | 33.9% | 40.9% | 48.6% | 13.0% |
| all_ensemble | 61.0% | 79.7% | 29.5% | 25.8% | 31.2% | 64.3% | 7.0% |

## 5. Crop comparison (same boxes)

| Crop | Prompts | Primary top-1 (macro) | Primary top-3 (macro) | Any-box top-1 (macro) | Box-level top-1 (micro over boxes) |
|---|---|---|---|---|---|
| box + 0% padding | original_ensemble | 41.2% | 72.4% | 67.6% | 28.6% |
| box + 5% padding | original_ensemble | 40.6% | 72.7% | 66.7% | 27.6% |
| box + 10% padding | original_ensemble | 41.2% | 73.7% | 67.7% | 28.8% |
| box + 20% padding | original_ensemble | 42.5% | 74.1% | 67.9% | 30.6% |
| whole photo (no box) | original_ensemble | 47.5% | 74.6% | - | - |
| box + 0% padding | detailed_ensemble | 47.1% | 81.4% | 72.4% | 31.5% |
| box + 5% padding | detailed_ensemble | 46.2% | 82.3% | 71.7% | 30.8% |
| box + 10% padding | detailed_ensemble | 46.3% | 81.1% | 72.4% | 31.7% |
| box + 20% padding | detailed_ensemble | 48.5% | 81.3% | 74.0% | 33.2% |
| whole photo (no box) | detailed_ensemble | 49.9% | 83.5% | - | - |

*whole photo* = CLIP on the full photo region, without any detection box (one prediction per image).

## 6. Examples

- **clip matches folder**: `examples/clip_matches_folder.jpg`
- **clip disagrees with gdino**: `examples/clip_disagrees_with_gdino.jpg`
- **near identical top2**: `examples/near_identical_top2.jpg`
- **incomplete object small box**: `examples/incomplete_object_small_box.jpg`
- **prompt changes prediction**: `examples/prompt_changes_prediction.jpg`

## 7. Findings and recommendations

(see console output)
