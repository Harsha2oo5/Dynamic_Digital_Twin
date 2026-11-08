# Zero-shot BTS component detection (experiment)

A standalone pipeline that detects and classifies the seven BTS components **without annotation or training**:

```
image ──► Grounding DINO (text prompts) ──► boxes + prompt + confidence
               │ class-agnostic duplicate suppression
               ├──► optional MobileSAM (box prompt → mask → tighter box)
               └──► CLIP (crop vs. class descriptions) ──► top-3 cosine similarities, UNKNOWN / UNCERTAIN
                         │
                         ▼
            final label = component only if Grounding DINO and CLIP agree; otherwise UNCERTAIN / UNKNOWN
```

It is **isolated from the application**. It doesn't touch the API, YOLO11n, ByteTrack, the frontend, the
dataset or any annotation. It only *reads* the backend's taxonomy (class ids 0–6) and its letterbox helper.

> **No ground truth exists.** The pipeline reports counts, agreement and timing, never precision, recall,
> mAP or accuracy. Agreement between Grounding DINO and CLIP is not proof of correctness.

## Setup

It uses the backend virtualenv; nothing extra to install (`transformers`, `torch` CPU, `ultralytics` are
already there). On first use it downloads:

| Model | Size | Where |
|---|---|---|
| `IDEA-Research/grounding-dino-tiny` | ~690 MB | Hugging Face cache (already present on the dev machine) |
| `openai/clip-vit-base-patch32` | ~600 MB | Hugging Face cache |
| `mobile_sam.pt` (only with `--sam`) | 39 MB | `experiments/zero_shot_bts/weights/` |

CPU only; no CUDA is installed or used.

## Run (from the project root)

```powershell
backend\.venv\Scripts\python.exe -m experiments.zero_shot_bts.run                    # all 543 images, Grounding DINO + CLIP
backend\.venv\Scripts\python.exe -m experiments.zero_shot_bts.run --limit 21         # 21 images, round-robin over the 7 folders
backend\.venv\Scripts\python.exe -m experiments.zero_shot_bts.run --mode gdino       # Grounding DINO only
backend\.venv\Scripts\python.exe -m experiments.zero_shot_bts.run --limit 7 --sam    # + MobileSAM refinement
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset PATH` | project dataset | dataset root (read-only) |
| `--limit N` | all | number of images, round-robin across class folders |
| `--mode gdino` / `gdino+clip` | `gdino+clip` | stages to run |
| `--sam` | off | MobileSAM box refinement (about +3 s/image) |
| `--box-threshold` | 0.25 | Grounding DINO confidence (max token probability) |
| `--nms-iou` | 0.5 | duplicate suppression across prompts |
| `--clip-min-similarity` | 0.20 | below this, CLIP's best class is UNCERTAIN |
| `--clip-min-margin` | 0.005 | top-1 minus top-2 similarity below this is UNCERTAIN |
| `--gdino-model`, `--clip-model` | tiny / ViT-B/32 | any compatible Hugging Face checkpoints |
| `--no-letterbox-crop` | crop on | by default detection runs on the photo inside phone screenshots |
| `--name` | timestamp | run folder name (never overwritten) |

Prompts and CLIP descriptions live in `config.py`; every run saves its exact configuration.

## Outputs: `experiments/zero_shot_bts/runs/<name>/` (not in git)

| Path | Content |
|---|---|
| `gallery.html` | open in a browser: filter by no detections, disagreements, UNCERTAIN/UNKNOWN, final label or folder; click an image to see every crop with the Grounding DINO and CLIP results side by side |
| `predictions/predictions.json` | every image and detection: box, Grounding DINO class, prompt, confidence, suppressed alternatives, CLIP top-3 and all seven similarities, margin, background similarity, decision reason, flags, optional SAM box |
| `predictions/predictions.jsonl` | the same, written image by image while running (survives interruption) |
| `visualizations/*.jpg` | photo region with boxes: solid = models agree, dashed = disagree / uncertain / unknown |
| `crops/*.jpg` | each crop as CLIP saw it |
| `reports/report.md`, `report.json` | counts, distributions, Grounding DINO → CLIP cross-table, flags, timing, memory, automatic example picks |
| `config/config.json` | models, prompts, thresholds, dataset path |

Because `predictions.json` keeps all seven similarities per crop, CLIP thresholds can be re-applied afterwards
without re-running the models.

## How a detection is labelled

1. **Grounding DINO**: one caption holds all prompts (for example "telecom antenna. cellular antenna. … rru.").
   A box's confidence is its maximum token probability. Its prompt and class come from the phrase with the
   highest mean token probability. Boxes below `--box-threshold` are dropped. Overlapping boxes from
   different prompts are merged, keeping the best and recording the others as `alternatives`.
2. **CLIP**: the padded crop is compared with averaged text embeddings of three descriptions per class, plus
   a set of "none of these" descriptions (wall, sky, person, text…).
   - **UNKNOWN**: a "none of these" description scores higher than every class.
   - **UNCERTAIN**: the best similarity is below the minimum, or the gap between the top two is below the margin.
   - Similarities are raw cosine values, not probabilities.
3. **Final label**: the component, if both models name the same one. Otherwise **UNCERTAIN** (with a
   `disagreement` flag) or **UNKNOWN**. Every detection also keeps both models' raw opinions.

## Results of the full run (2026-11-02, `runs/full_543_gdino_clip`, unverified)

Grounding DINO-tiny + CLIP ViT-B/32 ran on CPU over all 543 images: 75.6 min (8.3 s/image, of which
Grounding DINO 7.7 s and CLIP 0.5 s), peak memory 1.84 GB, 0 failures, 72 MB of outputs. MobileSAM was tested
separately on 3 images: about 3 s/image, and refined boxes overlap the originals with IoU 0.80–0.97.

| | Count |
|---|---|
| Images with detections / without | 543 / 0 |
| Candidate detections | 2432 (4.5 per image); 1555 below Grounding DINO confidence 0.35 |
| Grounding DINO class | cabinet 639, router 485, battery 293, antenna 283, PSU 261, radio unit 254, cable 217 |
| CLIP label | UNCERTAIN 819, antenna 279, PSU 277, router 269, cable 233, cabinet 201, UNKNOWN 188, battery 90, radio unit 76 |
| Agreement | agree 487 (20 %), disagree 938, CLIP uncertain 819, CLIP unknown 188 |

The folder's own component appears among the agreed labels in only some images: cabinet 48/70, antenna 36/100,
router 26/74, battery 12/62, cable 10/44, PSU 6/93, **radio unit 2/100**. This is a diagnostic, not accuracy;
folders are not ground truth.

What goes wrong (from the statistics and from inspecting the visualizations):
- **"Equipment cabinet" absorbs whole objects.** Grounding DINO's highest-confidence box usually frames the
  entire product and matches the cabinet prompt, whatever the product is (radio unit, battery, router). In the
  radio-unit folder, 97 of 290 boxes are "cabinet" and only 36 are "radio unit".
- **CLIP can't separate box-shaped equipment.** Similarities sit between about 0.21 and 0.34, and top-1/top-2
  margins are often below 0.005, so 34 % of crops are UNCERTAIN. Radio units are mostly ranked as PSU (51 of 100
  top boxes). CLIP is clearly better on antennas (57/100 top boxes ranked "antenna").
- **The agreement rule throws away some correct answers:** for example, CLIP ranked a radio unit correctly while
  Grounding DINO called it "cabinet". Raw opinions are kept in `predictions.json` for that reason.
- **Parts are detected as components.** Handles become "antenna", warning labels and terminal blocks become
  "PSU", and a router's fan module and latch become separate objects.
- **Look-alike pairs:** router ↔ PSU (73), antenna → cable (67; antenna photos usually include their feeder
  cable), battery ↔ PSU (49), cabinet ↔ router (137).

Conclusion: the zero-shot stack is useful for finding object *regions* (every image got boxes) but not for
naming BTS components reliably. That needs reviewed labels and a fine-tuned detector, which is the main
pipeline's path.

## Tests

```powershell
backend\.venv\Scripts\python.exe -m pytest experiments/zero_shot_bts/tests -q
```

These use fake models, so nothing is downloaded. They cover coordinate conversion, crop extraction, duplicate
suppression, prompt/token mapping, CLIP ranking and UNKNOWN/UNCERTAIN handling, label combination,
round-robin selection, and end-to-end output generation (dataset left untouched).
