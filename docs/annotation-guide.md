# Annotation guide: Digital Twin components

YOLO detection needs a **bounding box and class for every component instance** in every image.
The dataset currently has folder-level classes only, so these boxes must be drawn by people.
Synthetic or "whole-image" boxes must not be used.

## The seven classes (fixed order: IDs are positions)

| ID | Class | Source folder | What to box |
|---|---|---|---|
| 0 | `antenna` | `antenna/antenna/` | each antenna panel / mast antenna |
| 1 | `router` | `Archive (7)/` | each router / transmission chassis (e.g. Huawei ATN, Nokia units) |
| 2 | `cable` | `Archive (6)/` | each distinguishable cable, or each inseparable bundle (see rule 5) |
| 3 | `battery` | `Archive (5)/` | each battery module |
| 4 | `power_supply_unit` | `Archive (4)/` | each PSU / rectifier shelf |
| 5 | `equipment_cabinet` | `Equipment Cabinet/Equipment Cabinet/` | each outdoor/indoor cabinet (the whole enclosure) |
| 6 | `radio_unit` | `radio unit/radio unit/` | each radio unit (RRU) |

The order is defined in `backend/app/training/taxonomy.py`. The annotation tool **must** use exactly
this order, or class IDs will be swapped. The generated tool configs already have it. To verify them:

```powershell
cd backend
.venv\Scripts\python.exe -m app.training.annotation_kit --check     # prints the mapping, ends with "KIT OK"
```

## Annotation kit (`annotation/`)

| File | Use |
|---|---|
| `classes.txt` | class names in ID order |
| `cvat_labels.json` | CVAT project labels (paste into *Raw*, see below) |
| `label_studio_config.xml` | Label Studio labeling interface |
| `manifest.csv` | every image → folder class + near-duplicate group (used by validation and splitting) |
| `batches/batch_01/` | **initial batch**: 175 staged image copies, one subfolder per class, plus `batch_manifest.csv` |
| `batches/batch_02/` | second batch: 168 staged copies (cable 18), no near-duplicate group shared with batch 1 |
| `candidates/<run>/` | machine candidate boxes (never ground truth) |
| `reviewed/<batch>/` | **human review state**: `review_state.json` (status and boxes per image) and `review_log.jsonl` (audit trail). Tracked in git, because it is human work |
| `exports/` | reviewed exports (one subfolder per export); the only input `prepare_dataset` accepts |

## Review workflow: from candidates to training-ready labels

```
digital twin project datasets/   original images (read-only, never written)
        │ staged copies
annotation/batches/batch_01/     images the reviewer sees
        │ detector
annotation/candidates/<run>/     machine candidates (marker file: refused as training data)
        │ review_tool init (candidates loaded as UNVERIFIED boxes)
annotation/reviewed/batch_01/    human review state: status + boxes + audit log
        │ review_tool export (REVIEWED images only, plus review_manifest.json)
annotation/exports/<export>/     reviewed labels
        │ validate_annotations --require-review, then prepare_dataset (refuses anything else)
backend/dataset/digital_twin_yolo/   training-ready train/val/test
```

### An image becomes training-ready only when

1. A person opened it and every candidate box was **accepted, corrected or deleted** (unverified boxes block sign-off).
2. Missing objects were added, and false positives were removed (deleted candidates stay in the audit trail).
3. No box is marked **uncertain**. Uncertain images go to `NEEDS_SECOND_REVIEW` and are never exported.
4. The reviewer marked it **REVIEWED**. This needs a reviewer name and an explicit five-point checklist; nothing sets it automatically.
5. The export passes `validate_annotations --require-review`: class IDs, coordinates, duplicates, and each label's
   hash matching what the reviewer signed off.

Statuses: `NOT_REVIEWED` → `IN_PROGRESS` → `REVIEWED`, or → `NEEDS_SECOND_REVIEW` → `IN_PROGRESS`/`REVIEWED`.
Any edit to a `REVIEWED` image re-opens it (`IN_PROGRESS`).

### Commands (from `backend/`)

```powershell
# once per batch (already done for batch_01 on 2026-11-02: 175 images, 553 unverified candidate boxes)
.venv\Scripts\python.exe -m app.training.review_tool init --batch ..\annotation\batches\batch_01 --candidates ..\annotation\candidates\gdino_tiny_batch01

# review in the browser: http://127.0.0.1:8765  (every change is saved immediately; Ctrl+C to stop, resume any time)
.venv\Scripts\python.exe -m app.training.review_tool serve --review ..\annotation\reviewed\batch_01

# progress at any time
.venv\Scripts\python.exe -m app.training.review_tool status --review ..\annotation\reviewed\batch_01

# when enough images are REVIEWED: export, then validate
.venv\Scripts\python.exe -m app.training.review_tool export --review ..\annotation\reviewed\batch_01 --out ..\annotation\exports\2026-11-10_batch01_reviewed
.venv\Scripts\python.exe -m app.training.validate_annotations ..\annotation\exports\2026-11-10_batch01_reviewed --require-review
```

For later batches (e.g. `batch_02`) run `init` without `--candidates` to start with no boxes, or with the pilot
model's candidates.

### Using the review UI

- **Left panel.** Enter your name (needed to sign off). Filter by status, by folder, by *multiple candidate
  classes*, by *low-confidence candidate* (threshold editable), by *no candidate boxes* or by *uncertain boxes*.
  Click an image to open it.
- **Image.** Dashed boxes are unverified candidates, labelled `cand <class> <confidence>`. Solid boxes with ✓ are
  verified by a person, and red dashed boxes are marked uncertain. Drag a box to move it, drag a white corner to
  resize, and drag on empty space to draw a new box with the *New box class* (keys `1`–`7` choose it when no box is
  selected).
- **Right panel.** Each box lists what the model said (class, confidence, flags). Use the class dropdown (or keys
  `1`–`7`), **Accept** (`A`) for a box that is already right, **?** (`U`) to mark it uncertain, and **✕** (`Del`) to
  delete it. Changing a box's class or geometry counts as verifying it.
- **Sign-off.** *Mark REVIEWED* is enabled only when no box is unverified or uncertain, and asks you to confirm the
  checklist. An image with no boxes left becomes a **verified negative**. *Needs second review* takes a note for
  the second reviewer.
- **Keys.** `←`/`→` move between images (within the filter), `Tab` selects the next box, `Esc` deselects.

The Grounding DINO candidates have good boxes but unreliable classes (radio_unit is over-predicted), so expect
to change many classes. Never accept a candidate class without looking.

### CVAT as an alternative editor

CVAT is better for dense scenes (many cables), with zoom and SAM-assisted drawing. The review status still lives
in the review tool:

1. Check the candidate archives (done 2026-11-02: 7 archives, 175 images, 553 box lines all match `candidates.json`):
   ```powershell
   .venv\Scripts\python.exe -m app.training.review_tool check-cvat --candidates ..\annotation\candidates\gdino_tiny_batch01 --batch ..\annotation\batches\batch_01
   ```
   YOLO 1.1 carries only class and box, so the confidence and flags of every line are kept in
   `cvat_import/candidate_metadata.json` (archive → image → line number → class, confidence, tier, flags).
2. Create the project and tasks as in *CVAT: step by step* below, then for each task use **Actions → Upload
   annotations → YOLO 1.1** with `cvat_import/batch_01_<task>.zip` (for example, `batch_01_0_antenna.zip` into task
   `b01_0_antenna`). Images are matched by file name, and class IDs follow the project's label order (which must be
   the taxonomy order).
3. Correct every image in CVAT, then export the project as YOLO 1.1 (without images) and unzip it into a new folder.
4. Bring the corrected boxes back for sign-off:
   ```powershell
   .venv\Scripts\python.exe -m app.training.review_tool import-yolo --review ..\annotation\reviewed\batch_01 --export ..\annotation\exports\2026-11-10_batch01_cvat_raw
   ```
   Imported images become `IN_PROGRESS` with verified boxes. Open them in the review UI, check them, and mark
   each one REVIEWED. Images already REVIEWED are never overwritten by an import; re-open them first. Then
   `export` and validate as above.

A raw CVAT export **cannot** go straight to `prepare_dataset`: it has no review manifest, so the training gate
refuses it.

The kit is regenerated with `python -m app.training.annotation_kit`. Neither the kit nor the batches modify
`digital twin project datasets/`; batch images are **copies** with the original file names, so labels
map back to the originals through the manifest.

## Initial batch: `batch_01` (annotate this first)

25 images per class, 175 in total, selected with
`python -m app.training.annotation_batch --name batch_01 --per-class 25 --seed 42`:

- **No near-duplicates together.** The batch has at most one image per near-duplicate group, and that holds across
  all classes. The cabinet photo that appears in both the PSU and battery folders is used only once.
- **Diverse.** Within each class, farthest-point sampling on an image-content hash spreads the picks over
  visually different images: renders, field photos, different vendors. For each group the highest-resolution
  copy is chosen. The closest pair inside any class is 106 of 256 hash bits apart; duplicates are ≤ 30.
- **Scarce classes first.** Cable has only 41 independent images, so it gets first pick on contested images.
- **Reproducible.** The same seed always gives the same batch. `batch_manifest.csv` lists every file, its source and its group.

Later batches use `--name batch_02` (and so on). They automatically skip images that are already batched and prefer groups
not yet covered. Every image is annotated eventually; batch 1 exists to get a validated pilot model quickly.

## CVAT: step by step

CVAT is a separate application: either the cvat.ai cloud, or self-hosted with Docker (`git clone https://github.com/cvat-ai/cvat`,
then `docker compose up -d`, then create a user with `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`).
Nothing in this project installs it. Menu names below follow CVAT 2.x; older versions differ slightly.

### 1. Create the project (labels)

1. **Projects → + → Create a new project**. Name: `digital-twin`.
2. Under **Labels**, open the **Raw** tab, replace the content with the full contents of
   `annotation/cvat_labels.json`, and click **Done**.
3. Check that the label list shows, **in this order**: antenna, router, cable, battery,
   power_supply_unit, equipment_cabinet, radio_unit. All seven should be rectangle labels.
4. **Submit & Open**.

The order matters: CVAT numbers YOLO class IDs by label order, so this order makes CVAT's IDs match the taxonomy.

### 2. Import the images: one task per class

For each folder in `annotation/batches/batch_01/` (`0_antenna`, `1_router`, …, `6_radio_unit`):

1. **Tasks → + → Create a new task**.
2. **Name**: `b01_0_antenna` (batch + folder name). **Project**: `digital-twin`. The labels are inherited, so don't add any.
3. **Select files → My computer**: open the batch folder, select **all 25 images** (Ctrl+A), and upload.
   Upload only images; there are no `._*` metadata files in the batch folders.
4. Leave **Advanced configuration** at the defaults (one job per task is fine for 25 images; image quality 70 is fine).
5. **Submit & Continue**, then repeat for the next class.

You end up with 7 tasks of 25 images each. Assign tasks or jobs to annotators on the task page (**Assignee**).

> For later full-dataset tasks, upload from `digital twin project datasets/<folder>`, or stage a batch first.
> In the Archive folders, never select the `__MACOSX` subfolder.

### 3. Annotate

Open a job (task → **Job #…**). For every image:

1. In the left toolbar choose **Draw new rectangle** (the rectangle icon), set **Label** to the component,
   set **Drawing method: By 2 points**, and click **Shape** (not *Track*).
2. Drag from one corner to the opposite corner. Press **N** to repeat the last drawing tool. To change a box's label,
   select the box and use the label dropdown in the objects sidebar.
3. Box **every** component of **all seven classes** in the image (rules below), not only the task's class.
4. **Next image: F**, **previous: D**. **Save: Ctrl+S** (also saves automatically).
5. When the job is done: **Menu → Finish the job**, or set the job stage to *validation* for review.

Use the objects sidebar to check the box count and labels per image before moving on.

### 4. Review

A second person opens each job in the **validation** stage. They check that every component is boxed, that the labels
are right, and that boxes are tight. They fix issues, or raise them with *Issues* (right-click → *Open an issue*). Then
they set the job to **acceptance / completed**.

### 5. Export in YOLO detection format

1. **Projects → digital-twin → Actions (⋮) → Export dataset**.
2. **Format**: **YOLO 1.1**. *Ultralytics YOLO Detection 1.0* also works if your CVAT offers it.
3. **Save images**: **off**. Labels are enough, because the images are already on disk.
4. **OK**. Download the archive when it's ready (in newer CVAT versions, from the **Requests** page).
5. Unzip it into `annotation/exports/<YYYY-MM-DD>_batch01_cvat/`. A YOLO 1.1 export contains
   `obj.names`, `obj.data`, a subset list (e.g. `train.txt` or `default.txt`) and one `.txt` per image, each line being
   `class_id x_center y_center width height`, normalised to 0–1.

Exporting the project exports every task in it. Export only when all batch tasks are complete. Images in
unfinished tasks come out as **empty** label files, which validation reports as errors.

### 6. Validate

```powershell
cd backend
.venv\Scripts\python.exe -m app.training.validate_annotations ..\annotation\exports\2026-11-05_batch01_cvat
```

The output should say `VALID`, with ~175 images paired. The remaining ~368 images are listed as *not labelled yet*;
that's a warning, not an error. Fix every **ERROR** in CVAT, re-export and validate again. See *Review and validation* for what is checked.

This raw export is a structural check only. To make it training data, `review_tool import-yolo` it and sign off
each image (see *CVAT as an alternative editor* above). `prepare_dataset` refuses exports without a review manifest.

**Don't start training yourself.** Report the validated export, and the pilot run will be planned from there.

## Option B: Label Studio (single annotator, local)

1. Install Label Studio in its **own** environment, not the backend `.venv`
   (`py -3.12 -m venv ls-env && ls-env\Scripts\pip install label-studio`, then `ls-env\Scripts\label-studio`).
2. Create a project → *Labeling Interface → Code* → paste `label_studio_config.xml`.
3. Import a batch folder's images (one project, or one project per class).
4. Draw rectangles following the rules below.
5. *Export* → **YOLO** → unzip into `annotation/exports/<date>_labelstudio/`.
   Label Studio prefixes file names with an 8-character ID; validation handles that automatically.

## Annotation rules

1. **Box every instance of every class** visible in the image, not only the folder's class. An open
   cabinet photo in the PSU folder also gets `equipment_cabinet`, `battery` and `router` boxes where present.
   (The audit found the same cabinet photo in both the PSU and battery folders, so this matters.)
2. **Tight boxes**: edges touch the outermost visible pixels of the component; include mounting
   brackets only if they are part of the unit.
3. **Partially visible / occluded**: box the visible part if roughly a third or more of the component is visible
   and it's recognisable; otherwise skip it.
4. **Nested components**: box the cabinet *and* the components inside it, each with its own class.
5. **Cables** *(finalised team decision)*:
   - **Clearly distinguishable cables**: draw a **separate box for each cable**, including its connector.
   - **Visually inseparable bundles**: draw **one box around the entire bundle**.
   - **Never** draw several overlapping boxes for an inseparable bundle.

   If you're unsure whether cables can be told apart, treat them as a bundle.
6. **Screenshots**: ignore the black letterbox bars, phone UI and web-page text; box only the component.
7. **Product renders vs field photos**: annotate both the same way.
8. **Every image must contain at least one box of its folder's class.** If it truly doesn't show that
   component, note the file name and tell the dataset owner; don't force a box.
9. Consistency over speed: when unsure, ask and add the decision to this guide.
10. **One box per physical object.** Never keep two boxes on the same component (the model often proposes the same
    object twice under different classes; keep one and fix its class).
11. **Candidates are suggestions.** Correct the class whenever it is wrong, delete false positives, and add the
    objects the model missed. Don't guess: mark the box **uncertain**, which sends the image to second review.
12. The folder an image came from never limits what you annotate.

## Review and validation

`validate_annotations` checks:
- class list order against the taxonomy (`obj.names`, `classes.txt`, `data.yaml` or `notes.json`);
- 5 values per line;
- integer class IDs 0–6;
- normalised coordinates, with boxes inside the image;
- that every label file pairs with exactly one dataset image (subset list files are ignored);
- no empty label files, unless the image is listed in `verified_negatives.txt`;
- no duplicate same-class boxes (IoU > 0.95), which are an error;
- a box of the folder's class in every labelled image;
- no directory carrying the candidate marker, so machine candidates are refused;
- with `--require-review` (always on in `prepare_dataset`): a `review_manifest.json` must exist. Every label file must
  be listed there as `REVIEWED`, with a SHA-256 hash equal to the file's content. That catches unreviewed or
  candidate labels copied into the export, and labels edited after sign-off. Every listed image must have its label.

The report prints the number of images, annotations and invalid annotations, the per-class distribution, the
images ready for training and the images still requiring review (from the manifest). Write it in full with `--json`.
Images with several classes are normal and are never rejected for that.

It warns, for review, about:
- tiny boxes;
- near-whole-image boxes;
- overlapping cable boxes (the bundle rule says one box);
- images that are still unlabelled.
Exit code 0 means valid. Write the full report with `--json <file>`.

## Assisted annotation (candidates, never ground truth)

`backend/app/training/auto_annotate.py` runs a detector over a batch and writes **candidate** boxes for
human review. It applies the rules in this guide:
- same-class duplicates are suppressed;
- overlapping cable boxes are merged into **one bundle box** and flagged;
- cross-class conflicts, near-whole-image boxes, tiny boxes and ambiguous label mappings are flagged;
- no box is ever created from a folder name or as a whole-image placeholder. The folder is used only to flag
  "expected component not proposed".

### Directory meaning

| Directory | Content | Can it be trained on? |
|---|---|---|
| `annotation/candidates/<run>/` | machine candidates: `labels/`, `cvat_import/*.zip` (CVAT pre-annotations per task), `previews/{high_confidence,needs_review,no_detection}/`, `manual_review.csv`, `review.html`, `report.md` | **no**: the `CANDIDATES_NOT_GROUND_TRUTH.txt` marker makes validation and preparation refuse it |
| `annotation/rejected/<run>/` | candidate sets judged unusable, kept for traceability, with the reason in `REJECTED.md` | no |
| `annotation/reviewed/<batch>/` | review state: status, boxes with provenance (candidate / human / imported), deleted candidates, audit log | not directly; only through `review_tool export` |
| `annotation/exports/<export>/` | `review_tool export` output: labels of REVIEWED images + `review_manifest.json` (status, reviewer, label hash per image) | yes, after `validate_annotations --require-review` passes |
| `verified_negatives.txt` (inside an export) | images a reviewer confirmed contain none of the seven classes; their label files may be empty | yes |

Every candidate tier is only a review priority:
- **High confidence** (solid box in the preview): confidence ≥ 0.5, a direct label mapping and no flags.
- **Low confidence** (dashed box, `?`): anything else.
- **Image statuses**: `high_confidence`, `needs_review`, `no_detection`, plus a *multiple classes* tag.
  Open `review.html` in a browser to filter by status.

### What was tried on `batch_01` (2026-11-01, measured)

| Approach | Result |
|---|---|
| Pretrained COCO YOLO11n | not applicable: none of the seven classes exist in COCO |
| **YOLOE-11L prompt-free** (`yoloe-11l-seg-pf.pt`, 74 MB, CPU, built-in 4585-term vocabulary) | **unusable**: own component proposed in 10/175 images (1/175 at conf ≥ 0.5), 116/175 with no candidate, 0 high-confidence images. Raw labels were "darkness", "storage box", "bottle", "screw". Rejected: `annotation/rejected/yoloe_pf_batch01_eval/` |
| YOLOE / YOLO-World **text prompts** (your prompts are in `DEFAULT_PROMPTS`) | **not run**: needs the `clip` tokenizer package from GitHub; installing it was blocked pending your approval |
| **Grounding DINO-tiny** (`--mode gdino`, 2026-11-02; `transformers` from PyPI, ~690 MB model in the user's Hugging Face cache, CPU) | **boxes useful, class names not**: see *Grounding DINO candidates* below. Run: `annotation/candidates/gdino_tiny_batch01/` |
| SAM / SAM2 | class-agnostic, so it can't name the component. Possible later for tightening reviewed boxes, but not needed for YOLO boxes |

### Grounding DINO candidates (`gdino_tiny_batch01`)

```powershell
python -m app.training.auto_annotate --mode gdino --name gdino_tiny_batch01 --min-conf 0.25 --high-conf 0.45
```

How it differs from the YOLOE runs:
- **Letterbox crop.** Each image is trimmed to its photo content before inference (the phone screenshots are
  65–82% black/white bars), and boxes are mapped back to the staged image's pixels, so the CVAT zips match the
  uploaded images. This cut CPU time from ~18 s to ~7 s per image. `--no-crop` disables it.
- **One class per box.** Grounding DINO's stock decoding merges phrases ("power supply unit equipment cabinet").
  Instead, each box is scored per class (mean token probability of that class's phrase) and gets one class; boxes of
  different classes on the same object are collapsed into one, with the alternatives kept as `alt_class:` flags.

Measured on a 7-image probe before the full run: the boxes sit well on the main objects (battery, PSU, cable,
router), but **the class is mostly wrong**. The model labels most box-shaped equipment as whichever device word is
in the prompt ("radio unit", or "router" when the prompt was changed). Three prompt variants were tried; the bias
moves but doesn't go away. So:

- Use these candidates as **box pre-annotations**: in CVAT, keep or adjust the box and **set the correct label**
  (one dropdown change is faster than drawing). Delete false boxes and add missing ones.
- Never trust the candidate class or its confidence. Confidence is a detector score, not accuracy.
- Every candidate is low tier by design; there are no "high confidence" shortcuts.

Review `contact_sheets/<task>.jpg` (one grid per class) or `review.html` first. If a task's candidates are mostly
noise, skip the zip for that task and draw from scratch.

**Recommendation: annotate `batch_01` in CVAT (175 images), starting from the Grounding DINO boxes where they help,
then switch to model-assisted labelling with the pilot model.**

1. After `batch_01` is validated and the pilot model is trained, generate candidates for the next batch:
   ```powershell
   # batch_02 is already staged (2026-11-02): 168 images, cable 18 (the cable folder has only 41 independent images)
   python -m app.training.auto_annotate --mode custom --model models\custom\digital_twin_best.pt --batch ..\annotation\batches\batch_02 --name pilot_batch02
   ```
2. Review `annotation/candidates/pilot_batch02/review.html`. If the candidates look useful, upload each
   `cvat_import/<batch>_<task>.zip` into the matching CVAT task (*Actions → Upload annotations → YOLO 1.1*).
3. **Correct every image** in CVAT, export, and validate. Only that export is ground truth.

To try text prompts instead, first approve installing the tokenizer:
`.venv\Scripts\python.exe -m pip install ftfy regex "git+https://github.com/ultralytics/CLIP.git"`.
Then run `python -m app.training.auto_annotate --mode text --model yoloe-11l-seg.pt --name yoloe_text_batch01`
(this downloads the YOLOE text checkpoint and the MobileCLIP text encoder).
