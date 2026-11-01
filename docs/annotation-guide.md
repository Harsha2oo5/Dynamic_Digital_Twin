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
| `exports/` | put each YOLO export here (one subfolder per export) |

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
- no directory carrying the candidate marker, so machine candidates are refused.

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
| `annotation/exports/<export>/` | **human-reviewed** CVAT exports, the only ground truth | yes, after `validate_annotations` passes |
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
| Grounding DINO (`transformers`) | not attempted: large dependency (~700 MB model), slow on CPU |
| SAM / SAM2 | class-agnostic, so it can't name the component. Possible later for tightening reviewed boxes, but not needed for YOLO boxes |

**Recommendation: annotate `batch_01` by hand in CVAT (175 images), then use model-assisted labelling.**

1. After `batch_01` is validated and the pilot model is trained, generate candidates for the next batch:
   ```powershell
   python -m app.training.annotation_batch --name batch_02 --per-class 25
   python -m app.training.auto_annotate --mode custom --model models\custom\digital_twin_best.pt --batch ..\annotation\batches\batch_02 --name pilot_batch02
   ```
2. Review `annotation/candidates/pilot_batch02/review.html`. If the candidates look useful, upload each
   `cvat_import/<batch>_<task>.zip` into the matching CVAT task (*Actions → Upload annotations → YOLO 1.1*).
3. **Correct every image** in CVAT, export, and validate. Only that export is ground truth.

To try text prompts instead, first approve installing the tokenizer:
`.venv\Scripts\python.exe -m pip install ftfy regex "git+https://github.com/ultralytics/CLIP.git"`.
Then run `python -m app.training.auto_annotate --mode text --model yoloe-11l-seg.pt --name yoloe_text_batch01`
(this downloads the YOLOE text checkpoint and the MobileCLIP text encoder).
