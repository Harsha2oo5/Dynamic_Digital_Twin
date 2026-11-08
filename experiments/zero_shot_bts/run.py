"""Run the zero-shot BTS experiment over the dataset (read-only) and write a self-contained run folder.

From the project root, with the backend virtualenv:
    backend\\.venv\\Scripts\\python.exe -m experiments.zero_shot_bts.run                     # all 543 images, GD + CLIP
    backend\\.venv\\Scripts\\python.exe -m experiments.zero_shot_bts.run --limit 21          # 3 per folder
    backend\\.venv\\Scripts\\python.exe -m experiments.zero_shot_bts.run --mode gdino        # detection only
    backend\\.venv\\Scripts\\python.exe -m experiments.zero_shot_bts.run --limit 7 --sam     # + MobileSAM refinement

Output: experiments/zero_shot_bts/runs/<name>/
    config/config.json            exact models, prompts and thresholds
    predictions/predictions.json  every image and detection (machine-readable); predictions.jsonl while running
    visualizations/*.jpg          annotated photo regions
    crops/*.jpg                   every detection crop that CLIP saw
    reports/report.md|json        summary; no accuracy metrics (no ground truth exists)
    gallery.html                  browser gallery (open the file directly)
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

from experiments.zero_shot_bts import PROJECT_DIR
from experiments.zero_shot_bts.boxes import area, crop, nms, offset
from experiments.zero_shot_bts.config import UNCERTAIN, UNKNOWN, Config
from experiments.zero_shot_bts.models import ClipClassifier, GroundingDinoDetector, SamRefiner
from experiments.zero_shot_bts.outputs import (draw, pick_examples, report_markdown, save_crop, summarize,
                                               write_gallery, write_jpeg)

from app.training.audit_dataset import read_image
from app.training.auto_annotate import content_box
from app.training.dataset_validator import IMAGE_EXTS, is_macos_metadata
from app.training.taxonomy import CLASS_NAMES, FOLDER_TO_CLASS

EXP_DIR = Path(__file__).resolve().parent
LOW_GDINO = 0.35


def list_images(dataset: Path) -> list[tuple[Path, str]]:
    """(image, top-level folder) for every image of the mapped class folders, sorted."""
    out = []
    for folder in FOLDER_TO_CLASS:
        d = dataset / folder
        if d.is_dir():
            out += [(p, folder) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS and not is_macos_metadata(p)]
    return out


def select(images: list[tuple[Path, str]], limit: int | None) -> list[tuple[Path, str]]:
    """Round-robin across folders so a limited run still covers every class folder."""
    if not limit or limit >= len(images):
        return images
    by_folder: dict[str, list] = {}
    for item in images:
        by_folder.setdefault(item[1], []).append(item)
    out, i = [], 0
    while len(out) < limit:
        for items in by_folder.values():
            if i < len(items) and len(out) < limit:
                out.append(items[i])
        i += 1
    return out


def slug(folder: str, path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{folder}__{path.stem}").strip("_")


def combine(det: dict, use_clip: bool, photo_area: float) -> dict:
    """Final label = component only when Grounding DINO and CLIP agree; otherwise UNCERTAIN / UNKNOWN."""
    flags = []
    if det["gdino_confidence"] < LOW_GDINO:
        flags.append("low_gdino_confidence")
    if area(det["xyxy"]) > 0.9 * photo_area:
        flags.append("near_whole_photo")
    if not use_clip:
        det.update(agreement="no_clip", final_label=det["gdino_class"])
        flags.append("unverified_by_clip")
    elif det["clip_label"] == UNKNOWN:
        det.update(agreement="clip_unknown", final_label=UNKNOWN)
        flags.append("clip_unknown")
    elif det["clip_label"] == UNCERTAIN:
        det.update(agreement="clip_uncertain", final_label=UNCERTAIN)
        flags.append("clip_uncertain")
    elif det["clip_label"] == det["gdino_class"]:
        det.update(agreement="agree", final_label=det["gdino_class"])
    else:
        det.update(agreement="disagree", final_label=UNCERTAIN)
        flags.append(f"disagreement:gdino={det['gdino_class']},clip={det['clip_label']}")
    det["flags"] = flags
    return det


def process(path: Path, folder: str, cfg: Config, detector, classifier, refiner, out: Path) -> dict:
    timing = {}
    t0 = time.perf_counter()
    image = read_image(path)
    if image is None:
        raise ValueError(f"unreadable image {path}")
    h, w = image.shape[:2]
    cx0, cy0, cx1, cy1 = content_box(image) if cfg.letterbox_crop else (0, 0, w, h)
    photo = image[cy0:cy1, cx0:cx1]

    t = time.perf_counter()
    raw = [{**d, "xyxy": offset(d["xyxy"], cx0, cy0)} for d in detector.detect(photo)]
    dets = nms(raw, cfg.nms_iou)
    timing["gdino_s"] = round(time.perf_counter() - t, 3)

    if refiner is not None:
        t = time.perf_counter()
        for d, r in zip(dets, refiner.refine(image, [d["xyxy"] for d in dets])):
            d.update(r)
        timing["sam_s"] = round(time.perf_counter() - t, 3)

    sid = slug(folder, path)
    crops = []
    for i, d in enumerate(dets):
        d["id"] = i
        box = d["sam_xyxy"] if d.get("sam_used") else d["xyxy"]
        c = crop(image, box, cfg.crop_padding)
        crops.append(c)
        d["crop"] = f"crops/{sid}_{i}.jpg"
        save_crop(out / d["crop"], c)
    if classifier is not None:
        t = time.perf_counter()
        for d, r in zip(dets, classifier.classify(crops)):
            d.update(r)
        timing["clip_s"] = round(time.perf_counter() - t, 3)

    photo_area = float((cx1 - cx0) * (cy1 - cy0))
    for d in dets:
        combine(d, classifier is not None, photo_area)
        d["xyxy"] = [round(v, 1) for v in d["xyxy"]]
        d["gdino_confidence"] = round(d["gdino_confidence"], 4)
        d["prompt_score"] = round(d["prompt_score"], 4)
    rec = {"file": path.name, "folder": folder, "folder_class": CLASS_NAMES.get(FOLDER_TO_CLASS.get(folder)),
           "source_path": str(path), "width": w, "height": h, "photo_box": [cx0, cy0, cx1, cy1],
           "detections": dets, "visualization": f"visualizations/{sid}.jpg"}
    timing["total_s"] = round(time.perf_counter() - t0, 3)
    rec["timing"] = timing
    write_jpeg(out / rec["visualization"], draw(image, rec, (cx0, cy0, cx1, cy1)))
    return rec


def main() -> None:
    from app.core.config import get_settings

    cfg = Config()
    ap = argparse.ArgumentParser(description="Zero-shot BTS component detection (Grounding DINO -> CLIP -> optional SAM).")
    ap.add_argument("--dataset", type=Path, default=None, help="dataset root (read-only); default: the project dataset")
    ap.add_argument("--limit", type=int, default=None, help="process N images (round-robin over folders); default all")
    ap.add_argument("--mode", choices=["gdino", "gdino+clip"], default="gdino+clip")
    ap.add_argument("--sam", action="store_true", help="refine boxes with MobileSAM (optional, slower)")
    ap.add_argument("--box-threshold", type=float, default=cfg.box_threshold)
    ap.add_argument("--nms-iou", type=float, default=cfg.nms_iou)
    ap.add_argument("--clip-min-similarity", type=float, default=cfg.clip_min_similarity)
    ap.add_argument("--clip-min-margin", type=float, default=cfg.clip_min_margin)
    ap.add_argument("--gdino-model", default=cfg.gdino_model)
    ap.add_argument("--clip-model", default=cfg.clip_model)
    ap.add_argument("--no-letterbox-crop", action="store_true")
    ap.add_argument("--name", default=None, help="run folder name (default: timestamp + mode)")
    args = ap.parse_args()

    cfg.box_threshold, cfg.nms_iou = args.box_threshold, args.nms_iou
    cfg.clip_min_similarity, cfg.clip_min_margin = args.clip_min_similarity, args.clip_min_margin
    cfg.gdino_model, cfg.clip_model = args.gdino_model, args.clip_model
    cfg.use_clip, cfg.use_sam, cfg.letterbox_crop = args.mode == "gdino+clip", args.sam, not args.no_letterbox_crop

    dataset = (args.dataset or get_settings().resolve(get_settings().dataset_root)).resolve()
    images = select(list_images(dataset), args.limit)
    if not images:
        sys.exit(f"No images found under {dataset}")
    name = args.name or datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.mode.replace('+', '_')}{'_sam' if args.sam else ''}")
    out = EXP_DIR / "runs" / name
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    if out.resolve().is_relative_to(dataset):
        sys.exit("Output must not be inside the dataset")
    for sub in ("config", "predictions", "visualizations", "crops", "reports"):
        (out / sub).mkdir(parents=True)
    (out / "config" / "config.json").write_text(json.dumps({**cfg.to_dict(), "dataset": str(dataset),
                                                            "images_selected": len(images)}, indent=2), encoding="utf-8")

    detector = GroundingDinoDetector(cfg)
    classifier = ClipClassifier(cfg) if cfg.use_clip else None
    refiner = SamRefiner(cfg, EXP_DIR / "weights") if cfg.use_sam else None
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    records, failures, started = [], [], time.time()
    print(f"Zero-shot BTS run '{name}': {len(images)} images, mode {args.mode}{' + SAM' if args.sam else ''}, CPU")
    jsonl = (out / "predictions" / "predictions.jsonl").open("w", encoding="utf-8")
    try:
        for n, (path, folder) in enumerate(images, 1):
            try:
                rec = process(path, folder, cfg, detector, classifier, refiner, out)
            except Exception as exc:  # noqa: BLE001 - record and continue; reported at the end
                failures.append({"image": str(path), "error": f"{type(exc).__name__}: {exc}"})
                print(f"  [{n}/{len(images)}] FAILED {folder}/{path.name}: {exc}")
                continue
            records.append(rec)
            jsonl.write(json.dumps(rec) + "\n")
            peak_rss = max(peak_rss, proc.memory_info().rss)
            if n % cfg.batch_size == 0 or n == len(images) or n <= 2:
                jsonl.flush()
                el = time.time() - started
                print(f"  [{n}/{len(images)}] {folder}/{path.name}: {len(rec['detections'])} det, "
                      f"{rec['timing']['total_s']:.1f}s | elapsed {el / 60:.1f} min, ETA {el / n * (len(images) - n) / 60:.1f} min, "
                      f"RSS {peak_rss / 2**30:.2f} GB", flush=True)
    except KeyboardInterrupt:
        print("Interrupted: writing outputs for the images processed so far")
    finally:
        jsonl.close()

    runtime = {"wall_minutes": round((time.time() - started) / 60, 1), "peak_rss_gb": round(peak_rss / 2**30, 2),
               "model_load_seconds": {"gdino": detector.load_seconds, "clip": classifier and classifier.load_seconds,
                                      "sam": refiner and refiner.load_seconds},
               "device": "cpu", "failures": failures}
    (out / "predictions" / "predictions.json").write_text(json.dumps({"run": name, "images": records}, indent=1), encoding="utf-8")
    summary = summarize(records, cfg.to_dict(), runtime)
    examples = pick_examples(records)
    (out / "reports" / "report.json").write_text(json.dumps({**summary, "examples": examples}, indent=2), encoding="utf-8")
    (out / "reports" / "report.md").write_text(report_markdown(summary, examples), encoding="utf-8")
    write_gallery(out / "gallery.html", records, name)
    print(f"Done: {len(records)} images, {summary['total_detections']} detections, {len(failures)} failures, "
          f"{runtime['wall_minutes']} min, peak RSS {runtime['peak_rss_gb']} GB")
    print(f"Gallery: {out / 'gallery.html'}\nReport:  {out / 'reports' / 'report.md'}")


if __name__ == "__main__":
    main()
