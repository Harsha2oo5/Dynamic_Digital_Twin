"""Batch pseudo-labelling over the annotation manifest (all 543 images by default). The dataset is only read.

From backend/:
    python -m app.training.pseudo_label.run --name smoke_21 --limit 21
    python -m app.training.pseudo_label.run --name full_v1
    python -m app.training.pseudo_label.run --name full_v1_sam --sam
    python -m app.training.pseudo_label.run --name live_test --detections live --limit 7
    python -m app.training.pseudo_label.run --name strict --set min_final_margin=0.2 --set folder_prior_boost=1.0

Every run writes to annotation/pseudo_labels/<name>/ (never overwritten) and saves its exact configuration.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.core.config import PROJECT_DIR
from app.core.device import configure_torch
from app.training.annotation_kit import read_manifest
from app.training.audit_dataset import read_image
from app.training.pseudo_label import outputs as out_mod
from app.training.pseudo_label.config import DEFAULT_CONFIG_FILE, PseudoLabelConfig, load_config
from app.training.pseudo_label.pipeline import (
    CACHE_FILE,
    CachedScores,
    LiveDetections,
    StoredDetections,
    decide_image,
    score_image,
)
from app.training.pseudo_label.prototypes import Prototypes

DEFAULT_OUT_ROOT = PROJECT_DIR / "annotation" / "pseudo_labels"


def select_rows(manifest: dict[str, dict], limit: int | None) -> list[dict]:
    """Manifest rows sorted by path; with `limit`, round-robin across folder classes so every class is covered."""
    rows = sorted(manifest.values(), key=lambda r: r["original_path"])
    if limit is None or limit >= len(rows):
        return rows
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_folder[r["folder"]].append(r)
    out, i = [], 0
    while len(out) < limit:
        for items in by_folder.values():
            if i < len(items) and len(out) < limit:
                out.append(items[i])
        i += 1
    return out


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def run(name: str, cfg: PseudoLabelConfig, out_root: Path, limit: int | None = None, scorer=None, source=None,
        refiner=None, log=print, reuse_scores: Path | None = None) -> dict:
    """Run the pipeline. `scorer`, `source` and `refiner` default to the real models; tests inject fakes."""
    out = out_root / name
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    dataset_root = Path(cfg.dataset_root)
    rows = select_rows(read_manifest(Path(cfg.manifest)), limit)

    if cfg.offline:  # cached models only: never download anything
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from app.training.pseudo_label import models

    if scorer is None and reuse_scores is not None:
        scorer = CachedScores(reuse_scores, cfg)
    if source is None:
        source = (StoredDetections(Path(cfg.stored_run), dataset_root) if cfg.detections == "stored"
                  else LiveDetections(models.GroundingDino(cfg), cfg.letterbox_crop))
    if scorer is None:
        scorer = models.ClipScorer(cfg)
    if refiner is None and cfg.use_sam:
        refiner = models.SamRefiner(cfg)

    out.mkdir(parents=True)
    out_mod.write_marker(out)
    cfg_hash = cfg.hash()
    out_mod.write_json(out / "config.json", {"config_hash": cfg_hash, **cfg.to_dict()})

    t0 = time.time()
    entries, failures = [], []

    def fail(row: dict, exc: Exception, n: int) -> None:
        failures.append({"original_path": row["original_path"], "error": f"{type(exc).__name__}: {exc}"})
        log(f"[{n}/{len(rows)}] FAILED {row['original_path']}: {exc}")

    # pass 1: detections + CLIP scores and embeddings for every image
    for n, row in enumerate(rows, 1):
        try:
            image = read_image(dataset_root / row["original_path"])
            if image is None:
                raise ValueError("unreadable image")
            dets, photo_box = source.get(row["original_path"], image)
            entries.append(score_image(row, image, dets, photo_box, scorer, cfg))
        except Exception as exc:  # noqa: BLE001 - one bad image must not end a long run
            fail(row, exc, n)
            continue
        if n % 25 == 0 or n == len(rows):
            log(f"[pass 1: {n}/{len(rows)}] {time.time() - t0:.0f}s elapsed")
    cache = {}
    for e in entries:
        if len(e["emb"]) and e.get("flip_emb") is not None:
            cache[f"{e['row']['key']}__emb"] = e["emb"]
            cache[f"{e['row']['key']}__flip"] = e["flip_emb"]
    if cache:
        np.savez_compressed(out / CACHE_FILE, **cache)

    # pass 2: prototypes (leave-group-out) + fusion + suppression; previews need the image again
    protos = Prototypes(entries, cfg) if cfg.use_prototypes else None
    records = []
    with (out / "records.jsonl").open("w", encoding="utf-8") as jl:
        for n, e in enumerate(entries, 1):
            row = e["row"]
            try:
                image = read_image(dataset_root / row["original_path"]) if (cfg.write_previews or refiner) else None
                rec = decide_image(e, cfg, protos, refiner, image)
                if cfg.write_previews:
                    rec["preview"] = out_mod.write_preview(out, image, rec, cfg.preview_max_side)
            except Exception as exc:  # noqa: BLE001
                fail(row, exc, n)
                continue
            records.append(rec)
            jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elapsed = round(time.time() - t0, 1)

    hashes = {}
    for rec in records:
        h = out_mod.write_labels(out, rec)
        if h:
            hashes[rec["key"]] = h
    manifest = out_mod.build_manifest(name, cfg, cfg_hash, source.name, records, hashes)
    manifest["failures"] = failures
    out_mod.write_json(out / out_mod.MANIFEST, manifest)

    meta = {"run": name, "created_at": manifest["created_at"], "config_hash": cfg_hash,
            "detection_source": source.name, "scores": getattr(scorer, "name", "clip:live"),
            "folder_prior_boost": cfg.folder_prior_boost, "use_sam": cfg.use_sam,
            "use_prototypes": cfg.use_prototypes, "prototype_anchors": protos.anchors() if protos else None,
            "elapsed_s": elapsed, "failures": len(failures)}
    summary = out_mod.summarize(records, meta)
    out_mod.write_json(out / "report.json", summary)
    (out / "report.md").write_text(out_mod.report_markdown(summary), encoding="utf-8")
    if cfg.write_previews:
        out_mod.write_gallery(out, name, records)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run folder name under the output root")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="YAML config (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="number of images, round-robin across folders")
    p.add_argument("--detections", choices=["stored", "live"], default=None)
    p.add_argument("--stored-run", default=None, help="zero-shot run folder with predictions/predictions.json")
    p.add_argument("--sam", action="store_true", help="refine accepted boxes with MobileSAM (~+3 s/image)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (JSON)")
    p.add_argument("--reuse-scores", type=Path, default=None, metavar="RUN_DIR",
                   help="reuse the CLIP scores of an earlier run (same CLIP settings) to re-tune thresholds quickly")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    a = p.parse_args()

    overrides = parse_overrides(a.set)
    if a.detections:
        overrides["detections"] = a.detections
    if a.stored_run:
        overrides["stored_run"] = a.stored_run
    if a.sam:
        overrides["use_sam"] = True
    try:
        cfg = load_config(a.config if a.config and a.config.is_file() else None, overrides)
        configure_torch(0)
        s = run(a.name, cfg, a.out_root, a.limit, reuse_scores=a.reuse_scores)
    except (ValueError, KeyError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\n{s['images']} images · {s['training_images']} usable for training · boxes {s['boxes_by_tier']}")
    print(f"Report: {a.out_root / a.name / 'report.md'}")
    print(out_mod.DISCLAIMER)


if __name__ == "__main__":
    main()
