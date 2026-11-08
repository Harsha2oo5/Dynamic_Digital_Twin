"""Class-wise weak-label diagnostics + CLIP prompt and crop experiments on the existing zero-shot run.

From the project root:
    backend\\.venv\\Scripts\\python.exe -m experiments.zero_shot_bts.diagnostics.run
    (--run <folder under runs/>, default full_543_gdino_clip; --no-clip for the stored-results analysis only)

Reads the run (never modifies it) and the original images (read-only). Grounding DINO boxes are reused exactly;
only CLIP crops and text prompts vary. Writes reports to experiments/zero_shot_bts/diagnostics/ (embedding cache in cache/)."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.zero_shot_bts.boxes import crop
from experiments.zero_shot_bts.diagnostics import clip_eval as ce
from experiments.zero_shot_bts.diagnostics import weak_labels as wl

from app.training.audit_dataset import read_image

DIAG_DIR = Path(__file__).resolve().parent
EXP_DIR = DIAG_DIR.parent
DISCLAIMER = ("Weak-label diagnostics: folder names are image-level labels, not object-level ground truth. An image "
              "can show several components and a box can be a part, so these rates are NOT accuracy, precision, "
              "recall or mAP, and matching a folder does not prove a prediction is correct.")


# --- embeddings ---------------------------------------------------------------------------------------
def embed_all(records: list[dict], embedder, cache: Path, log=print) -> tuple[dict, list, np.ndarray]:
    """Embed every detection crop at every padding plus the whole photo of each image, reading each original
    image once. Cached (keys must match) so the analysis can be re-run without CLIP."""
    keys = [(i, d["id"]) for i, r in enumerate(records) for d in r["detections"]]
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        if [tuple(k) for k in data["keys"]] == keys and len(data["photo"]) == len(records):
            log(f"Using cached embeddings {cache}")
            return ({p: data[f"pad_{p}"].astype(np.float32) for p in ce.PADDINGS}, keys, data["photo"].astype(np.float32))
    bufs = {p: [] for p in ce.PADDINGS}
    out = {p: [] for p in ce.PADDINGS}
    photo_buf, photo_out = [], []
    t0 = time.time()

    def flush(force=False):
        for p in ce.PADDINGS:
            if bufs[p] and (force or len(bufs[p]) >= 32):
                out[p].append(embedder.images(bufs[p]))
                bufs[p].clear()
        if photo_buf and (force or len(photo_buf) >= 32):
            photo_out.append(embedder.images(photo_buf))
            photo_buf.clear()

    for i, r in enumerate(records):
        img = read_image(Path(r["source_path"]))
        if img is None:
            raise ValueError(f"cannot read {r['source_path']}")
        for d in r["detections"]:
            for p in ce.PADDINGS:
                bufs[p].append(crop(img, d["xyxy"], p))
        x0, y0, x1, y1 = r["photo_box"]
        photo_buf.append(img[y0:y1, x0:x1])
        flush()
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            log(f"  embedded {i + 1}/{len(records)} images ({time.time() - t0:.0f}s)")
    flush(force=True)
    emb = {p: np.concatenate(out[p]) for p in ce.PADDINGS}
    photo = np.concatenate(photo_out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, keys=np.array(keys, dtype=object), photo=photo.astype(np.float16),
                        **{f"pad_{p}": e.astype(np.float16) for p, e in emb.items()})
    return emb, keys, photo


def ranks_dict(keys: list, sims: np.ndarray) -> dict:
    return dict(zip(keys, ce.rankings(sims)))


def photo_records(records: list[dict]) -> list[dict]:
    """One pseudo-detection per image covering the whole photo (classification without detection)."""
    return [{**r, "detections": [{"id": 0, "gdino_confidence": 1.0, "gdino_class": "-", "xyxy": r["photo_box"]}]}
            for r in records]


def split_half(records: list[dict], ranks: dict) -> tuple[float, float]:
    """primary_top1 macro on even- vs odd-indexed images: a stability check against picking a configuration
    that only fits this particular set of weak labels."""
    halves = []
    for parity in (0, 1):
        idx = [i for i in range(len(records)) if i % 2 == parity]
        sub = [records[i] for i in idx]
        sub_ranks = {(n, k[1]): v for n, i in enumerate(idx) for k, v in ranks.items() if k[0] == i}
        halves.append(wl.weak_metrics(sub, sub_ranks)["overall"]["primary_top1_macro"])
    return tuple(halves)


# --- examples -------------------------------------------------------------------------------------------
def tile(img: np.ndarray, lines: list[str], w: int = 300, h: int = 230) -> np.ndarray:
    canvas = np.full((h + 16 * len(lines) + 10, w, 3), 24, np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    for n, t in enumerate(lines):
        cv2.putText(canvas, t[:46], (6, h + 16 + 16 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def grid(tiles: list[np.ndarray], title: str, cols: int = 4) -> np.ndarray:
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 24))
    rows = [np.hstack([cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(60, 60, 60)) for t in tiles[i:i + cols]])
            for i in range(0, len(tiles), cols)]
    body = np.vstack(rows)
    head = np.full((34, body.shape[1], 3), 15, np.uint8)
    cv2.putText(head, title[:150], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([head, body])


def top3_text(sims_row: np.ndarray) -> str:
    order = np.argsort(-sims_row)[:3]
    return " ".join(f"{ce.CLASSES[j][:7]} {sims_row[j]:.3f}" for j in order)


def make_examples(records, keys, sims_orig, sims_best, best_name, run_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    kidx = {k: n for n, k in enumerate(keys)}
    crops_cache = {}

    def load(i, d):
        p = run_dir / d["crop"]
        if p not in crops_cache:
            crops_cache[p] = read_image(p)
        return crops_cache[p]

    def caption(i, d, extra=None):
        r, n = records[i], kidx[(i, d["id"])]
        return [f"{r['folder'][:18]}/{r['file'][:22]}", f"folder: {r['folder_class']}",
                f"GD: {d['gdino_class']} {d['gdino_confidence']:.2f}", f"CLIP: {top3_text(sims_orig[n])}"] + (extra or [])

    picks: dict[str, list] = {k: [] for k in ["clip_matches_folder", "clip_disagrees_with_gdino", "near_identical_top2",
                                              "incomplete_object_small_box", "prompt_changes_prediction"]}
    seen_cls = {k: set() for k in picks}
    for i, r in enumerate(records):
        p = wl.primary(r)
        for d in r["detections"]:
            n = kidx[(i, d["id"])]
            o = ce.CLASSES[int(np.argmax(sims_orig[n]))]
            b = ce.CLASSES[int(np.argmax(sims_best[n]))]
            srt = np.sort(sims_orig[n])
            cls = r["folder_class"]
            cand = []
            if d is p and o == cls:
                cand.append(("clip_matches_folder", []))
            if d.get("agreement") == "disagree":
                cand.append(("clip_disagrees_with_gdino", []))
            if srt[-1] - srt[-2] < 0.001:
                cand.append(("near_identical_top2", [f"margin {srt[-1] - srt[-2]:.4f}"]))
            if wl.box_fraction(r, d) < 0.05:
                cand.append(("incomplete_object_small_box", [f"box = {100 * wl.box_fraction(r, d):.1f}% of photo"]))
            if d is p and o != b:
                cand.append(("prompt_changes_prediction", [f"{best_name}: {top3_text(sims_best[n])}"]))
            for cat, extra in cand:
                if len(picks[cat]) < 8 and cls not in seen_cls[cat]:  # spread examples over classes first
                    picks[cat].append((i, d, extra))
                    seen_cls[cat].add(cls)
                    if len(seen_cls[cat]) == len(ce.CLASSES):
                        seen_cls[cat] = set()
    files = {}
    for cat, items in picks.items():
        if not items:
            continue
        tiles = [tile(load(i, d), caption(i, d, extra)) for i, d, extra in items]
        path = out / f"{cat}.jpg"
        cv2.imencode(".jpg", grid(tiles, f"{cat.replace('_', ' ')} (weak-label diagnostic, unverified)"),
                     [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(path))
        files[cat] = {"file": f"examples/{cat}.jpg", "items": [f"{records[i]['folder']}/{records[i]['file']}#{d['id']}"
                                                                for i, d, _ in items]}
    return files


# --- reporting -------------------------------------------------------------------------------------------
def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def write_confusion_csv(path: Path, tables: dict[str, dict]) -> None:
    cols = ce.CLASSES + ["(no detection)"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "folder_class", *cols, "images"])
        for source, table in tables.items():
            for cls, row in table.items():
                w.writerow([source, cls, *[row.get(c, 0) for c in cols], sum(row.values())])


def report_md(res: dict) -> str:
    st, lines = res["stored"], []
    lines += ["# Class-wise diagnostic report (zero-shot Grounding DINO + CLIP)", "", f"> {DISCLAIMER}", "",
              f"Run analysed: `{res['run']}` ({st['metrics']['overall']['images']} images, {res['detections']} detections).",
              "Definitions: *primary box* = highest Grounding DINO confidence in the image; *any box* = at least one box; "
              "*box level* = share of all boxes in the folder's images.", "",
              "## 1. Weak-label rates of the stored run (original prompts, 5 % padding)", "",
              "| Folder class | Images | CLIP top-1 (primary) | CLIP top-3 (primary) | CLIP top-1 (any box) | CLIP top-3 (any box) | "
              "GD class (primary) | GD class (any box) | Box-level CLIP top-1 |", "|---|---|---|---|---|---|---|---|---|"]
    for cls, m in st["metrics"]["per_class"].items():
        lines.append(f"| {cls} | {m['images']} | {pct(m['primary_top1'])} | {pct(m['primary_top3'])} | {pct(m['any_top1'])} | "
                     f"{pct(m['any_top3'])} | {pct(m['gdino_primary'])} | {pct(m['gdino_any'])} | {pct(m['box_level_top1'])} |")
    o = st["metrics"]["overall"]
    lines.append(f"| **all (micro / macro)** | {o['images']} | {pct(o['primary_top1_micro'])} / {pct(o['primary_top1_macro'])} | "
                 f"{pct(o['primary_top3_micro'])} / {pct(o['primary_top3_macro'])} | {pct(o['any_top1_micro'])} | "
                 f"{pct(o['any_top3_micro'])} | {pct(o['gdino_primary_micro'])} | {pct(o['gdino_any_micro'])} | |")
    lines += ["", "Chance level for a top-1 pick among 7 classes is 14.3 %, and 42.9 % for top-3.", ""]
    for source in ("clip_primary", "gdino_primary"):
        lines += [f"### Confusion: folder class → {source.replace('_', ' ')} top-1", "",
                  "| folder \\ predicted | " + " | ".join(c[:9] for c in ce.CLASSES) + " |",
                  "|---" * (len(ce.CLASSES) + 1) + "|"]
        for cls, row in st["confusion"][source].items():
            lines.append(f"| {cls} | " + " | ".join(str(row.get(c, 0)) for c in ce.CLASSES) + " |")
        lines.append("")
    lines += ["## 2. Score distributions (stored run)", "",
              "| Folder class | CLIP top-1 sim (median, p25-p75) | top-1 − top-2 margin (median) | near ties (< 0.005) | "
              "GD confidence (median, p25-p75) |", "|---|---|---|---|---|"]
    for cls, d in st["distributions"].items():
        s, mg, g = d["clip_top1_similarity"], d["clip_margin_top1_top2"], d["gdino_confidence"]
        lines.append(f"| {cls} | {s['median']} ({s['p25']}-{s['p75']}) | {mg['median']} | {pct(d['near_tie_share_margin_lt_0.005'])} | "
                     f"{g['median']} ({g['p25']}-{g['p75']}) |")
    lines += ["", "## 3. Box size vs. folder match (stored run)", "", "| Box size (share of photo) | Boxes | CLIP top-1 = folder | CLIP UNKNOWN |",
              "|---|---|---|---|"]
    for k, v in st["size"]["by_size"].items():
        lines.append(f"| {k} | {v['boxes']} | {pct(v['top1_is_folder'])} | {pct(v['clip_unknown'])} |")
    lines.append(f"\nPrimary-box sizes: {st['size']['primary_box_size']}")
    if "prompts" in res:
        lines += ["", "## 4. Prompt comparison (same boxes, 5 % padding)", "",
                  f"Sanity check: the re-computed `original_ensemble` reproduces the stored top-1 for "
                  f"{pct(res['sanity_top1_reproduced'])} of boxes.", "",
                  "| Configuration | Primary top-1 (macro) | Primary top-3 (macro) | Any-box top-1 | Half A / half B (top-1 macro) | "
                  "Decided / uncertain / unknown | Median margin | Top-1 changed vs original |", "|---|---|---|---|---|---|---|---|"]
        for name, p in res["prompts"].items():
            m = p["metrics"]["overall"]
            ds = p["decided"]
            lines.append(f"| {name} | {pct(m['primary_top1_macro'])} | {pct(m['primary_top3_macro'])} | {pct(m['any_top1_macro'])} | "
                         f"{pct(p['split_half'][0])} / {pct(p['split_half'][1])} | {pct(ds['decided'])} / {pct(ds['uncertain'])} / "
                         f"{pct(ds['unknown'])} | {ds['median_margin']} | {pct(p['top1_changed_vs_original'])} |")
        lines += ["", "Per class, primary-box top-1 (macro shown above):", "",
                  "| Configuration | " + " | ".join(c[:9] for c in ce.CLASSES) + " |", "|---" * (len(ce.CLASSES) + 1) + "|"]
        for name, p in res["prompts"].items():
            lines.append(f"| {name} | " + " | ".join(pct(p["metrics"]["per_class"][c]["primary_top1"]) for c in ce.CLASSES) + " |")
        lines += ["", "## 5. Crop comparison (same boxes)", "",
                  "| Crop | Prompts | Primary top-1 (macro) | Primary top-3 (macro) | Any-box top-1 (macro) | Box-level top-1 (micro over boxes) |",
                  "|---|---|---|---|---|---|"]
        for row in res["crops"]:
            lines.append(f"| {row['crop']} | {row['prompts']} | {pct(row['primary_top1_macro'])} | {pct(row['primary_top3_macro'])} | "
                         f"{pct(row['any_top1_macro'])} | {pct(row['box_level_top1'])} |")
        lines += ["", "*whole photo* = CLIP on the full photo region, without any detection box (one prediction per image).", "",
                  "## 6. Examples", ""]
        for cat, info in res["examples"].items():
            lines.append(f"- **{cat.replace('_', ' ')}**: `{info['file']}`")
    lines += ["", "## 7. Findings and recommendations", ""] + res.get("findings", ["(see console output)"])
    return "\n".join(lines) + "\n"


def box_level_micro(records, ranks) -> float:
    n = hit = 0
    for i, r in enumerate(records):
        for d in r["detections"]:
            n += 1
            hit += ranks[(i, d["id"])][0] == r["folder_class"]
    return round(hit / n, 4) if n else None


def main() -> None:
    from experiments.zero_shot_bts.config import Config

    ap = argparse.ArgumentParser(description="Weak-label class-wise diagnostics and CLIP prompt/crop experiments.")
    ap.add_argument("--run", default="full_543_gdino_clip")
    ap.add_argument("--no-clip", action="store_true", help="only analyse the stored results (no CLIP re-scoring)")
    args = ap.parse_args()
    run_dir = EXP_DIR / "runs" / args.run
    out = DIAG_DIR
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((run_dir / "predictions" / "predictions.json").read_text(encoding="utf-8"))["images"]
    stored = wl.stored_rankings(records)
    res = {"disclaimer": DISCLAIMER, "run": args.run, "detections": len(stored),
           "stored": {"metrics": wl.weak_metrics(records, stored),
                      "confusion": {"clip_primary": wl.confusion(records, stored, "clip"),
                                    "gdino_primary": wl.confusion(records, stored, "gdino")},
                      "distributions": wl.distributions(records), "size": wl.size_analysis(records, stored)}}
    write_confusion_csv(out / "class_confusion.csv", res["stored"]["confusion"])

    if not args.no_clip:
        t0 = time.time()
        embedder = ce.ClipEmbedder(Config().clip_model)
        emb, keys, photo = embed_all(records, embedder, DIAG_DIR / "cache" / f"{args.run}_embeddings.npz")
        neg = embedder.text(ce.NEGATIVES)
        texts = {name: embedder.prompt_embeddings(prompts) for name, prompts, _ in ce.PROMPT_CONFIGS}
        res["clip_seconds"] = round(time.time() - t0, 1)

        def score(name, how, img):
            return ce.class_similarities(img, texts[name], how)

        base = score("original_ensemble", "mean", emb[0.05])
        base_ranks = ranks_dict(keys, base)
        res["sanity_top1_reproduced"] = round(sum(base_ranks[k][0] == stored[k][0] for k in keys) / len(keys), 4)
        res["prompts"], all_sims = {}, {}
        prompt_rows = []
        for name, _, how in ce.PROMPT_CONFIGS:
            sims = score(name, how, emb[0.05])
            all_sims[name] = sims
            rk = ranks_dict(keys, sims)
            m = wl.weak_metrics(records, rk)
            res["prompts"][name] = {"aggregation": how, "metrics": m, "split_half": split_half(records, rk),
                                    "decided": ce.decided_share(sims, (emb[0.05] @ neg.T).max(axis=1)),
                                    "top1_changed_vs_original": wl.rank_change_rate(rk, base_ranks)}
            for cls, v in m["per_class"].items():
                prompt_rows.append({"configuration": name, "aggregation": how, "folder_class": cls, **v})
        # pick the best configuration on half A only, report half B (guards against fitting the weak labels)
        best = max(res["prompts"], key=lambda n: res["prompts"][n]["split_half"][0])
        res["best_prompt_config"] = {"name": best, "selected_on": "half A (even-indexed images)",
                                     "half_b_primary_top1_macro": res["prompts"][best]["split_half"][1]}
        best_how = next(h for n, _, h in ce.PROMPT_CONFIGS if n == best)
        res["crops"] = []
        for name, how in (("original_ensemble", "mean"), (best, best_how)):
            for p in ce.PADDINGS:
                rk = ranks_dict(keys, score(name, how, emb[p]))
                m = wl.weak_metrics(records, rk)["overall"]
                res["crops"].append({"crop": f"box + {int(p * 100)}% padding", "prompts": name,
                                     "primary_top1_macro": m["primary_top1_macro"], "primary_top3_macro": m["primary_top3_macro"],
                                     "any_top1_macro": m["any_top1_macro"], "box_level_top1": box_level_micro(records, rk)})
            pr = photo_records(records)
            prk = {(i, 0): r for i, r in enumerate(ce.rankings(score(name, how, photo)))}
            m = wl.weak_metrics(pr, prk)
            res["crops"].append({"crop": "whole photo (no box)", "prompts": name, "primary_top1_macro": m["overall"]["primary_top1_macro"],
                                 "primary_top3_macro": m["overall"]["primary_top3_macro"], "any_top1_macro": None,
                                 "box_level_top1": None, "per_class_top1": {c: v["primary_top1"] for c, v in m["per_class"].items()}})
            if name == best:
                break
        res["examples"] = make_examples(records, keys, all_sims["original_ensemble"], all_sims[best], best, run_dir, out / "examples")
        with (out / "prompt_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prompt_rows[0]))
            w.writeheader()
            w.writerows(prompt_rows)
        with (out / "crop_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["crop", "prompts", "primary_top1_macro", "primary_top3_macro", "any_top1_macro", "box_level_top1"])
            w.writeheader()
            w.writerows({k: row.get(k) for k in w.fieldnames} for row in res["crops"])

    findings_file = DIAG_DIR / "findings.md"
    if findings_file.is_file():  # human-written interpretation, kept separate from computed numbers
        res["findings"] = findings_file.read_text(encoding="utf-8").splitlines()
    (out / "diagnostic_results.json").write_text(json.dumps(res, indent=2, default=dict), encoding="utf-8")
    (out / "class_wise_report.md").write_text(report_md(res), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
