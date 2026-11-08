"""Run outputs: annotated visualisations, crops, predictions JSON, summary report and the HTML gallery.

Everything here describes AUTOMATED predictions. There is no ground truth, so no accuracy, precision, recall
or mAP is computed; model agreement is reported as agreement, never as correctness."""

import html
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.training.taxonomy import CLASS_NAMES
from experiments.zero_shot_bts.boxes import area
from experiments.zero_shot_bts.config import UNCERTAIN, UNKNOWN

COLORS = {name: c for name, c in zip(CLASS_NAMES.values(), [(75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67),
                                                             (49, 130, 245), (180, 30, 145), (244, 212, 66)])}
COLORS[UNCERTAIN] = (40, 160, 230)
COLORS[UNKNOWN] = (150, 150, 150)
LABELS = list(CLASS_NAMES.values()) + [UNCERTAIN, UNKNOWN]
SHORT = {"antenna": "antenna", "router": "router", "cable": "cable", "battery": "battery", "power_supply_unit": "PSU",
         "equipment_cabinet": "cabinet", "radio_unit": "RRU", UNCERTAIN: "UNCERTAIN", UNKNOWN: "UNKNOWN"}


def write_jpeg(path: Path, img: np.ndarray, quality: int = 85) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tofile(str(path))


def save_crop(path: Path, crop: np.ndarray, max_side: int = 224) -> None:
    s = min(1.0, max_side / max(crop.shape[:2]))
    write_jpeg(path, cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop)


def _dashed_rect(img, p1, p2, color, thickness, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw(image: np.ndarray, rec: dict, view_box, max_side: int = 1100) -> np.ndarray:
    """Annotated view of the photo region: box, final label, Grounding DINO class/confidence, CLIP top-1 and
    similarity. Solid = both models agree; dashed = disagreement / uncertain / unknown / no CLIP."""
    x0, y0, x1, y1 = (int(v) for v in view_box)
    view = image[y0:y1, x0:x1].copy()
    s = min(1.0, max_side / max(view.shape[:2]))
    if s < 1:
        view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    fs = 0.42
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    band = 26 + th + 10  # banner + room for labels of boxes touching the top edge
    view = cv2.copyMakeBorder(view, band, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    thick = max(2, int(round(2 * max(view.shape[:2]) / 900)))
    for d in sorted(rec["detections"], key=lambda d: -area(d["xyxy"])):  # small boxes drawn last, on top
        box = d.get("sam_xyxy") if d.get("sam_used") else d["xyxy"]
        bx1, by1, bx2, by2 = (int((v - o) * s) for v, o in zip(box, (x0, y0, x0, y0)))
        by1, by2 = by1 + band, by2 + band
        color = COLORS.get(d["final_label"], (200, 200, 200))
        if d["agreement"] == "agree":
            cv2.rectangle(view, (bx1, by1), (bx2, by2), color, thick)
        else:
            _dashed_rect(view, (bx1, by1), (bx2, by2), color, thick)
        top = d["clip_top3"][0] if d.get("clip_top3") else None
        text = (f"#{d['id']} {SHORT.get(d['final_label'], d['final_label'])} | GD {SHORT[d['gdino_class']]} "
                f"{d['gdino_confidence']:.2f}" + (f" | CLIP {SHORT[top['class']]} {top['similarity']:.3f}" if top else ""))
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(by1, 26 + th + 6)
        cv2.rectangle(view, (bx1, ty - th - 6), (bx1 + tw + 6, ty), color, -1)
        cv2.putText(view, text, (bx1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15), 1, cv2.LINE_AA)
    banner = f"{rec['folder']}/{rec['file']}  |  {len(rec['detections'])} detections  |  AUTOMATED, UNVERIFIED"
    cv2.putText(view, banner[:140], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return view


# --- report ---------------------------------------------------------------------------------------
def summarize(records: list[dict], cfg: dict, runtime: dict) -> dict:
    dets = [d for r in records for d in r["detections"]]
    with_clip = [d for d in dets if d.get("clip_label")]
    times = [r["timing"]["total_s"] for r in records]

    def stage(name):
        v = [r["timing"].get(name) for r in records if r["timing"].get(name) is not None]
        return {"mean": round(statistics.mean(v), 3), "median": round(statistics.median(v), 3),
                "max": round(max(v), 3)} if v else None

    cross = Counter((d["gdino_class"], d["clip_label"]) for d in with_clip)
    per_folder = {}
    for r in records:
        f = per_folder.setdefault(r["folder_class"] or r["folder"], Counter())
        f["images"] += 1
        f["with_detections"] += bool(r["detections"])
        f["folder_component_in_final_labels"] += r["folder_class"] in {d["final_label"] for d in r["detections"]}
    return {
        "disclaimer": "Automated zero-shot predictions. No human-verified ground truth exists, so no precision, "
                      "recall, mAP or accuracy is reported. Agreement between models is not correctness.",
        "images_processed": len(records),
        "images_with_detections": sum(bool(r["detections"]) for r in records),
        "images_without_detections": sum(not r["detections"] for r in records),
        "total_detections": len(dets),
        "detections_per_image_mean": round(len(dets) / len(records), 2) if records else 0,
        "gdino_class_counts": dict(Counter(d["gdino_class"] for d in dets).most_common()),
        "clip_label_counts": dict(Counter(d["clip_label"] for d in with_clip).most_common()),
        "final_label_counts": dict(Counter(d["final_label"] for d in dets).most_common()),
        "agreement_counts": dict(Counter(d["agreement"] for d in dets).most_common()),
        "gdino_vs_clip": {f"{g} -> {c}": n for (g, c), n in cross.most_common()},
        "low_gdino_confidence": sum("low_gdino_confidence" in d["flags"] for d in dets),
        "flag_counts": dict(Counter(f for d in dets for f in d["flags"]).most_common()),
        "per_folder_diagnostic": {k: dict(v) for k, v in per_folder.items()},
        "timing_seconds": {"per_image_total": stage("total_s"), "gdino": stage("gdino_s"), "clip": stage("clip_s"),
                           "sam": stage("sam_s"), "sum": round(sum(times), 1)},
        "runtime": runtime,
        "config": cfg,
    }


def pick_examples(records: list[dict], n: int = 6) -> dict:
    """Automatic picks for the report. 'Agreeing' examples are NOT verified successes."""
    agree = sorted(((d["gdino_confidence"] + (d["clip_top3"][0]["similarity"] if d.get("clip_top3") else 0), r, d)
                    for r in records for d in r["detections"] if d["agreement"] == "agree"), key=lambda t: -t[0])
    disagree = [(r, d) for r in records for d in r["detections"] if d["agreement"] == "disagree"]
    empty = [r for r in records if not r["detections"]]
    whole = [(r, d) for r in records for d in r["detections"] if "near_whole_photo" in d["flags"]]
    fmt = lambda r, d=None: {"image": f"{r['folder']}/{r['file']}", "visualization": r["visualization"],  # noqa: E731
                             **({"detection": d["id"], "gdino": f"{d['gdino_class']} {d['gdino_confidence']:.2f}",
                                 "clip": d.get("clip_label")} if d else {})}
    return {"highest_confidence_agreement_unverified": [fmt(r, d) for _, r, d in agree[:n]],
            "disagreements": [fmt(r, d) for r, d in disagree[:n]],
            "no_detections": [fmt(r) for r in empty[:n]],
            "near_whole_photo_boxes": [fmt(r, d) for r, d in whole[:n]]}


def report_markdown(s: dict, examples: dict) -> str:
    t = s["timing_seconds"]
    lines = ["# Zero-shot BTS detection report", "", f"> {s['disclaimer']}", "",
             f"- Images processed: **{s['images_processed']}**; with detections: {s['images_with_detections']}; "
             f"without: {s['images_without_detections']}",
             f"- Candidate detections: **{s['total_detections']}** ({s['detections_per_image_mean']} per image)",
             f"- Low Grounding DINO confidence (< 0.35): {s['low_gdino_confidence']}",
             f"- Timing per image (s): {t['per_image_total']}; Grounding DINO {t['gdino']}; CLIP {t['clip']}; SAM {t['sam']}",
             f"- Runtime: {s['runtime']}", "",
             "## Grounding DINO class (from the matched prompt)", ""]
    lines += [f"- {k}: {v}" for k, v in s["gdino_class_counts"].items()]
    lines += ["", "## CLIP label", ""]
    lines += [f"- {k}: {v}" for k, v in s["clip_label_counts"].items()] or ["- (CLIP not run)"]
    lines += ["", "## Final label (agreement required; otherwise UNCERTAIN / UNKNOWN)", ""]
    lines += [f"- {k}: {v}" for k, v in s["final_label_counts"].items()]
    lines += ["", "## Agreement", ""] + [f"- {k}: {v}" for k, v in s["agreement_counts"].items()]
    lines += ["", "## Grounding DINO -> CLIP (most frequent pairs)", ""]
    lines += [f"- {k}: {v}" for k, v in list(s["gdino_vs_clip"].items())[:20]]
    lines += ["", "## Per source folder (diagnostic only: folder names are not ground truth)", "",
              "| Folder class | Images | With detections | Folder's component among final labels |", "|---|---|---|---|"]
    lines += [f"| {k} | {v.get('images', 0)} | {v.get('with_detections', 0)} | {v.get('folder_component_in_final_labels', 0)} |"
              for k, v in s["per_folder_diagnostic"].items()]
    lines += ["", "## Flags", ""] + [f"- {k}: {v}" for k, v in s["flag_counts"].items()]
    lines += ["", "## Automatic example picks (unverified)", ""]
    for k, items in examples.items():
        lines += [f"### {k}", ""] + [f"- {json.dumps(i)}" for i in items] + [""]
    return "\n".join(lines) + "\n"


# --- gallery --------------------------------------------------------------------------------------
GALLERY = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zero-shot BTS Gallery</title><style>
:root{--bg:#0b0f14;--p:#111821;--p2:#17202b;--l:#263241;--i:#e6edf3;--i2:#a4b1c0;--i3:#6b7a8c;--a:#2fb5d2;--g:#3fb950;--w:#d29922;--b:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--i);font:13px/1.45 system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--p);border-bottom:1px solid var(--l);padding:10px 16px}
h1{font-size:15px;margin:0 0 4px}.warn{color:var(--w);font-size:12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
select,input,button{background:var(--p2);color:var(--i);border:1px solid var(--l);border-radius:6px;padding:4px 8px;font:inherit}
button{cursor:pointer}button:disabled{opacity:.4}.chip{color:var(--i2);font-size:12px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;padding:16px}
.card{background:var(--p);border:1px solid var(--l);border-radius:8px;overflow:hidden;cursor:pointer}
.card:hover{border-color:var(--a)}.card img{width:100%;height:190px;object-fit:contain;background:#000;display:block}
.m{padding:8px}.fn{font-family:monospace;font-size:11px;color:var(--i2);word-break:break-all}
.b{display:inline-block;border:1px solid var(--l);border-radius:4px;padding:0 5px;font-size:11px;margin:2px 3px 0 0;color:var(--i2)}
.agree{color:var(--g);border-color:var(--g)}.disagree{color:var(--b);border-color:var(--b)}.unc{color:var(--w);border-color:var(--w)}
dialog{background:var(--p);color:var(--i);border:1px solid var(--l);border-radius:10px;width:min(1200px,96vw);max-height:92vh;padding:0}
dialog::backdrop{background:rgba(0,0,0,.7)}.dh{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--l)}
.db{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px;padding:12px;max-height:80vh;overflow:auto}
.db img.big{width:100%;background:#000}@media(max-width:800px){.db{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{border-top:1px solid var(--l);padding:4px;text-align:left;vertical-align:top}
th{color:var(--i3);font-weight:500}td img{width:72px;height:72px;object-fit:contain;background:#000;border-radius:4px}
.empty{padding:40px;text-align:center;color:var(--i3)}
</style></head><body><header><h1>Zero-shot BTS detection: __RUN__</h1>
<div class="warn">Automated Grounding DINO + CLIP predictions. Nothing here is verified ground truth; agreement between models is not correctness.</div>
<div class="bar"><select id="f"><option value="all">All images</option><option value="none">No detections</option>
<option value="disagree">Has disagreement</option><option value="unc">Has UNCERTAIN / UNKNOWN</option><option value="agree">Has agreeing detection</option></select>
<select id="cls"><option value="">any final label</option></select><select id="folder"><option value="">any folder</option></select>
<input id="q" placeholder="file name"><span class="chip" id="count"></span>
<button id="prev">Prev page</button><span class="chip" id="page"></span><button id="next">Next page</button></div></header>
<main id="grid"></main>
<dialog id="dlg"><div class="dh"><b id="dt"></b><span><button id="dp">&larr;</button> <button id="dn">&rarr;</button> <button id="dc">Close</button></span></div><div class="db" id="dbody"></div></dialog>
<script>
const D=__DATA__,LABELS=__LABELS__,PER=48;let view=[],page=0,cur=-1;const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
LABELS.forEach(l=>$("cls").insertAdjacentHTML("beforeend",`<option>${l}</option>`));
[...new Set(D.map(r=>r.folder))].sort().forEach(f=>$("folder").insertAdjacentHTML("beforeend",`<option>${esc(f)}</option>`));
const agr=a=>a==="agree"?"agree":a==="disagree"?"disagree":"unc";
function apply(){const f=$("f").value,c=$("cls").value,fo=$("folder").value,q=$("q").value.toLowerCase();
 view=D.filter(r=>{const ds=r.detections;if(f==="none"&&ds.length)return false;if(f==="disagree"&&!ds.some(d=>d.agreement==="disagree"))return false;
  if(f==="unc"&&!ds.some(d=>d.final_label==="UNCERTAIN"||d.final_label==="UNKNOWN"))return false;if(f==="agree"&&!ds.some(d=>d.agreement==="agree"))return false;
  if(c&&!ds.some(d=>d.final_label===c))return false;if(fo&&r.folder!==fo)return false;if(q&&!r.file.toLowerCase().includes(q))return false;return true});
 page=0;render()}
function render(){const pages=Math.max(1,Math.ceil(view.length/PER));$("count").textContent=`${view.length} of ${D.length} images`;$("page").textContent=`${page+1} / ${pages}`;
 $("prev").disabled=page===0;$("next").disabled=page>=pages-1;const g=$("grid");
 const items=view.slice(page*PER,page*PER+PER);if(!items.length){g.innerHTML='<div class="empty">No images match.</div>';return}
 g.innerHTML=items.map((r,i)=>{const n=Counter(r.detections.map(d=>d.agreement));
  return `<div class="card" data-i="${page*PER+i}"><img loading="lazy" src="${esc(r.visualization)}" alt=""><div class="m"><div class="fn">${esc(r.folder)}/${esc(r.file)}</div>
  <span class="b">${r.detections.length} det</span>${n.agree?`<span class="b agree">${n.agree} agree</span>`:""}${n.disagree?`<span class="b disagree">${n.disagree} disagree</span>`:""}
  ${(r.detections.length-(n.agree||0)-(n.disagree||0))?`<span class="b unc">${r.detections.length-(n.agree||0)-(n.disagree||0)} other</span>`:""}</div></div>`}).join("");
 g.querySelectorAll(".card").forEach(el=>el.onclick=()=>open(+el.dataset.i))}
function Counter(a){const o={};a.forEach(x=>o[x]=(o[x]||0)+1);return o}
function open(i){cur=i;const r=view[i];$("dt").textContent=`${r.folder}/${r.file} (${i+1}/${view.length})`;
 const rows=r.detections.map(d=>`<tr><td>${d.crop?`<img src="${esc(d.crop)}" alt="">`:""}</td><td><b>#${d.id} ${esc(d.final_label)}</b><br><span class="b ${agr(d.agreement)}">${esc(d.agreement)}</span>
  <br>flags: ${esc(d.flags.join(", ")||"-")}</td><td>GD: ${esc(d.gdino_class)} ${d.gdino_confidence.toFixed(2)}<br><span style="color:var(--i3)">prompt "${esc(d.prompt)}"</span>
  ${d.alternatives&&d.alternatives.length?`<br><span style="color:var(--i3)">also: ${d.alternatives.slice(0,3).map(a=>esc(a.prompt)+" "+a.gdino_confidence).join(", ")}</span>`:""}</td>
  <td>${d.clip_top3?d.clip_top3.map(t=>`${esc(t.class)} ${t.similarity.toFixed(3)}`).join("<br>")+`<br><span style="color:var(--i3)">margin ${d.clip_margin} · bg ${d.clip_negative_similarity}</span>`:"CLIP not run"}
  ${d.clip_reason?`<br><span style="color:var(--w)">${esc(d.clip_reason)}</span>`:""}${d.sam_used!==undefined?`<br>SAM: ${d.sam_used?"refined (IoU "+d.sam_iou_with_original+")":esc(d.sam_reason)}`:""}</td></tr>`).join("");
 $("dbody").innerHTML=`<div><img class="big" src="${esc(r.visualization)}" alt=""><div class="chip">folder (context only, not ground truth): ${esc(r.folder_class||"?")} · ${r.timing.total_s}s</div></div>
  <div>${r.detections.length?`<table><tr><th>crop</th><th>final</th><th>Grounding DINO</th><th>CLIP top-3 (cosine)</th></tr>${rows}</table>`:'<div class="empty">No detections above the threshold.</div>'}</div>`;
 if(!$("dlg").open)$("dlg").showModal()}
$("dp").onclick=()=>cur>0&&open(cur-1);$("dn").onclick=()=>cur<view.length-1&&open(cur+1);$("dc").onclick=()=>$("dlg").close();
document.addEventListener("keydown",e=>{if(!$("dlg").open)return;if(e.key==="ArrowRight")$("dn").click();if(e.key==="ArrowLeft")$("dp").click()});
["f","cls","folder"].forEach(id=>$(id).onchange=apply);$("q").oninput=apply;$("prev").onclick=()=>{page--;render()};$("next").onclick=()=>{page++;render()};apply();
</script></body></html>"""


def write_gallery(path: Path, records: list[dict], run_name: str) -> None:
    data = json.dumps(records).replace("</", "<\\/")
    path.write_text(GALLERY.replace("__RUN__", html.escape(run_name)).replace("__LABELS__", json.dumps(LABELS))
                    .replace("__DATA__", data), encoding="utf-8")
