"""The three model stages, each loaded lazily on first use (CPU only).

Stage A  GroundingDinoDetector  text-prompted boxes; each box keeps its best-matching prompt and class
Stage B  ClipClassifier         crop vs. class descriptions (cosine similarity), with UNKNOWN / UNCERTAIN
Stage C  SamRefiner             optional MobileSAM box-prompted masks -> tighter boxes
"""

import time
from pathlib import Path

import cv2
import numpy as np

from app.training.taxonomy import CLASS_NAMES
from experiments.zero_shot_bts.boxes import area, clip_xyxy, cxcywh_norm_to_xyxy, iou, mask_to_box
from experiments.zero_shot_bts.config import UNCERTAIN, UNKNOWN, Config


def _torch():
    import torch

    torch.set_grad_enabled(False)
    return torch


# --- Stage A ------------------------------------------------------------------------------------
def build_prompt(prompts: dict[int, list[str]]) -> tuple[str, list[tuple[int, str, int, int]]]:
    """One Grounding DINO caption ("a. b. c.") and the character span of every phrase."""
    text, spans = "", []
    for cid in sorted(prompts):
        for p in prompts[cid]:
            p = p.lower().strip()
            spans.append((cid, p, len(text), len(text) + len(p)))
            text += p + ". "
    return text.strip(), spans


def phrase_token_index(offsets: list[tuple[int, int]], spans) -> list[tuple[int, str, list[int]]]:
    """Token indices that fall inside each phrase span (tokenizer offset mapping)."""
    out = []
    for cid, phrase, a, b in spans:
        toks = [i for i, (s, e) in enumerate(offsets) if e > s and s >= a and e <= b]
        if toks:
            out.append((cid, phrase, toks))
    return out


class GroundingDinoDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = self.processor = None
        self.load_seconds = None

    def _load(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        t = time.time()
        self.processor = AutoProcessor.from_pretrained(self.cfg.gdino_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.gdino_model).eval()
        self.text, spans = build_prompt({int(k): v for k, v in self.cfg.gdino_prompts.items()})
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        self.phrases = phrase_token_index(enc["offset_mapping"], spans)
        self.load_seconds = round(time.time() - t, 1)

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Boxes in `image_bgr` pixels. gdino_confidence = max token probability over the whole caption
        (Grounding DINO's usual box score); the prompt is the phrase with the highest mean token probability."""
        if self.model is None:
            self._load()
        torch = _torch()
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.text, return_tensors="pt",
                                size={"shortest_edge": self.cfg.gdino_short_side, "longest_edge": self.cfg.gdino_long_side})
        out = self.model(**inputs)
        probs = out.logits[0].sigmoid()  # (queries, tokens)
        conf = probs.max(dim=1).values
        phrase_scores = torch.stack([probs[:, toks].mean(dim=1) for _, _, toks in self.phrases], dim=1)
        best = phrase_scores.argmax(dim=1)
        dets = []
        for q in torch.nonzero(conf >= self.cfg.box_threshold).flatten().tolist():
            xyxy = clip_xyxy(cxcywh_norm_to_xyxy(out.pred_boxes[0][q].tolist(), w, h), w, h)
            if min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) < self.cfg.min_box_px:
                continue
            cid, phrase, _ = self.phrases[int(best[q])]
            dets.append({"xyxy": xyxy, "gdino_confidence": float(conf[q]), "prompt": phrase, "gdino_class_id": cid,
                         "gdino_class": CLASS_NAMES[cid], "prompt_score": float(phrase_scores[q, best[q]])})
        return dets


# --- Stage B ------------------------------------------------------------------------------------
def decide(class_sims: dict[int, float], negative_sim: float, min_sim: float, min_margin: float) -> dict:
    """Turn raw cosine similarities into a cautious decision. Similarities are NOT probabilities.

    UNKNOWN    the best "none of the seven" description beats every component class
    UNCERTAIN  best class similarity below `min_sim`, or top-1/top-2 margin below `min_margin`
    otherwise  the top-1 class"""
    ranked = sorted(class_sims.items(), key=lambda kv: -kv[1])
    top3 = [{"class_id": c, "class": CLASS_NAMES[c], "similarity": round(s, 4)} for c, s in ranked[:3]]
    (c1, s1), (_, s2) = ranked[0], ranked[1]
    margin = s1 - s2
    if negative_sim > s1:
        label, reason = UNKNOWN, f"background description wins ({negative_sim:.3f} > {s1:.3f})"
    elif s1 < min_sim:
        label, reason = UNCERTAIN, f"best similarity {s1:.3f} < {min_sim}"
    elif margin < min_margin:
        label, reason = UNCERTAIN, f"top-1/top-2 margin {margin:.4f} < {min_margin}"
    else:
        label, reason = CLASS_NAMES[c1], None
    return {"clip_label": label, "clip_top3": top3, "clip_margin": round(margin, 4),
            "clip_negative_similarity": round(negative_sim, 4), "clip_reason": reason,
            "clip_all": {CLASS_NAMES[c]: round(s, 4) for c, s in class_sims.items()}}


class ClipClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = self.processor = None
        self.load_seconds = None

    def _load(self):
        torch = _torch()
        from transformers import CLIPModel, CLIPProcessor

        t = time.time()
        self.processor = CLIPProcessor.from_pretrained(self.cfg.clip_model)
        self.model = CLIPModel.from_pretrained(self.cfg.clip_model).eval()
        # text embeddings are computed once and cached for the whole run (prompt ensembling per class)
        descs = {int(k): v for k, v in self.cfg.clip_descriptions.items()}
        self.class_ids = sorted(descs)
        self.class_emb = torch.stack([self._text(descs[c]).mean(dim=0) for c in self.class_ids])
        self.class_emb = self.class_emb / self.class_emb.norm(dim=-1, keepdim=True)
        self.neg_emb = self._text(self.cfg.clip_negatives)
        self.load_seconds = round(time.time() - t, 1)

    def _text(self, texts: list[str]):
        inp = self.processor(text=texts, return_tensors="pt", padding=True)
        e = self.model.get_text_features(**inp)
        return e / e.norm(dim=-1, keepdim=True)

    def classify(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        if not crops_bgr:
            return []
        if self.model is None:
            self._load()
        from PIL import Image

        pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inp = self.processor(images=pils, return_tensors="pt")
        e = self.model.get_image_features(**inp)
        e = e / e.norm(dim=-1, keepdim=True)
        sims = (e @ self.class_emb.T).tolist()
        neg = (e @ self.neg_emb.T).max(dim=1).values.tolist()
        return [decide(dict(zip(self.class_ids, row)), n, self.cfg.clip_min_similarity, self.cfg.clip_min_margin)
                for row, n in zip(sims, neg)]


# --- Stage C ------------------------------------------------------------------------------------
class SamRefiner:
    def __init__(self, cfg: Config, weights_dir: Path):
        self.cfg, self.weights_dir = cfg, weights_dir
        self.model = None
        self.load_seconds = None

    def _load(self):
        from ultralytics import SAM

        t = time.time()
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        path = self.weights_dir / self.cfg.sam_weights
        if not path.is_file():  # Ultralytics downloads its released asset to this path
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(path))
        self.model = SAM(str(path))
        self.load_seconds = round(time.time() - t, 1)

    def refine(self, image_bgr: np.ndarray, boxes: list) -> list[dict]:
        """For each box: SAM mask -> tight box. The refined box is used only when the mask fills enough of the
        original box; otherwise the original is kept and the reason recorded."""
        if not boxes:
            return []
        if self.model is None:
            self._load()
        h, w = image_bgr.shape[:2]
        res = self.model(image_bgr, bboxes=[list(b) for b in boxes], device="cpu", verbose=False)[0]
        masks = res.masks.data.cpu().numpy() if res.masks is not None else []
        out = []
        for i, box in enumerate(boxes):
            if i >= len(masks):
                out.append({"sam_used": False, "sam_reason": "no mask returned"})
                continue
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            tight = mask_to_box(m > 0.5)
            fill = float((m > 0.5).sum()) / max(area(box), 1.0)
            if tight is None or fill < self.cfg.sam_min_fill:
                out.append({"sam_used": False, "sam_reason": f"mask fill {fill:.2f} < {self.cfg.sam_min_fill}"})
            else:
                tight = clip_xyxy(tight, w, h)
                out.append({"sam_used": True, "sam_xyxy": [round(v, 1) for v in tight], "sam_mask_fill": round(fill, 3),
                            "sam_iou_with_original": round(iou(tight, box), 3)})
        return out
