"""Model stages (CPU, loaded lazily). Ported from experiments/zero_shot_bts/models.py, which stays as the record.

GroundingDino   text-prompted regions; each box keeps its best prompt, class and the prompts it suppressed
ClipScorer      cosine similarities of crops (and mirrored crops) to class prompt ensembles and negatives
SamRefiner      optional MobileSAM box-prompted masks -> tighter boxes (local weights only, never downloaded)
"""

import time
from pathlib import Path

import cv2
import numpy as np

from app.training.pseudo_label.config import CLASSES, PseudoLabelConfig
from app.training.pseudo_label.geometry import area, clip_box, cxcywh_norm_to_xyxy, iou, mask_to_box


def _torch():
    import torch

    torch.set_grad_enabled(False)
    return torch


def build_caption(prompts: dict[str, list[str]]) -> tuple[str, list[tuple[str, str, int, int]]]:
    """One caption "a. b. c." and the character span of every phrase."""
    text, spans = "", []
    for cls in CLASSES:
        for p in prompts[cls]:
            p = p.lower().strip()
            spans.append((cls, p, len(text), len(text) + len(p)))
            text += p + ". "
    return text.strip(), spans


def phrase_tokens(offsets: list[tuple[int, int]], spans) -> list[tuple[str, str, list[int]]]:
    out = []
    for cls, phrase, a, b in spans:
        toks = [i for i, (s, e) in enumerate(offsets) if e > s and s >= a and e <= b]
        if toks:
            out.append((cls, phrase, toks))
    return out


def suppress_proposals(dets: list[dict], iou_thr: float) -> list[dict]:
    """Class-agnostic NMS across prompts; suppressed prompts are kept on the survivor as `alternatives`."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["gdino_confidence"]):
        owner = next((k for k in kept if iou(k["xyxy"], d["xyxy"]) > iou_thr), None)
        if owner is None:
            kept.append({**d, "alternatives": list(d.get("alternatives", []))})
        else:
            owner["alternatives"].append({"prompt": d["prompt"], "gdino_class": d["gdino_class"],
                                          "gdino_confidence": round(d["gdino_confidence"], 4)})
    return kept


class GroundingDino:
    def __init__(self, cfg: PseudoLabelConfig):
        self.cfg = cfg
        self.model = self.processor = None
        self.load_seconds: float | None = None

    def _load(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        t = time.time()
        self.processor = AutoProcessor.from_pretrained(self.cfg.gdino_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.gdino_model).eval()
        self.text, spans = build_caption(self.cfg.gdino_prompts)
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        self.phrases = phrase_tokens(enc["offset_mapping"], spans)
        self.load_seconds = round(time.time() - t, 1)

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Boxes in `image_bgr` pixels, after cross-prompt suppression."""
        if self.model is None:
            self._load()
        torch = _torch()
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.text, return_tensors="pt",
                                size={"shortest_edge": self.cfg.gdino_short_side,
                                      "longest_edge": self.cfg.gdino_long_side})
        out = self.model(**inputs)
        probs = out.logits[0].sigmoid()
        conf = probs.max(dim=1).values
        phrase_scores = torch.stack([probs[:, toks].mean(dim=1) for _, _, toks in self.phrases], dim=1)
        best = phrase_scores.argmax(dim=1)
        dets = []
        for q in torch.nonzero(conf >= self.cfg.box_threshold).flatten().tolist():
            xyxy = clip_box(cxcywh_norm_to_xyxy(out.pred_boxes[0][q].tolist(), w, h), w, h)
            if min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) < 8:
                continue
            cls, phrase, _ = self.phrases[int(best[q])]
            dets.append({"xyxy": xyxy, "gdino_confidence": float(conf[q]), "prompt": phrase, "gdino_class": cls})
        return suppress_proposals(dets, self.cfg.proposal_nms_iou)


class ClipScorer:
    def __init__(self, cfg: PseudoLabelConfig):
        self.cfg = cfg
        self.model = self.processor = None
        self.load_seconds: float | None = None

    def _text(self, texts: list[str]):
        inp = self.processor(text=texts, return_tensors="pt", padding=True)
        e = self.model.get_text_features(**inp)
        return e / e.norm(dim=-1, keepdim=True)

    def _class_matrix(self, prompts: dict[str, list[str]]):
        torch = _torch()
        m = torch.stack([self._text(prompts[c]).mean(dim=0) for c in CLASSES])
        return m / m.norm(dim=-1, keepdim=True)

    def _load(self) -> None:
        _torch()
        from transformers import CLIPModel, CLIPProcessor

        t = time.time()
        self.processor = CLIPProcessor.from_pretrained(self.cfg.clip_model)
        self.model = CLIPModel.from_pretrained(self.cfg.clip_model).eval()
        self.main = self._class_matrix(self.cfg.clip_prompts)
        self.alt = self._class_matrix(self.cfg.clip_alt_prompts)
        self.neg = self._text(self.cfg.clip_negatives)
        self.load_seconds = round(time.time() - t, 1)

    def score(self, crops_bgr: list[np.ndarray]) -> list[dict]:
        """Per crop: similarities (main prompts), mirrored-crop similarities, alt-prompt similarities, best negative."""
        if not crops_bgr:
            return []
        if self.model is None:
            self._load()
        from PIL import Image

        imgs = crops_bgr + [cv2.flip(c, 1) for c in crops_bgr]
        pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in imgs]
        e = self.model.get_image_features(**self.processor(images=pils, return_tensors="pt"))
        e = e / e.norm(dim=-1, keepdim=True)
        n = len(crops_bgr)
        main, alt = (e @ self.main.T).tolist(), (e[:n] @ self.alt.T).tolist()
        neg = (e[:n] @ self.neg.T).max(dim=1).values.tolist()
        emb = e.numpy().astype(np.float32)
        return [{"sims": dict(zip(CLASSES, main[i])), "flip_sims": dict(zip(CLASSES, main[n + i])),
                 "alt_sims": dict(zip(CLASSES, alt[i])), "negative_sim": neg[i], "emb": emb[i], "flip_emb": emb[n + i]}
                for i in range(n)]


class SamRefiner:
    def __init__(self, cfg: PseudoLabelConfig, min_fill: float = 0.15, min_iou: float = 0.5):
        self.cfg, self.min_fill, self.min_iou = cfg, min_fill, min_iou
        self.model = None
        self.load_seconds: float | None = None

    def _load(self) -> None:
        from ultralytics import SAM

        path = Path(self.cfg.sam_weights)
        if not path.is_file():
            raise FileNotFoundError(f"MobileSAM weights not found at {path}; this pipeline never downloads them")
        t = time.time()
        self.model = SAM(str(path))
        self.load_seconds = round(time.time() - t, 1)

    def refine(self, image_bgr: np.ndarray, boxes: list) -> list[dict]:
        """Per box: the tight mask box if the mask fills enough of the box and stays close to it."""
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
            o = iou(tight, box) if tight else 0.0
            if tight is None or fill < self.min_fill or o < self.min_iou:
                out.append({"sam_used": False, "sam_reason": f"mask fill {fill:.2f}, IoU {o:.2f}"})
            else:
                out.append({"sam_used": True, "sam_xyxy": [round(v, 1) for v in clip_box(tight, w, h)],
                            "sam_iou_with_original": round(o, 3)})
        return out
